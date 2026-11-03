#!/usr/bin/env python3
"""Replay native HT-S decoder features through NVCR's IProviderSession boundary."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys

from probe_hts_export import sha256, verify_reference

REPO = Path(__file__).resolve().parents[3]


def capture_sources(root, output):
    """Rename only the full proxy; add read-only snapshots after decompression.

    Supporting operators still link to the supplied unmodified upstream binary.
    The renamed class avoids changing the definition/ABI of that binary's class.
    No arithmetic, entropy, graph, or state-transition code is edited.
    """
    source = root / "src/layers/extensions/inference"
    header = (source / "dmc_hts_proxy.h").read_text()
    implementation = (source / "dmc_hts_proxy.cpp").read_text()
    # The capture class only decodes. Omit the unused encoder entry point,
    # whose Python-array return helper is hidden in the upstream shared library.
    start = implementation.index("py::tuple DMCHTSProxy::compress(")
    end = implementation.index("std::vector<at::Tensor> DMCHTSProxy::decompress(", start)
    implementation = implementation[:start] + implementation[end:]
    marker = "    DMCHTSProxy();"
    if header.count(marker) != 1 or implementation.count("DMCHTSProxy::DMCHTSProxy()") != 1:
        raise ValueError("unexpected pinned proxy source layout")
    header = header.replace(marker, marker + "\n"
        "    std::vector<at::Tensor> snapshot() const {\n"
        "        return {m_feature_p.clone(), m_feature_i.clone(), m_y_hat.clone(),\n"
        "                m_ctx.clone(), m_q_decoder[m_qp].clone()};\n"
        "    }\n"
        "    std::vector<at::Tensor> entropy_snapshot() const {\n"
        "        auto indexes = at::from_blob(m_y_to_decode->data(), {*m_size_ptr},\n"
        "            at::TensorOptions().dtype(at::kByte)).clone();\n"
        "        return {indexes, m_y_decoded.narrow(0, 0, *m_size_ptr).cpu().clone(),\n"
        "                m_z_hat_io.permute({0, 2, 3, 1}).contiguous().flatten().cpu().clone()};\n"
        "    }\n").replace("DMCHTSProxy", "NVCRCaptureHTSProxy")
    implementation = ('// Copyright (c) Microsoft Corporation. MIT License.\n'
                      '#include <c10/cuda/CUDAGuard.h>\n#include <torch/torch.h>\n'
                      '#include "common_cpp.h"\n#include "cuda_check.h"\n'
                      '#include "def_cutlass.h"\n#include "def_elementwise.h"\n'
                      '#include "capture_hts.h"\n#include "memory_pool.h"\n' +
                      implementation[implementation.index("DMCHTSProxy::DMCHTSProxy()"):]
                      .replace("DMCHTSProxy", "NVCRCaptureHTSProxy"))
    (output / "capture_hts.h").write_text(header)
    (output / "capture_hts.cpp").write_text(implementation)


def load_binding(root, extension, output, nvcr_build):
    from torch.utils.cpp_extension import CUDA_HOME, load
    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME is required")
    build = output / "binding"
    build.mkdir()
    capture_sources(root, build)
    backend = REPO / "src/dcvcuf/backend/upstream"
    entropy = REPO / "src/dcvcuf/entropy"
    rans = REPO / "third_party/dcvcuf_rans/src"
    return load(name="nvcr_dcvcuf_session_probe", sources=[
        str(Path(__file__).with_name("session_probe.cpp")),
        str(backend / "recon_session.cpp"), str(REPO / "src/common/error.cpp"),
        str(REPO / "src/common/sha256.cpp"),
        str(entropy / "uf_rans_codec.cpp"), str(rans / "rans.cpp"),
        str(build / "capture_hts.cpp")], build_directory=str(build),
        extra_include_paths=[str(build), str(backend), str(entropy), str(rans),
                             str(REPO / "src/common"), str(REPO / "include"),
                             str(nvcr_build / "generated/include"),
                             str(root / "src/layers/extensions/inference"),
                             str(root / "src/cpp/py_rans"), str(Path(CUDA_HOME) / "include")],
        extra_cflags=["-O2", "-std=c++20"], extra_ldflags=[str(extension)], with_cuda=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-root", type=Path, required=True)
    for name in ("image-checkpoint", "checkpoint", "extension", "stream"):
        p.add_argument(f"--{name}", type=Path, required=True)
        p.add_argument(f"--{name}-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--nvcr-build", type=Path, default=REPO / "build-cpu",
                   help="Configured NVCR build supplying generated version headers")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = args.reference_root.resolve()
    revision = verify_reference(root)
    for name in ("image_checkpoint", "checkpoint", "extension", "stream"):
        path = getattr(args, name).resolve()
        if sha256(path) != getattr(args, name + "_sha256"):
            raise ValueError(f"SHA-256 mismatch: {path}")
        setattr(args, name, path)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from export_weights_artifact import export_state_dict
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no fallback")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(args.extension.parent))
    import inference_extensions_cuda as upstream
    if Path(upstream.__file__).resolve() != args.extension:
        raise ValueError("imported upstream binary does not match verified extension")
    from src.models.image_model import DMCI
    from src.models.video_model_ht import DMC
    from src.utils.common import get_state_dict, set_torch_env
    from src.utils.stream_helper import (SPSHelper, NalType, read_header,
                                         read_sps_remaining, read_ip_remaining)
    # Match the reference worker's full setup, including deterministic CUBLAS
    # and cuDNN selection, before creating streams or executing model operations.
    set_torch_env()
    # The pinned CUDA proxy uses empty scratch tensors. Keep deterministic
    # initialization enabled so every native reference decode writes the same
    # complete tensor before the provider stages consume it.
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.cuda.set_stream(torch.cuda.Stream())
    image_model = DMCI().eval()
    image_model.load_state_dict(get_state_dict(args.image_checkpoint))
    image_model.update(0)
    image_model = image_model.half().cuda().to(memory_format=torch.channels_last)
    image_model.proxy = upstream.DMCIProxy()
    image_model.proxy.set_param(image_model.add_cdf_to_state_dict(image_model.state_dict()), 0)
    video_model = DMC().eval()
    video_model.load_state_dict(get_state_dict(args.checkpoint))
    video_model.update(0)
    video_model = video_model.half().cuda().to(memory_format=torch.channels_last)
    state = video_model.add_cdf_to_state_dict(video_model.state_dict())
    z_cdf = state["bit_estimator_z.quantized_cdf"].cpu().contiguous()
    z_cdf_sizes = state["bit_estimator_z.cdf_length"].cpu().contiguous()
    y_cdf = state["gaussian_encoder.quantized_cdf"].cpu().contiguous()
    y_cdf_sizes = state["gaussian_encoder.cdf_length"].cpu().contiguous()
    stage_state = {name: value for name, value in state.items()
                   if name.startswith(("decoder.", "recon_head."))}
    artifact = export_state_dict(
        stage_state, output / "artifacts/cvpr2026_video_hts_decode.pt")
    binding = load_binding(root, args.extension, output, args.nvcr_build.resolve())
    original = upstream.DMCHTSProxy()
    original.set_param(state, 0)
    capture = binding.CaptureHTS()
    capture.set_param(state, 0)
    stream = io.BytesIO(args.stream.read_bytes())
    sps_helper = SPSHelper()
    checks = []
    arrays = {}
    for name, tensor in (("z_cdf", z_cdf), ("z_cdf_sizes", z_cdf_sizes),
                         ("y_cdf", y_cdf), ("y_cdf_sizes", y_cdf_sizes)):
        arrays[name] = tensor.numpy()
    intra_replays = 0
    session = other = None
    with torch.inference_mode():
        while stream.tell() < len(stream.getbuffer()):
            header = read_header(stream)
            if header['nal_type'] == NalType.NAL_SPS:
                sps_helper.add_sps_by_id(read_sps_remaining(stream, header['sps_id']))
                continue
            sps = sps_helper.get_sps_by_id(header['sps_id'])
            qp, ec_part, reset, payload = read_ip_remaining(stream)
            if header['nal_type'] == NalType.NAL_I:
                torch.cuda.synchronize()
                decoded = image_model.decompress(payload, sps, qp, ec_part)['x_hat'].clone()
                torch.cuda.synchronize()
                repeated_intra = image_model.decompress(payload, sps, qp, ec_part)['x_hat'].clone()
                torch.cuda.synchronize()
                if not torch.equal(decoded, repeated_intra):
                    raise ValueError("native I-frame repeat decode differs; stream/library pair is not a stable reference")
                arrays[f"intra{intra_replays}_frame"] = decoded.cpu().numpy()
                intra_replays += 1
                original.add_ref_feature_from_frame(decoded, False)
                capture.add_ref_feature_from_frame(decoded, False)
                continue
            if header['nal_type'] != NalType.NAL_P:
                raise ValueError("unsupported native NAL")
            encoded = np.frombuffer(payload, dtype=np.uint8).copy()
            torch.cuda.synchronize()
            expected = [x.clone() for x in original.decompress(
                encoded, qp, sps['height'], sps['width'], ec_part, reset)]
            torch.cuda.synchronize()
            captured_frames = [x.clone() for x in capture.decompress(
                encoded, qp, sps['height'], sps['width'], ec_part, reset)]
            feature, reset_feature, latent, context, quant = capture.snapshot()
            y_indexes, expected_y, expected_z = capture.entropy_snapshot()
            actual_z, actual_y = binding.decode_entropy(
                z_cdf, z_cdf_sizes, y_cdf, y_cdf_sizes,
                torch.from_numpy(encoded), y_indexes, ec_part, qp, 128,
                (sps['height'] + 63) // 64, (sps['width'] + 63) // 64)
            torch.cuda.synchronize()
            if session is None:
                session = binding.SessionProbe(artifact["path"], artifact["sha256"],
                                               feature.shape[2], feature.shape[3], 0)
                other = binding.SessionProbe(artifact["path"], artifact["sha256"],
                                             feature.shape[2], feature.shape[3], 0)
            value = feature.permute(0, 2, 3, 1).contiguous().cpu()
            preserved, frames, actual_reset = session.run(value)
            expected_frames = torch.cat(expected).permute(0, 2, 3, 1).contiguous().cpu()
            expected_reset = reset_feature.permute(0, 2, 3, 1).contiguous().cpu()
            session.reset()
            repeated = session.run(value)
            independent = other.run(value)
            synthesis_inputs = [x.permute(0, 2, 3, 1).contiguous().cpu()
                                for x in (latent, context, quant)]
            pipeline = session.run_pipeline(*synthesis_inputs)
            session.reset()
            pipeline_repeated = session.run_pipeline(*synthesis_inputs)
            pipeline_independent = other.run_pipeline(*synthesis_inputs)
            check = {
                "capture_matches_original": all(torch.equal(a, b) for a, b in
                                                zip(expected, captured_frames, strict=True)),
                "frames_exact": torch.equal(frames, expected_frames),
                "reset_exact": torch.equal(actual_reset, expected_reset),
                "input_preserved": torch.equal(preserved, value),
                "reset_reuse_exact": all(torch.equal(a, b) for a, b in
                                          zip((preserved, frames, actual_reset), repeated, strict=True)),
                "independent_session_exact": all(torch.equal(a, b) for a, b in
                                                  zip(repeated, independent, strict=True)),
            }
            check.update({
                "decoder_inputs_preserved": all(torch.equal(a, b) for a, b in
                                                zip(synthesis_inputs, pipeline[:3], strict=True)),
                "decoder_feature_exact": torch.equal(pipeline[3], value),
                "pipeline_frames_exact": torch.equal(pipeline[4], expected_frames),
                "pipeline_reset_exact": torch.equal(pipeline[5], expected_reset),
                "entropy_z_exact": torch.equal(actual_z, expected_z),
                "entropy_y_exact": torch.equal(actual_y, expected_y),
                "pipeline_reset_reuse_exact": all(torch.equal(a, b) for a, b in
                                                 zip(pipeline, pipeline_repeated, strict=True)),
                "pipeline_independent_session_exact": all(torch.equal(a, b) for a, b in
                    zip(pipeline, pipeline_independent, strict=True)),
            })
            index = len(checks)
            for name, tensor in (("feature", value), ("frames", expected_frames),
                                 ("reset", expected_reset), ("nvcr_frames", frames),
                                 ("nvcr_reset", actual_reset),
                                 ("latent", synthesis_inputs[0]), ("context", synthesis_inputs[1]),
                                 ("quant", synthesis_inputs[2]), ("nvcr_feature", pipeline[3]),
                                 ("y_indexes", y_indexes), ("y_symbols", expected_y),
                                 ("z_symbols", expected_z),
                                 ("pipeline_frames", pipeline[4]), ("pipeline_reset", pipeline[5])):
                arrays[f"chunk{index}_{name}"] = tensor.numpy()
            checks.append({"qp": qp, "reset_feature_memory": reset,
                           "feature_shape_nhwc": list(value.shape), "checks": check})
        if not checks:
            raise ValueError("native stream must contain a predicted chunk")
        rejection_checks = session.rejection_checks(other)
    np.savez(output / "vectors.npz", **arrays)
    report = {
        "scope": "experimental_nvcr_provider_session_hts_synthesis",
        "reference_commit": revision, "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "execution_setup": "upstream src.utils.common.set_torch_env",
        "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING", "0"),
        "session_binary_sha256": sha256(Path(binding.__file__)),
        "stage_artifact": artifact,
        "generated_versions_sha256": sha256(
            args.nvcr_build / "generated/include/nvcr/common/versions.hpp"),
        "inputs": {name: {"path": str(getattr(args, name)),
                          "sha256": getattr(args, name + "_sha256")}
                   for name in ("image_checkpoint", "checkpoint", "extension", "stream")},
        "source_sha256": {str(path.relative_to(REPO)): sha256(path) for path in (
            Path(__file__), Path(__file__).with_name("session_probe.cpp"),
            Path(__file__).with_name("export_weights_artifact.py"),
            REPO / "src/dcvcuf/backend/upstream/recon_session.cpp",
            REPO / "src/dcvcuf/backend/upstream/recon_session.hpp",
            REPO / "src/common/sha256.cpp", REPO / "src/common/sha256.hpp",
            REPO / "src/dcvcuf/entropy/uf_rans_codec.cpp",
            REPO / "src/dcvcuf/entropy/uf_rans_codec.hpp",
            REPO / "third_party/dcvcuf_rans/src/rans.cpp",
            REPO / "third_party/dcvcuf_rans/src/rans.hpp",
            REPO / "include/nvcr/provider/experimental/session.hpp")},
        "capture_source_sha256": {name: sha256(output / "binding" / name)
                                  for name in ("capture_hts.h", "capture_hts.cpp")},
        "vectors_sha256": sha256(output / "vectors.npz"),
        "tensor_sha256": {name: hashlib.sha256(value.tobytes()).hexdigest()
                          for name, value in arrays.items()},
        "chunks": checks, "contract_checks": rejection_checks,
        "intra_repeat_checks_passed": intra_replays,
        "acceptance": "exact equality for native entropy symbols, decoder feature, reconstructed frames, reset feature, and preserved inputs",
        "passed": all(all(item['checks'].values()) for item in checks),
        "c7_complete": False,
        "limitations": ["entropy decode, decoder, and reconstruction stages only; no registered codec or provider",
                        "synchronous Torch/upstream bridge, not a standalone deployment",
                        "remaining image/context/prior/encoder stages and codec lifecycle still pending"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == "__main__":
    sys.exit(main())
