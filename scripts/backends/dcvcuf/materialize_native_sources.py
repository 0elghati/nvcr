#!/usr/bin/env python3
"""Materialize pinned DCVC-UF fused sources with a Python-free C++ ABI."""

import argparse
from pathlib import Path
import shutil


def replace_once(text, old, new, source):
    count = text.count(old)
    if count != 1:
        raise ValueError(f"{source}: expected one occurrence, found {count}: {old!r}")
    return text.replace(old, new)


def materialize(reference_root, output):
    source = reference_root / "src/layers/extensions/inference"
    required = (
        "dmc_common.cpp", "dmc_common.h", "dmci_proxy.cpp", "dmci_proxy.h",
        "dmc_hts_proxy.cpp", "dmc_hts_proxy.h", "layers_proxy.cpp",
        "layers_proxy.h", "memory_pool.cpp", "memory_pool.h",
    )
    for name in required:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)

    final_output = output
    output = output.with_name(output.name + ".staging")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if any(part in {"build", "inference_extensions_cuda.egg-info", "__pycache__"}
               for part in relative.parts):
            continue
        if path.suffix not in {".h", ".hpp", ".cpp", ".cu"}:
            continue
        if path.name == "bind.cpp" or path.name.startswith(("dmc_htl_", "dmc_ld_")):
            continue
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = path.read_text()
        text = text.replace("#include <torch/extension.h>", "#include <torch/torch.h>")
        text = text.replace('#include "py_rans.h"', '#include "native_rans.hpp"')
        destination.write_text(text)


    image_h = output / "dmci_proxy.h"
    text = image_h.read_text()
    class_marker = "class DMCIProxy : public DMCCommon {"
    image_result = """struct DMCIEncodeResult final {
    std::vector<std::uint8_t> bit_stream;
    at::Tensor reconstruction;
    int entropy_coder_parallel{1};
};

"""
    text = replace_once(text, class_marker, image_result + class_marker, image_h)
    text = replace_once(
        text,
        "    py::tuple compress(const at::Tensor& x, const int qp, const int padding_b, const int padding_r);",
        "    DMCIEncodeResult compress(const at::Tensor& x, int qp, int padding_b, int padding_r);",
        image_h)
    text = replace_once(
        text,
        "    at::Tensor decompress(const py::array_t<uint8_t>& bit_stream, const int qp, const int height,\n"
        "                          const int width, const int entropy_coder_parallel);",
        "    at::Tensor decompress(const std::uint8_t* bit_stream, int bit_stream_size, int qp,\n"
        "                          int height, int width, int entropy_coder_parallel);",
        image_h)
    image_h.write_text(text)

    image_cpp = output / "dmci_proxy.cpp"
    text = image_cpp.read_text()
    text = replace_once(
        text,
        "py::tuple DMCIProxy::compress(const at::Tensor& x, const int qp, const int padding_b, const int padding_r)",
        "DMCIEncodeResult DMCIProxy::compress(const at::Tensor& x, int qp, int padding_b, int padding_r)",
        image_cpp)
    text = replace_once(
        text,
        "        return py::make_tuple(m_entropy_encoder.get_encoded_stream(), m_x_hat, m_entropy_coder_parallel);",
        "        return {m_entropy_encoder.get_encoded_stream(), m_x_hat, m_entropy_coder_parallel};",
        image_cpp)
    text = replace_once(
        text,
        "    return py::make_tuple(py::array_t<uint8_t>(0), at::Tensor(), 1);",
        "    return {};",
        image_cpp)
    text = replace_once(
        text,
        "at::Tensor DMCIProxy::decompress(const py::array_t<uint8_t>& bit_stream, const int qp,\n"
        "                                 const int height, const int width, const int entropy_coder_parallel)",
        "at::Tensor DMCIProxy::decompress(const std::uint8_t* bit_stream, int bit_stream_size, int qp,\n"
        "                                 int height, int width, int entropy_coder_parallel)",
        image_cpp)
    text = replace_once(
        text,
        "    py::buffer_info bit_stream_buf = bit_stream.request();\n"
        "    m_bit_stream = static_cast<uint8_t*>(bit_stream_buf.ptr);\n"
        "    m_bit_stream_size = static_cast<int>(bit_stream.size());",
        "    m_bit_stream = const_cast<std::uint8_t*>(bit_stream);\n"
        "    m_bit_stream_size = bit_stream_size;",
        image_cpp)
    image_cpp.write_text(text)

    video_h = output / "dmc_hts_proxy.h"
    text = video_h.read_text()
    class_marker = "class DMCHTSProxy : public DMCCommon {"
    video_result = """struct DMCHTSEncodeResult final {
    std::vector<std::uint8_t> bit_stream;
    int entropy_coder_parallel{1};
};

"""
    text = replace_once(text, class_marker, video_result + class_marker, video_h)
    text = replace_once(
        text,
        "    py::tuple compress(const at::Tensor& x, const int qp, const bool reset_feature_memory,\n"
        "                       const int padding_b, const int padding_r);",
        "    DMCHTSEncodeResult compress(const at::Tensor& x, int qp, bool reset_feature_memory,\n"
        "                                int padding_b, int padding_r);",
        video_h)
    text = replace_once(
        text,
        "    std::vector<at::Tensor> decompress(const py::array_t<uint8_t>& bit_stream, const int qp,\n"
        "                                       const int height, const int width,\n"
        "                                       const int entropy_coder_parallel,\n"
        "                                       const bool reset_feature_memory);",
        "    std::vector<at::Tensor> decompress(const std::uint8_t* bit_stream, int bit_stream_size,\n"
        "                                          int qp, int height, int width,\n"
        "                                          int entropy_coder_parallel,\n"
        "                                          bool reset_feature_memory);",
        video_h)
    # Read-only views and a side-effect-free reconstruction-head call for NVCR's
    # test-only G1 diagnostics (src/dcvcuf/diagnostics.hpp). Coding is unchanged.
    text = replace_once(
        text,
        "                                          bool reset_feature_memory);\n",
        "                                          bool reset_feature_memory);\n"
        "    at::Tensor nvcr_y_hat() const { return m_y_hat; }\n"
        "    at::Tensor nvcr_feature() const { return m_feature_p; }\n"
        "    at::Tensor nvcr_memory() const { return m_memory; }\n"
        "    at::Tensor nvcr_context() const { return m_ctx; }\n"
        "    std::vector<at::Tensor> nvcr_reconstruct(const at::Tensor& feature)\n"
        "    {\n"
        "        return std::get<0>(m_recon_head.forward(feature));\n"
        "    }\n",
        video_h)
    video_h.write_text(text)

    video_cpp = output / "dmc_hts_proxy.cpp"
    text = video_cpp.read_text()
    text = replace_once(
        text,
        "py::tuple DMCHTSProxy::compress(const at::Tensor& x, const int qp, const bool reset_feature_memory,\n"
        "                                const int padding_b, const int padding_r)",
        "DMCHTSEncodeResult DMCHTSProxy::compress(const at::Tensor& x, int qp, bool reset_feature_memory,\n"
        "                                         int padding_b, int padding_r)",
        video_cpp)
    text = replace_once(
        text,
        "        return py::make_tuple(m_entropy_encoder.get_encoded_stream(), m_entropy_coder_parallel);",
        "        return {m_entropy_encoder.get_encoded_stream(), m_entropy_coder_parallel};",
        video_cpp)
    text = replace_once(
        text,
        "    return py::make_tuple(py::array_t<uint8_t>(0), 1);",
        "    return {};",
        video_cpp)
    text = replace_once(
        text,
        "std::vector<at::Tensor> DMCHTSProxy::decompress(const py::array_t<uint8_t>& bit_stream,\n"
        "                                                const int qp, const int height, const int width,\n"
        "                                                const int entropy_coder_parallel,\n"
        "                                                const bool reset_feature_memory)",
        "std::vector<at::Tensor> DMCHTSProxy::decompress(const std::uint8_t* bit_stream,\n"
        "                                                int bit_stream_size, int qp, int height, int width,\n"
        "                                                int entropy_coder_parallel, bool reset_feature_memory)",
        video_cpp)
    text = replace_once(
        text,
        "    py::buffer_info bit_stream_buf = bit_stream.request();\n"
        "    m_bit_stream = static_cast<uint8_t*>(bit_stream_buf.ptr);\n"
        "    m_bit_stream_size = static_cast<int>(bit_stream.size());",
        "    m_bit_stream = const_cast<std::uint8_t*>(bit_stream);\n"
        "    m_bit_stream_size = bit_stream_size;",
        video_cpp)
    video_cpp.write_text(text)

    remaining = []
    for path in output.rglob("*"):
        if path.is_file() and path.suffix in {".h", ".hpp", ".cpp", ".cu"}:
            text = path.read_text()
            if "py::" in text or "pybind" in text or "Python.h" in text:
                remaining.append(str(path.relative_to(output)))
    if remaining:
        raise ValueError(f"Python ABI remains in materialized sources: {remaining}")

    final_output.mkdir(parents=True, exist_ok=True)
    staged = {path.relative_to(output) for path in output.rglob("*") if path.is_file()}
    for destination in [path for path in final_output.rglob("*") if path.is_file()]:
        if destination.relative_to(final_output) not in staged:
            destination.unlink()
    for source_path in [path for path in output.rglob("*") if path.is_file()]:
        destination = final_output / source_path.relative_to(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = source_path.read_bytes()
        if not destination.exists() or destination.read_bytes() != data:
            destination.write_bytes(data)
    shutil.rmtree(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.reference_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
