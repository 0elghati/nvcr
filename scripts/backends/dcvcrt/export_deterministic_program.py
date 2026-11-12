#!/usr/bin/env python3
"""Convert an exported DCVC-RT ONNX stage into an NVCR deterministic program.

The deterministic executor (src/deterministic) runs a small, fixed op set with
an exactly specified arithmetic order so that every target computes bit-identical
results (issue #216). This tool lowers one ONNX stage into that op set:

  conv            dense or depthwise convolution with bias
  mul, add        same-shape elementwise
  mul_scalar      elementwise by a constant
  scale_channels  elementwise by a [1, C, 1, 1] graph input (quantization steps)
  sigmoid
  clamp           Clip with constant bounds
  slice_channels  a static channel range
  concat          two tensors along channels
  crop            Slice of H or W from 0 to another tensor's H or W
  depth_to_space  CRD mode (PixelShuffle)
  im2col          receptive fields of a dense spatial conv, which becomes pointwise
  space_to_depth  the Unsqueeze/Reshape/Transpose/Squeeze chain of PixelUnshuffle

Integer shape subgraphs (Shape/Gather/Add/Div/Mul feeding Slice) are folded into
static channel ranges; the tool evaluates them at two frame sizes and rejects
any range that depends on the frame size. Anything else is rejected rather than
approximated. Output: <name>.detprog.json and <name>.detw (raw float16 weights).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

FORMAT = "nvcr.deterministic-program"
VERSION = 1
NUMERICS = "nvcr.intconv.v1"
PROBE_SIZES = ((7, 11), (13, 5))


class ExportError(RuntimeError):
    pass


# Stage-local module prefixes -> pinned DCVC-RT module names, per stage. Layer
# names in an integer map are "I.<module path>" (DMCI) or "P.<module path>"
# (DMC).
STAGE_MODULES = {
    "i_hyper_synthesis": ("I", {"hyper_dec": "hyper_dec", "prior_fusion": "y_prior_fusion",
                                "prior_reduction": "y_spatial_prior_reduction"}),
    "i_spatial_prior_1": ("I", {"adaptor": "y_spatial_prior_adaptor_1", "prior": "y_spatial_prior"}),
    "i_spatial_prior_2": ("I", {"adaptor": "y_spatial_prior_adaptor_2", "prior": "y_spatial_prior"}),
    "i_spatial_prior_3": ("I", {"adaptor": "y_spatial_prior_adaptor_3", "prior": "y_spatial_prior"}),
    "i_synthesis": ("I", {"": "dec"}),
    "p_reference_frame": ("P", {"adaptor": "feature_adaptor_i", "extractor": "feature_extractor"}),
    "p_reference_feature": ("P", {"adaptor": "feature_adaptor_p", "extractor": "feature_extractor"}),
    "p_prior": ("P", {"hyper_decoder": "hyper_decoder", "temporal": "temporal_prior_encoder",
                      "fusion": "y_prior_fusion"}),
    "p_spatial_prior": ("P", {"": "y_spatial_prior"}),
    "p_synthesis": ("P", {"decoder": "decoder", "reconstruction": "recon_generation_net"}),
}


def reference_layer(stage: str, weight_name: str) -> str:
    """Pinned DCVC-RT layer name for a stage-local conv weight name."""
    layer = weight_name.removesuffix(".weight")
    if stage not in STAGE_MODULES:
        raise ExportError(f"no layer naming for stage {stage}")
    model, modules = STAGE_MODULES[stage]
    for local, reference in modules.items():
        if not local:
            return f"{model}.{reference}.{layer}"
        if layer == local or layer.startswith(local + "."):
            return f"{model}.{reference}{layer[len(local):]}"
    raise ExportError(f"{stage} weight {weight_name} has no pinned module")


def integer_widths(spec: str) -> dict:
    """w<bits>a<bits> to a program node's integer widths."""
    import re
    match = re.fullmatch(r"w(8|16)a(8|16)", spec)
    if match is None:
        raise ExportError(f"unsupported integer widths {spec}")
    return {"weight_bits": int(match[1]), "activation_bits": int(match[2])}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def attributes(node: onnx.NodeProto) -> dict:
    return {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}


class Lowering:
    def __init__(self, model: onnx.ModelProto) -> None:
        self.graph = model.graph
        self.initializers = {
            item.name: numpy_helper.to_array(item) for item in self.graph.initializer}
        self.channels: dict[str, int] = {}
        self.scalars: dict[str, float] = {}
        self.integer_nodes: dict[str, onnx.NodeProto] = {}
        self.names: dict[str, str] = {}
        self.weights: list[np.ndarray] = []
        self.weight_count = 0
        self.nodes: list[dict] = []
        self.unit_spatial: set[str] = set()
        self.consumed: set[int] = set()
        self.producers = {output: node for node in self.graph.node for output in node.output}
        self.consumers: dict[str, list[int]] = {}
        for index, node in enumerate(self.graph.node):
            for item in node.input:
                self.consumers.setdefault(item, []).append(index)

    def name(self, onnx_name: str) -> str:
        if onnx_name not in self.names:
            self.names[onnx_name] = f"t{len(self.names)}"
        return self.names[onnx_name]

    def add_weight(self, values: np.ndarray) -> int:
        offset = self.weight_count
        flat = np.ascontiguousarray(values, dtype=np.float16).reshape(-1)
        self.weights.append(flat)
        self.weight_count += flat.size
        return offset

    # Integer shape arithmetic, evaluated for one probe frame size.
    def evaluate_integer(self, tensor: str, height: int, width: int, cache: dict) -> np.ndarray:
        if tensor in cache:
            return cache[tensor]
        if tensor in self.initializers:
            return self.initializers[tensor].astype(np.int64)
        node = self.integer_nodes.get(tensor)
        if node is None:
            raise ExportError(f"integer value {tensor} has no supported producer")
        attrs = attributes(node)
        inputs = [self.evaluate_integer(item, height, width, cache)
                  for item in node.input if node.op_type != "Shape"]
        if node.op_type == "Constant":
            value = numpy_helper.to_array(attrs["value"]).astype(np.int64)
        elif node.op_type == "Shape":
            source = node.input[0]
            if source not in self.channels:
                raise ExportError(f"Shape of unknown tensor {source}")
            value = np.array([1, self.channels[source], height, width], dtype=np.int64)
        elif node.op_type == "Gather":
            value = np.take(inputs[0], inputs[1], axis=int(attrs.get("axis", 0)))
        elif node.op_type == "Add":
            value = inputs[0] + inputs[1]
        elif node.op_type == "Sub":
            value = inputs[0] - inputs[1]
        elif node.op_type == "Mul":
            value = inputs[0] * inputs[1]
        elif node.op_type == "Div":
            value = np.floor_divide(inputs[0], inputs[1])
        elif node.op_type == "Unsqueeze":
            value = np.reshape(inputs[0], (-1,))
        elif node.op_type == "Concat":
            value = np.concatenate([np.reshape(item, (-1,)) for item in inputs])
        else:
            raise ExportError(f"unsupported integer op {node.op_type}")
        cache[tensor] = np.asarray(value, dtype=np.int64)
        return cache[tensor]

    def static_channel_slice(self, node: onnx.NodeProto) -> tuple[int, int]:
        ranges = set()
        for height, width in PROBE_SIZES:
            cache: dict = {}
            starts = self.evaluate_integer(node.input[1], height, width, cache).reshape(-1)
            ends = self.evaluate_integer(node.input[2], height, width, cache).reshape(-1)
            axes = (self.evaluate_integer(node.input[3], height, width, cache).reshape(-1)
                    if len(node.input) > 3 and node.input[3] else np.arange(starts.size))
            steps = (self.evaluate_integer(node.input[4], height, width, cache).reshape(-1)
                     if len(node.input) > 4 and node.input[4] else np.ones(starts.size))
            if axes.tolist() != [1] or steps.tolist() != [1]:
                raise ExportError(f"Slice {node.name} is not a unit-step channel slice")
            channels = self.channels[node.input[0]]
            begin = int(min(max(starts[0], 0), channels))
            end = int(min(max(ends[0], 0), channels))
            ranges.add((begin, end))
        if len(ranges) != 1:
            raise ExportError(f"Slice {node.name} depends on the frame size")
        return ranges.pop()

    def is_integer_value(self, tensor: str) -> bool:
        return tensor in self.integer_nodes or (
            tensor in self.initializers and self.initializers[tensor].dtype.kind in "iu")

    def lower(self) -> None:
        for graph_input in self.graph.input:
            dims = graph_input.type.tensor_type.shape.dim
            if len(dims) != 4 or dims[0].dim_value != 1 or dims[1].dim_value <= 0:
                raise ExportError(f"input {graph_input.name} must be [1, C, H, W] with static C")
            self.channels[graph_input.name] = dims[1].dim_value
            self.names[graph_input.name] = graph_input.name
            if dims[2].dim_value == 1 and dims[3].dim_value == 1:
                self.unit_spatial.add(graph_input.name)
        for graph_output in self.graph.output:
            self.names[graph_output.name] = graph_output.name

        for index, node in enumerate(self.graph.node):
            if index in self.consumed:
                continue
            attrs = attributes(node)
            op = node.op_type
            if op == "Unsqueeze" and node.input[0] in self.channels:
                self.lower_space_to_depth(index)
                continue
            if op == "Constant":
                value = numpy_helper.to_array(attrs["value"])
                if value.dtype.kind == "f":
                    if value.size != 1:
                        raise ExportError(f"non-scalar float Constant {node.name}")
                    self.scalars[node.output[0]] = float(value.reshape(-1)[0])
                else:
                    self.integer_nodes[node.output[0]] = node
                continue
            if op == "Shape" or (op in ("Gather", "Div", "Unsqueeze", "Concat", "Sub")
                                 and all(self.is_integer_value(item) for item in node.input)):
                self.integer_nodes[node.output[0]] = node
                continue
            if op in ("Add", "Mul") and all(self.is_integer_value(item) for item in node.input):
                self.integer_nodes[node.output[0]] = node
                continue
            self.lower_float(node, op, attrs)

        missing = [item.name for item in self.graph.output if item.name not in self.channels]
        if missing:
            raise ExportError(f"outputs were not produced: {missing}")

    def constant_ints(self, tensor: str) -> list[int]:
        producer = self.producers.get(tensor)
        if producer is not None and producer.op_type == "Constant":
            return numpy_helper.to_array(attributes(producer)["value"]).astype(np.int64).reshape(-1).tolist()
        return self.evaluate_integer(tensor, *PROBE_SIZES[0], {}).reshape(-1).tolist()

    def sole_consumer(self, tensor: str, op: str) -> tuple[int, onnx.NodeProto]:
        consumers = self.consumers.get(tensor, [])
        if len(consumers) != 1 or self.graph.node[consumers[0]].op_type != op:
            raise ExportError(f"{tensor} does not feed exactly one {op}")
        return consumers[0], self.graph.node[consumers[0]]

    def lower_space_to_depth(self, index: int) -> None:
        """PixelUnshuffle(b) as exported by PyTorch:
        Unsqueeze(3) -> Reshape[0,0,-1,b,0] -> Reshape[0,0,0,0,-1,b]
        -> Transpose[0,1,3,5,2,4] -> Reshape[0,-1,1,1,0,0] -> Squeeze(2,3)."""
        unsqueeze = self.graph.node[index]
        source = unsqueeze.input[0]
        if self.constant_ints(unsqueeze.input[1]) != [3]:
            raise ExportError(f"unsupported Unsqueeze {unsqueeze.name}")
        chain = [index]
        first_index, first = self.sole_consumer(unsqueeze.output[0], "Reshape")
        second_index, second = self.sole_consumer(first.output[0], "Reshape")
        transpose_index, transpose = self.sole_consumer(second.output[0], "Transpose")
        third_index, third = self.sole_consumer(transpose.output[0], "Reshape")
        squeeze_index, squeeze = self.sole_consumer(third.output[0], "Squeeze")
        chain += [first_index, second_index, transpose_index, third_index, squeeze_index]
        shape1 = self.constant_ints(first.input[1])
        block = shape1[3] if len(shape1) == 5 else 0
        if (block <= 1 or shape1 != [0, 0, -1, block, 0]
                or self.constant_ints(second.input[1]) != [0, 0, 0, 0, -1, block]
                or attributes(transpose).get("perm") != [0, 1, 3, 5, 2, 4]
                or self.constant_ints(third.input[1]) != [0, -1, 1, 1, 0, 0]
                or self.constant_ints(squeeze.input[1]) != [2, 3]):
            raise ExportError(f"Unsqueeze {unsqueeze.name} does not start a PixelUnshuffle")
        output = squeeze.output[0]
        self.nodes.append({"op": "space_to_depth", "input": self.name(source),
                           "output": self.name(output), "block": block})
        self.channels[output] = self.channels[source] * block * block
        self.consumed.update(chain)

    def shape_source(self, tensor: str) -> tuple[str, int] | None:
        """(T, k) when tensor is Unsqueeze(Gather(Shape(T), k))."""
        node = self.producers.get(tensor)
        if node is None or node.op_type != "Unsqueeze":
            return None
        gather = self.producers.get(node.input[0])
        if gather is None or gather.op_type != "Gather":
            return None
        shape = self.producers.get(gather.input[0])
        if shape is None or shape.op_type != "Shape" or shape.input[0] not in self.channels:
            return None
        index = self.constant_ints(gather.input[1])
        return (shape.input[0], index[0]) if len(index) == 1 else None

    def spatial_crop(self, node: onnx.NodeProto) -> tuple[int, str] | None:
        """(axis, reference) for x[:, :, :ref.H] or x[:, :, :, :ref.W]."""
        if len(node.input) < 4 or not node.input[3]:
            return None
        axes = self.constant_ints(node.input[3])
        if axes not in ([2], [3]):
            return None
        steps = self.constant_ints(node.input[4]) if len(node.input) > 4 and node.input[4] else [1]
        source = self.shape_source(node.input[2])
        if self.constant_ints(node.input[1]) != [0] or steps != [1] or source is None \
                or source[1] != axes[0]:
            raise ExportError(f"Slice {node.name} is not a crop to another tensor's size")
        return axes[0], source[0]

    def lower_float(self, node: onnx.NodeProto, op: str, attrs: dict) -> None:
        output = node.output[0]
        if op == "Conv":
            weight = self.initializers.get(node.input[1])
            bias = self.initializers.get(node.input[2]) if len(node.input) > 2 else None
            if weight is None or bias is None:
                raise ExportError(f"Conv {node.name} needs constant weight and bias")
            if any(item != 1 for item in attrs.get("dilations", [1, 1])):
                raise ExportError(f"Conv {node.name} uses dilation")
            groups = int(attrs.get("group", 1))
            in_channels = self.channels[node.input[0]]
            out_channels = int(weight.shape[0])
            if groups not in (1, in_channels) or (groups == in_channels and out_channels != in_channels):
                raise ExportError(f"Conv {node.name} must be dense or depthwise")
            kernel = [int(item) for item in weight.shape[2:]]
            stride = [int(item) for item in attrs.get("strides", [1, 1])]
            pads = [int(item) for item in attrs.get("pads", [0, 0, 0, 0])]
            source = self.name(node.input[0])
            if groups == 1 and (kernel != [1, 1] or stride != [1, 1] or pads != [0, 0, 0, 0]):
                # A dense spatial conv is im2col plus a pointwise conv over the
                # same [O][C][kh][kw] weights: the fma order is unchanged, and
                # The integer conv quantizes each receptive field with one scale.
                columns = f"{node.output[0]}::im2col"
                self.nodes.append({
                    "op": "im2col", "input": source, "output": self.name(columns),
                    "kernel": kernel, "stride": stride, "pads": pads})
                source = self.name(columns)
                in_channels *= kernel[0] * kernel[1]
                kernel, stride, pads = [1, 1], [1, 1], [0, 0, 0, 0]
            self.nodes.append({
                "op": "conv",
                "input": source,
                "output": self.name(output),
                "in_channels": in_channels,
                "out_channels": out_channels,
                "depthwise": groups > 1,
                "kernel": kernel,
                "stride": stride,
                "pads": pads,
                "weight_name": node.input[1],
                "weight_offset": self.add_weight(weight),
                "bias_offset": self.add_weight(bias),
            })
            self.channels[output] = out_channels
        elif op in ("Mul", "Add"):
            left, right = node.input
            scalar = [item for item in (left, right) if item in self.scalars]
            unit = [item for item in (left, right) if item in self.unit_spatial]
            if op == "Mul" and len(unit) == 1 and not scalar:
                tensor = right if left in self.unit_spatial else left
                if self.channels[tensor] != self.channels[unit[0]]:
                    raise ExportError(f"Mul {node.name} scales with a mismatched channel count")
                self.nodes.append({
                    "op": "scale_channels", "inputs": [self.name(tensor), self.name(unit[0])],
                    "output": self.name(output)})
                self.channels[output] = self.channels[tensor]
            elif scalar and op == "Mul":
                tensor = right if left in self.scalars else left
                self.nodes.append({
                    "op": "mul_scalar", "input": self.name(tensor), "output": self.name(output),
                    "value": self.scalars[scalar[0]]})
                self.channels[output] = self.channels[tensor]
            elif scalar:
                raise ExportError(f"Add with a scalar constant in {node.name}")
            else:
                if self.channels[left] != self.channels[right]:
                    raise ExportError(f"{op} {node.name} broadcasts across channels")
                self.nodes.append({
                    "op": op.lower(), "inputs": [self.name(left), self.name(right)],
                    "output": self.name(output)})
                self.channels[output] = self.channels[left]
        elif op == "Sigmoid":
            self.nodes.append({
                "op": "sigmoid", "input": self.name(node.input[0]), "output": self.name(output)})
            self.channels[output] = self.channels[node.input[0]]
        elif op == "Slice" and self.spatial_crop(node) is not None:
            axis, reference = self.spatial_crop(node)
            self.nodes.append({
                "op": "crop", "inputs": [self.name(node.input[0]), self.name(reference)],
                "output": self.name(output), "axis": axis})
            self.channels[output] = self.channels[node.input[0]]
        elif op == "Concat":
            if int(attrs.get("axis", 0)) != 1 or len(node.input) != 2:
                raise ExportError(f"Concat {node.name} is not a two-tensor channel concat")
            left, right = node.input
            self.nodes.append({
                "op": "concat", "inputs": [self.name(left), self.name(right)],
                "output": self.name(output)})
            self.channels[output] = self.channels[left] + self.channels[right]
        elif op == "Clip":
            bounds = [self.scalars.get(item) for item in node.input[1:3]]
            if len(bounds) != 2 or None in bounds:
                raise ExportError(f"Clip {node.name} needs constant bounds")
            self.nodes.append({
                "op": "clamp", "input": self.name(node.input[0]), "output": self.name(output),
                "min": bounds[0], "max": bounds[1]})
            self.channels[output] = self.channels[node.input[0]]
        elif op == "Slice":
            begin, end = self.static_channel_slice(node)
            self.nodes.append({
                "op": "slice_channels", "input": self.name(node.input[0]),
                "output": self.name(output), "begin": begin, "end": end})
            self.channels[output] = end - begin
        elif op == "DepthToSpace":
            mode = attrs.get("mode", b"DCR")
            mode = mode.decode() if isinstance(mode, bytes) else mode
            block = int(attrs["blocksize"])
            if mode != "CRD":
                raise ExportError(f"DepthToSpace {node.name} uses {mode}; only CRD is supported")
            channels = self.channels[node.input[0]]
            if channels % (block * block):
                raise ExportError(f"DepthToSpace {node.name} channel count is not divisible")
            self.nodes.append({
                "op": "depth_to_space", "input": self.name(node.input[0]),
                "output": self.name(output), "block": block})
            self.channels[output] = channels // (block * block)
        else:
            raise ExportError(f"unsupported op {op} ({node.name})")


def export(onnx_path: Path, output_dir: Path, name: str, integer_map: dict | None = None) -> dict:
    lowering = Lowering(onnx.load(str(onnx_path)))
    lowering.lower()
    if integer_map is not None:
        default = integer_map.get("default", "w16a16")
        layers = integer_map.get("layers", {})
        for node in lowering.nodes:
            pointwise = (node["op"] == "conv" and not node["depthwise"] and node["kernel"] == [1, 1]
                         and node["stride"] == [1, 1] and node["pads"] == [0, 0, 0, 0])
            if pointwise:
                layer = reference_layer(name, node["weight_name"])
                node["layer"] = layer
                node["integer"] = integer_widths(layers.get(layer, default))
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / f"{name}.detw"
    blob = np.concatenate(lowering.weights) if lowering.weights else np.zeros(0, np.float16)
    weights_path.write_bytes(blob.astype("<f2").tobytes())
    program = {
        "format": FORMAT,
        "version": VERSION,
        "numerics": NUMERICS,
        "name": name,
        "source": {"onnx": onnx_path.name, "onnx_sha256": sha256(onnx_path)},
        "weights": {
            "file": weights_path.name,
            "dtype": "float16",
            "count": int(blob.size),
            "sha256": sha256(weights_path),
        },
        "inputs": [
            {"name": item.name, "channels": lowering.channels[item.name]}
            for item in lowering.graph.input],
        "outputs": [
            {"name": item.name, "channels": lowering.channels[item.name]}
            for item in lowering.graph.output],
        "nodes": lowering.nodes,
    }
    if integer_map is not None:
        program["integer_map_sha256"] = hashlib.sha256(
            json.dumps(integer_map, sort_keys=True).encode()).hexdigest()
    program_path = output_dir / f"{name}.detprog.json"
    program_path.write_text(json.dumps(program, indent=1) + "\n", encoding="utf-8")
    return {"program": str(program_path), "nodes": len(lowering.nodes),
            "weights": int(blob.size)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="directory with the exported DCVC-RT ONNX stages")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--integer-map", type=Path,
                        help='integer widths: {"default": "w16a16", "layers": {NAME: SPEC}} '
                             "keyed by pinned DCVC-RT layer names")
    parser.add_argument("stages", nargs="+", help="stage names, e.g. i_hyper_synthesis")
    args = parser.parse_args()
    integer_map = (json.loads(args.integer_map.read_text(encoding="utf-8"))
                   if args.integer_map else None)
    for stage in args.stages:
        try:
            summary = export(args.model_dir / f"{stage}.onnx", args.output_dir, stage, integer_map)
        except ExportError as error:
            print(f"{stage}: not exportable: {error}", file=sys.stderr)
            return 1
        print(f"{stage}: {summary['nodes']} nodes, {summary['weights']:,} weights -> "
              f"{summary['program']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
