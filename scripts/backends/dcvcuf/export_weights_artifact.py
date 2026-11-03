#!/usr/bin/env python3
"""Export DCVC-UF state dictionaries as digest-bound LibTorch artifacts."""

import hashlib
from pathlib import Path

import torch

_ESCAPE = "__nvcr_dot__"


class WeightArchive(torch.nn.Module):
    def forward(self, value):
        return value


def encode_weight_name(name):
    if not name or _ESCAPE in name:
        raise ValueError(f"unsupported weight name: {name!r}")
    return name.replace(".", _ESCAPE)


def export_state_dict(state_dict, output, *, preserve_channels_last=False):
    """Write a deterministic-name TorchScript archive loadable from C++."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    archive = WeightArchive()
    for name in sorted(state_dict):
        value = state_dict[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"weight {name!r} is not a tensor")
        value = value.detach().cpu()
        if preserve_channels_last and value.dim() == 4 and value.is_floating_point():
            value = value.contiguous(memory_format=torch.channels_last)
        else:
            value = value.contiguous()
        archive.register_buffer(encode_weight_name(name), value)
    torch.jit.script(archive).save(str(output))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return {"path": str(output.resolve()), "sha256": digest,
            "tensor_count": len(state_dict)}
