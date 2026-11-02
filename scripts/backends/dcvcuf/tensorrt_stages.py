"""DCVC-UF TensorRT stage definitions shared by the C8 export and check tools.

Each stage wraps pinned upstream PyTorch sub-networks exactly as upstream's
inference path composes them. Codec glue (masks, quantization, entropy, crops,
pixel conversion) stays outside the stages, where the fused path also has it:

- the image hyper-decoder stage returns prior-fusion params and the reduced
  common params at the padded latent size; glue crops both (the reduction is a
  1x1 convolution, so it commutes with the crop);
- the HT-S hyper-decoder output is cropped in glue to the temporal prior's size
  before the prior stage fuses them, as in the fused path;
- the reconstruction head stays unclamped, as upstream's PyTorch head; the
  fused path clamps frames for display.
"""

from __future__ import annotations

# Axis classes for 1/1 padded frames, 1/8 features, 1/16 latents, 1/16
# latents padded to a multiple of 4 (pad_for_y), and 1/64 hyper latents.
AXES = {
    "frame": 1,
    "feature": 8,
    "latent": 16,
    "latent_pad4": 16,
    "hyper": 64,
    "scalar": None,
}

# 1080p shape profile on padded frame sizes (multiples of 64 at the minimum).
PROFILE = {
    "name": "1080p",
    "frame_min": (64, 64),
    "frame_opt": (1088, 1920),
    "frame_max": (1088, 1920),
}

# name -> (inputs, outputs); each tensor is (name, channels, axis class).
STAGES = {
    "i_encoder": (
        [("frame", 3, "frame"), ("q_enc", 384, "scalar")],
        [("y", 256, "latent")]),
    "i_hyper_encoder": (
        [("y_padded", 256, "latent_pad4")],
        [("z", 128, "hyper")]),
    "i_hyper_decoder": (
        [("z_hat", 128, "hyper")],
        [("params", 512, "latent_pad4"), ("common", 256, "latent_pad4")]),
    "i_spatial_prior_1": (
        [("y_hat", 256, "latent"), ("common", 256, "latent")],
        [("scales_means", 512, "latent")]),
    "i_spatial_prior_2": (
        [("y_hat", 256, "latent"), ("common", 256, "latent")],
        [("scales_means", 512, "latent")]),
    "i_spatial_prior_3": (
        [("y_hat", 256, "latent"), ("common", 256, "latent")],
        [("scales_means", 512, "latent")]),
    "i_decoder": (
        [("y_hat", 256, "latent"), ("q_dec", 384, "scalar")],
        [("frame_hat", 3, "frame")]),
    "p_feature_adaptor_i": (
        [("feature", 192, "feature")],
        [("memory", 512, "feature")]),
    "p_feature_adaptor_m": (
        [("memory", 512, "feature"), ("feature", 512, "feature")],
        [("memory_out", 512, "feature")]),
    "p_feature_extractor": (
        [("memory", 512, "feature")],
        [("context", 512, "feature")]),
    "p_encoder": (
        [("frames", 24, "frame"), ("context", 512, "feature"), ("q_encoder", 512, "scalar")],
        [("y", 256, "latent")]),
    "p_hyper_encoder": (
        [("y_padded", 256, "latent_pad4")],
        [("z", 128, "hyper")]),
    "p_hyper_decoder": (
        [("z_hat", 128, "hyper")],
        [("hyper_params", 256, "latent_pad4")]),
    "p_prior": (
        [("hyper_params", 256, "latent"), ("memory", 512, "feature"),
         ("q_feature", 512, "scalar")],
        [("params", 768, "latent"), ("common", 256, "latent")]),
    "p_spatial_prior_1": (
        [("y_hat", 256, "latent"), ("common", 256, "latent")],
        [("means", 256, "latent")]),
    "p_spatial_prior_2": (
        [("y_hat", 256, "latent"), ("common", 256, "latent")],
        [("means", 256, "latent")]),
    "p_spatial_prior_3": (
        [("y_hat", 256, "latent"), ("common", 256, "latent")],
        [("means", 256, "latent")]),
    "p_decoder": (
        [("y_hat", 256, "latent"), ("context", 512, "feature"), ("q_decoder", 512, "scalar")],
        [("feature", 512, "feature")]),
    "p_reconstruction": (
        [("feature", 512, "feature")],
        [("frames", 24, "frame")]),
    "p_reset": (
        [("feature", 512, "feature")],
        [("reset_feature", 192, "feature")]),
}


def axis_size(frame_size, axis):
    divisor = AXES[axis]
    if divisor is None:
        return 1
    size = frame_size // divisor
    if axis == "latent_pad4":
        size = (size + 3) // 4 * 4
    return size


def tensor_shape(channels, axis, frame_hw):
    if axis == "scalar":
        return [1, channels, 1, 1]
    return [1, channels, axis_size(frame_hw[0], axis), axis_size(frame_hw[1], axis)]


def profile_shapes(channels, axis):
    return {key: tensor_shape(channels, axis, PROFILE[f"frame_{key}"])
            for key in ("min", "opt", "max")}


def modules(image, video):
    """Returns name -> torch module computing the stage from upstream modules."""
    import torch
    from torch import nn

    class Stage(nn.Module):
        def __init__(self, function, owner):
            super().__init__()
            self.owner = owner  # registers upstream weights for export
            self.function = function

        def forward(self, *inputs):
            return self.function(*inputs)

    def image_prior(adaptor):
        return lambda y_hat, common: image.y_spatial_prior(
            adaptor(torch.cat((y_hat, common), dim=1)))

    def video_prior(adaptor):
        return lambda y_hat, common: video.y_spatial_prior(adaptor(y_hat, common))

    def image_hyper_decoder(z_hat):
        params = image.y_prior_fusion(image.hyper_dec(z_hat))
        return params, image.y_spatial_prior_reduction(params)

    def video_prior_params(hyper_params, memory, q_feature):
        params = video.y_prior_fusion(
            hyper_params, video.temporal_prior_encoder(memory, q_feature))
        return params, video.y_spatial_prior_reduction(params)

    functions = {
        "i_encoder": (lambda frame, q_enc: image.enc(frame, q_enc), image),
        "i_hyper_encoder": (lambda y: image.hyper_enc(y), image),
        "i_hyper_decoder": (image_hyper_decoder, image),
        "i_spatial_prior_1": (image_prior(image.y_spatial_prior_adaptor_1), image),
        "i_spatial_prior_2": (image_prior(image.y_spatial_prior_adaptor_2), image),
        "i_spatial_prior_3": (image_prior(image.y_spatial_prior_adaptor_3), image),
        "i_decoder": (lambda y_hat, q_dec: image.dec(y_hat, q_dec), image),
        "p_feature_adaptor_i": (lambda feature: video.feature_adaptor_i(feature), video),
        "p_feature_adaptor_m": (
            lambda memory, feature: video.feature_adaptor_m(memory, feature), video),
        "p_feature_extractor": (lambda memory: video.feature_extractor(memory), video),
        "p_encoder": (
            lambda frames, context, q: video.encoder(frames, context, q), video),
        "p_hyper_encoder": (lambda y: video.hyper_encoder(y), video),
        "p_hyper_decoder": (lambda z_hat: video.hyper_decoder(z_hat), video),
        "p_prior": (video_prior_params, video),
        "p_spatial_prior_1": (video_prior(video.y_spatial_prior_adaptor_1), video),
        "p_spatial_prior_2": (video_prior(video.y_spatial_prior_adaptor_2), video),
        "p_spatial_prior_3": (video_prior(video.y_spatial_prior_adaptor_3), video),
        "p_decoder": (
            lambda y_hat, context, q: video.decoder(y_hat, context, q), video),
        "p_reconstruction": (
            lambda feature: torch.cat(video.recon_head(feature), dim=1), video),
        "p_reset": (lambda feature: video.recon_head(feature, for_reset=True), video),
    }
    return {name: Stage(function, owner).eval()
            for name, (function, owner) in functions.items()}
