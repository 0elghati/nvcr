#pragma once

namespace nvcr::provider::onnxruntime {

// Registers one provider per ONNX Runtime execution provider available in the
// linked runtime: onnxruntime-cpu, onnxruntime-cuda, and onnxruntime-qnn.
// Safe to call multiple times; re-registration is idempotent.
void register_providers();

}  // namespace nvcr::provider::onnxruntime
