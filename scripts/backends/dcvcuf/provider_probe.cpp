// Diagnostic bindings only; all model execution comes from the upstream library.
#include <torch/extension.h>
#include "common_cpp.h"
#include "def_cutlass.h"
#include "dmc_hts_proxy.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    py::class_<DMCHTSReconHeadProxy>(m, "ReconHead", py::module_local())
        .def(py::init<>())
        .def("set_param", &DMCHTSReconHeadProxy::set_param)
        .def("pre_allocate_tensors", &DMCHTSReconHeadProxy::pre_allocate_tensors)
        .def("forward", &DMCHTSReconHeadProxy::forward)
        .def("forward_reset", &DMCHTSReconHeadProxy::forward_reset);
    m.def("conv1x1", [](const at::Tensor& x, const at::Tensor& weight,
                        const at::Tensor& bias, bool wsilu) {
        auto out = at::empty({1, weight.size(0), x.size(2), x.size(3)},
                             x.options().memory_format(at::MemoryFormat::ChannelsLast));
        if (wsilu) {
            return conv1x1_bias_wsilu(get_gpu_sm(), x, weight, bias, out);
        }
        return conv1x1_bias(get_gpu_sm(), x, weight, bias, out);
    });
}
