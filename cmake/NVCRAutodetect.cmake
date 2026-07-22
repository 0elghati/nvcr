# Auto-detects the CUDA compiler and the target GPU architecture so that a
# plain `cmake -S . -B build -DNVCR_ENABLE_TENSORRT=ON` configures correctly on
# both Jetson (Tegra, shared SoC memory) and discrete-GPU hosts (RTX,
# datacenter, or any other CUDA GPU) without the caller having to know
# `nvcc`'s path or the GPU's compute capability. See NVCR_CUDA_ARCH_SET in
# NVCROptions.cmake to switch between single-machine auto-detection (default)
# and a portable multi-arch release build.
#
# Must be included and invoked (via nvcr_autodetect_cuda()) before
# enable_language(CUDA) runs.

function(nvcr_autodetect_cuda)
    # 1. CUDA compiler: fall back to common install locations when `nvcc` is
    #    not already resolvable, which is common on Jetson images where
    #    /usr/local/cuda/bin is not on the default shell PATH.
    if(NOT CMAKE_CUDA_COMPILER)
        find_program(NVCR_DETECTED_NVCC nvcc)
        if(NOT NVCR_DETECTED_NVCC)
            file(GLOB _nvcr_cuda_candidates
                "/usr/local/cuda/bin/nvcc"
                "/usr/local/cuda-*/bin/nvcc")
            if(_nvcr_cuda_candidates)
                list(SORT _nvcr_cuda_candidates COMPARE NATURAL ORDER DESCENDING)
                list(GET _nvcr_cuda_candidates 0 NVCR_DETECTED_NVCC)
            endif()
        endif()
        if(NVCR_DETECTED_NVCC)
            set(CMAKE_CUDA_COMPILER "${NVCR_DETECTED_NVCC}" CACHE FILEPATH "Auto-detected nvcc")
            message(STATUS "NVCR: auto-detected CMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}")
        else()
            message(WARNING "NVCR: could not locate nvcc automatically; pass -DCMAKE_CUDA_COMPILER=/path/to/nvcc")
        endif()
    else()
        message(STATUS "NVCR: using user-specified CMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}")
    endif()

    # 2. GPU architecture. Three modes, selected by NVCR_CUDA_ARCH_SET
    #    (see NVCROptions.cmake), unless CMAKE_CUDA_ARCHITECTURES is already
    #    set explicitly (e.g. -DCMAKE_CUDA_ARCHITECTURES=90 for custom builds):
    #      - "auto" (default): detect this machine's single GPU via
    #        nvidia-smi's reported compute capability, falling back to
    #        CMAKE_CUDA_ARCHITECTURES=native. Fastest build; the resulting
    #        binary targets only the GPU present on the build host. This is
    #        the right choice for scripts/install.sh local/dev installs.
    #      - "portable": build a redistributable fat binary covering a
    #        curated list of common Jetson and discrete RTX/datacenter GPU
    #        architectures in one build, independent of what GPU (if any) is
    #        present on the build host. This is the right choice for release
    #        packaging that must run out of the box on many target machines
    #        of the same host CPU architecture (still build once per host
    #        CPU architecture, e.g. once for x86_64 and once for aarch64).
    #    nvidia-smi reports compute_cap correctly on Jetson devices even
    #    though it reports memory.total/memory.free as N/A there (unified
    #    memory), so this single code path covers both platform families.
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        if(NVCR_CUDA_ARCH_SET STREQUAL "portable")
            # Turing (75), Ampere datacenter (80), Ampere consumer/Jetson Orin
            # (86, 87), Ada (89), and Hopper (90). Extend this list as new GPU
            # generations need coverage; each additional architecture adds to
            # build time and binary size.
            set(CMAKE_CUDA_ARCHITECTURES "75;80;86;87;89;90" CACHE STRING "Portable multi-arch release build")
            message(STATUS "NVCR: NVCR_CUDA_ARCH_SET=portable; building fat binary for CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
        else()
            find_program(NVCR_NVIDIA_SMI nvidia-smi)
            set(_nvcr_arch_list "")
            if(NVCR_NVIDIA_SMI)
                execute_process(
                    COMMAND "${NVCR_NVIDIA_SMI}" --query-gpu=compute_cap --format=csv,noheader
                    OUTPUT_VARIABLE _nvcr_compute_caps
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET
                    RESULT_VARIABLE _nvcr_smi_result)
                if(_nvcr_smi_result EQUAL 0 AND _nvcr_compute_caps)
                    string(REPLACE "\n" ";" _nvcr_compute_cap_list "${_nvcr_compute_caps}")
                    foreach(_nvcr_cap ${_nvcr_compute_cap_list})
                        string(STRIP "${_nvcr_cap}" _nvcr_cap)
                        string(REPLACE "." "" _nvcr_arch "${_nvcr_cap}")
                        if(_nvcr_arch MATCHES "^[0-9]+$")
                            list(APPEND _nvcr_arch_list "${_nvcr_arch}")
                        endif()
                    endforeach()
                    if(_nvcr_arch_list)
                        list(REMOVE_DUPLICATES _nvcr_arch_list)
                    endif()
                endif()
            endif()
            if(_nvcr_arch_list)
                set(CMAKE_CUDA_ARCHITECTURES "${_nvcr_arch_list}" CACHE STRING "Auto-detected via nvidia-smi compute_cap")
                message(STATUS "NVCR: auto-detected CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} (nvidia-smi compute_cap=${_nvcr_compute_caps})")
            else()
                set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "Auto-detected (native fallback)")
                message(STATUS "NVCR: nvidia-smi unavailable or returned no GPUs; falling back to CMAKE_CUDA_ARCHITECTURES=native")
            endif()
        endif()
    else()
        message(STATUS "NVCR: using user-specified CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
    endif()
endfunction()
