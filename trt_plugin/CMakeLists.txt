cmake_minimum_required(VERSION 3.18)
project(jam_plugin LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- TensorRT ---
if(NOT DEFINED TRT_ROOT)
    file(GLOB TRT_CANDIDATES /usr/local/TensorRT-*)
    list(GET TRT_CANDIDATES 0 TRT_ROOT)
endif()
message(STATUS "TRT: ${TRT_ROOT}")
find_library(NVINFER_LIB nvinfer PATHS ${TRT_ROOT}/lib NO_DEFAULT_PATH)
find_path(TRT_INCLUDE_DIR NvInfer.h PATHS ${TRT_ROOT}/include NO_DEFAULT_PATH)

# --- libtorch (at::Tensor) ---
execute_process(COMMAND python3 -c "import torch; print(torch.__path__[0])"
    OUTPUT_VARIABLE TORCH_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "Torch: ${TORCH_ROOT}")
set(TORCH_LIB_DIR "${TORCH_ROOT}/lib")

# --- selective_scan_cuda.so ---
execute_process(COMMAND python3 -c
    "import importlib.util; s=importlib.util.find_spec('selective_scan_cuda'); print(s.origin if s else '')"
    OUTPUT_VARIABLE SSC_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "SSC: ${SSC_PATH}")

# --- CUDA ---
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA: ${CUDAToolkit_VERSION}")

# --- Build ---
add_library(jam_plugin SHARED selective_scan_plugin.cpp selective_scan_plugin.cu)
target_include_directories(jam_plugin PRIVATE
    ${TRT_INCLUDE_DIR}
    ${TORCH_ROOT}/include
    ${TORCH_ROOT}/include/torch/csrc/api/include)
target_link_libraries(jam_plugin ${NVINFER_LIB} CUDA::cudart)

# Link libtorch
foreach(lib torch c10 c10_cuda)
    target_link_libraries(jam_plugin "${TORCH_LIB_DIR}/lib${lib}.so")
endforeach()

# Link selective_scan_cuda.so (IMPORTED to avoid -l prefix)
if(SSC_PATH)
    add_library(ssc_lib SHARED IMPORTED)
    set_target_properties(ssc_lib PROPERTIES IMPORTED_LOCATION "${SSC_PATH}")
    target_link_libraries(jam_plugin ssc_lib)
endif()

set_target_properties(jam_plugin PROPERTIES
    BUILD_RPATH "${TRT_ROOT}/lib;${TORCH_LIB_DIR}"
    INSTALL_RPATH "${TRT_ROOT}/lib;${TORCH_LIB_DIR}")
