# Copyright (c) 2024, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.26.4)

include(../set_cuda_architecture.cmake)

rapids_cuda_init_architectures(tpch_example)
rapids_cuda_set_architectures(RAPIDS)

project(
  tpch_example
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

# This function takes in a example name and example source and handles setting all of the associated
# properties and linking to build the example
function(ConfigureTPCHExample CMAKE_TPCH_EXAMPLE_NAME)
  add_executable(${CMAKE_TPCH_EXAMPLE_NAME} ${ARGN})
  target_link_libraries(${CMAKE_TPCH_EXAMPLE_NAME} PUBLIC cudf::cudf cudf::cudf_tpch_datagen)
  target_compile_features(${CMAKE_TPCH_EXAMPLE_NAME} PUBLIC cxx_std_17)
endfunction()

ConfigureTPCHExample(dbgen dbgen.cpp)
ConfigureTPCHExample(tpch_q1 q1.cpp)
ConfigureTPCHExample(tpch_q5 q5.cpp)
ConfigureTPCHExample(tpch_q6 q6.cpp)
ConfigureTPCHExample(tpch_q9 q9.cpp)
ConfigureTPCHExample(tpch_q10 q10.cpp)
