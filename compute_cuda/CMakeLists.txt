cmake_minimum_required(VERSION 3.12.4)

option(unittests "unittests" OFF)
set_property(GLOBAL PROPERTY GLOBAL_FETCHDEPS_BASE ${CMAKE_CURRENT_SOURCE_DIR}/al2o3 )
include(FetchContent)
FetchContent_Declare( al2o3 GIT_REPOSITORY https://github.com/DeanoC/al2o3 GIT_TAG master )
FetchContent_GetProperties(al2o3)
if(NOT al2o3_POPULATED)
	FetchContent_Populate(al2o3)
	add_subdirectory(${al2o3_SOURCE_DIR} ${al2o3_BINARY_DIR})
endif()
INIT_AL2O3(${CMAKE_CURRENT_SOURCE_DIR})

SET_MIN_VERSIONS()
set(LibName compute_cuda)
project(${LibName} C CXX CUDA)
file(GLOB_RECURSE Src CONFIGURE_DEPENDS include/*.h include/*.hpp src/*.c src/*.cpp src/*.cu)

set(Deps
		al2o3_platform
		al2o3_memory
		)
ADD_LIB2(${LibName} "${Src}" "${Deps}")

set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ptxs)
set(CUDA_VERBOSE_BUILD OFF)
set(CUDA_NVCC_FLAGS "-gencode arch=compute_61,code=sm_61")
find_package(CUDA REQUIRED)
cuda_compile_ptx(
		cuda_ptx_files
		kernels/test.cu
)

add_custom_target(ptx ALL
		DEPENDS ${cuda_ptx_files} kernels/test.cu
		SOURCES kernels/test.cu)