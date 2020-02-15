// License Summary: MIT see LICENSE file

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ float const * A;
__device__ float const * B;
__device__ float  * C;

extern "C" __global__ void vectorAdd() {
	cg::thread_block block = cg::this_thread_block();

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	C[i] = A[i] + B[i];
}