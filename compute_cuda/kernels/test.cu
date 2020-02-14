// License Summary: MIT see LICENSE file

__global__ void vectorAdd(const float *A, const float *B, float* C) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	C[i] = A[i] + B[i];
}