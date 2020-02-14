// License Summary: MIT see LICENSE file
#include "al2o3_platform/platform.h"
#include "al2o3_memory/memory.h"
#include "../include/accelcuda.h"
#include <cuda.h>
#include <nvrtc.h>

inline int _ConvertSMVer2Cores(int major, int minor) {
	// Defines for GPU Architecture types (using the SM version to determine
	// the # of cores per SM
	typedef struct {
		int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
		// and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = {
			{0x30, 192},
			{0x32, 192},
			{0x35, 192},
			{0x37, 192},
			{0x50, 128},
			{0x52, 128},
			{0x53, 128},
			{0x60, 64},
			{0x61, 128},
			{0x62, 128},
			{0x70, 64},
			{0x72, 64},
			{0x75, 64},
			{-1, -1}};

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	// If we don't find the values, we default use the previous one
	// to run properly
	LOGINFO("MapSMtoCores for SM %d.%d is undefined. Default to use %d Cores/SM\n",
					major, minor, nGpuArchCoresPerSM[index - 1].Cores);
	return nGpuArchCoresPerSM[index - 1].Cores;
}

inline const char *_ConvertSMVer2ArchName(int major, int minor) {
	// Defines for GPU Architecture types (using the SM version to determine
	// the GPU Arch name)
	typedef struct {
		int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
		// and m = SM minor version
		const char *name;
	} sSMtoArchName;

	sSMtoArchName nGpuArchNameSM[] = {
			{0x30, "Kepler"},
			{0x32, "Kepler"},
			{0x35, "Kepler"},
			{0x37, "Kepler"},
			{0x50, "Maxwell"},
			{0x52, "Maxwell"},
			{0x53, "Maxwell"},
			{0x60, "Pascal"},
			{0x61, "Pascal"},
			{0x62, "Pascal"},
			{0x70, "Volta"},
			{0x72, "Xavier"},
			{0x75, "Turing"},
			{-1, "Graphics Device"}};

	int index = 0;

	while (nGpuArchNameSM[index].SM != -1) {
		if (nGpuArchNameSM[index].SM == ((major << 4) + minor)) {
			return nGpuArchNameSM[index].name;
		}

		index++;
	}

	// If we don't find the values, we default use the previous one
	// to run properly
	LOGINFO("MapSMtoArchName for SM %d.%d is undefined. Default to use %s\n",
					major,
					minor,
					nGpuArchNameSM[index - 1].name);
	return nGpuArchNameSM[index - 1].name;
}

void checkCUDA(cudaError_t result, char const *const func, const char *const file,
					 int const line) {
	if (result) {
		LOGERROR("CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
						 static_cast<unsigned int>(result), cudaGetErrorName(result), func);
	}
}

void errCheck(CUresult result, char const *const func, const char *const file,
					 int const line) {
	if (result) {
		char const *str = nullptr;
		cuGetErrorString(result, &str);
		LOGERROR("CU driver error at %s:%d code=%d(%s) \"%s\" \n", file, line,
						 static_cast<unsigned int>(result), str, func);
	}
}

void errCheck(nvrtcResult result, char const *const func, const char *const file,
					 int const line) {
	if (result) {
		LOGERROR("NVRTC error at %s:%d code=%d(%s) \"%s\" \n", file, line,
						 static_cast<unsigned int>(result), nvrtcGetErrorString(result), func);
	}
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) checkCUDA((val), #val, __FILE__, __LINE__)
#define checkErrors(val) errCheck((val), #val, __FILE__, __LINE__)

struct AccelCUDA_Cuda {
	int deviceIndex;
	CUdevice device;
	CUcontext context;

	cudaStream_t currentStream; // defaults to 0 the default stream
};

AL2O3_EXTERN_C AccelCUDA_Cuda *AccelCUDA_Create() {
	cuInit(0);

	int deviceCount;
	int pickedDeviceIndex = -1;
	int pickedTotalCores = -1;
	checkErrors(cuDeviceGetCount(&deviceCount));

	LOGINFO("--- CUDA Devices ---");
	for (int i = 0u; i < deviceCount; ++i) {
		CUdevice currentDevice;
		checkErrors( cuDeviceGet(&currentDevice, i));

		int devProps[CU_DEVICE_ATTRIBUTE_MAX] {};
		for(int j=1;j < CU_DEVICE_ATTRIBUTE_MAX;++j) {
			checkErrors(cuDeviceGetAttribute(devProps + j, (CUdevice_attribute)j, currentDevice));
		}

		int version[2] = {
			devProps[CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR],
			devProps[CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR],			
		};

		int const coresPerSM = _ConvertSMVer2Cores(version[0], version[1]);
		int const totalCores = coresPerSM * devProps[CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT];
		int const computePerf = totalCores * (devProps[CU_DEVICE_ATTRIBUTE_CLOCK_RATE]/1024);

		char name[2048];
		cuDeviceGetName(name, 2048, currentDevice);

		LOGINFO("%d: %s %s (%d.%d)", i,
						name, _ConvertSMVer2ArchName(version[0], version[1]), version[0], version[1]);
		LOGINFO("%d: SMs %d, Cores %d, Total Cores %d Clock %d ~GFLOPs %f", i,
						devProps[CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT], coresPerSM, totalCores, devProps[CU_DEVICE_ATTRIBUTE_CLOCK_RATE]/1024/1024, ((float)2 * computePerf)/1024.0f);

		// for now just pick the biggest new enough device
		if (totalCores > pickedTotalCores) {
			pickedDeviceIndex = i;
			pickedTotalCores = totalCores;
		}
	}

	LOGINFO("---");
	int nvrtcMajor = 0;
	int nvrtcMinor = 0;
	checkErrors(nvrtcVersion( &nvrtcMajor, &nvrtcMajor));
	LOGINFO("NVRTC V %i.%i", nvrtcMajor, nvrtcMinor);

	if (pickedDeviceIndex == -1) {
		return nullptr;
	}

	AccelCUDA_Cuda* cuda = (AccelCUDA_Cuda*)MEMORY_CALLOC(1, sizeof(AccelCUDA_Cuda));
	if(!cuda) return nullptr;

	cuda->deviceIndex = pickedDeviceIndex;
	checkErrors(cuDeviceGet(&cuda->device, pickedDeviceIndex));
	checkErrors(cuDevicePrimaryCtxRetain(&cuda->context, cuda->device));
	checkErrors(cuCtxSetCurrent(cuda->context));

	return cuda;
}

AL2O3_EXTERN_C void AccelCUDA_Destroy(AccelCUDA_Cuda *cuda) {
	if(!cuda) return;

	checkErrors(cuCtxSynchronize());
	checkErrors(cuDevicePrimaryCtxRelease(cuda->device));

	MEMORY_FREE(cuda);
}

AL2O3_EXTERN_C AccelCUDA_Stream AccelCUDA_StreamCreate(AccelCUDA_Cuda* cuda) {
	CUstream stream;
	checkErrors(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
	return (AccelCUDA_Stream)stream;
}

AL2O3_EXTERN_C void AccelCUDA_StreamDestroy(AccelCUDA_Cuda* cuda, AccelCUDA_Stream stream) {
	checkErrors(cuStreamDestroy((CUstream)stream));
}

AL2O3_EXTERN_C bool AccelCUDA_StreamIsIdle(AccelCUDA_Stream stream) {
	return cuStreamQuery((CUstream)stream) == CUDA_SUCCESS;
}

AL2O3_EXTERN_C void AccelCUDA_StreamSynchronize(AccelCUDA_Stream stream) {
	checkErrors(cuStreamSynchronize((CUstream)stream));
}

AL2O3_EXTERN_C AccelCUDA_DeviceMemory AccelCUDA_DeviceMalloc(AccelCUDA_Cuda*, uint64_t size) {
	CUdeviceptr dptr;
	checkErrors(cuMemAlloc(&dptr, size));
	return (AccelCUDA_DeviceMemoryPtr*)dptr;
}
AL2O3_EXTERN_C AccelCUDA_DevicePitchedMemory AccelCUDA_DeviceMalloc2D(AccelCUDA_Cuda*, uint64_t width, uint64_t height) {
	AccelCUDA_DevicePitchedMemoryPtr * dptr = (AccelCUDA_DevicePitchedMemoryPtr *) MEMORY_CALLOC(1, sizeof(AccelCUDA_DevicePitchedMemoryPtr));
	checkErrors(cuMemAllocPitch((CUdeviceptr*)&dptr->ptr, &dptr->pitch, width, height, 4));
	return dptr;
}

AL2O3_EXTERN_C void AccelCUDA_FreeDeviceMemory(AccelCUDA_Cuda*, AccelCUDA_DeviceMemory memory) {
	checkErrors(cuMemFree((CUdeviceptr)memory));
}

AL2O3_EXTERN_C void AccelCUDA_FreeDeviceMemory2D(AccelCUDA_Cuda*, AccelCUDA_DevicePitchedMemory memory) {
	checkErrors(cuMemFree((CUdeviceptr)memory->ptr));
	MEMORY_FREE(memory);
}

AL2O3_EXTERN_C void* AccelCUDA_HostMalloc(AccelCUDA_Cuda* cuda, uint64_t size) {
	void* dptr = nullptr;
	checkErrors(cuMemAllocHost(&dptr, size));
	return dptr;
}
AL2O3_EXTERN_C void AccelCUDA_FreeHostMemory(AccelCUDA_Cuda* , void* memory) {
	checkErrors(cuMemFreeHost(memory));
}

AL2O3_EXTERN_C void AccelCUDA_CopyHostToDevice(AccelCUDA_Stream stream, void const* hostMem,  AccelCUDA_DeviceMemory devMemory, size_t bytes) {
	checkErrors(cuMemcpyHtoDAsync((CUdeviceptr)devMemory, hostMem, bytes, (cudaStream_t)stream));
}

AL2O3_EXTERN_C void AccelCUDA_CopyDeviceToHost(AccelCUDA_Stream stream, AccelCUDA_DeviceMemory devMemory, void * hostMem, size_t bytes) {
	checkErrors(cuMemcpyDtoHAsync(hostMem, (CUdeviceptr)devMemory, bytes, (cudaStream_t)stream));
}

AL2O3_EXTERN_C void AccelCUDA_CopyDeviceToDevice(AccelCUDA_Stream stream, AccelCUDA_DeviceMemory srcDevMemory, AccelCUDA_DeviceMemory dstDevMemory, size_t bytes) {
	checkErrors(cuMemcpyDtoDAsync((CUdeviceptr)dstDevMemory, (CUdeviceptr)srcDevMemory, bytes, (cudaStream_t)stream));
}