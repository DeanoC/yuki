// License Summary: MIT see LICENSE file
#include "al2o3_platform/platform.h"
#include "al2o3_memory/memory.h"
#include "accelcuda.h"
#include <cuda.h>
#include <nvrtc.h>


struct _AccelCUDA_Cuda {
	int deviceIndex;
	CUdevice device;
	
	CUcontext context;
};

struct _AccelCUDA_Function {
	CUfunction function;

	size_t sharedMemBytes; 
	uint64_t blockDimX;
	uint64_t blockDimY;
	uint64_t blockDimZ;

	CUfunc_cache cacheConfig;
};

static_assert(sizeof(uintptr_t) == sizeof(CUstream), "sizeof CUstream error");
static_assert(sizeof(uintptr_t) == sizeof(CUdeviceptr), "sizeof CUdeviceptr error");
static_assert(sizeof(uintptr_t) == sizeof(CUmodule), "sizeof CUModule error");

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


AL2O3_EXTERN_C AccelCUDA_Cuda AccelCUDA_Create() {

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

	_AccelCUDA_Cuda* cuda = (_AccelCUDA_Cuda*)MEMORY_CALLOC(1, sizeof(_AccelCUDA_Cuda));
	if(!cuda) return nullptr;

	cuda->deviceIndex = pickedDeviceIndex;
	checkErrors(cuDeviceGet(&cuda->device, pickedDeviceIndex));
	checkErrors(cuDevicePrimaryCtxRetain(&cuda->context, cuda->device));
	checkErrors(cuCtxSetCurrent(cuda->context));

	return cuda;
}

AL2O3_EXTERN_C void AccelCUDA_Destroy(AccelCUDA_Cuda cuda) {
	if(!cuda) return;

	checkErrors(cuCtxSynchronize());
	checkErrors(cuDevicePrimaryCtxRelease(cuda->device));

	MEMORY_FREE(cuda);
}

AL2O3_EXTERN_C AccelCUDA_Stream AccelCUDA_StreamCreate(AccelCUDA_Cuda cuda) {
	CUstream stream;
	checkErrors(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
	return (AccelCUDA_Stream)stream;
}

AL2O3_EXTERN_C void AccelCUDA_StreamDestroy(AccelCUDA_Cuda cuda, AccelCUDA_Stream stream) {
	checkErrors(cuStreamDestroy((CUstream)stream));
}

AL2O3_EXTERN_C AccelCUDA_Module AccelCUDA_ModuleCreateFromPTX(AccelCUDA_Cuda cuda, char const *ptx) {
	CUmodule module;
	checkErrors( cuModuleLoadData(&module, ptx));

	return (AccelCUDA_Module)module;
}

AL2O3_EXTERN_C void AccelCUDA_ModuleDestroy(AccelCUDA_Cuda cuda, AccelCUDA_Module module) {
	checkErrors(cuModuleUnload((CUmodule)module));
}

AL2O3_EXTERN_C AccelCUDA_Function AccelCUDA_FunctionCreate(AccelCUDA_Cuda cuda, AccelCUDA_Module module, char const *name) {
	_AccelCUDA_Function* func = (_AccelCUDA_Function*)MEMORY_CALLOC(1, sizeof(_AccelCUDA_Function));
	if(!func) return (AccelCUDA_Function)nullptr;

	CUfunction function;
	checkErrors(cuModuleGetFunction(&function, (CUmodule)module, name));
	func->function = function;
	AccelCUDA_FunctionSetCacheConfig(func, ACCC_PREFER_L1);

	return func;
}

AL2O3_EXTERN_C void AccelCUDA_FunctionDestroy(AccelCUDA_Cuda cuda, AccelCUDA_Function func) {
	MEMORY_FREE(func);
}

AL2O3_EXTERN_C bool AccelCUDA_StreamIsIdle(AccelCUDA_Stream stream) {
	return cuStreamQuery((CUstream)stream) == CUDA_SUCCESS;
}

AL2O3_EXTERN_C void AccelCUDA_StreamSynchronize(AccelCUDA_Stream stream) {
	checkErrors(cuStreamSynchronize((CUstream)stream));
}

AL2O3_EXTERN_C AccelCUDA_DeviceMemory AccelCUDA_ModuleGetGlobal(AccelCUDA_Module module, char const *name) {
	CUdeviceptr memory;
	checkErrors(cuModuleGetGlobal(&memory, nullptr, (CUmodule)module, name));
	return (AccelCUDA_DeviceMemory)memory;
}

AL2O3_EXTERN_C size_t AccelCUDA_ModuleGetGlobalSize(AccelCUDA_Module module, char const *name) {
	size_t size;
	checkErrors(cuModuleGetGlobal(nullptr, &size, (CUmodule)module, name));
	return size;
}

AL2O3_EXTERN_C AccelCUDA_DeviceMemory AccelCUDA_DeviceMalloc(AccelCUDA_Cuda cuda, uint64_t size) {
	CUdeviceptr dptr;
	checkErrors(cuMemAlloc(&dptr, size));
	return (AccelCUDA_DeviceMemory)dptr;
}

AL2O3_EXTERN_C AccelCUDA_DevicePitchedMemory AccelCUDA_DeviceMalloc2D(AccelCUDA_Cuda cuda, uint64_t width, uint64_t height) {
	_AccelCUDA_DevicePitchedMemory * dptr = (_AccelCUDA_DevicePitchedMemory *) MEMORY_CALLOC(1, sizeof(_AccelCUDA_DevicePitchedMemory));
	checkErrors(cuMemAllocPitch((CUdeviceptr*)&dptr->ptr, &dptr->pitch, width, height, 4));
	return dptr;
}

AL2O3_EXTERN_C void AccelCUDA_FreeDeviceMemory(AccelCUDA_Cuda cuda, AccelCUDA_DeviceMemory memory) {
	checkErrors(cuMemFree((CUdeviceptr)memory));
}


AL2O3_EXTERN_C void AccelCUDA_FreeDeviceMemory2D(AccelCUDA_Cuda cuda, AccelCUDA_DevicePitchedMemory memory) {
	checkErrors(cuMemFree((CUdeviceptr)memory->ptr));
	MEMORY_FREE((void*)memory);
}

AL2O3_EXTERN_C void* AccelCUDA_HostMalloc(AccelCUDA_Cuda cuda, size_t size) {
	void* dptr = nullptr;
	checkErrors(cuMemAllocHost(&dptr, size));
	return dptr;
}

AL2O3_EXTERN_C void AccelCUDA_FreeHostMemory(AccelCUDA_Cuda cuda, void* memory) {
	checkErrors(cuMemFreeHost(memory));
}

AL2O3_EXTERN_C void AccelCUDA_CopyHostToDevice(AccelCUDA_Stream stream, void const* hostMemory,  AccelCUDA_DeviceMemory devMemory, size_t bytes) {
	checkErrors(cuMemcpyHtoDAsync((CUdeviceptr)devMemory, hostMemory, bytes, (cudaStream_t)stream));
}

AL2O3_EXTERN_C void AccelCUDA_CopyDeviceToHost(AccelCUDA_Stream stream, AccelCUDA_DeviceMemory devMemory, void * hostMem, size_t bytes) {
	checkErrors(cuMemcpyDtoHAsync(hostMem, (CUdeviceptr)devMemory, bytes, (cudaStream_t)stream));
}

AL2O3_EXTERN_C void AccelCUDA_CopyDeviceToDevice(AccelCUDA_Stream stream, AccelCUDA_DeviceMemory srcDevMemory, AccelCUDA_DeviceMemory dstDevMemory, size_t bytes) {
	checkErrors(cuMemcpyDtoDAsync((CUdeviceptr)dstDevMemory, (CUdeviceptr)srcDevMemory, bytes, (cudaStream_t)stream));
}

AL2O3_EXTERN_C void AccelCUDA_SetDeviceMemoryToUInt8(AccelCUDA_Stream stream, AccelCUDA_DeviceMemory memory, size_t size, uint8_t val) {
	checkErrors(cuMemsetD8Async((CUdeviceptr)memory, val, size, (CUstream)stream));
}

AL2O3_EXTERN_C void AccelCUDA_SetDeviceMemoryToUInt16(AccelCUDA_Stream stream, AccelCUDA_DeviceMemory memory, size_t size, uint16_t val) {
	checkErrors(cuMemsetD16Async((CUdeviceptr)memory, val, size, (CUstream)stream));
}

AL2O3_EXTERN_C void AccelCUDA_SetDeviceMemoryToUInt32(AccelCUDA_Stream stream, AccelCUDA_DeviceMemory memory, size_t size, uint32_t val) {
	checkErrors(cuMemsetD32Async((CUdeviceptr)memory, val, size, (CUstream)stream));
}

AL2O3_EXTERN_C void AccelCUDA_SetDeviceMemoryToFloat(AccelCUDA_Stream stream, AccelCUDA_DeviceMemory memory, size_t size, float val) {
	union { float f; uint32_t i; } fv;
	fv.f = val;
	checkErrors(cuMemsetD32Async((CUdeviceptr)memory, fv.i, size, (CUstream)stream));
}
AL2O3_EXTERN_C size_t AccelCUDA_GetSizeOfDeviceMemory(AccelCUDA_DeviceMemory memory) {
	size_t size = 0;
	checkErrors(cuMemGetAddressRange(nullptr, &size, memory));
	return size;
}

AL2O3_EXTERN_C void AccelCUDA_StreamPointGlobalTo(AccelCUDA_Stream stream, AccelCUDA_DeviceMemory src, AccelCUDA_DeviceMemory global) {
	CUdeviceptr address;

	checkErrors(cuMemGetAddressRange(&address, nullptr, (CUdeviceptr) src));
	checkErrors(cuMemcpyHtoDAsync(global, &address, sizeof(CUdeviceptr), (CUstream) stream));
}

AL2O3_EXTERN_C void AccelCUDA_ModulePointGlobalTo(AccelCUDA_Stream stream, AccelCUDA_Module module, char const *name, AccelCUDA_DeviceMemory memory) {
	CUdeviceptr buf;
	CUdeviceptr address;
	size_t bytes;
	size_t psize;

	checkErrors(cuModuleGetGlobal(&buf, &bytes, (CUmodule) module, name));
	checkErrors(cuMemGetAddressRange(&address, &psize, memory));
	checkErrors(cuMemcpyHtoDAsync(buf, &address, bytes, (CUstream) stream));
}

AL2O3_EXTERN_C int AccelCUDA_FunctionGetAttribute(AccelCUDA_Function function, AccelCUDA_FunctionAttribute attribute) {
	int value;
	checkErrors(cuFuncGetAttribute(&value, (CUfunction_attribute)attribute, (CUfunction)function));
	return value;
}

AL2O3_EXTERN_C void AccelCUDA_FunctionSetMaxDynamicSharedBytes(AccelCUDA_Function function, int size) {
	checkErrors(cuFuncSetAttribute((CUfunction)function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, size));
}

AL2O3_EXTERN_C void AccelCUDA_FunctionSetPreferredSharedMemoryCarveOutHint(AccelCUDA_Function function, int size) {
	checkErrors(cuFuncSetAttribute((CUfunction)function, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, size));
}

AL2O3_EXTERN_C AccelCUDA_CacheConfig AccelCUDA_FunctionGetCacheConfig(AccelCUDA_Function function) {
	switch(function->cacheConfig) {
		case CU_FUNC_CACHE_PREFER_SHARED:
			return ACCC_PREFER_SHARED;
		case CU_FUNC_CACHE_PREFER_L1:
			return ACCC_PREFER_L1;
		case CU_FUNC_CACHE_PREFER_EQUAL:
			return ACCC_PREFER_EQUAL;
		case CU_FUNC_CACHE_PREFER_NONE:
		default:
			return ACCC_PREFER_L1;
	}
}

AL2O3_EXTERN_C void AccelCUDA_FunctionSetCacheConfig(AccelCUDA_Function func, AccelCUDA_CacheConfig config) {
	switch(config) {
		case ACCC_PREFER_SHARED: func->cacheConfig = CU_FUNC_CACHE_PREFER_SHARED; break;
		case ACCC_PREFER_L1:  func->cacheConfig = CU_FUNC_CACHE_PREFER_L1; break;
		case ACCC_PREFER_EQUAL: func->cacheConfig = CU_FUNC_CACHE_PREFER_EQUAL; break;
		default: LOGERROR("Invalid Cache Config"); return;
	}

	checkErrors(cuFuncSetCacheConfig((CUfunction) func->function, func->cacheConfig));
}

AL2O3_EXTERN_C void AccelCUDA_FunctionSetDynamicSharedMemorySize(AccelCUDA_Function func, size_t size) {
	func->sharedMemBytes = size;
}

AL2O3_EXTERN_C void AccelCUDA_FunctionSetBlockDims(AccelCUDA_Function func, uint64_t x, uint64_t y, uint64_t z) {
	func->blockDimX = x;
	func->blockDimY = y;
	func->blockDimZ = z;

}

AL2O3_EXTERN_C uint32_t AccelCUDA_FunctionGetMaxActiveBlocksPerMultiprocessor(AccelCUDA_Function func) {
	int numBlocks = 0; 
	int totalBlockSize = (int)(func->blockDimX * func->blockDimY * func->blockDimZ);
	cuOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, func->function, totalBlockSize, func->sharedMemBytes);
	return (uint32_t) numBlocks;
}

AL2O3_EXTERN_C void AccelCUDA_StreamLaunchCoopFunction(AccelCUDA_Stream stream, AccelCUDA_Function func, 
																												uint64_t gridDimX, uint64_t gridDimY, uint64_t gridDimZ) {
	ASSERT(func->blockDimX != 0);
	ASSERT(func->blockDimY != 0);
	ASSERT(func->blockDimZ != 0);


	checkErrors(cuLaunchCooperativeKernel((CUfunction)func->function, 
		(unsigned int)gridDimX, (unsigned int)gridDimY, (unsigned int)gridDimZ,
		(unsigned int)func->blockDimX, (unsigned int)func->blockDimY, (unsigned int)func->blockDimZ, 
		(unsigned int)func->sharedMemBytes,
		(CUstream)stream,	nullptr));

}
AL2O3_EXTERN_C void AccelCUDA_StreamHostCallback(AccelCUDA_Stream stream, AccelCUDA_HostCallback callback, void* userData) {
	checkErrors(cuLaunchHostFunc((CUstream)stream, callback, userData)); 
}

AL2O3_EXTERN_C void AccelCUDA_StreamLaunchFunction(AccelCUDA_Stream stream, AccelCUDA_Function func, 
																												uint64_t gridDimX, uint64_t gridDimY, uint64_t gridDimZ) {
	ASSERT(func->blockDimX != 0);
	ASSERT(func->blockDimY != 0);
	ASSERT(func->blockDimZ != 0);

	checkErrors(cuLaunchKernel((CUfunction)func->function, 
		(unsigned int)gridDimX, (unsigned int)gridDimY, (unsigned int)gridDimZ,
		(unsigned int)func->blockDimX, (unsigned int)func->blockDimY, (unsigned int)func->blockDimZ, 
		(unsigned int)func->sharedMemBytes,
		(CUstream)stream,	nullptr, nullptr));

}
