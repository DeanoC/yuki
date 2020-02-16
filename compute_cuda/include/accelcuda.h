// License Summary: MIT see LICENSE file
#pragma once

#include "al2o3_platform/platform.h"

typedef struct _AccelCUDA_Cuda *AccelCUDA_Cuda;
typedef struct _AccelCUDA_Function *AccelCUDA_Function;
typedef uintptr_t const AccelCUDA_DeviceMemory;
typedef uintptr_t const AccelCUDA_Stream;
typedef uintptr_t const AccelCUDA_Module;
typedef uintptr_t const AccelCUDA_Surface;
typedef uintptr_t const AccelCUDA_Texture;

typedef struct _AccelCUDA_DevicePitchedMemory {
	size_t pitch;
	AccelCUDA_DeviceMemory ptr;
} _AccelCUDA_DevicePitchedMemory;
typedef struct _AccelCUDA_DevicePitchedMemory const * const AccelCUDA_DevicePitchedMemory;

typedef enum AccelCUDA_FunctionAttribute {
	ACFA_MAX_THREADS_PER_BLOCK = 0,
	ACFA_MAX_SHARED_SIZE_BYTES = 1,
	ACFA_MAX_CONST_SIZE_BYTES = 2,
	ACFA_MAX_LOCAL_SIZE_BYTES = 3,
	ACFA_MAX_NUM_REGS = 4,
	ACFA_MAX_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
	ACFA_MAX_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
} AccelCUDA_FunctionAttribute;

typedef enum AccelCUDA_CacheConfig {
	ACCC_PREFER_SHARED = 1,
	ACCC_PREFER_L1 = 2,
	ACCC_PREFER_EQUAL = 3
} AccelCUDA_CacheConfig;

typedef void(*AccelCUDA_HostCallback)(void*  userData);


AL2O3_EXTERN_C AccelCUDA_Cuda AccelCUDA_Create();
AL2O3_EXTERN_C void AccelCUDA_Destroy(AccelCUDA_Cuda cuda);

// create and destroy functions
AL2O3_EXTERN_C AccelCUDA_Stream AccelCUDA_StreamCreate(AccelCUDA_Cuda cuda);
AL2O3_EXTERN_C void AccelCUDA_StreamDestroy(AccelCUDA_Cuda cuda, AccelCUDA_Stream stream);
AL2O3_EXTERN_C AccelCUDA_Module AccelCUDA_ModuleCreateFromPTX(AccelCUDA_Cuda cuda, char const *ptx);
AL2O3_EXTERN_C void AccelCUDA_ModuleDestroy(AccelCUDA_Cuda cuda, AccelCUDA_Module module);
AL2O3_EXTERN_C AccelCUDA_Function AccelCUDA_FunctionCreate(AccelCUDA_Cuda cuda, AccelCUDA_Module module, char const *name);
AL2O3_EXTERN_C void AccelCUDA_FunctionDestroy(AccelCUDA_Cuda cuda, AccelCUDA_Function function);

// stream functions
AL2O3_EXTERN_C bool AccelCUDA_StreamIsIdle(AccelCUDA_Stream stream);
AL2O3_EXTERN_C void AccelCUDA_StreamSynchronize(AccelCUDA_Stream stream);
AL2O3_EXTERN_C void AccelCUDA_StreamPointGlobalTo(AccelCUDA_Stream stream, AccelCUDA_DeviceMemory src, AccelCUDA_DeviceMemory global);
AL2O3_EXTERN_C void AccelCUDA_StreamLaunchCoopFunction(AccelCUDA_Stream stream, AccelCUDA_Function function, 
																												uint64_t gridDimX, uint64_t gridDimY, uint64_t gridDimZ);
AL2O3_EXTERN_C void AccelCUDA_StreamHostCallback(AccelCUDA_Stream stream, AccelCUDA_HostCallback callback, void* userData);
AL2O3_EXTERN_C void AccelCUDA_StreamLaunchFunction(AccelCUDA_Stream stream, AccelCUDA_Function function, 
																												uint64_t gridDimX, uint64_t gridDimY, uint64_t gridDimZ);


// memory functions
AL2O3_EXTERN_C AccelCUDA_DeviceMemory AccelCUDA_DeviceMalloc(AccelCUDA_Cuda cuda, size_t size);
AL2O3_EXTERN_C AccelCUDA_DevicePitchedMemory AccelCUDA_DeviceMalloc2D(AccelCUDA_Cuda cuda, uint64_t width, uint64_t height);

AL2O3_EXTERN_C void AccelCUDA_FreeDeviceMemory(AccelCUDA_Cuda cuda, AccelCUDA_DeviceMemory memory);
AL2O3_EXTERN_C void AccelCUDA_FreeDeviceMemory2D(AccelCUDA_Cuda cuda , AccelCUDA_DevicePitchedMemory memory);

AL2O3_EXTERN_C void* AccelCUDA_HostMalloc(AccelCUDA_Cuda cuda, size_t size);
AL2O3_EXTERN_C void AccelCUDA_FreeHostMemory(AccelCUDA_Cuda cuda , void* memory);

AL2O3_EXTERN_C void AccelCUDA_CopyHostToDevice(AccelCUDA_Stream stream, void const* hostMem,  AccelCUDA_DeviceMemory devMemory, size_t bytes);
AL2O3_EXTERN_C void AccelCUDA_CopyDeviceToHost(AccelCUDA_Stream stream, AccelCUDA_DeviceMemory devMemory, void * hostMem, size_t bytes);
AL2O3_EXTERN_C void AccelCUDA_CopyDeviceToDevice(AccelCUDA_Stream stream, AccelCUDA_DeviceMemory srcDevMemory,  AccelCUDA_DeviceMemory dstDevMemory, size_t bytes);
AL2O3_EXTERN_C void AccelCUDA_SetDeviceMemoryToUInt8(AccelCUDA_Stream stream, AccelCUDA_DeviceMemory memory, size_t size, uint8_t val);
AL2O3_EXTERN_C void AccelCUDA_SetDeviceMemoryToUInt16(AccelCUDA_Stream stream, AccelCUDA_DeviceMemory memory, size_t size, uint16_t val);
AL2O3_EXTERN_C void AccelCUDA_SetDeviceMemoryToUInt32(AccelCUDA_Stream stream, AccelCUDA_DeviceMemory memory, size_t size, uint32_t val);
AL2O3_EXTERN_C void AccelCUDA_SetDeviceMemoryToFloat(AccelCUDA_Stream stream, AccelCUDA_DeviceMemory memory, size_t size, float val);

AL2O3_EXTERN_C size_t AccelCUDA_GetSizeOfDeviceMemory(AccelCUDA_DeviceMemory memory);

// module functions
AL2O3_EXTERN_C AccelCUDA_DeviceMemory AccelCUDA_ModuleGetGlobal(AccelCUDA_Module module, char const *name);
AL2O3_EXTERN_C size_t AccelCUDA_ModuleGetGlobalSize(AccelCUDA_Module module, char const *name);

// function functions
AL2O3_EXTERN_C int AccelCUDA_FunctionGetAttribute(AccelCUDA_Function function, AccelCUDA_FunctionAttribute attribute);
AL2O3_EXTERN_C void AccelCUDA_FunctionSetMaxDynamicSharedBytes(AccelCUDA_Function function, int size);
AL2O3_EXTERN_C void AccelCUDA_FunctionSetPreferredSharedMemoryCarveOutHint(AccelCUDA_Function function, int size);
AL2O3_EXTERN_C AccelCUDA_CacheConfig AccelCUDA_FunctionGetCacheConfig(AccelCUDA_Function function);
AL2O3_EXTERN_C void AccelCUDA_FunctionSetCacheConfig(AccelCUDA_Function function, AccelCUDA_CacheConfig config);
AL2O3_EXTERN_C void AccelCUDA_FunctionSetDynamicSharedMemorySize(AccelCUDA_Function function, size_t size);
AL2O3_EXTERN_C void AccelCUDA_FunctionSetBlockDims(AccelCUDA_Function function, uint64_t x, uint64_t y, uint64_t z);
AL2O3_EXTERN_C uint32_t AccelCUDA_FunctionGetMaxActiveBlocksPerMultiprocessor(AccelCUDA_Function func);

AL2O3_EXTERN_C void AccelCUDA_ModulePointGlobalAtDeviceMemory(AccelCUDA_Stream stream, AccelCUDA_Module module, char const *name, AccelCUDA_DeviceMemory memory);
