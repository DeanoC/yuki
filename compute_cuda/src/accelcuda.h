// License Summary: MIT see LICENSE file
#pragma once

#include "al2o3_platform/platform.h"

typedef struct AccelCUDA_Cuda AccelCUDA_Cuda;
typedef struct AccelCUDA_DeviceMemoryPtr* AccelCUDA_DeviceMemory;

typedef struct AccelCUDA_DevicePitchedMemoryPtr {
	size_t pitch;
	AccelCUDA_DeviceMemory ptr;
} AccelCUDA_DevicePitchedMemoryPtr;
typedef struct AccelCUDA_DevicePitchedMemoryPtr* AccelCUDA_DevicePitchedMemory;

typedef struct AccelCUDA_StreamPtr* AccelCUDA_Stream;

AL2O3_EXTERN_C AccelCUDA_Cuda* AccelCUDA_Create();
AL2O3_EXTERN_C void AccelCUDA_Destroy(AccelCUDA_Cuda* cuda);

AL2O3_EXTERN_C AccelCUDA_Stream AccelCUDA_StreamCreate(AccelCUDA_Cuda* cuda);
AL2O3_EXTERN_C void AccelCUDA_StreamDestroy(AccelCUDA_Cuda* cuda, AccelCUDA_Stream stream);
AL2O3_EXTERN_C bool AccelCUDA_StreamIsIdle(AccelCUDA_Stream stream);
AL2O3_EXTERN_C void AccelCUDA_StreamSynchronize(AccelCUDA_Stream stream);

AL2O3_EXTERN_C AccelCUDA_DeviceMemory AccelCUDA_DeviceMalloc(AccelCUDA_Cuda* cuda, uint64_t size);
AL2O3_EXTERN_C AccelCUDA_DevicePitchedMemory AccelCUDA_DeviceMalloc2D(AccelCUDA_Cuda* cuda, uint64_t width, uint64_t height);

AL2O3_EXTERN_C void AccelCUDA_FreeDeviceMemory(AccelCUDA_Cuda* , AccelCUDA_DeviceMemory memory);
AL2O3_EXTERN_C void AccelCUDA_FreeDeviceMemory2D(AccelCUDA_Cuda* , AccelCUDA_DevicePitchedMemory memory);

AL2O3_EXTERN_C void* AccelCUDA_HostMalloc(AccelCUDA_Cuda* cuda, uint64_t size);
AL2O3_EXTERN_C void AccelCUDA_FreeHostMemory(AccelCUDA_Cuda* , void* memory);

AL2O3_EXTERN_C void AccelCUDA_CopyHostToDevice(AccelCUDA_Stream stream, void const* hostMem,  AccelCUDA_DeviceMemory devMemory, size_t bytes);
AL2O3_EXTERN_C void AccelCUDA_CopyDeviceToHost(AccelCUDA_Stream stream, AccelCUDA_DeviceMemory devMemory, void * hostMem, size_t bytes);
AL2O3_EXTERN_C void AccelCUDA_CopyDeviceToDevice(AccelCUDA_Stream stream, AccelCUDA_DeviceMemory srcDevMemory,  AccelCUDA_DeviceMemory dstDevMemory, size_t bytes);

AL2O3_EXTERN_C void AccelCUDA_PrepareDeviceCall(uint32_t gridX, uint32_t gridY, uint32_t gridZ,
																						uint32_t blockX, uint32_t blockY, uint32_t blockZ,
																						size_t sharedMem,
																						AccelCUDA_Stream stream
																						);
AL2O3_EXTERN_C void AccelCUDA_DeviceCall(char const* functionName);