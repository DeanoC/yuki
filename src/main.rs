#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
include!(concat!(env!("OUT_DIR"), "/accelcuda.rs"));

use std::{slice, fs, env};
use std::ffi::CString;


fn main() {
    let cuda = unsafe { AccelCUDA_Create() };

    let memSize : u64 = 1024 * 1024;
    let elementCount = memSize / std::mem::size_of::<f32>() as u64;
    let blockSize = 1024;

    let mem1D = unsafe { AccelCUDA_DeviceMalloc(cuda, memSize) };
    let mem2D = unsafe { AccelCUDA_DeviceMalloc2D(cuda, 1024, 1024) };

    let hostMem1D = unsafe { AccelCUDA_HostMalloc(cuda, memSize) };

    let hostMem1DSlice = unsafe {
        slice::from_raw_parts_mut(hostMem1D as *mut f32, elementCount as usize)
    };

    for f in hostMem1DSlice {
        *f = 1.0f32;
    }
    let stream = unsafe { AccelCUDA_StreamCreate(cuda) };

    let cwd = env::current_dir().unwrap();
    let resourcePath = cwd.join("resources");
    let gpuPath = resourcePath.join("gpu");

    let contents = fs::read_to_string(gpuPath.join("test.cu.ptx"))
        .expect("Something went wrong reading the file");
    let module = unsafe { AccelCUDA_ModuleCreateFromPTX(cuda, CString::new(contents).unwrap().as_ptr())};

    let func = unsafe { AccelCUDA_FunctionCreate(cuda, module, CString::new("vectorAdd").unwrap().as_ptr())};
    unsafe { AccelCUDA_FunctionSetBlockDims(func, blockSize, 1, 1) };
    unsafe { AccelCUDA_FunctionSetDynamicSharedMemorySize(func, 0) };

    let gpuA = unsafe { AccelCUDA_ModuleGetGlobal(module,CString::new("A").unwrap().as_ptr()) };
    let gpuB = unsafe { AccelCUDA_ModuleGetGlobal(module,CString::new("B").unwrap().as_ptr()) };
    let gpuC = unsafe { AccelCUDA_ModuleGetGlobal(module,CString::new("C").unwrap().as_ptr()) };
    let gpuASize = unsafe { AccelCUDA_ModuleGetGlobalSize(module,CString::new("A").unwrap().as_ptr()) };
    let gpuBSize = unsafe { AccelCUDA_ModuleGetGlobalSize(module,CString::new("B").unwrap().as_ptr()) };
    let gpuCSize = unsafe { AccelCUDA_ModuleGetGlobalSize(module,CString::new("C").unwrap().as_ptr()) };
    assert_eq!(gpuASize, 8);
    assert_eq!(gpuBSize, 8);
    assert_eq!(gpuCSize, 8);

    let memC = unsafe { AccelCUDA_DeviceMalloc(cuda, memSize) };
    let memCsize = unsafe { AccelCUDA_GetSizeOfDeviceMemory(memC) };
    assert_eq!(memCsize, memSize);

    unsafe { AccelCUDA_CopyHostToDevice(stream, hostMem1D, mem1D, memSize) };
    unsafe { AccelCUDA_SetDeviceMemoryToFloat(stream, memC, elementCount, 1.0f32) };

    unsafe { AccelCUDA_StreamPointGlobalTo(stream, mem1D, gpuA) };
    unsafe { AccelCUDA_StreamPointGlobalTo(stream, mem1D, gpuB) };
    unsafe { AccelCUDA_StreamPointGlobalTo(stream, memC, gpuC) };

    unsafe { AccelCUDA_StreamLaunchFunction(stream, func, elementCount / blockSize, 1, 1)};

    unsafe { AccelCUDA_CopyDeviceToHost(stream, memC, hostMem1D, memSize); }
    let isDone = unsafe { AccelCUDA_StreamIsIdle(stream) };
    println!("Stream isDone? {}", isDone);
    unsafe { AccelCUDA_StreamSynchronize(stream) };
    assert!( unsafe { AccelCUDA_StreamIsIdle(stream) } );

    let hostMemC = unsafe {
        slice::from_raw_parts_mut(hostMem1D as *mut f32, elementCount as usize)
    };

    for f in hostMemC {
        assert!( (*f - 2.0f32).abs() < 1e-5f32 );
    }

    unsafe { AccelCUDA_FreeDeviceMemory(cuda, memC) };
  //  unsafe { AccelCUDA_FunctionDestroy(cuda, func)}
    unsafe { AccelCUDA_ModuleDestroy(cuda, module) };
    unsafe { AccelCUDA_FreeHostMemory(cuda, hostMem1D) };
    unsafe { AccelCUDA_FreeDeviceMemory2D(cuda, mem2D) };
    unsafe { AccelCUDA_FreeDeviceMemory(cuda, mem1D) };

    unsafe { AccelCUDA_Destroy(cuda) };

}
