#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!("./accelcuda.rs");

use std::slice;


fn main() {
    let cuda = unsafe { AccelCUDA_Create() };

    println!("Hello, world!");

    let mem1D = unsafe { AccelCUDA_DeviceMalloc(cuda, 1024 * 1024) };
    let mem2D = unsafe { AccelCUDA_DeviceMalloc2D(cuda, 1024, 1024) };

    let hostMem1D = unsafe { AccelCUDA_HostMalloc(cuda, 1024 * 1024) };

    let hostMem1DSlice = unsafe {
        slice::from_raw_parts_mut(hostMem1D as *mut f32, 1024 * 1024 / std::mem::size_of::<f32>())
    };

    for f in hostMem1DSlice {
        *f = 0.0;
    }
    let stream = unsafe { AccelCUDA_StreamCreate(cuda) };

    unsafe { AccelCUDA_CopyHostToDevice(stream, hostMem1D, mem1D, 1024 * 1024); }
    unsafe { AccelCUDA_CopyDeviceToHost(stream, mem1D, hostMem1D, 1024 * 1024); }
    let isDone = unsafe { AccelCUDA_StreamIsIdle(stream) };
    println!("Stream isDone? {}", isDone);
    unsafe { AccelCUDA_StreamSynchronize(stream) };
    assert!( unsafe { AccelCUDA_StreamIsIdle(stream) } );

    unsafe { AccelCUDA_FreeHostMemory(cuda, hostMem1D) };
    unsafe { AccelCUDA_FreeDeviceMemory2D(cuda, mem2D) };
    unsafe { AccelCUDA_FreeDeviceMemory(cuda, mem1D) };

    unsafe { AccelCUDA_Destroy(cuda) };

}
