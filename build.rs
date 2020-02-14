use cuda_config::*;

fn main()
{
    if cfg!(target_os = "windows") {
        println!(
            "cargo:rustc-link-search=native={}",
            find_cuda_windows().display()
        );
    } else {
        for path in find_cuda() {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    };

    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=nvrtc");
    println!("cargo:rerun-if-env-changed=CUDA_LIBRARY_PATH");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=compute_cuda/out_libs/*");
    println!("cargo:rustc-link-search=native=compute_cuda/out_libs");
    println!("cargo:rustc-link-lib=static=compute_cuda");
    println!("cargo:rustc-link-lib=static=al2o3_platform");
    println!("cargo:rustc-link-lib=static=al2o3_memory");

    let bindings = bindgen::Builder::default()
        .header("compute_cuda/src/accelcuda.h")
        .clang_arg("-Icompute_cuda/al2o3/al2o3_platform-src/include")
        .whitelist_function("AccelCUDA_.*")
        .generate()
        .expect("Unable to generate bindings");
    bindings.write_to_file("src/accelcuda.rs").expect("Unable to write bindings");

}