use cuda_config::*;
use std::path::Path;
use std::{env, fs};

fn copy(target_dir_path: &Path, file_name: &Path) {
    let wd = env::current_dir().unwrap();
    let fnonly = Path::new(file_name).file_name().unwrap();
    let src = Path::new( &wd).join(file_name);
    let dst = Path::new(&target_dir_path).join(fnonly);

    print!("copying {} to {}", src.display(), dst.display());
    fs::copy(src, dst).unwrap();
}

const PTX_FILE: &str = "compute_cuda\\ptxs\\cuda_compile_ptx_1_generated_test.cu.ptx";

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

    println!("cargo:rerun-if-changed=compute_cuda/out_libs/*");
    println!("cargo:rustc-link-search=native=compute_cuda/out_libs");
    println!("cargo:rustc-link-lib=static=compute_cuda");
    println!("cargo:rustc-link-lib=static=al2o3_platform");
    println!("cargo:rustc-link-lib=static=al2o3_memory");

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("accelcuda.rs");

    let bindings = bindgen::Builder::default()
        .header("compute_cuda/include/accelcuda.h")
        .clang_arg("-Icompute_cuda/al2o3/al2o3_platform-src/include")
        .whitelist_function("AccelCUDA_.*")
        .generate()
        .expect("Unable to generate bindings");
    bindings.write_to_file(dest_path).expect("Unable to write bindings");

    let target_dir_path = env::var("OUT_DIR").unwrap();
    copy(Path::new(&target_dir_path), Path::new(&PTX_FILE));

}