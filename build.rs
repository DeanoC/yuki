use cuda_config::*;
use std::path::Path;
use std::{env, fs, io};
use std::fs::DirEntry;

// one possible implementation of walking a directory only visiting files
fn visit_dirs<F>(ref dir: &Path, cb: F) -> io::Result<()> where F: Fn(&DirEntry), F: Copy {
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit_dirs(&path, cb)?;
            } else {
                cb(&entry);
            }
        }
    }
    Ok(())
}

fn copy_ptxs(ref src_dir_path: &Path, ref dst_dir_path: &Path) {
    visit_dirs(src_dir_path, |e: &DirEntry| {
        let p = e.path();
        match p.extension() {
            Some(ext) if ext == "ptx" => {
                let pc = p.clone();
                let v: Vec<&str> = pc.file_name().unwrap().to_str().unwrap()
                    .rsplitn(2, "_generated_").collect();
                if !v.is_empty() {
                    let src = p;
                    let dst = dst_dir_path.join(v[0]);
                    println!("copying {}, {}", src.to_path_buf().display(), dst.to_path_buf().display());
                    fs::copy(src.to_path_buf(), dst.to_path_buf()).expect("copy failed");
                }
            }
            Some(..) | None => ()
        }
    }).unwrap();
}

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
    println!("{}", dest_path.display());

    let bindings = bindgen::Builder::default()
        .header("compute_cuda/include/accelcuda.h")
        .clang_arg("-Icompute_cuda/al2o3/al2o3_platform-src/include")
        .whitelist_function("AccelCUDA_.*")
        .generate()
        .expect("Unable to generate bindings");
    bindings.write_to_file(dest_path).expect("Unable to write bindings");

    let target_dir_path = env::current_dir().unwrap();//env::var_os("OUT_DIR").unwrap();
    let target_dir_path = Path::new(&target_dir_path);
    let ptx_dir_path = target_dir_path.join("resources").join("gpu");

    fs::create_dir_all(&ptx_dir_path).unwrap();

    copy_ptxs("compute_cuda\\ptxs".as_ref(), &ptx_dir_path);
}