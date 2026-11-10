use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    // transcribe-cpp-sys reconstructs its link line from a generic Unix
    // manifest that lists `pthread` and the C++ runtime. Bionic has neither a
    // separate libpthread (pthreads live in libc) nor a full libstdc++ (the
    // real C++ runtime is libc++_shared, which the app already bundles), so
    // satisfy the former with an empty archive and link the latter explicitly.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "android" {
        let out = PathBuf::from(env::var("OUT_DIR").unwrap());
        fs::write(out.join("libpthread.a"), b"!<arch>\n").unwrap();
        println!("cargo:rustc-link-search=native={}", out.display());
        println!("cargo:rustc-link-lib=dylib=c++_shared");
    }

    // NPU build: the C++ wrapper around ONNX Runtime (qnn/qnn_whisper.cpp),
    // compiled against the onnxruntime-android-qnn AAR unpacked in
    // QNN_ORT_DIR (headers/ and jni/arm64-v8a/libonnxruntime.so).
    #[cfg(feature = "qnn")]
    {
        let ort = PathBuf::from(env::var("QNN_ORT_DIR").expect("QNN_ORT_DIR: the unpacked onnxruntime-android-qnn AAR"));
        println!("cargo:rerun-if-changed=qnn/qnn_whisper.cpp");
        println!("cargo:rerun-if-env-changed=QNN_ORT_DIR");
        cc::Build::new()
            .cpp(true)
            .std("c++17")
            .file("qnn/qnn_whisper.cpp")
            .include(ort.join("headers"))
            .cpp_link_stdlib(None)
            .compile("qnn_whisper");
        println!("cargo:rustc-link-search=native={}", ort.join("jni/arm64-v8a").display());
        println!("cargo:rustc-link-lib=dylib=onnxruntime");
    }
}
