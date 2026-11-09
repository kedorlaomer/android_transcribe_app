#!/bin/bash
# Builds the debug APK with the GPU (OpenCL, Adreno) engine. Not part of the
# regular or F-Droid build: it compiles transcribe-cpp-sys from patched sources
# instead of the crates.io release.
#
#   ggml           llama.cpp PR #29740 (karusrus/llama.cpp, branch
#                  hexagon-int-hmx-nofp16), plus gpu/ggml.patch: the ggml
#                  changes transcribe.cpp 0.4.0 carries, and the Adreno 825
#                  added to the OpenCL backend's list of known GPUs
#   transcribe.cpp transcribe-cpp-sys 0.4.0 plus gpu/transcribe-cpp-sys.patch:
#                  TRANSCRIBE_WHISPER_CPU_DECODER (encoder on the GPU, decoder
#                  on the CPU), TRANSCRIBE_WHISPER_ENC_FA_F16 (F16 K/V for
#                  encoder attention, for the Hexagon NPU) and the Whisper
#                  KV-cache view fix
#   OpenCL         headers and the libOpenCL.so link stub of NDK r29 (the
#                  ghcr.io/snapdragon-toolchain/arm64-android:v0.8 image); at
#                  run time the app uses the phone's /vendor/lib64/libOpenCL.so
#
# Needs: docker, git, cargo-ndk, the Android NDK and JDK 17 (paths below).
# Work dir: $WORK (default /tmp/voice-ime-gpu). Output:
# app/build/outputs/apk/fdroid/debug/app-fdroid-debug.apk
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-/tmp/voice-ime-gpu}
LLAMA_REPO=https://github.com/karusrus/llama.cpp
LLAMA_COMMIT=08c83734f9c5c3b497e98b96f1a380f5a6084468
SYS_CRATE=https://static.crates.io/crates/transcribe-cpp-sys/transcribe-cpp-sys-0.4.0.crate
SYS_SHA256=7f0976932ee2a0d9bcfb7a90947d0a73771cee383b163bd7189a8e0d67da1157
NDK_IMAGE=ghcr.io/snapdragon-toolchain/arm64-android:v0.8

NDK=${ANDROID_NDK_HOME:-$HOME/Library/Android/sdk/ndk/28.0.13004108}
export JAVA_HOME=${JAVA_HOME:-/opt/homebrew/opt/openjdk@17}
export PATH=$JAVA_HOME/bin:$HOME/.cargo/bin:$PATH
case $(uname) in Darwin) HOST=darwin-x86_64 ;; *) HOST=linux-x86_64 ;; esac

mkdir -p "$WORK"
SRC=$WORK/transcribe-cpp-sys
OCL=$WORK/opencl

# --- Patched transcribe-cpp-sys source tree -------------------------------
if [ ! -e "$SRC/.patched" ]; then
    rm -rf "$SRC" "$WORK/llama.cpp"
    curl -sSfL "$SYS_CRATE" -o "$WORK/sys.crate"
    echo "$SYS_SHA256  $WORK/sys.crate" | shasum -a 256 -c -
    mkdir -p "$SRC" && tar -xzf "$WORK/sys.crate" -C "$SRC" --strip-components=1

    git init -q "$WORK/llama.cpp"
    git -C "$WORK/llama.cpp" fetch -q --depth 1 "$LLAMA_REPO" "$LLAMA_COMMIT"
    git -C "$WORK/llama.cpp" checkout -q FETCH_HEAD
    # The crate's ggml/ is replaced with the PR's, keeping only what the
    # crate ships (its include list; e.g. no tests or docs).
    for d in CMakeLists.txt cmake include src; do
        rm -rf "$SRC/ggml/$d"
        cp -R "$WORK/llama.cpp/ggml/$d" "$SRC/ggml/"
    done

    patch -s -d "$SRC" -p1 < "$REPO/gpu/ggml.patch"
    patch -s -d "$SRC" -p1 < "$REPO/gpu/transcribe-cpp-sys.patch"
    touch "$SRC/.patched"
fi

# --- OpenCL headers + link stub --------------------------------------------
if [ ! -e "$OCL/lib/libOpenCL.so" ]; then
    mkdir -p "$OCL"
    docker run --rm --platform linux/amd64 -v "$OCL:/o" "$NDK_IMAGE" bash -c '
        S=/opt/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr
        mkdir -p /o/CL /o/lib
        cp -L $S/include/CL/*.h $S/include/CL/*.hpp /o/CL/
        cp -L $S/lib/aarch64-linux-android/libOpenCL.so /o/lib/
        chmod -R u+w /o'
fi

# --- Native library ---------------------------------------------------------
cd "$REPO"
export ANDROID_NDK_HOME=$NDK ANDROID_NDK_ROOT=$NDK ANDROID_NDK=$NDK
export TRANSCRIBE_CMAKE_ARGS="-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 -DGGML_OPENCL=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=ON -DGGML_OPENCL_EMBED_KERNELS=ON -DOpenCL_INCLUDE_DIR=$OCL -DOpenCL_LIBRARY=$OCL/lib/libOpenCL.so"
export CARGO_TARGET_DIR=$WORK/target
# The [patch] rewrites Cargo.lock; the repo keeps its crates.io pin.
cp Cargo.lock "$WORK/Cargo.lock.orig"
trap 'cp "$WORK/Cargo.lock.orig" "$REPO/Cargo.lock"' EXIT
# rustflags repeat .cargo/config.toml's (a --config value replaces them) and
# link libOpenCL: the static ggml-opencl archive doesn't carry the dependency.
cargo ndk -t arm64-v8a -o "$WORK/jni" build --release \
    --config "patch.crates-io.transcribe-cpp-sys.path=\"$SRC\"" \
    --config "target.aarch64-linux-android.rustflags=[\"-C\",\"link-arg=-Wl,-z,max-page-size=16384\",\"-C\",\"link-arg=-lc++_shared\",\"-C\",\"link-arg=-L$OCL/lib\",\"-C\",\"link-arg=-lOpenCL\"]"

# --- APK: Gradle without its own cargo step ---------------------------------
JNI=app/src/main/jniLibs/arm64-v8a
mkdir -p "$JNI"
cp "$WORK/jni/arm64-v8a/libandroid_transcribe_app.so" "$JNI/"
cp "$NDK/toolchains/llvm/prebuilt/$HOST/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" "$JNI/"
"$NDK/toolchains/llvm/prebuilt/$HOST/bin/llvm-strip" --strip-debug "$JNI"/*.so
./gradlew -q assembleFdroidDebug -x cargoNdkBuild
ls -la app/build/outputs/apk/fdroid/debug/app-fdroid-debug.apk
