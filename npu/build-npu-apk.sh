#!/bin/bash
# Builds the debug APK with the NPU engine: Whisper on the Qualcomm Hexagon
# NPU through ONNX Runtime's QNN execution provider (cargo feature `qnn`).
# Not part of the regular or F-Droid builds: the QNN libraries are
# Qualcomm's, closed source.
#
#   ONNX Runtime   com.microsoft.onnxruntime:onnxruntime-android-qnn
#                  (Maven Central): libonnxruntime.so and its headers
#   QNN            com.qualcomm.qti:qnn-runtime (Maven Central): the HTP
#                  backend (libQnnHtp.so), the CPU-side stub and NPU-side
#                  skel for Hexagon v73, libQnnSystem.so. The online graph
#                  compiler (libQnnHtpPrepare.so, 85 MB) is left out: the
#                  models are precompiled context binaries.
#
# The GPU build's patched transcribe.cpp is not used; the regular one from
# crates.io handles the language ID model and non-NPU models on the CPU.
#
# Needs: cargo-ndk, the Android NDK and JDK 17 (paths below).
# Work dir: $WORK (default /tmp/voice-ime-npu). Output:
# app/build/outputs/apk/fdroid/debug/app-fdroid-debug.apk
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-/tmp/voice-ime-npu}
MAVEN=https://repo1.maven.org/maven2
ORT_VERSION=1.29.0
QNN_VERSION=2.50.0
ORT_SHA256=cc4cfd72a47703aee65d8e6ce7448fa7488e3fff6556f9d6d354ec1687347cfd
QNN_SHA256=b507656e4031d8aa6a7ab0fecf725b941096737ad7440ad7ea288f7c9cc9d743
QNN_LIBS="libQnnHtp.so libQnnHtpV73Stub.so libQnnHtpV73Skel.so libQnnSystem.so"

NDK=${ANDROID_NDK_HOME:-$HOME/Library/Android/sdk/ndk/28.0.13004108}
export JAVA_HOME=${JAVA_HOME:-/opt/homebrew/opt/openjdk@17}
export PATH=$JAVA_HOME/bin:$HOME/.cargo/bin:$PATH
case $(uname) in Darwin) HOST=darwin-x86_64 ;; *) HOST=linux-x86_64 ;; esac

fetch() {  # fetch <url> <sha256> <dest>
    [ -e "$3" ] || curl -sSfL "$1" -o "$3"
    echo "$2  $3" | shasum -a 256 -c - >/dev/null
}

# --- ONNX Runtime and QNN ---------------------------------------------------
mkdir -p "$WORK"
ORT=$WORK/ort
if [ ! -e "$ORT/jni/arm64-v8a/libonnxruntime.so" ]; then
    fetch "$MAVEN/com/microsoft/onnxruntime/onnxruntime-android-qnn/$ORT_VERSION/onnxruntime-android-qnn-$ORT_VERSION.aar" \
        "$ORT_SHA256" "$WORK/ort.aar"
    rm -rf "$ORT" && mkdir -p "$ORT" && unzip -q "$WORK/ort.aar" 'headers/*' 'jni/arm64-v8a/libonnxruntime.so' -d "$ORT"
fi
QNN=$WORK/qnn
if [ ! -e "$QNN/libQnnHtp.so" ]; then
    fetch "$MAVEN/com/qualcomm/qti/qnn-runtime/$QNN_VERSION/qnn-runtime-$QNN_VERSION.aar" \
        "$QNN_SHA256" "$WORK/qnn.aar"
    rm -rf "$QNN" && mkdir -p "$QNN"
    for lib in $QNN_LIBS; do unzip -q -j "$WORK/qnn.aar" "jni/arm64-v8a/$lib" -d "$QNN"; done
fi

# --- Native library ---------------------------------------------------------
cd "$REPO"
export ANDROID_NDK_HOME=$NDK ANDROID_NDK_ROOT=$NDK ANDROID_NDK=$NDK
export TRANSCRIBE_CMAKE_ARGS="-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16"
export CARGO_TARGET_DIR=$WORK/target QNN_ORT_DIR=$ORT
cargo ndk -t arm64-v8a -o "$WORK/jni" build --release --features qnn

# --- APK: Gradle without its own cargo step ---------------------------------
JNI=app/src/main/jniLibs/arm64-v8a
rm -rf "$JNI" && mkdir -p "$JNI"
cp "$WORK/jni/arm64-v8a/libandroid_transcribe_app.so" "$ORT/jni/arm64-v8a/libonnxruntime.so" "$JNI/"
for lib in $QNN_LIBS; do cp "$QNN/$lib" "$JNI/"; done
cp "$NDK/toolchains/llvm/prebuilt/$HOST/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" "$JNI/"
"$NDK/toolchains/llvm/prebuilt/$HOST/bin/llvm-strip" --strip-debug "$JNI/libandroid_transcribe_app.so" "$JNI/libc++_shared.so"
./gradlew -q assembleFdroidDebug -x cargoNdkBuild -PnpuBuild=true
ls -la app/build/outputs/apk/fdroid/debug/app-fdroid-debug.apk
