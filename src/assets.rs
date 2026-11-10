//! Extraction of the bundled speech model from APK assets into filesDir.

use jni::objects::JObject;
use jni::JNIEnv;
use std::path::{Path, PathBuf};

/// Asset directory (and filesDir subdirectory) holding the bundled GGUF.
const BUILTIN_MODEL_DIR: &str = "builtin-model";
/// Marker file written after a successful extraction. If this file is missing,
/// the directory is assumed to be incomplete (e.g. interrupted mid-extraction)
/// and the assets will be re-extracted.
const EXTRACTION_COMPLETE_MARKER: &str = ".extraction_complete";
/// Model directory of the pre-GGUF (ONNX) app versions; deleted on sight so
/// upgrades don't leave ~670 MB of dead files behind.
const LEGACY_MODEL_DIR: &str = "parakeet-tdt-0.6b-v3-int8";

/// Resolves the app's `filesDir` via the given Context.
pub fn files_dir(env: &mut JNIEnv, context: &JObject) -> anyhow::Result<PathBuf> {
    let files_dir_obj = env
        .call_method(context, "getFilesDir", "()Ljava/io/File;", &[])?
        .l()?;
    let path_str_obj = env
        .call_method(
            &files_dir_obj,
            "getAbsolutePath",
            "()Ljava/lang/String;",
            &[],
        )?
        .l()?;
    let path_string: String = env.get_string(&path_str_obj.into())?.into();
    Ok(PathBuf::from(path_string))
}

/// The directory the APK's native libraries are extracted to
/// (`ApplicationInfo.nativeLibraryDir`).
#[cfg(feature = "qnn")]
pub fn native_library_dir(env: &mut JNIEnv, context: &JObject) -> anyhow::Result<PathBuf> {
    let info = env
        .call_method(context, "getApplicationInfo", "()Landroid/content/pm/ApplicationInfo;", &[])?
        .l()?;
    let dir = env.get_field(&info, "nativeLibraryDir", "Ljava/lang/String;")?.l()?;
    let dir: String = env.get_string(&dir.into())?.into();
    Ok(PathBuf::from(dir))
}

/// Resolves the app's shared-storage media directory
/// (`Android/media/<package>/`), where the per-language models are placed.
pub fn media_dir(env: &mut JNIEnv, context: &JObject) -> anyhow::Result<PathBuf> {
    use jni::objects::JObjectArray;
    let dirs: JObjectArray = env
        .call_method(context, "getExternalMediaDirs", "()[Ljava/io/File;", &[])?
        .l()?
        .into();
    let first = env.get_object_array_element(&dirs, 0)?;
    if first.is_null() {
        anyhow::bail!("shared storage unavailable");
    }
    let path_str_obj = env
        .call_method(&first, "getAbsolutePath", "()Ljava/lang/String;", &[])?
        .l()?;
    let path_string: String = env.get_string(&path_str_obj.into())?.into();
    Ok(PathBuf::from(path_string))
}

/// True if this build bundles a model (F-Droid builds don't). Judged on the
/// APK assets rather than an earlier extraction, so it matches the app's
/// `ModelStore.hasBuiltinModel`.
pub fn has_builtin_model(env: &mut JNIEnv, context: &JObject) -> bool {
    match list_builtin_assets(env, context) {
        Ok(names) => names
            .iter()
            .any(|n| n.to_ascii_lowercase().ends_with(".gguf")),
        Err(e) => {
            let _ = env.exception_clear();
            log::warn!("cannot list bundled model assets: {}", e);
            false
        }
    }
}

fn list_builtin_assets(env: &mut JNIEnv, context: &JObject) -> anyhow::Result<Vec<String>> {
    use jni::objects::JObjectArray;

    let asset_manager = env
        .call_method(
            context,
            "getAssets",
            "()Landroid/content/res/AssetManager;",
            &[],
        )?
        .l()?;
    let dir = env.new_string(BUILTIN_MODEL_DIR)?;
    let list: JObjectArray = env
        .call_method(
            &asset_manager,
            "list",
            "(Ljava/lang/String;)[Ljava/lang/String;",
            &[(&dir).into()],
        )?
        .l()?
        .into();
    let mut names = Vec::new();
    for i in 0..env.get_array_length(&list)? {
        let name = env.get_object_array_element(&list, i)?;
        names.push(env.get_string(&name.into())?.into());
    }
    Ok(names)
}

/// Extracts the bundled model from APK assets (if not already done) and
/// returns the path of its GGUF file.
///
/// The keyboard, popup and speech service may all call this at the same time
/// on first start. Extraction is therefore serialized with a file lock and
/// written to a temporary directory that is renamed into place only once
/// complete, so no caller ever sees, deletes or loads another one's
/// half-written model.
pub fn extract_builtin_model(env: &mut JNIEnv, context: &JObject) -> anyhow::Result<PathBuf> {
    let base_path = files_dir(env, context)?;
    let model_dir = base_path.join(BUILTIN_MODEL_DIR);
    let marker_file = model_dir.join(EXTRACTION_COMPLETE_MARKER);

    // The marker only ever appears together with a complete directory.
    if marker_file.exists() {
        return find_gguf(&model_dir);
    }

    let _lock = ExtractionLock::acquire(&base_path.join(EXTRACTION_LOCK))?;
    // Another process may have finished extracting while we waited.
    if marker_file.exists() {
        return find_gguf(&model_dir);
    }

    let legacy_dir = base_path.join(LEGACY_MODEL_DIR);
    if legacy_dir.exists() {
        log::info!("Removing legacy ONNX model directory");
        let _ = std::fs::remove_dir_all(&legacy_dir);
    }

    let tmp_root = base_path.join(EXTRACTION_TMP_DIR);
    if tmp_root.exists() {
        std::fs::remove_dir_all(&tmp_root)?;
    }
    std::fs::create_dir_all(&tmp_root)?;

    // Don't leave a partial copy (hundreds of MB) behind if this fails, e.g.
    // because storage is full.
    let extracted = extract_into(env, context, &tmp_root);
    if extracted.is_err() {
        let _ = std::fs::remove_dir_all(&tmp_root);
    }
    let tmp_model_dir = extracted?;

    if model_dir.exists() {
        log::info!("Replacing incomplete model directory");
        std::fs::remove_dir_all(&model_dir)?;
    }
    std::fs::rename(&tmp_model_dir, &model_dir)?;
    let _ = std::fs::remove_dir_all(&tmp_root);
    log::info!("Asset extraction complete");

    find_gguf(&model_dir)
}

/// Copies the bundled model into `tmp_root` and marks it complete; returns
/// the model directory inside it.
fn extract_into(env: &mut JNIEnv, context: &JObject, tmp_root: &Path) -> anyhow::Result<PathBuf> {
    let asset_manager_obj = env
        .call_method(
            context,
            "getAssets",
            "()Landroid/content/res/AssetManager;",
            &[],
        )?
        .l()?;
    copy_assets_recursively(env, &asset_manager_obj, BUILTIN_MODEL_DIR, tmp_root)?;

    // Only mark (and publish) the extraction if the model actually arrived.
    let tmp_model_dir = tmp_root.join(BUILTIN_MODEL_DIR);
    find_gguf(&tmp_model_dir)?;
    std::fs::write(tmp_model_dir.join(EXTRACTION_COMPLETE_MARKER), "ok")?;
    Ok(tmp_model_dir)
}

/// Lock file in filesDir serializing extraction across processes.
const EXTRACTION_LOCK: &str = ".builtin-model.lock";
/// Scratch directory in filesDir that extraction writes into.
const EXTRACTION_TMP_DIR: &str = ".builtin-model.tmp";

/// An exclusive `flock` on a file, released when dropped (or when the
/// process dies, so a killed extraction never leaves a stale lock).
struct ExtractionLock(std::fs::File);

impl ExtractionLock {
    fn acquire(path: &Path) -> anyhow::Result<ExtractionLock> {
        use std::os::unix::io::AsRawFd;
        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(path)?;
        loop {
            if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } == 0 {
                return Ok(ExtractionLock(file));
            }
            let err = std::io::Error::last_os_error();
            if err.kind() != std::io::ErrorKind::Interrupted {
                return Err(err.into());
            }
        }
    }
}

impl Drop for ExtractionLock {
    fn drop(&mut self) {
        use std::os::unix::io::AsRawFd;
        unsafe { libc::flock(self.0.as_raw_fd(), libc::LOCK_UN) };
    }
}

/// Removes the extraction marker so the next load re-extracts the bundled
/// model — called when a load fails on (likely corrupt) extracted files.
pub fn invalidate_builtin_model(model_path: &Path) {
    if let Some(dir) = model_path.parent() {
        let marker = dir.join(EXTRACTION_COMPLETE_MARKER);
        if marker.exists() {
            log::warn!("Model load failed, removing extraction marker for re-extraction");
            let _ = std::fs::remove_file(&marker);
        }
    }
}

/// Returns the single `.gguf` file in `dir`.
fn find_gguf(dir: &Path) -> anyhow::Result<PathBuf> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().is_some_and(|ext| ext.eq_ignore_ascii_case("gguf")) {
            return Ok(path);
        }
    }
    anyhow::bail!("no GGUF file found in {}", dir.display())
}

fn copy_assets_recursively(
    env: &mut JNIEnv,
    asset_manager: &JObject,
    path: &str,
    target_root: &Path,
) -> anyhow::Result<()> {
    use jni::objects::JObjectArray;

    let path_jstring = env.new_string(path)?;
    let list_array_obj = env
        .call_method(
            asset_manager,
            "list",
            "(Ljava/lang/String;)[Ljava/lang/String;",
            &[(&path_jstring).into()],
        )?
        .l()?;

    let list_array: JObjectArray = list_array_obj.into();
    let len = env.get_array_length(&list_array)?;

    if len == 0 {
        return copy_asset_file(env, asset_manager, path, target_root);
    }

    let target_dir = target_root.join(path);
    std::fs::create_dir_all(&target_dir)?;

    for i in 0..len {
        let file_name_obj = env.get_object_array_element(&list_array, i)?;
        let file_name: String = env.get_string(&file_name_obj.into())?.into();

        let child_path = if path.is_empty() {
            file_name
        } else {
            format!("{}/{}", path, file_name)
        };

        copy_assets_recursively(env, asset_manager, &child_path, target_root)?;
    }
    Ok(())
}

fn copy_asset_file(
    env: &mut JNIEnv,
    asset_manager: &JObject,
    asset_path: &str,
    target_root: &Path,
) -> anyhow::Result<()> {
    let path_jstring = env.new_string(asset_path)?;
    let result = env.call_method(
        asset_manager,
        "open",
        "(Ljava/lang/String;)Ljava/io/InputStream;",
        &[(&path_jstring).into()],
    );

    let stream_obj = match result {
        Ok(stream_val) => stream_val.l()?,
        Err(e) => {
            // A failed Java call leaves an exception pending, which would
            // abort the process on the next JNI call.
            let _ = env.exception_clear();
            anyhow::bail!("cannot open asset {}: {}", asset_path, e);
        }
    };
    let target_file_path = target_root.join(asset_path);
    let copied = copy_stream(env, &stream_obj, &target_file_path);
    let _ = env.call_method(&stream_obj, "close", "()V", &[]);
    if copied.is_err() {
        let _ = env.exception_clear();
    }
    copied?;
    log::info!("Extracted: {:?}", target_file_path);
    Ok(())
}

/// Copies a Java InputStream to a file, flushed to disk before returning.
fn copy_stream(env: &mut JNIEnv, stream: &JObject, target: &Path) -> anyhow::Result<()> {
    use std::io::Write;
    // Large chunks: every read is a JNI round trip, and the model is ~500 MB.
    const CHUNK: usize = 1 << 20;
    let mut file = std::fs::File::create(target)?;
    let mut buffer = vec![0u8; CHUNK];
    let buffer_j = env.new_byte_array(CHUNK as i32)?;
    loop {
        let bytes_read = env
            .call_method(stream, "read", "([B)I", &[(&buffer_j).into()])?
            .i()?;
        if bytes_read < 0 {
            break;
        }
        let n = bytes_read as usize;
        let slice = unsafe { std::slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut i8, n) };
        env.get_byte_array_region(&buffer_j, 0, slice)?;
        file.write_all(&buffer[..n])?;
    }
    file.sync_all()?;
    Ok(())
}
