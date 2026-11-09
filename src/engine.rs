//! Loads and serves the speech-to-text engine (transcribe.cpp, GGUF models).
//!
//! The engine is a process-wide singleton behind `Arc<Mutex<..>>`: a
//! transcribe.cpp session may only be used by one thread at a time, which the
//! mutex guarantees. Model selection is read from marker files in filesDir
//! (written by `ModelsActivity`): an imported GGUF if one is selected, the
//! bundled model otherwise — with the bundled model as fallback if the
//! imported one fails to load.

use once_cell::sync::Lazy;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::SystemTime;

use jni::objects::{GlobalRef, JObject};
use jni::JNIEnv;

use crate::assets;

/// File in filesDir naming the imported GGUF (a file under `models/`) to use
/// instead of the bundled model. Absent or empty = bundled model.
const ACTIVE_MODEL_FILE: &str = "active_model";
/// File in filesDir with an optional language hint — a locale like `en-US`
/// or `auto`. Absent or empty = let the model autodetect.
const MODEL_LANGUAGE_FILE: &str = "model_language";
/// Marker file in filesDir: when present, models that support translation
/// (e.g. Whisper) translate speech to English instead of transcribing it.
/// Ignored by models without translation support.
const MODEL_TRANSLATE_FILE: &str = "model_translate";
/// Optional file in filesDir with the CPU thread count for inference.
/// Absent/invalid/0 = default (all cores).
const MODEL_THREADS_FILE: &str = "model_threads";

/// Per-language Whisper models, placed in the app's shared media directory
/// (`Android/media/<package>/models/`) rather than bundled or imported. When
/// the language setting names one of these languages, its model is loaded in
/// place of the imported/bundled one. English and German share stock
/// large-v3-turbo; Slovak and Hebrew use fine-tunes, which measured clearly
/// better than stock (FLEURS: sk 6.4% vs 10.1% WER, he 18.4% vs 29.5%).
const LANGUAGE_MODELS: &[(&str, &str)] = &[
    ("en", "whisper-large-v3-turbo-q8_0.bin"),
    ("de", "whisper-large-v3-turbo-q8_0.bin"),
    ("sk", "whisper-large-v3-turbo-sk-q8_0.bin"),
    ("he", "whisper-large-v3-turbo-he-q8_0.bin"),
];

/// Language ID model, in the same directory as the per-language models. With
/// it present and the language setting on automatic, each run is classified
/// among the [`LANGUAGE_MODELS`] languages and sent to that language's model.
/// On whole dictation chunks it picked the right one of the four languages
/// for 98-100% of FLEURS clips and for all 29 chunks of the test dictation.
const LANGID_MODEL: &str = "lang-id-voxlingua107-ecapa-Q8_0.gguf";
/// Below this probability (over the four languages) a run keeps the previous
/// run's language rather than trusting a guess.
const LANGID_MIN_P: f32 = 0.5;
/// Language assumed until a run has been identified confidently.
const LANGID_DEFAULT: &str = "en";

/// The language-model file for a language tag (`de-DE` matches `de`), if any.
fn language_model_file(language: Option<&str>) -> Option<&'static str> {
    let primary = language?.split('-').next()?.to_ascii_lowercase();
    LANGUAGE_MODELS
        .iter()
        .find(|(lang, _)| *lang == primary)
        .map(|(_, file)| *file)
}

/// Longest audio passed to the model in one run (15 s). Whisper decodes a
/// 30 s window either way, but on continuous dictation, independent runs of
/// at most 15 s cut at quiet points were clearly more accurate than whole
/// recordings, where the decoder drifted or looped. Longer input is split at
/// quiet points and the texts joined.
const MAX_RUN_SAMPLES: usize = 15 * 16_000;
/// When splitting, search this far back from the hard boundary for the
/// quietest point so words aren't cut mid-syllable.
const SPLIT_SEARCH_SAMPLES: usize = 7 * 16_000;

/// Length of the next piece of `samples` to transcribe in one model run:
/// everything if it fits, otherwise up to the quietest point near the limit.
pub fn next_run_len(samples: &[f32]) -> usize {
    if samples.len() <= MAX_RUN_SAMPLES {
        samples.len()
    } else {
        crate::audio::find_quietest_split(
            samples,
            MAX_RUN_SAMPLES - SPLIT_SEARCH_SAMPLES,
            MAX_RUN_SAMPLES,
        )
    }
}

/// True if `samples` is long enough that [`next_run_len`] would split it.
pub fn exceeds_one_run(samples: &[f32]) -> bool {
    samples.len() > MAX_RUN_SAMPLES
}

/// Per-run language routing (language setting on automatic).
struct AutoLanguage {
    langid: transcribe_cpp::LangIdSession,
    /// Where the per-language models live.
    models_dir: PathBuf,
    /// The most recently used model other than the current one, kept loaded
    /// so alternating between two languages doesn't reload each time.
    spare: Option<(PathBuf, transcribe_cpp::Session)>,
    threads: i32,
    /// Language of the last run, used when identification is unsure.
    last: String,
}

/// A loaded transcribe.cpp session plus the options applied to every run.
pub struct Engine {
    session: transcribe_cpp::Session,
    session_path: PathBuf,
    language: Option<String>,
    auto: Option<AutoLanguage>,
    task: transcribe_cpp::Task,
    /// Family-specific decode options attached to every run; `None` for
    /// models that don't take the whisper run extension.
    run_ext: Option<transcribe_cpp::RunExtension>,
    /// Status reported once loading succeeded; carries a warning when the
    /// translate setting can't do what the user expects with this model.
    ready_status: &'static str,
}

impl Engine {
    fn load(
        model_path: &Path,
        language: Option<String>,
        translate: bool,
        threads: i32,
    ) -> Result<Engine, String> {
        let model = load_model(model_path)?;
        // Translation is gated on the model's capabilities: models without it
        // (e.g. Parakeet) silently keep transcribing, so the setting can stay
        // on while switching models.
        let supports_translate = model
            .capabilities()
            .map(|c| c.supports_translate)
            .unwrap_or(false);
        let task = if translate && supports_translate {
            transcribe_cpp::Task::Translate
        } else {
            if translate {
                log::info!("translate requested but unsupported by this model; transcribing");
            }
            transcribe_cpp::Task::Transcribe
        };
        // Silently doing something other than what the translate switch says
        // looks like a bug (still-untranslated subtitles), so say it in the
        // status. Whisper Turbo is special-cased: it advertises translation
        // but was distilled without translation data — verified on-device to
        // keep transcribing German as German with task=Translate.
        let ready_status = if translate && task == transcribe_cpp::Task::Transcribe {
            "Ready (this model can't translate)"
        } else if translate && model.variant().contains("turbo") {
            "Ready (note: Whisper Turbo translates poorly; use another Whisper)"
        } else {
            "Ready"
        };
        // Whisper's stock recipe re-decodes a chunk at up to five higher
        // temperatures when its quality gates fail, so one noisy chunk can
        // cost several full decodes. For an interactive app a single greedy
        // pass is the better trade: worst case is a worse line of text, not
        // a multiplied wait. temperature_inc = 0 turns the retry ladder off;
        // models that don't take the whisper run extension are unaffected.
        let run_ext = if model.accepts_ext(
            transcribe_cpp::ExtSlot::Run,
            transcribe_cpp::sys::TRANSCRIBE_EXT_KIND_WHISPER_RUN,
        ) {
            // Each run also starts without the previous run's text as a
            // prompt: carried-over context made the Slovak fine-tune loop.
            Some(transcribe_cpp::RunExtension::Whisper(
                transcribe_cpp::WhisperRunOptions {
                    temperature_inc: Some(0.0),
                    condition_on_prev_tokens: Some(false),
                    ..Default::default()
                },
            ))
        } else {
            None
        };

        log::info!(
            "engine: {} threads, task {:?}, single-pass decode: {}",
            threads,
            task,
            run_ext.is_some()
        );
        let session = open_session(&model, threads)?;
        Ok(Engine {
            session,
            session_path: model_path.to_path_buf(),
            language,
            auto: None,
            task,
            run_ext,
            ready_status,
        })
    }

    /// Loads the engine for automatic language routing: starts on the model
    /// for [`LANGID_DEFAULT`] and loads the others on demand.
    fn load_auto(models_dir: &Path, translate: bool, threads: i32) -> Result<Engine, String> {
        // Language ID is small and runs once per run; on the CPU it is faster
        // than on the GPU and doesn't compete with the encoder for it.
        let langid_model = load_model_with(&models_dir.join(LANGID_MODEL), transcribe_cpp::Backend::Cpu)?;
        let langid = langid_model
            .langid_session_with(&transcribe_cpp::LangIdSessionOptions { n_threads: threads })
            .map_err(|e| e.to_string())?;
        let first = language_model_file(Some(LANGID_DEFAULT)).expect("default language has a model");
        let mut engine = Engine::load(
            &models_dir.join(first),
            Some(LANGID_DEFAULT.to_string()),
            translate,
            threads,
        )?;
        engine.auto = Some(AutoLanguage {
            langid,
            models_dir: models_dir.to_path_buf(),
            spare: None,
            threads,
            last: LANGID_DEFAULT.to_string(),
        });
        log::info!("engine: automatic language routing");
        Ok(engine)
    }

    /// Transcribes 16 kHz mono f32 samples to text. Input longer than
    /// [`MAX_RUN_SAMPLES`] is transcribed in quiet-point chunks.
    pub fn transcribe(&mut self, samples: Vec<f32>) -> Result<String, String> {
        if samples.len() <= MAX_RUN_SAMPLES {
            return self.route_and_run(&samples);
        }

        let mut text = String::new();
        let mut rest: &[f32] = &samples;
        while !rest.is_empty() {
            let take = next_run_len(rest);
            let piece = self.route_and_run(&rest[..take])?;
            let piece = piece.trim();
            if !piece.is_empty() {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(piece);
            }
            rest = &rest[take..];
        }
        Ok(text)
    }

    /// One run; with automatic routing, first identifies the run's language
    /// and switches to its model. A decode loop at the end of the run's text
    /// is cut off (see [`crate::repetition`]).
    fn route_and_run(&mut self, samples: &[f32]) -> Result<String, String> {
        if self.auto.is_some() {
            let lang = self.identify(samples);
            self.switch_to(&lang)?;
        }
        let text = self.run(samples)?;
        let trimmed = crate::repetition::trim_loop(&text);
        if trimmed.len() != text.len() {
            log::warn!(
                "dropped a repeating tail: {} of {} words kept",
                trimmed.split_whitespace().count(),
                text.split_whitespace().count()
            );
        }
        Ok(trimmed)
    }

    /// The run's language among [`LANGUAGE_MODELS`], or the previous run's
    /// when identification fails (e.g. under 0.5 s of audio) or is unsure.
    fn identify(&mut self, samples: &[f32]) -> String {
        let auto = self.auto.as_mut().expect("automatic routing");
        let allowed = LANGUAGE_MODELS.iter().map(|(lang, _)| lang.to_string()).collect();
        let options = transcribe_cpp::LangIdOptions { allowed: Some(allowed) };
        match auto.langid.run(samples, &options) {
            Ok(result) => {
                let top = result.candidates.first();
                // VoxLingua107 labels Hebrew with the old code `iw`.
                let code = top.map(|c| if c.code == "iw" { "he" } else { c.code.as_str() });
                let p = top.map(|c| c.p).unwrap_or(0.0);
                match code {
                    Some(code) if p >= LANGID_MIN_P => {
                        log::info!("language: {} (p={:.2})", code, p);
                        auto.last = code.to_string();
                    }
                    _ => log::info!("language unsure ({:?}, p={:.2}); keeping {}", code, p, auto.last),
                }
            }
            Err(e) => log::info!("language ID failed ({}); keeping {}", e, auto.last),
        }
        auto.last.clone()
    }

    /// Makes the current session the one for `lang`'s model, swapping in the
    /// spare or loading it (the current one then becomes the spare).
    fn switch_to(&mut self, lang: &str) -> Result<(), String> {
        self.language = Some(lang.to_string());
        let auto = self.auto.as_mut().expect("automatic routing");
        let file = language_model_file(Some(lang)).ok_or_else(|| format!("no model for {}", lang))?;
        let path = auto.models_dir.join(file);
        if path == self.session_path {
            return Ok(());
        }
        let (path, session) = match auto.spare.take() {
            Some((spare_path, spare)) if spare_path == path => (spare_path, spare),
            other => {
                // Free the old spare first: it is the least recently used.
                drop(other);
                let started = std::time::Instant::now();
                let session = open_session(&load_model(&path)?, auto.threads)?;
                log::info!("loaded {} in {:.2}s", file, started.elapsed().as_secs_f64());
                (path, session)
            }
        };
        let previous_path = std::mem::replace(&mut self.session_path, path);
        let previous = std::mem::replace(&mut self.session, session);
        auto.spare = Some((previous_path, previous));
        Ok(())
    }

    /// One model run. A rejected language hint is degraded instead of
    /// failing the transcription: `de-DE` retries as `de`, then as no hint
    /// (each model knows a different set of tags — e.g. Parakeet v3 takes
    /// locales/short codes, English-only models take none). The degraded
    /// value is kept so later runs skip the rejected attempts.
    fn run(&mut self, samples: &[f32]) -> Result<String, String> {
        loop {
            let opts = transcribe_cpp::RunOptions {
                language: self.language.clone(),
                task: self.task,
                family: self.run_ext.clone(),
                ..Default::default()
            };
            match self.session.run(samples, &opts) {
                Ok(t) => return Ok(t.text),
                // Some model families stop a decode that hit its length
                // budget or started repeating itself; the text up to that
                // point is still the best answer, so keep it.
                Err(transcribe_cpp::Error::OutputTruncated { message, partial })
                | Err(transcribe_cpp::Error::OutputRepetition { message, partial }) => {
                    log::warn!("decode stopped early ({}); using the partial text", message);
                    return Ok(partial.map(|t| t.text).unwrap_or_default());
                }
                Err(transcribe_cpp::Error::Unsupported(msg)) if self.language.is_some() => {
                    let lang = self.language.take().unwrap();
                    self.language = lang.split_once('-').map(|(primary, _)| primary.to_string());
                    log::warn!(
                        "language hint '{}' rejected ({}); retrying with {:?}",
                        lang,
                        msg,
                        self.language
                    );
                }
                Err(e) => return Err(e.to_string()),
            }
        }
    }
}

/// Loads a model file.
fn load_model(path: &Path) -> Result<transcribe_cpp::Model, String> {
    load_model_with(path, transcribe_cpp::Backend::Auto)
}

/// Loads a model on `backend` (Auto: the GPU where this build has one).
fn load_model_with(path: &Path, backend: transcribe_cpp::Backend) -> Result<transcribe_cpp::Model, String> {
    if !path.is_file() {
        return Err(format!("model file not found: {}", path.display()));
    }
    let options = transcribe_cpp::ModelOptions { backend, device: None };
    let model = transcribe_cpp::Model::load_with(path, &options).map_err(|e| e.to_string())?;
    log::info!("loaded {} on {}", path.display(), model.backend());
    Ok(model)
}

/// Environment for GPU builds, set before the first model load: the
/// encoder runs on the GPU and the decoder on the CPU (one decode step at a
/// time is slow on the GPU), and compiled GPU kernels are cached in the
/// app's cache dir (there is no TMPDIR in an app process; without a cache
/// every start recompiles them, which takes tens of seconds).
fn configure_gpu_env(files_dir: &Path) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        std::env::set_var("TRANSCRIBE_WHISPER_CPU_DECODER", "1");
        if let Some(cache) = files_dir.parent().map(|app| app.join("cache").join("cl-cache")) {
            let _ = std::fs::create_dir_all(&cache);
            std::env::set_var("GGML_OPENCL_KERNEL_CACHE_DIR", cache);
        }
    });
}

/// Opens a transcription session on a loaded model.
fn open_session(model: &transcribe_cpp::Model, threads: i32) -> Result<transcribe_cpp::Session, String> {
    let options = transcribe_cpp::SessionOptions {
        n_threads: threads,
        ..Default::default()
    };
    model.session_with(&options).map_err(|e| e.to_string())
}

/// Holds the loaded engine singleton.
static GLOBAL_ENGINE: Lazy<Mutex<Option<Arc<Mutex<Engine>>>>> =
    Lazy::new(|| Mutex::new(None));

/// Where the engine's settings and per-language models live.
#[derive(Debug, Clone)]
struct ConfigDirs {
    files_dir: PathBuf,
    media_dir: Option<PathBuf>,
}

/// The settings the loaded engine was built from, and where they live.
/// Comparing against the files on disk before transcribing makes any user of
/// the engine pick up a changed model or language, whichever screen changed
/// it, instead of keeping the old one until the process dies.
static LOADED_CONFIG: Lazy<Mutex<Option<(ConfigDirs, EngineConfig)>>> =
    Lazy::new(|| Mutex::new(None));

#[derive(Debug, Clone, PartialEq)]
struct EngineConfig {
    /// The per-language model for the language setting, if it has one;
    /// takes precedence over `active_model`.
    language_model: Option<PathBuf>,
    /// The per-language models' directory when the language setting is
    /// automatic and the language ID model is present there; takes
    /// precedence over `active_model`.
    auto_models_dir: Option<PathBuf>,
    active_model: Option<String>,
    /// Size and mtime of the model file to load, so replacing a file under
    /// the same name also reloads.
    model_stamp: Option<(u64, Option<SystemTime>)>,
    language: Option<String>,
    translate: bool,
    threads: Option<i32>,
}

impl EngineConfig {
    fn read(files_dir: &Path, media_dir: Option<&Path>) -> EngineConfig {
        // Absent/empty or "auto" = no hint (the model's automatic mode).
        // A tag the model doesn't know is degraded per run (Engine::run).
        let language = read_config(&files_dir.join(MODEL_LANGUAGE_FILE))
            .filter(|l| !l.eq_ignore_ascii_case("auto"));
        let models_dir = media_dir.map(|dir| dir.join("models"));
        let language_model = models_dir
            .as_ref()
            .zip(language_model_file(language.as_deref()))
            .map(|(dir, file)| dir.join(file));
        let auto_models_dir = models_dir
            .filter(|dir| language.is_none() && dir.join(LANGID_MODEL).is_file());
        let active_model = read_config(&files_dir.join(ACTIVE_MODEL_FILE));
        let model_path = language_model
            .clone()
            .or_else(|| auto_models_dir.as_ref().map(|dir| dir.join(LANGID_MODEL)))
            .or_else(|| active_model.as_ref().map(|name| files_dir.join("models").join(name)));
        let model_stamp = model_path.and_then(|path| {
            let meta = std::fs::metadata(path).ok()?;
            Some((meta.len(), meta.modified().ok()))
        });
        EngineConfig {
            language_model,
            auto_models_dir,
            active_model,
            model_stamp,
            language,
            translate: files_dir.join(MODEL_TRANSLATE_FILE).exists(),
            threads: read_config(&files_dir.join(MODEL_THREADS_FILE))
                .and_then(|s| s.parse::<i32>().ok())
                .filter(|&n| n > 0),
        }
    }
}

/// True if the model settings changed since the engine was loaded.
fn config_is_stale() -> bool {
    match &*LOADED_CONFIG.lock().unwrap() {
        Some((dirs, loaded)) => {
            EngineConfig::read(&dirs.files_dir, dirs.media_dir.as_deref()) != *loaded
        }
        None => false,
    }
}

/// Drops the engine if its settings are outdated, so the next load picks up
/// the current ones. Re-checked under the load lock, so concurrent callers
/// don't throw away a load that already uses the new settings.
fn reset_if_stale() {
    let (lock, cvar) = &*LOAD_STATE;
    let mut state = lock.lock().unwrap();
    while *state == LoadState::Loading {
        state = cvar.wait(state).unwrap();
    }
    if config_is_stale() {
        log::info!("model settings changed; reloading the engine");
        *GLOBAL_ENGINE.lock().unwrap() = None;
        *state = LoadState::Idle;
    }
}

/// True if the engine is loaded and matches the current settings.
pub fn is_current() -> bool {
    is_engine_loaded() && !config_is_stale()
}

/// Loading coordination state + condvar for waiters.
static LOAD_STATE: Lazy<(Mutex<LoadState>, Condvar)> =
    Lazy::new(|| (Mutex::new(LoadState::Idle), Condvar::new()));

#[derive(Debug, Clone, PartialEq)]
enum LoadState {
    /// No load in progress
    Idle,
    /// A thread is currently loading the model
    Loading,
    /// Loading completed successfully
    Done,
    /// Loading failed
    Failed(String),
}

pub fn get_engine() -> Option<Arc<Mutex<Engine>>> {
    GLOBAL_ENGINE.lock().unwrap().clone()
}

/// Number of voice-input transcriptions (keyboard, popup, speech service,
/// shared files) waiting for or using the engine. All features share one
/// engine, which runs one transcription at a time, so live subtitles yield to
/// these: a subtitle job doesn't start while one is pending, and voice input
/// waits for at most the subtitle run already in flight.
static INTERACTIVE: Lazy<(Mutex<usize>, Condvar)> = Lazy::new(|| (Mutex::new(0), Condvar::new()));

struct InteractiveTurn;

impl InteractiveTurn {
    fn begin() -> InteractiveTurn {
        *INTERACTIVE.0.lock().unwrap() += 1;
        InteractiveTurn
    }
}

impl Drop for InteractiveTurn {
    fn drop(&mut self) {
        let (lock, cvar) = &*INTERACTIVE;
        let mut pending = lock.lock().unwrap();
        *pending -= 1;
        if *pending == 0 {
            cvar.notify_all();
        }
    }
}

/// Transcribes for someone waiting on the result (voice input, a shared
/// file, the benchmark): takes precedence over live subtitles.
pub fn transcribe_shared(engine: &Arc<Mutex<Engine>>, samples: Vec<f32>) -> Result<String, String> {
    let _turn = InteractiveTurn::begin();
    transcribe_guarded(engine, samples)
}

/// [`transcribe_shared`] for a recording or file: long quiet stretches
/// before and after the speech are left out (see
/// [`crate::endpoint::speech_range`]).
pub fn transcribe_speech(engine: &Arc<Mutex<Engine>>, samples: Vec<f32>) -> Result<String, String> {
    let speech = crate::endpoint::speech_range(&samples);
    if speech.is_empty() {
        log::info!("only digital silence recorded; nothing to transcribe");
        return Ok(String::new());
    }
    let samples = if speech.len() < samples.len() {
        log::info!(
            "transcribing {:.1}-{:.1}s of {:.1}s (quiet start/end left out)",
            speech.start as f64 / 16_000.0,
            speech.end as f64 / 16_000.0,
            samples.len() as f64 / 16_000.0
        );
        samples[speech].to_vec()
    } else {
        samples
    };
    transcribe_shared(engine, samples)
}

/// Transcribes live subtitles: waits while voice input needs the engine.
pub fn transcribe_background(
    engine: &Arc<Mutex<Engine>>,
    samples: Vec<f32>,
) -> Result<String, String> {
    {
        let (lock, cvar) = &*INTERACTIVE;
        let mut pending = lock.lock().unwrap();
        while *pending > 0 {
            pending = cvar.wait(pending).unwrap();
        }
    }
    transcribe_guarded(engine, samples)
}

/// Runs a transcription on the shared engine. Two layers of hardening keep a
/// single bad run from freezing every later one (the symptom would be an IME
/// stuck at "Processing" until its process dies, since notify callbacks stop
/// coming): a panic anywhere in the engine stack is caught and surfaced as a
/// normal error, and a lock poisoned by an earlier panic is recovered instead
/// of propagating the poison forever.
fn transcribe_guarded(engine: &Arc<Mutex<Engine>>, samples: Vec<f32>) -> Result<String, String> {
    let audio_secs = samples.len() as f64 / 16_000.0;
    let started = std::time::Instant::now();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut guard = engine.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.transcribe(samples)
    }))
    .unwrap_or_else(|_| {
        log::error!("transcription panicked; reporting as error");
        Err("transcription failed unexpectedly, please try again".to_string())
    });
    log::info!(
        "transcribed {:.1}s audio in {:.2}s",
        audio_secs,
        started.elapsed().as_secs_f64()
    );
    result
}

pub fn is_engine_loaded() -> bool {
    GLOBAL_ENGINE.lock().unwrap().is_some()
}

/// Drops the loaded engine and clears the load state so the next
/// `ensure_loaded*` call reloads with the current model selection. Waits for
/// an in-flight load to finish first, so a reload can't race a load of the
/// previous selection. Call from a background thread.
pub fn reset() {
    let (lock, cvar) = &*LOAD_STATE;
    let mut state = lock.lock().unwrap();
    while *state == LoadState::Loading {
        state = cvar.wait(state).unwrap();
    }
    *GLOBAL_ENGINE.lock().unwrap() = None;
    *state = LoadState::Idle;
}

fn notify_status(env: &mut JNIEnv, obj: &JObject, msg: &str) {
    if let Ok(jmsg) = env.new_string(msg) {
        let _ = env.call_method(
            obj,
            "onStatusUpdate",
            "(Ljava/lang/String;)V",
            &[(&jmsg).into()],
        );
    }
}

/// Ensures the engine is loaded. Safe to call from multiple threads
/// concurrently; reports status via the target's `onStatusUpdate` callback.
///
/// Convenience wrapper around [`ensure_loaded_from_thread`] for JNI entry
/// points that already hold an attached `JNIEnv`.
pub fn ensure_loaded(env: &mut JNIEnv, context: &JObject) -> Result<(), String> {
    let jvm = Arc::new(env.get_java_vm().map_err(|e| e.to_string())?);
    let context_ref = env.new_global_ref(context).map_err(|e| e.to_string())?;
    ensure_loaded_from_thread(&jvm, &context_ref)
}

/// Ensures the engine is loaded. Safe to call from multiple threads concurrently.
///
/// - If already loaded, returns immediately.
/// - If another thread is loading, waits for it to finish.
/// - If no one is loading, this thread takes ownership of loading.
/// - If a previous load failed, retries.
///
/// Reports status via the target's `onStatusUpdate` JNI callback.
pub fn ensure_loaded_from_thread(
    jvm: &Arc<jni::JavaVM>,
    target_ref: &GlobalRef,
) -> Result<(), String> {
    let notify = |msg: &str| {
        if let Ok(mut env) = jvm.attach_current_thread() {
            notify_status(&mut env, target_ref.as_obj(), msg);
        }
    };

    if config_is_stale() {
        reset_if_stale();
    }

    // Fast path: already loaded
    if is_engine_loaded() {
        notify("Ready");
        return Ok(());
    }

    let (lock, cvar) = &*LOAD_STATE;
    let mut state = lock.lock().unwrap();

    // Re-check under lock
    if is_engine_loaded() {
        notify("Ready");
        return Ok(());
    }

    match &*state {
        LoadState::Loading => {
            // Another thread is loading — wait for it
            notify("Waiting for model...");
            while *state == LoadState::Loading {
                state = cvar.wait(state).unwrap();
            }
            drop(state);

            if is_engine_loaded() {
                notify("Ready");
                Ok(())
            } else {
                let msg = "Model failed to load".to_string();
                notify(&format!("Error: {}", msg));
                Err(msg)
            }
        }
        LoadState::Done => {
            notify("Ready");
            Ok(())
        }
        LoadState::Idle | LoadState::Failed(_) => {
            // We take ownership of loading (retry on previous failure)
            *state = LoadState::Loading;
            drop(state);

            let result = if let Ok(mut env) = jvm.attach_current_thread() {
                do_load(&mut env, target_ref.as_obj())
            } else {
                Err("Failed to attach JNI thread".to_string())
            };

            let mut state = lock.lock().unwrap();
            match &result {
                Ok(()) => *state = LoadState::Done,
                Err(msg) => *state = LoadState::Failed(msg.clone()),
            }
            cvar.notify_all();
            result
        }
    }
}

/// Inference thread count: the number of performance cores, capped at 4.
///
/// Phone SoCs are heterogeneous: fast cores paired with slow efficiency
/// cores. ggml synchronizes all threads after each operation, so a thread
/// on a slow core stalls the whole pool, and using every core is slower
/// than using only the fast ones. Cores are classified by their maximum
/// frequency from sysfs: within 70% of the fastest core counts as fast.
/// The count is capped at 4 because grabbing every fast core leaves none
/// for the rest of the system (audio pipeline, the app playing the sound),
/// and any preempted worker stalls the pool at the next op barrier; past
/// 4 threads the matmuls are memory-bound on phone-class SoCs anyway, and
/// more threads mainly build up heat. If sysfs is unreadable, falls back
/// to a conservative 4. The `model_threads` config file (user-settable in
/// the Models screen) overrides the heuristic.
fn performance_core_count() -> i32 {
    let mut freqs: Vec<u64> = Vec::new();
    for i in 0..64 {
        let path = format!("/sys/devices/system/cpu/cpu{i}/cpufreq/cpuinfo_max_freq");
        match std::fs::read_to_string(&path) {
            Ok(s) => match s.trim().parse::<u64>() {
                Ok(f) => freqs.push(f),
                Err(_) => break,
            },
            Err(_) => break,
        }
    }
    match freqs.iter().max() {
        Some(&max) if max > 0 => {
            let fast = freqs.iter().filter(|&&f| f * 10 >= max * 7).count();
            let threads = fast.clamp(1, 4);
            log::info!(
                "cpu clusters {:?} kHz -> {} performance cores -> {} threads",
                freqs,
                fast,
                threads
            );
            threads as i32
        }
        _ => std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(4) as i32,
    }
}

/// CPU features this build's ggml kernels require (see GGML_CPU_ARM_ARCH in
/// the gradle build): dot-product and half-precision SIMD, present on arm64
/// cores since ~2018. Without this check, an older CPU would crash with an
/// illegal instruction mid-inference instead of showing an error.
#[cfg(target_arch = "aarch64")]
fn check_cpu_features() -> Result<(), String> {
    const HWCAP_ASIMHP: libc::c_ulong = 1 << 10; // FEAT_FP16 (asimdhp)
    const HWCAP_ASIMDDP: libc::c_ulong = 1 << 20; // FEAT_DotProd (asimddp)
    let hwcap = unsafe { libc::getauxval(libc::AT_HWCAP) };
    if hwcap & HWCAP_ASIMDDP == 0 || hwcap & HWCAP_ASIMHP == 0 {
        return Err(
            "this device's CPU is too old for this app version (needs arm64 \
             dotprod/fp16, available on phones from ~2018 on)"
                .to_string(),
        );
    }
    Ok(())
}

#[cfg(not(target_arch = "aarch64"))]
fn check_cpu_features() -> Result<(), String> {
    Ok(())
}

/// Reads a single-value config file (trimmed); `None` if absent or empty.
fn read_config(path: &Path) -> Option<String> {
    let s = std::fs::read_to_string(path).ok()?;
    let s = s.trim();
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

/// Status reported when there is nothing to load: no imported model is
/// selected and this build bundles none (F-Droid). The app matches this exact
/// text (`ModelStore.NO_MODEL_STATUS`) to show its translated hint instead.
const NO_MODEL_STATUS: &str = "Error: no speech model installed";

/// Performs the model load: the selected imported GGUF if any (falling back
/// to the bundled model on failure), otherwise the bundled model.
fn do_load(env: &mut JNIEnv, context: &JObject) -> Result<(), String> {
    if let Err(msg) = check_cpu_features() {
        notify_status(env, context, &format!("Error: {}", msg));
        return Err(msg);
    }

    let files_dir = assets::files_dir(env, context).map_err(|e| {
        let msg = format!("Failed to resolve filesDir: {}", e);
        notify_status(env, context, &format!("Error: {}", msg));
        msg
    })?;
    // No shared storage just means no per-language models.
    let media_dir = assets::media_dir(env, context)
        .map_err(|e| log::warn!("no media dir: {}", e))
        .ok();
    configure_gpu_env(&files_dir);
    let dirs = ConfigDirs {
        files_dir: files_dir.clone(),
        media_dir,
    };
    let config = EngineConfig::read(&dirs.files_dir, dirs.media_dir.as_deref());
    let language = config.language.clone();
    let translate = config.translate;
    let threads = config.threads.unwrap_or_else(performance_core_count);
    let has_builtin = assets::has_builtin_model(env, context);

    // The language's own model, if it has one. A missing file is an error
    // rather than a silent fallback to a model that is worse at the language.
    if let Some(path) = config.language_model.clone() {
        let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        notify_status(env, context, &format!("Loading model {}...", name));
        return match Engine::load(&path, language, translate, threads) {
            Ok(engine) => {
                let status = engine.ready_status;
                *GLOBAL_ENGINE.lock().unwrap() = Some(Arc::new(Mutex::new(engine)));
                *LOADED_CONFIG.lock().unwrap() = Some((dirs, config));
                notify_status(env, context, status);
                Ok(())
            }
            Err(e) => {
                log::error!("Language model {} failed to load: {}", path.display(), e);
                let msg = format!("Error loading {}: {}", name, e);
                notify_status(env, context, &msg);
                Err(msg)
            }
        };
    }

    // Automatic language: route each run to its language's model. Like a
    // fixed language's model, a load failure is an error, not a fallback.
    if let Some(dir) = config.auto_models_dir.clone() {
        notify_status(env, context, "Loading models...");
        return match Engine::load_auto(&dir, translate, threads) {
            Ok(engine) => {
                let status = engine.ready_status;
                *GLOBAL_ENGINE.lock().unwrap() = Some(Arc::new(Mutex::new(engine)));
                *LOADED_CONFIG.lock().unwrap() = Some((dirs, config));
                notify_status(env, context, status);
                Ok(())
            }
            Err(e) => {
                log::error!("Automatic language setup in {} failed: {}", dir.display(), e);
                let msg = format!("Error loading models: {}", e);
                notify_status(env, context, &msg);
                Err(msg)
            }
        };
    }

    if let Some(name) = config.active_model.clone() {
        let path = files_dir.join("models").join(&name);
        notify_status(env, context, &format!("Loading model {}...", name));
        match Engine::load(&path, language.clone(), translate, threads) {
            Ok(engine) => {
                let status = engine.ready_status;
                *GLOBAL_ENGINE.lock().unwrap() = Some(Arc::new(Mutex::new(engine)));
                *LOADED_CONFIG.lock().unwrap() = Some((dirs, config));
                notify_status(env, context, status);
                return Ok(());
            }
            Err(e) => {
                log::error!("Imported model {} failed to load: {}", path.display(), e);
                if !has_builtin {
                    let msg = format!("Error loading {}: {}", name, e);
                    notify_status(env, context, &msg);
                    return Err(msg);
                }
                notify_status(
                    env,
                    context,
                    &format!("Error loading {}: {} — using built-in model", name, e),
                );
                // fall through to the bundled model
            }
        }
    }

    if !has_builtin {
        notify_status(env, context, NO_MODEL_STATUS);
        return Err(NO_MODEL_STATUS.trim_start_matches("Error: ").to_string());
    }

    notify_status(env, context, "Checking assets...");

    let path = assets::extract_builtin_model(env, context).map_err(|e| {
        let msg = format!("Asset error: {}", e);
        notify_status(env, context, &format!("Error: {}", msg));
        msg
    })?;

    notify_status(env, context, "Loading model...");

    match Engine::load(&path, language, translate, threads) {
        Ok(engine) => {
            let status = engine.ready_status;
            *GLOBAL_ENGINE.lock().unwrap() = Some(Arc::new(Mutex::new(engine)));
            *LOADED_CONFIG.lock().unwrap() = Some((dirs, config));
            notify_status(env, context, status);
            Ok(())
        }
        Err(e) => {
            // Load failed — likely corrupt/incomplete extraction. Invalidate
            // it so the next attempt re-extracts from the APK.
            assets::invalidate_builtin_model(&path);
            let msg = format!("Model error: {}", e);
            notify_status(env, context, &format!("Error: {}", msg));
            Err(msg)
        }
    }
}
