use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use jni::objects::{GlobalRef, JObject};
use jni::JNIEnv;

use crate::early_runs::EarlyRuns;
use crate::endpoint::Endpointer;
use crate::engine;

// --- Optional auto-stop endpointing ------------------------------------------
/// Trailing silence after speech that triggers auto-stop.
const AUTO_STOP_SILENCE_MS: u64 = 2000;
/// If no speech is ever detected, auto-stop after this long.
const AUTO_STOP_NO_SPEECH_MS: u64 = 8000;

pub struct SendStream(#[allow(dead_code)] pub cpal::Stream);
unsafe impl Send for SendStream {}
unsafe impl Sync for SendStream {}

pub struct VoiceSessionState {
    pub stream: Option<SendStream>,
    pub audio_buffer: Arc<Mutex<Vec<f32>>>,
    pub jvm: Arc<jni::JavaVM>,
    pub target_ref: GlobalRef,
    pub last_level_sent: Arc<Mutex<std::time::Instant>>,
    /// True while the current recording runs; flipped off on stop/cancel so
    /// the auto-stop monitor (if any) exits.
    pub session_active: Arc<AtomicBool>,
    /// Transcribes the current recording's full runs while it goes on (see
    /// [`crate::early_runs`]); returns them once the recording ends.
    early_runs: Option<JoinHandle<EarlyRuns>>,
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

fn notify_level(env: &mut JNIEnv, obj: &JObject, level: f32) {
    let _ = env.call_method(obj, "onAudioLevel", "(F)V", &[level.into()]);
}

fn notify_text(env: &mut JNIEnv, obj: &JObject, text: &str) {
    if let Ok(jtxt) = env.new_string(text) {
        let _ = env.call_method(
            obj,
            "onTextTranscribed",
            "(Ljava/lang/String;)V",
            &[(&jtxt).into()],
        );
    }
}

pub fn init_session(env: JNIEnv, target: JObject) -> VoiceSessionState {
    android_logger::init_once(
        android_logger::Config::default().with_max_level(log::LevelFilter::Info),
    );

    let vm = env.get_java_vm().expect("Failed to get JavaVM");
    let vm_arc = Arc::new(vm);
    let target_ref = env.new_global_ref(&target).expect("Failed to ref target");

    let state = VoiceSessionState {
        stream: None,
        audio_buffer: Arc::new(Mutex::new(Vec::new())),
        jvm: vm_arc.clone(),
        target_ref: target_ref.clone(),
        last_level_sent: Arc::new(Mutex::new(std::time::Instant::now())),
        session_active: Arc::new(AtomicBool::new(false)),
        early_runs: None,
    };

    // Load engine in background
    let vm_clone = vm_arc.clone();
    let target_ref_clone = target_ref.clone();

    std::thread::spawn(move || {
        let _ = engine::ensure_loaded_from_thread(&vm_clone, &target_ref_clone);
    });

    state
}

/// Begin microphone capture. With `auto_stop` set, a monitor thread watches
/// for trailing silence after speech (or a no-speech timeout) and invokes the
/// Java-side `onAutoStop()` callback, which is expected to stop the recording
/// the same way a manual tap would.
pub fn start_recording(mut env: JNIEnv, state: &mut VoiceSessionState, auto_stop: bool) {
    let host = cpal::default_host();
    let device = match host.default_input_device() {
        Some(d) => d,
        None => {
            notify_status(
                &mut env,
                state.target_ref.as_obj(),
                "Error: no microphone available. Check permissions.",
            );
            return;
        }
    };

    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(16000),
        buffer_size: cpal::BufferSize::Default,
    };

    // A new buffer rather than clearing the old one: the previous
    // recording's early-runs worker may still hold it.
    state.audio_buffer = Arc::new(Mutex::new(Vec::new()));
    let buffer_clone = state.audio_buffer.clone();

    // End any previous session's monitor, then arm a fresh flag.
    state.session_active.store(false, Ordering::SeqCst);
    let session_active = Arc::new(AtomicBool::new(true));
    state.session_active = session_active.clone();

    let endpoint = if auto_stop {
        Some(Arc::new(Mutex::new(Endpointer::new())))
    } else {
        None
    };

    let jvm = state.jvm.clone();
    let target_ref = state.target_ref.clone();
    let last_sent = state.last_level_sent.clone();
    let endpoint_cb = endpoint.clone();

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &_| {
            buffer_clone.lock().unwrap().extend_from_slice(data);

            // compute RMS
            let mut sum = 0.0f32;
            for &x in data {
                sum += x * x;
            }
            let rms = (sum / (data.len().max(1) as f32)).sqrt();
            let level = (rms * 6.0).clamp(0.0, 1.0);

            if let Some(ep) = &endpoint_cb {
                ep.lock().unwrap().push(data);
            }

            // throttle updates
            let mut last = last_sent.lock().unwrap();
            if last.elapsed() >= std::time::Duration::from_millis(50) {
                *last = std::time::Instant::now();

                if let Ok(mut env) = jvm.attach_current_thread() {
                    let obj = target_ref.as_obj();
                    notify_level(&mut env, obj, level);
                }
            }
        },
        |e| log::error!("Stream err: {}", e),
        None,
    );

    match stream {
        Ok(s) => {
            s.play().ok();
            state.stream = Some(SendStream(s));
            notify_status(&mut env, state.target_ref.as_obj(), "Listening...");

            let buffer = state.audio_buffer.clone();
            let active = session_active.clone();
            state.early_runs = Some(std::thread::spawn(move || run_early(buffer, active)));

            // Settings changed in the app since this process loaded the
            // model: reload while the user speaks rather than after.
            if engine::is_engine_loaded() && !engine::is_current() {
                let jvm = state.jvm.clone();
                let target_ref = state.target_ref.clone();
                std::thread::spawn(move || {
                    let _ = engine::ensure_loaded_from_thread(&jvm, &target_ref);
                });
            }

            if let Some(ep) = endpoint {
                let jvm = state.jvm.clone();
                let target_ref = state.target_ref.clone();
                let started_at = Instant::now();
                std::thread::spawn(move || loop {
                    std::thread::sleep(Duration::from_millis(100));
                    if !session_active.load(Ordering::SeqCst) {
                        return;
                    }
                    let (speech, silence_ms) = {
                        let ep = ep.lock().unwrap();
                        (ep.speech_started(), ep.silence_ms())
                    };
                    let done = (speech && silence_ms >= AUTO_STOP_SILENCE_MS)
                        || (!speech
                            && started_at.elapsed()
                                >= Duration::from_millis(AUTO_STOP_NO_SPEECH_MS));
                    if done {
                        // Claim the session so a simultaneous manual stop and
                        // this monitor can't both fire.
                        if session_active.swap(false, Ordering::SeqCst) {
                            if let Ok(mut env) = jvm.attach_current_thread() {
                                let _ = env.call_method(
                                    target_ref.as_obj(),
                                    "onAutoStop",
                                    "()V",
                                    &[],
                                );
                            }
                        }
                        return;
                    }
                });
            }
        }
        Err(e) => {
            notify_status(
                &mut env,
                state.target_ref.as_obj(),
                &format!("Error: failed to open microphone: {}", e),
            );
        }
    }
}

pub fn stop_recording(mut env: JNIEnv, state: &mut VoiceSessionState) {
    // Drop the stream to stop recording; end the auto-stop monitor if running.
    state.session_active.store(false, Ordering::SeqCst);
    state.stream = None;

    let buffer = state.audio_buffer.lock().unwrap().clone();

    // Guard against empty buffer (mic permission denied, instant stop, etc.)
    if buffer.is_empty() {
        notify_status(
            &mut env,
            state.target_ref.as_obj(),
            "Error: no audio recorded. Check microphone permissions.",
        );
        return;
    }

    let jvm = state.jvm.clone();
    let target_ref = state.target_ref.clone();
    let early_runs = state.early_runs.take();

    notify_status(&mut env, target_ref.as_obj(), "Transcribing...");

    std::thread::spawn(move || {
        let mut env = match jvm.attach_current_thread() {
            Ok(e) => e,
            Err(_) => return,
        };
        let obj = target_ref.as_obj();

        // Wait for the engine if it is still loading, or reload it if the
        // model settings changed since it was loaded.
        if !engine::is_current() {
            if let Err(_) = engine::ensure_loaded(&mut env, obj) {
                return;
            }
        }

        if let Some(eng_arc) = engine::get_engine() {
            // Wait for a run still in flight, then transcribe what's left.
            let early = early_runs.and_then(|worker| worker.join().ok());
            let res = match early.as_ref().and_then(|early| early.rest(&buffer)) {
                Some(rest) => {
                    let rest_text = if rest.is_empty() {
                        Ok(String::new())
                    } else {
                        engine::transcribe_shared(&eng_arc, buffer[rest].to_vec())
                    };
                    rest_text.map(|rest_text| early.expect("rest implies runs").text(&rest_text))
                }
                None => engine::transcribe_speech(&eng_arc, buffer),
            };

            match res {
                Ok(text) => {
                    notify_status(&mut env, obj, "Ready");
                    notify_text(&mut env, obj, &text);
                }
                Err(e) => notify_status(&mut env, obj, &format!("Error: {}", e)),
            }
        } else {
            notify_status(&mut env, obj, "Error: model not loaded");
        }
    });
}

pub fn cancel_recording(mut env: JNIEnv, state: &mut VoiceSessionState) {
    state.session_active.store(false, Ordering::SeqCst);
    state.stream = None;
    // The early-runs worker finishes a run in flight and exits; its texts
    // are dropped.
    state.early_runs = None;
    state.audio_buffer.lock().unwrap().clear();
    notify_status(&mut env, state.target_ref.as_obj(), "Canceled");
}

/// How often the early-runs worker checks for a run it can transcribe.
const EARLY_RUN_POLL: Duration = Duration::from_millis(500);

/// Transcribes full runs of the recording in `buffer` while `active`, and
/// returns them once it ends (after finishing a run in flight). Runs only
/// with an up-to-date engine; until then the audio waits for the next check
/// or for the end of the recording.
fn run_early(buffer: Arc<Mutex<Vec<f32>>>, active: Arc<AtomicBool>) -> EarlyRuns {
    let mut early = EarlyRuns::default();
    while active.load(Ordering::SeqCst) {
        std::thread::sleep(EARLY_RUN_POLL);
        let next = {
            let recorded = buffer.lock().unwrap();
            early
                .next_run(&recorded)
                .map(|run| (run.clone(), recorded[run].to_vec()))
        };
        let Some((run, samples)) = next else { continue };
        if !active.load(Ordering::SeqCst) || !engine::is_current() {
            continue;
        }
        let Some(eng) = engine::get_engine() else { continue };
        match engine::transcribe_shared(&eng, samples) {
            Ok(text) => early.finished(run, text),
            Err(e) => {
                log::warn!("early run failed ({}); transcribing after the recording", e);
                early.failed();
                break;
            }
        }
    }
    early
}
