//! Transcribing a long dictation while it is still being recorded.
//!
//! A recording longer than one model run is transcribed in runs of at most
//! [`crate::engine::next_run_len`]. Where a run ends depends only on the
//! audio up to that run's maximum length, so each run can start as soon as
//! that much is recorded, and only the last run is left for after the user
//! stops. A run is only started once speech continues past its maximum
//! length, so the quiet end of the recording, which is trimmed off, can't
//! change where it ends. The runs then match those of transcribing the
//! finished recording at once, except that the quiet start is trimmed based
//! on the first seconds only (see [`crate::endpoint::speech_start`]), and
//! that speech detection measures against the noise floor of the audio
//! recorded so far.

use std::ops::Range;

use crate::endpoint;
use crate::engine;

/// Audio recorded before the start of the speech is decided: the noise floor
/// is taken over it, so it should be several seconds. Runs can't start
/// earlier anyway, since the first needs a whole run's worth of audio.
const MIN_AUDIO_FOR_START: usize = 15 * 16_000;

/// The runs of one recording transcribed so far.
#[derive(Default)]
pub struct EarlyRuns {
    /// Where the transcribed audio starts, once decided.
    start: Option<usize>,
    /// End of the audio transcribed so far.
    done_until: usize,
    texts: Vec<String>,
    /// A run failed; the recording is then transcribed at once after all.
    failed: bool,
}

impl EarlyRuns {
    /// The next run of `recorded` that can be transcribed now, if any.
    pub fn next_run(&mut self, recorded: &[f32]) -> Option<Range<usize>> {
        if self.failed {
            return None;
        }
        if self.start.is_none() {
            if recorded.len() < MIN_AUDIO_FOR_START {
                return None;
            }
            let start = endpoint::speech_start(recorded)?;
            self.start = Some(start);
            self.done_until = start;
        }
        let rest = &recorded[self.done_until..];
        if !engine::exceeds_one_run(rest) {
            return None;
        }
        // Wait until speech continues past the run's maximum length; until
        // then the user may stop, and the run would be cut differently.
        let speech_end = endpoint::speech_end(recorded)?;
        if !engine::exceeds_one_run(&recorded[self.done_until..speech_end]) {
            return None;
        }
        Some(self.done_until..self.done_until + engine::next_run_len(rest))
    }

    /// Records the text of `run`, as returned by [`EarlyRuns::next_run`].
    pub fn finished(&mut self, run: Range<usize>, text: String) {
        self.done_until = run.end;
        self.texts.push(text);
    }

    /// A run failed: everything is transcribed after the recording instead.
    pub fn failed(&mut self) {
        self.failed = true;
    }

    /// The audio of the finished recording `recorded` still to transcribe,
    /// or `None` if no early run is usable and all of it is to be
    /// transcribed as usual. May be empty.
    pub fn rest(&self, recorded: &[f32]) -> Option<Range<usize>> {
        if self.failed || self.texts.is_empty() {
            return None;
        }
        let end = endpoint::speech_range(recorded).end.max(self.done_until);
        Some(self.done_until..end)
    }

    /// The whole text: the early runs' texts followed by `rest_text`, the
    /// text of [`EarlyRuns::rest`], joined like the runs of one transcription.
    pub fn text(self, rest_text: &str) -> String {
        let mut text = String::new();
        for piece in self.texts.iter().map(String::as_str).chain([rest_text]) {
            let piece = piece.trim();
            if !piece.is_empty() {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(piece);
            }
        }
        text
    }
}
