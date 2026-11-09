//! Removes a decode loop from the end of a model run's text.
//!
//! Whisper sometimes gets stuck after the real speech and repeats a phrase or
//! a whole sentence until it runs out of budget, or appends a second copy of
//! what it just transcribed; quiet or very short audio makes it likelier. The
//! loop always sits at the end of the run: a block of words repeated back to
//! back, often followed by a partial copy whose last word is cut off. This
//! keeps the text up to the end of the first copy, or to the last sentence
//! end within it (a loop usually starts after real speech that ended a
//! sentence), and drops the rest.
//!
//! Words are compared case-insensitively and without punctuation, so
//! "Die Darm der Darm, die Darm der Darm," counts as two copies. Real speech
//! repeats itself too ("very, very, very"), so short blocks need more copies
//! before they count as a loop.
//!
//! A long loop can also drift: a word changes or drops out from one copy to
//! the next, so the copies stop being exact. Those are found as a long stretch
//! at the end in which most three-word sequences repeat earlier text.

use std::collections::HashSet;

/// A block of at least this many words counts as a loop with 2 copies.
const LONG_BLOCK: usize = 8;
/// Shorter blocks need at least this many copies...
const MIN_COPIES_SHORT: usize = 3;
/// ...together spanning at least this many words (so a 1-word block needs 8).
const MIN_SPAN_SHORT: usize = 8;
/// A last word that breaks the repetition still continues it, as a cut-off
/// copy of its counterpart, if the two share this many leading characters.
const CUT_WORD_PREFIX: usize = 3;

/// A drifting loop is a stretch of at least this many words...
const DRIFT_MIN_SPAN: usize = 16;
/// ...in which every window of this many words...
const DRIFT_WINDOW: usize = 8;
/// ...has at least this many words ending a three-word sequence seen before.
const DRIFT_MIN_REPEATS: usize = 5;
/// The stretch may be followed by words that repeat little (a last copy that
/// goes off the rails before the decode stops): this many, or up to half the
/// stretch's length if that is more.
const DRIFT_TAIL_SLACK: usize = 6;
/// The cut moves back to a sentence end at most this many words earlier.
const DRIFT_SENTENCE_BACKOFF: usize = 8;

/// Copies a `block`-word block needs to count as a loop.
fn copies_needed(block: usize) -> usize {
    if block >= LONG_BLOCK {
        2
    } else {
        MIN_COPIES_SHORT.max(MIN_SPAN_SHORT.div_ceil(block))
    }
}

/// Comparison key of a word: lowercase letters and digits only.
fn key(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// True if `word` looks like a cut-off copy of `counterpart`.
fn is_cut_off(word: &str, counterpart: &str) -> bool {
    let shared = word
        .chars()
        .zip(counterpart.chars())
        .take_while(|(a, b)| a == b)
        .count();
    word != counterpart && shared >= CUT_WORD_PREFIX
}

/// Number of words to keep: `exact_loop` or `drifting_loop`, whichever cuts
/// more, or all words if neither finds a loop. `ends_sentence[i]`: word `i`
/// ends a sentence.
fn words_to_keep(keys: &[String], ends_sentence: &[bool]) -> usize {
    let n = keys.len();
    exact_loop(keys, ends_sentence)
        .unwrap_or(n)
        .min(drifting_loop(keys, ends_sentence).unwrap_or(n))
}

/// Words to keep before a block that repeats exactly at the end: up to the
/// end of its first copy, or to the last sentence end within that copy.
fn exact_loop(keys: &[String], ends_sentence: &[bool]) -> Option<usize> {
    let n = keys.len();
    for block in 1..=n / 2 {
        // The repetition runs to the end, or stops at a cut-off last word.
        let mut ends = vec![n];
        if is_cut_off(&keys[n - 1], &keys[n - 1 - block]) {
            ends.push(n - 1);
        }
        for end in ends {
            // Leftmost start of the stretch before `end` that repeats with
            // period `block`: keys[i] == keys[i + block] throughout.
            let mut start = end - block;
            while start > 0 && keys[start - 1] == keys[start - 1 + block] {
                start -= 1;
            }
            // The stretch holds this many copies, the last possibly partial.
            if (end - start) / block >= copies_needed(block) {
                let first_copy_end = start + block;
                return Some(
                    (start.max(1)..=first_copy_end)
                        .rev()
                        .find(|&c| ends_sentence[c - 1])
                        .unwrap_or(first_copy_end),
                );
            }
        }
    }
    None
}

/// Words to keep before a drifting loop at the end: up to where it starts
/// repeating earlier text, or to a sentence end shortly before that.
fn drifting_loop(keys: &[String], ends_sentence: &[bool]) -> Option<usize> {
    let n = keys.len();
    if n < DRIFT_MIN_SPAN + 2 {
        return None;
    }
    // repeats[i]: the three words ending at word i occurred earlier.
    let mut seen = HashSet::new();
    let mut repeats = vec![false; n];
    for i in 2..n {
        repeats[i] = !seen.insert((&keys[i - 2], &keys[i - 1], &keys[i]));
    }
    // The stretch is the last run of windows that are mostly repeats; it has
    // to reach close to the end.
    let dense = |at: usize| {
        repeats[at..at + DRIFT_WINDOW]
            .iter()
            .filter(|&&r| r)
            .count()
            >= DRIFT_MIN_REPEATS
    };
    let last = (0..=n - DRIFT_WINDOW).rev().find(|&at| dense(at))?;
    let end = last + DRIFT_WINDOW;
    let mut start = last;
    while start > 0 && dense(start - 1) {
        start -= 1;
    }
    if end - start < DRIFT_MIN_SPAN || n - end > DRIFT_TAIL_SLACK.max((end - start) / 2) {
        return None;
    }
    // Cut where the first repeated sequence in the stretch starts, or at a
    // sentence end a few words before it.
    let cut = (start..end).find(|&i| repeats[i])? - 2;
    Some(
        (cut.saturating_sub(DRIFT_SENTENCE_BACKOFF).max(1)..=cut)
            .rev()
            .find(|&c| ends_sentence[c - 1])
            .unwrap_or(cut),
    )
}

/// `text` with a decode loop at its end reduced to its first copy. Text
/// without one is returned unchanged.
pub fn trim_loop(text: &str) -> String {
    // Byte range of each whitespace-separated word, to cut the original text
    // (keeping its punctuation and spacing) rather than rebuild it.
    let mut spans = Vec::new();
    let mut start = None;
    for (i, c) in text.char_indices() {
        match (c.is_whitespace(), start) {
            (true, Some(s)) => {
                spans.push((s, i));
                start = None;
            }
            (false, None) => start = Some(i),
            _ => {}
        }
    }
    if let Some(s) = start {
        spans.push((s, text.len()));
    }

    let keys: Vec<String> = spans.iter().map(|&(s, e)| key(&text[s..e])).collect();
    let ends_sentence: Vec<bool> = spans
        .iter()
        .map(|&(s, e)| text[s..e].ends_with(['.', '!', '?', '…']))
        .collect();
    let keep = words_to_keep(&keys, &ends_sentence);
    if keep == spans.len() {
        return text.to_string();
    }
    text[..spans[keep - 1].1].to_string()
}

#[cfg(test)]
mod tests {
    use super::trim_loop;

    #[test]
    fn leaves_normal_text_alone() {
        for t in [
            "",
            "Hello.",
            "Am 15. August 1940 fielen die Alliierten in Südfrankreich ein.",
            "Das ist sehr, sehr, sehr gut.",
            "No, no, no, I said tomorrow.",
            "ha ha ha ha ha",
            "Er sagte: Ich komme. Sie sagte: Ich komme auch.",
            "הרכבת מתעכבת הגיעה בעוד עשר דקות בערך.",
        ] {
            assert_eq!(trim_loop(t), t, "changed: {t}");
        }
    }

    #[test]
    fn drops_a_repeated_sentence() {
        let s = "Der Kongress begann im August 2005 mit der Finanzierung der \
                 Anti-Obszönitätsinitiative und legte fest, dass das FBI 10 Agenten \
                 für die Erwachsenenpornografie einsetzen muss.";
        assert_eq!(trim_loop(&format!("{s} {s} ")), s);
    }

    #[test]
    fn drops_a_short_looping_phrase() {
        let t = "Nachdem der Damm 1963 erbaut worden war, kamen die Überflutungen \
                 zum Stillstand. Die Darm der Darm, die Darm der Darm, die Darm der Darm, ";
        assert_eq!(
            trim_loop(t),
            "Nachdem der Damm 1963 erbaut worden war, kamen die Überflutungen \
             zum Stillstand."
        );
    }

    #[test]
    fn drops_a_cut_off_last_copy() {
        let t = "Prevádzkové frekvencie štandardu 802 11n sú 2,4 GHz a 5 GHz. \
                 Vážitom prvom ráde a výšetkové frekvencie štandardu 802 11n sú 2,4 GHz a 5 GHz. \
                 Vážitom prvom ráde a výšetkové frekvencie štandardu 802 11n sú 2,4 GHz a 5 GHz. \
                 Vážitom prvom ráde a výšetkové frekvenských";
        // The loop starts inside the first sentence ("frekvencie ..."), which
        // ends within the first copy: cut there, which is all that was said.
        assert_eq!(
            trim_loop(t),
            "Prevádzkové frekvencie štandardu 802 11n sú 2,4 GHz a 5 GHz."
        );
    }

    #[test]
    fn drops_a_loop_that_drifts() {
        // Real 4-bit output on a quiet Slovak clip: the copies change slightly
        // as the loop goes on ("frekvencie" -> "frekvenie", "80211n sú 2,4"
        // dropped), and the last copy is cut off.
        let t = "prevádzkové frekvencie štandardu 80211n sú 2,4 GHz a 5 GHz. \
                 Všetkové frekvencie štandardu 80211n sú 2,4 GHz a 5 GHz. \
                 Všetkové frekvencie štandardu 80211n sú 2,4 GHz a 5 GHz. \
                 Všetkové frekvenie štandardu 80211n sú 2,4 GHz a 5 GHz. \
                 Všetkové frekvenie štandardu 80211n sú 2,4 GHz a 5 GHz. \
                 Všetkové frekvenie štandardu 802 GHz a 5 GHz. \
                 Všetkové frekvenie štandardu 802 GHz a";
        assert_eq!(
            trim_loop(t),
            "prevádzkové frekvencie štandardu 80211n sú 2,4 GHz a 5 GHz."
        );
        // Another 4-bit variant: the last copy goes off the rails into new words.
        let t = "prevádzkové frekvencie štandardu 80211n sú 2,4 GHz a 5 GHz. \
                 Výšetkové frekvencie štandardu 80211n sú 2,4 GHz a 5 GHz. \
                 Výšetkové frekvencie štandardu 80211n sú 2,4 GHz a 5 GHz. \
                 Výšetkové frekvencie štandardu 80211n sú 2,4 GHz a 5 GHz. \
                 Výšetkové frekvenie štandardu 802 GHz a 5 GHz. \
                 Výšetkové frekvenie štandardov Národnej rady je";
        assert_eq!(
            trim_loop(t),
            "prevádzkové frekvencie štandardu 80211n sú 2,4 GHz a 5 GHz."
        );
    }

    #[test]
    fn needs_more_copies_of_short_blocks() {
        // 2 copies of a 4-word block: could be deliberate, kept.
        assert_eq!(
            trim_loop("Go left, go left, then stop."),
            "Go left, go left, then stop."
        );
        let two = "eins zwei drei vier eins zwei drei vier";
        assert_eq!(trim_loop(two), two);
        // 3 copies of it: a loop.
        assert_eq!(
            trim_loop("eins zwei drei vier eins zwei drei vier eins zwei drei vier"),
            "eins zwei drei vier"
        );
        // A single word needs 8 copies.
        assert_eq!(
            trim_loop("the the the the the the the"),
            "the the the the the the the"
        );
        assert_eq!(
            trim_loop("Okay the the the the the the the the"),
            "Okay the"
        );
    }

    #[test]
    fn partial_copies_must_continue_the_block() {
        let block = "eins zwei drei vier fünf sechs sieben acht";
        // Words after the copies that don't continue the block: kept.
        let t = format!("{block} {block} neun zehn");
        assert_eq!(trim_loop(&t), t);
        // A partial copy, the last word cut off: dropped.
        assert_eq!(trim_loop(&format!("{block} {block} eins zwei dre")), block);
    }
}
