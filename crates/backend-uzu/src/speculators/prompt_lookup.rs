//! Draft-model-free speculator: proposes the continuation that followed the
//! most recent occurrence of the current n-gram suffix earlier in the token
//! history (prompt lookup / n-gram speculation). Drafts cost no GPU work and
//! no weights; the existing tree-verification pass checks them exactly, so
//! greedy output is token-identical to plain decoding. Pays off on
//! copy-heavy generation (summarization with quotes, code edits, RAG) where
//! the model reproduces spans of its own context; the stream's acceptance
//! feedback shuts speculation down when drafts stop landing.

/// Suffix lengths matched against the history, longest first. The floor
/// matters as much as the ceiling: a unigram fallback matches any recurring
/// token and drafts an unrelated continuation, which burns batched passes on
/// junk and trips the acceptance cooldown before real copy stretches begin
/// (observed on gemma: the probe window landed on the answer preamble and
/// paused speculation for the whole generation). Trigram-only matching makes
/// drafting rare in free prose and dense inside verbatim spans.
const MAX_NGRAM: usize = 3;
const MIN_NGRAM: usize = 3;

/// Draft tokens proposed per pass. With the verification root this makes a
/// batch of 4 — the fitted batched-GEMV tile (see ADR-11); larger batches
/// cost more per pass than their marginal drafts return.
const DRAFT_LEN: usize = 3;

pub struct PromptLookupSpeculator {
    max_ngram: usize,
    min_ngram: usize,
    draft_len: usize,
}

impl Default for PromptLookupSpeculator {
    fn default() -> Self {
        Self {
            max_ngram: MAX_NGRAM,
            min_ngram: MIN_NGRAM,
            draft_len: draft_len_override().unwrap_or(DRAFT_LEN),
        }
    }
}

/// Sweep probe for the draft length. Only lengths that keep the verification
/// batch on an instantiated `M_TILE` (3 and 7, for batches of 4 and 8) are
/// worth measuring; anything else falls back to one pass per token.
fn draft_len_override() -> Option<usize> {
    std::env::var("UZU_SPEC_DRAFT_LEN").ok()?.parse::<usize>().ok().filter(|value| *value > 0)
}

impl PromptLookupSpeculator {
    /// Verification batch: the root token plus the drafts.
    pub fn batch_size(&self) -> u32 {
        (self.draft_len + 1) as u32
    }

    /// How many drafts the lookup would have got right at the position
    /// `draft_len` tokens back, judged against what the model actually went on
    /// to produce.
    ///
    /// This is the signal the stream re-enters drafting on, and it is the same
    /// quantity the acceptance window later measures — the tokens a drafted
    /// pass would win. The older signal only asked whether a draft existed,
    /// which is a property of the history repeating, not of the model
    /// following: on LFM2 free-form prose the lookup drafts constantly and the
    /// model then diverges, so drafting entered a losing regime over and over.
    /// Returns `None` until the history is long enough to judge.
    pub fn hindsight_hits(
        &self,
        history: &[u64],
    ) -> Option<usize> {
        let outcome_start = history.len().checked_sub(self.draft_len)?;
        let root_index = outcome_start.checked_sub(1)?;
        let drafts = self.propose(&history[..root_index], history[root_index]);
        let hits = drafts
            .iter()
            .zip(&history[outcome_start..])
            .take_while(|(drafted, actual)| drafted == actual)
            .count();
        Some(hits)
    }

    /// Tokens that followed the latest earlier occurrence of the suffix
    /// ending in `root_token`; empty when no suffix of length >= 1 recurs.
    pub fn propose(
        &self,
        history: &[u64],
        root_token: u64,
    ) -> Vec<u64> {
        for n in (self.min_ngram..=self.max_ngram).rev() {
            if n > history.len() + 1 {
                continue;
            }
            // The suffix to match: the last n-1 history tokens plus the root.
            let tail = &history[history.len() - (n - 1)..];
            let Some(found) = self.rfind_ngram(history, tail, root_token) else {
                continue;
            };
            let continuation = &history[found + 1..];
            let drafts: Vec<u64> = continuation.iter().take(self.draft_len).copied().collect();
            if !drafts.is_empty() {
                return drafts;
            }
        }
        Vec::new()
    }

    /// Index of the last token of the latest match of `tail ++ [root]` in
    /// `history`, excluding a match that ends at the history's end (that is
    /// the query itself when root is already appended by the caller's flow).
    fn rfind_ngram(
        &self,
        history: &[u64],
        tail: &[u64],
        root: u64,
    ) -> Option<usize> {
        let n = tail.len() + 1;
        if history.len() < n {
            return None;
        }
        // end is the index of the candidate root-position match.
        for end in (n - 1..history.len() - 1).rev() {
            if history[end] != root {
                continue;
            }
            if history[end - tail.len()..end] == *tail {
                return Some(end);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use proc_macros::uzu_test;

    use super::*;

    #[uzu_test]
    fn proposes_continuation_of_repeated_ngram() {
        let speculator = PromptLookupSpeculator::default();
        // history: ... 7 8 9 10 ... 7 8   -> root 9 matched via (7 8 9),
        // continuation after the earlier 9 is 10 11 12.
        let history = [1, 7, 8, 9, 10, 11, 12, 5, 7, 8];
        assert_eq!(speculator.propose(&history, 9), vec![10, 11, 12]);
    }

    #[uzu_test]
    fn unigram_repeat_is_not_enough() {
        let speculator = PromptLookupSpeculator::default();
        // Root 4 recurs but no trigram does: junk drafts are worse than none.
        let history = [4, 20, 21, 22, 1, 2, 3];
        assert!(speculator.propose(&history, 4).is_empty());
    }

    #[uzu_test]
    fn hindsight_counts_only_the_drafts_the_model_followed() {
        let speculator = PromptLookupSpeculator::default();
        // The trigram (7 8 9) recurs and 10 11 12 followed it the first time.
        // The model then produced 10 11 40: two drafts would have landed.
        let history = [1, 7, 8, 9, 10, 11, 12, 5, 7, 8, 9, 10, 11, 40];
        assert_eq!(speculator.hindsight_hits(&history), Some(2));
    }

    #[uzu_test]
    fn hindsight_is_zero_when_the_model_diverges_immediately() {
        let speculator = PromptLookupSpeculator::default();
        let history = [1, 7, 8, 9, 10, 11, 12, 5, 7, 8, 9, 30, 31, 32];
        assert_eq!(speculator.hindsight_hits(&history), Some(0));
    }

    #[uzu_test]
    fn hindsight_needs_history() {
        let speculator = PromptLookupSpeculator::default();
        assert_eq!(speculator.hindsight_hits(&[1, 2]), None);
    }

    #[uzu_test]
    fn no_match_returns_empty() {
        let speculator = PromptLookupSpeculator::default();
        assert!(speculator.propose(&[1, 2, 3], 9).is_empty());
        assert!(speculator.propose(&[], 9).is_empty());
    }

    #[uzu_test]
    fn prefers_latest_occurrence() {
        let speculator = PromptLookupSpeculator::default();
        // The trigram (8 9 -> 5) occurs twice; the later continuation wins.
        let history = [8, 9, 5, 100, 0, 8, 9, 5, 200, 201, 202, 7, 8, 9];
        assert_eq!(speculator.propose(&history, 5), vec![200, 201, 202]);
    }
}

#[cfg(test)]
mod cost {
    use std::time::Instant;

    use proc_macros::uzu_test;

    use super::*;

    /// What the linear history scan costs per drafted token, against the
    /// token budget it has to fit into: the fastest stand model decodes at
    /// ~370 t/s (2.7 ms per token), the slowest 8B at ~24 t/s (41 ms).
    #[uzu_test]
    #[ignore]
    fn bench_propose_over_history() {
        let speculator = PromptLookupSpeculator::default();
        for length in [1_024usize, 8_192, 32_768, 131_072] {
            // Worst case for the scan: the root token never recurs, so the
            // whole history is walked before giving up.
            let history: Vec<u64> = (0..length as u64).map(|index| index % 50_000 + 1).collect();
            let started = Instant::now();
            let iterations = 200;
            for _ in 0..iterations {
                std::hint::black_box(speculator.propose(&history, 0));
            }
            let per_call = started.elapsed().as_secs_f64() * 1e6 / iterations as f64;
            println!("history {length:7}: {per_call:8.1} us per propose");
        }
    }
}

/// When drafting is on, and when it stops.
///
/// This is the speculator's own schedule, not the decode loop's. It used to
/// live on `LanguageModelStream` as six counters and five constants behind
/// three `matches!(speculator, PromptLookup(_))` guards, which meant every
/// model carried it — including those with no speculator at all — and a second
/// speculator would have added a fourth guard rather than an impl.
pub struct DraftingSchedule {
    /// Drafting starts paused: the lookup has to show a verbatim span before
    /// the first batched pass is spent.
    paused: bool,
    /// Hindsight over already generated tokens, deciding when to resume.
    entry_hits: u32,
    entry_samples: u32,
    /// Acceptance over drafted passes, deciding when to pause again.
    window_passes: u32,
    window_accepted: u32,
    /// Consecutive passes on which the lookup proposed nothing.
    idle_passes: u32,
}

/// Already generated tokens the re-entry decision is judged over. Long enough
/// that a single lucky repeat cannot restart drafting, short enough to catch a
/// verbatim span while it is still being produced.
const ENTRY_WINDOW: u32 = 16;

/// Drafted passes an acceptance window covers.
const WINDOW_PASSES: u32 = 4;

/// Consecutive passes with nothing drafted before drafting pauses again.
///
/// The batch stays requested while drafting is on, which costs a blocking
/// resolve per token whether or not the lookup found anything, so a stream
/// that has left its verbatim span has to notice. Counted separately from
/// acceptance: mixing the two averages an idle pass together with a good one
/// and pauses mid-span.
const IDLE_PASSES_BEFORE_PAUSE: u32 = 8;

/// Accepted tokens per drafted pass a window must average to keep drafting,
/// scaled by 100.
///
/// A constant, and not for want of trying to measure it. The right value
/// depends on what a batched pass costs relative to a plain one, which is a
/// property of the model: on Foundation-Sec-8B bf16, where a pass is dominated
/// by reading 16 GB of weights, dropping this to 115 turns +5.1% into +11.7%,
/// while the same value costs LFM2-350M 10% because its pass is not weight
/// bound. Deriving it from the two classes' own GPU times was implemented and
/// reverted twice: the first attempt could never collect a plain-pass sample,
/// since a plain pass is fully accepted and a fully accepted pass is chained
/// rather than waited on; the second timed both alike and produced the same
/// answer, which throughput then contradicted. The ratio is not the whole cost
/// of drafting — the trie build, the lookup itself and the dispatches around
/// the batched pass are not in it. `UZU_SPEC_MIN_ACCEPT` sweeps this.
const MIN_ACCEPT_PER_PASS_X100: u32 = 300;

fn min_accept_override() -> Option<u32> {
    static VALUE: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("UZU_SPEC_MIN_ACCEPT").ok().and_then(|raw| raw.parse::<u32>().ok()).filter(|value| *value >= 100)
    })
}

fn accept_bar_x100() -> u32 {
    min_accept_override().unwrap_or(MIN_ACCEPT_PER_PASS_X100).max(101)
}

impl Default for DraftingSchedule {
    fn default() -> Self {
        Self {
            paused: true,
            entry_hits: 0,
            entry_samples: 0,
            window_passes: 0,
            window_accepted: 0,
            idle_passes: 0,
        }
    }
}

impl DraftingSchedule {
    pub fn is_drafting(&self) -> bool {
        !self.paused
    }

    /// The batch the next pass should request.
    ///
    /// While paused, the decision comes from hindsight: over the last
    /// `ENTRY_WINDOW` already generated tokens, count the drafts the lookup
    /// *would* have got right and resume only when that predicts a pass worth
    /// more than it costs — the same quantity the acceptance window measures
    /// once drafting is on. Both signals run on the CPU over known tokens, so
    /// they cost nothing on the GPU and lag by one pass, which is fine for a
    /// trigger.
    pub fn requested_batch(
        &mut self,
        speculator: &PromptLookupSpeculator,
        tokens: &[u64],
    ) -> u32 {
        if !self.paused {
            return speculator.batch_size();
        }
        if let Some(hits) = speculator.hindsight_hits(tokens) {
            self.entry_hits = self.entry_hits.saturating_add(hits as u32);
            self.entry_samples = self.entry_samples.saturating_add(1);
        }
        if self.entry_samples < ENTRY_WINDOW {
            return 1;
        }
        // Predicted accept per pass is the root token plus the drafts that
        // would have landed.
        let predicted_x100 = 100 + self.entry_hits * 100 / self.entry_samples;
        self.entry_hits = 0;
        self.entry_samples = 0;
        if predicted_x100 < accept_bar_x100() {
            return 1;
        }
        self.paused = false;
        speculator.batch_size()
    }

    /// Records what a finished pass did, and pauses drafting when it stops
    /// paying.
    ///
    /// A pass that drafted nothing has a one-token batch and is counted apart:
    /// averaging it together with the drafted ones throws speculation out
    /// mid-span, which measured as copy-heavy gemma falling from 183 to 150
    /// t/s.
    pub fn observe_pass(
        &mut self,
        batch: u32,
        accepted: u32,
    ) {
        if self.paused {
            return;
        }
        if batch <= 1 {
            self.idle_passes += 1;
            if self.idle_passes >= IDLE_PASSES_BEFORE_PAUSE {
                self.pause();
            }
            return;
        }
        self.idle_passes = 0;
        self.window_passes += 1;
        self.window_accepted += accepted;
        if self.window_passes == WINDOW_PASSES {
            if self.window_accepted * 100 < WINDOW_PASSES * accept_bar_x100() {
                self.pause();
            }
            self.window_passes = 0;
            self.window_accepted = 0;
        }
    }

    fn pause(&mut self) {
        self.paused = true;
        self.window_passes = 0;
        self.window_accepted = 0;
        self.idle_passes = 0;
    }
}
