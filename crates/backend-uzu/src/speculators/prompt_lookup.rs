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
            draft_len: DRAFT_LEN,
        }
    }
}

impl PromptLookupSpeculator {
    /// Verification batch: the root token plus the drafts.
    pub fn batch_size(&self) -> u32 {
        (self.draft_len + 1) as u32
    }

    /// Whether the lookup would draft a full-length chain here — the cheap
    /// CPU signal the stream uses to leave a drafting pause: dense full hits
    /// mean a verbatim span is being reproduced.
    pub fn would_draft_fully(
        &self,
        history: &[u64],
        root_token: u64,
    ) -> bool {
        self.propose(history, root_token).len() == self.draft_len
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
