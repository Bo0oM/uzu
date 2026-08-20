//! Stream-level invariant: one token in, one step of state.
//!
//! This covers the seam that unit tests otherwise leave open. Inside
//! `generate()` a fully accepted pass is chained into the next one, while a
//! partially accepted one is waited on and resolved through the trie -- two
//! routes that must advance mixer state identically. A change that sent fully
//! accepted passes down the second route left all 605 tests green and still
//! crashed LFM2-350M with "Called short conv state encode accept on a state
//! with nothing to accept", because nothing exercised a stateful mixer
//! through the stream.
//!
//! Run against a short-convolution or GDN model to cover that class:
//! `TEST_MODEL=LFM2-350M cargo test -p backend-uzu --lib stream_state`.
//! Under a plain attention model this still guards the accounting, which is
//! the other half of what the two routes have to agree on.

use proc_macros::uzu_test;
use test_runner::path::get_test_model_path;

use crate::{
    encodable_block::sampling::SamplingMethod,
    engine::{
        Engine,
        language_model::stream::{LanguageModelStream, LanguageModelStreamOptions},
    },
    tests::helpers::for_each_non_cpu_backend,
};

const PROMPT_TOKENS: u64 = 24;
const DECODE_TOKENS: usize = 12;

#[uzu_test]
fn stream_advances_state_once_per_token() {
    for_each_non_cpu_backend!(|B| {
        let engine = Engine::<B>::new().expect("engine");
        let model = engine.load_language_model(&get_test_model_path()).expect("model");
        let mut state = model.create_empty_state(Some(1024)).expect("state");
        let input: Vec<u64> = (0..PROMPT_TOKENS).map(|token| 1000 + token).collect();
        let options = LanguageModelStreamOptions {
            sampling_method: SamplingMethod::Greedy,
            #[cfg(grammar)]
            grammar: None,
        };

        let mut produced = 0usize;
        {
            let mut stream = LanguageModelStream::new(&model, &input, &mut state, options).expect("stream");
            for step in 0..DECODE_TOKENS {
                match stream.next() {
                    Some(Ok(_)) => produced += 1,
                    Some(Err(error)) => panic!("decode failed at step {step}: {error:?}"),
                    None => panic!("stream ended early at step {step}, after {produced} tokens"),
                }
            }
        }

        // The context must have grown by exactly one token per step. A mixer
        // state advanced twice, or advanced with nothing to advance on, lands
        // here as a count that does not match -- or, on a short-convolution
        // model, as the panic this test exists to catch.
        assert_eq!(
            state.tokens().len(),
            PROMPT_TOKENS as usize + produced,
            "context is {} tokens after a {PROMPT_TOKENS}-token prompt and {produced} decoded",
            state.tokens().len(),
        );
    });
}
