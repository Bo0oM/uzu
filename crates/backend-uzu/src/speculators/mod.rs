pub mod dflash_tfm;
pub mod prompt_lookup;

use crate::backends::common::Backend;

/// Draft providers for speculative decoding. DFlash is a trained GPU
/// speculator shipped next to the model; PromptLookup drafts from the token
/// history on the CPU and needs no weights, no state, and no hidden
/// features.
pub enum Speculator<B: Backend> {
    DFlash(dflash_tfm::DFlashTfmSpeculator<B>),
    PromptLookup(prompt_lookup::PromptLookupSpeculator),
}

impl<B: Backend> Speculator<B> {
    pub fn hidden_feature_layer_indices(&self) -> Option<&[u32]> {
        match self {
            Speculator::DFlash(speculator) => Some(speculator.hidden_feature_layer_indices()),
            Speculator::PromptLookup(_) => None,
        }
    }
}
