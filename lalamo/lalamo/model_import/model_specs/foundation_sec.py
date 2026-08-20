from lalamo.model_import.model_configs import HFLlamaConfig
from lalamo.model_import.model_spec import LanguageModelSpec
from lalamo.model_import.model_specs.output_parser_regexes import OPTIONAL_THINKING_OUTPUT_PARSER_REGEX
from lalamo.model_import.origins import HuggingFaceOrigin

__all__ = ["FOUNDATION_SEC_MODELS"]

# Cisco Foundation AI's security models are Llama-3.1-8B derivatives: same
# LlamaForCausalLM graph, head_dim 128, GQA with 8 KV heads, llama3 rope
# scaling. Only the vocabulary is extended (128276) with role and thinking
# tags of their own chat format, which the "foundation-sec" hanashi template
# renders.
FOUNDATION_SEC_END_OF_THINKING_TAG = "\n</think>"

FOUNDATION_SEC_MODELS = [
    LanguageModelSpec(
        vendor="Foundation AI",
        family="Foundation-Sec",
        name="Foundation-Sec-8B-Reasoning",
        size="8B",
        origin=HuggingFaceOrigin(repo="fdtn-ai/Foundation-Sec-8B-Reasoning"),
        config_type=HFLlamaConfig,
        output_parser_regex=OPTIONAL_THINKING_OUTPUT_PARSER_REGEX,
        end_of_thinking_tag=FOUNDATION_SEC_END_OF_THINKING_TAG,
    ),
]

# Quantized builds no longer need a spec of their own: `lalamo convert
# fdtn-ai/Foundation-Sec-8B-Reasoning --quantize 8` produces them in one step.
#
# The Instruct siblings (Foundation-Sec-8B-Instruct, Foundation-Sec-1.1-8B-Instruct)
# share this architecture and the same role tags, but render a different system
# preamble and no forced thinking block, so they need a template of their own
# before they can be listed here.
