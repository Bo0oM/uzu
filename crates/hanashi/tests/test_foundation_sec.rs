//! Round-trip for the foundation-sec template: render a chat the way Cisco's
//! Foundation-Sec models were trained (their own `<|role|>` tags plus a forced
//! `<think>` block), then feed a completion back through the parser and check
//! it lands as one assistant message with reasoning and response separated.
//!
//! Skipped unless the converted model is present, since it supplies the
//! tokenizer that knows the extended vocabulary.

use std::{path::PathBuf, sync::Arc};

use hanashi::{
    Encoding as EncodingTrait,
    chat::hanashi::{HanashiEncodingImpl, config::HanashiConfig},
};
use shoji::types::session::chat::{ChatContentBlock, ChatMessage, ChatRole};
use tokenizers::Tokenizer;

fn model_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("workspace")
        .join("models")
        .join("0.5.16")
        .join("Foundation-Sec-8B-Reasoning")
}

fn text_message(
    role: ChatRole,
    text: &str,
) -> ChatMessage {
    let mut message = ChatMessage::for_role(role);
    message.content.push(ChatContentBlock::Text {
        value: text.to_string(),
    });
    message
}

#[test]
fn foundation_sec_round_trip() {
    let directory = model_directory();
    if !directory.exists() {
        println!("skipping: {} is not converted", directory.display());
        return;
    }
    let tokenizer = Tokenizer::from_file(directory.join("tokenizer.json").to_str().unwrap()).unwrap();
    let mut encoding = HanashiEncodingImpl::new(HanashiConfig::FoundationSec, Arc::new(tokenizer.clone())).unwrap();

    encoding
        .encode(vec![
            text_message(ChatRole::System {}, "You are a senior detection engineer."),
            text_message(ChatRole::User {}, "Which ATT&CK technique covers ntdsutil IFM dumps?"),
        ])
        .unwrap();

    let prompt = encoding.state().text();
    assert!(prompt.contains("<|system|>"), "system tag missing: {prompt}");
    assert!(prompt.contains("<|user|>"), "user tag missing: {prompt}");
    assert!(prompt.ends_with("<|assistant|>\n<think>"), "generation prompt is not seeded: {prompt}");

    let completion = "ntdsutil creates a shadow copy of the directory database.\n</think>\nThat is T1003.003.<|end_of_text|>";
    let completion_ids = tokenizer.encode(completion, false).unwrap().get_ids().to_vec();
    encoding.decode(completion_ids).unwrap();

    let last = encoding.state().messages.last().expect("no message was parsed");
    assert_eq!(last.role, ChatRole::Assistant {}, "parsed role: {:?}", last.role);
    let reasoning = last.content.iter().any(|block| matches!(block, ChatContentBlock::Reasoning { .. }));
    let response = last.content.iter().find_map(|block| match block {
        ChatContentBlock::Text {
            value,
        } => Some(value.clone()),
        _ => None,
    });
    assert!(reasoning, "reasoning block missing: {:?}", last.content);
    assert_eq!(response.as_deref().map(str::trim), Some("That is T1003.003."));
}
