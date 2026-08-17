//! Fills in a missing encoding.json for side-loaded models. The downloader
//! and lalamo both ship a model without one, and without it the chat
//! pipeline fails at startup ("Failed to parse encoding config"), so the
//! local registries infer the hanashi template from the model's name and
//! persist the result next to the weights.

use std::{fs, path::Path};

/// Known hanashi template names, matched against the lowercased model name.
/// Order matters: more specific families first (functiongemma before gemma,
/// qwen3.6/3.5 before qwen3, lfm2.5 before lfm2).
fn infer_template(name: &str) -> Option<&'static str> {
    let name = name.to_ascii_lowercase();
    let has = |needle: &str| name.contains(needle);
    if has("functiongemma") {
        return Some("functiongemma");
    }
    if has("gemma-4") || has("gemma4") {
        return Some("gemma-4");
    }
    if has("gemma") {
        return Some("gemma-3");
    }
    if has("gpt-oss") {
        return Some("gpt-oss");
    }
    if has("lfm2.5") || has("lfm2-5") {
        return if has("think") {
            Some("lfm2.5-thinking")
        } else {
            Some("lfm2.5-instruct")
        };
    }
    if has("lfm2") {
        return Some("lfm2");
    }
    if has("llama") {
        return Some("llama-3.2");
    }
    if has("qwen3.6") || has("qwen3-6") {
        return Some("qwen3.6");
    }
    if has("qwen3.5") || has("qwen3-5") {
        return Some("qwen3.5");
    }
    if has("qwen3") {
        return if has("think") {
            Some("qwen3-thinking")
        } else if has("instruct") {
            Some("qwen3-instruct")
        } else {
            Some("qwen3")
        };
    }
    if has("muse") {
        return Some("muse-glimmer");
    }
    None
}

/// The repo id recorded by the downloader, when the model came from the
/// cloud registry; a stronger family signal than a renamed directory.
fn benchmark_repo_id(model_path: &Path) -> Option<String> {
    let text = fs::read_to_string(model_path.join("benchmark_task.json")).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    value.get("repo_id").and_then(|v| v.as_str()).map(|s| s.to_string())
}

/// Infers the hanashi encoding entries for a model directory without an
/// encoding.json, persisting them best-effort so later launches (and other
/// tools) see a complete model.
pub(crate) fn infer_and_persist_encodings(model_path: &Path) -> Vec<serde_json::Value> {
    let dir_name = model_path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
    let template = benchmark_repo_id(model_path)
        .as_deref()
        .and_then(infer_template)
        .or_else(|| infer_template(dir_name));
    let Some(template) = template else {
        tracing::warn!(
            path = %model_path.display(),
            "model has no encoding.json and its family is not recognized; \
             chat will not start for it — add an encoding.json with the \
             hanashi template name"
        );
        return vec![];
    };
    let entries = vec![serde_json::json!({ "type": "hanashi", "name": template })];
    let serialized = serde_json::Value::Array(entries.clone()).to_string();
    if let Err(error) = fs::write(model_path.join("encoding.json"), &serialized) {
        // Read-only model dirs still work for this launch; we just re-infer
        // next time.
        tracing::debug!(?error, path = %model_path.display(), "could not persist inferred encoding.json");
    } else {
        tracing::info!(path = %model_path.display(), template, "inferred and wrote encoding.json");
    }
    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infers_known_families() {
        assert_eq!(infer_template("gemma-3-1b-it-4bit"), Some("gemma-3"));
        assert_eq!(infer_template("google/functiongemma-270m-it"), Some("functiongemma"));
        assert_eq!(infer_template("Qwen3-0.6B"), Some("qwen3"));
        assert_eq!(infer_template("Qwen3.5-0.8B-M"), Some("qwen3.5"));
        assert_eq!(infer_template("Qwen/Qwen3-4B-Instruct-2507"), Some("qwen3-instruct"));
        assert_eq!(infer_template("LFM2-350M"), Some("lfm2"));
        assert_eq!(infer_template("LiquidAI/LFM2.5-1.2B-Thinking"), Some("lfm2.5-thinking"));
        assert_eq!(infer_template("LiquidAI/LFM2.5-1.2B-Instruct"), Some("lfm2.5-instruct"));
        assert_eq!(infer_template("Llama-3.2-1B-Instruct"), Some("llama-3.2"));
        assert_eq!(infer_template("openai_gpt-oss-20b"), Some("gpt-oss"));
    }

    #[test]
    fn unknown_family_yields_none() {
        assert_eq!(infer_template("mystery-model-7b"), None);
    }

    #[test]
    fn persists_inferred_encoding_and_prefers_repo_id() {
        let dir = std::env::temp_dir().join(format!("uzu-enc-infer-{}", std::process::id()));
        // Directory named unrecognizably; the recorded repo id decides.
        let model_dir = dir.join("my-side-loaded-model");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("benchmark_task.json"), r#"{"repo_id":"google/gemma-3-1b-it"}"#).unwrap();

        let entries = infer_and_persist_encodings(&model_dir);
        assert_eq!(entries, vec![serde_json::json!({"type":"hanashi","name":"gemma-3"})]);
        let written: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(model_dir.join("encoding.json")).unwrap()).unwrap();
        assert_eq!(written, serde_json::json!([{"type":"hanashi","name":"gemma-3"}]));

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
