use std::path::PathBuf;

const MODEL_DIR_NAME: &str = "Llama-3.2-1B-Instruct";
const MODEL_FILE_NAME: &str = "model.safetensors";

fn get_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

pub fn get_test_model_path() -> PathBuf {
    // On-device runs (dinghy) ship the model inside the app bundle and point
    // here; relative values resolve against the executable's directory.
    if let Ok(dir) = std::env::var("UZU_TEST_MODEL_DIR") {
        let mut path = PathBuf::from(&dir);
        if path.is_relative()
            && let Ok(exe) = std::env::current_exe()
            && let Some(exe_dir) = exe.parent()
        {
            path = exe_dir.join(path);
        }
        if !path.exists() {
            panic!("UZU_TEST_MODEL_DIR set but no model at {path:?}");
        }
        return path;
    }
    let model_dir = std::env::var("TEST_MODEL").unwrap_or_else(|_| MODEL_DIR_NAME.to_string());
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("workspace")
        .join("models")
        .join(get_version())
        .join(model_dir);
    if !path.exists() {
        panic!("Test model not found at {:?}. Set TEST_MODEL env var or run `./scripts/download_test_model.sh`.", path);
    }
    path
}

pub fn get_test_weights_path() -> PathBuf {
    get_test_model_path().join(MODEL_FILE_NAME)
}

#[cfg(target_os = "ios")]
pub fn ios_set_current_dir() {
    use objc2_foundation::{NSSearchPathDirectory, NSSearchPathDomainMask, NSSearchPathForDirectoriesInDomains};
    let paths = NSSearchPathForDirectoriesInDomains(
        NSSearchPathDirectory(9),  // NSDocumentDirectory
        NSSearchPathDomainMask(1), // NSUserDomainMask
        true,
    );
    if let Some(docs) = paths.firstObject() {
        let _ = std::env::set_current_dir(docs.to_string());
    }
}
