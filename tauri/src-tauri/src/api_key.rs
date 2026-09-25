//! API key handling for the desktop shell.
//!
//! The backend requires `Authorization: Bearer <key>` on every request. The
//! shell and the sidecar share one plaintext key file, `<data-dir>/api_key`,
//! where `<data-dir>` is the directory the sidecar receives as `--data-dir`.
//! Both sides create the file when it is missing (create-if-absent on each
//! side, so start-up order does not matter); the shell does it before
//! spawning so it knows the key immediately.
//!
//! The key must never be logged (sidecar stderr and shell logs are forwarded
//! to every webview as `server-log` events), never put on the sidecar's argv
//! (visible in `ps`) and never handed to the sidecar through an env var.
//! Error strings from this module may name the file path, never its content.

use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

const API_KEY_FILE_NAME: &str = "api_key";
const API_KEY_PREFIX: &str = "vbx_";
/// 32 random bytes encode to the 43 base64url characters the backend expects
/// after the `vbx_` prefix (it uses `secrets.token_urlsafe(32)`).
const API_KEY_RANDOM_BYTES: usize = 32;
/// How long to wait for another creator to finish writing a key file we lost
/// the creation race to (a few dozen bytes, written right after the open).
const RACE_READ_ATTEMPTS: u32 = 20;
const RACE_READ_DELAY: Duration = Duration::from_millis(25);

fn api_key_file(data_dir: &Path) -> PathBuf {
    data_dir.join(API_KEY_FILE_NAME)
}

/// Read and trim a key file. `Ok(None)` when the file is missing or empty.
fn read_key_file(path: &Path) -> Result<Option<String>, String> {
    match std::fs::read_to_string(path) {
        Ok(contents) => {
            let trimmed = contents.trim();
            if trimmed.is_empty() {
                Ok(None)
            } else {
                Ok(Some(trimmed.to_string()))
            }
        }
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("Failed to read API key file {:?}: {}", path, e)),
    }
}

fn generate_key() -> String {
    use base64::Engine as _;
    use rand::RngCore as _;

    let mut bytes = [0u8; API_KEY_RANDOM_BYTES];
    rand::thread_rng().fill_bytes(&mut bytes);
    format!(
        "{}{}",
        API_KEY_PREFIX,
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
    )
}

/// Create `path` exclusively (`create_new`, i.e. O_EXCL) with owner-only
/// permissions on Unix and write `key` into it. `Ok(false)` means another
/// process created the file first; the caller should read that one instead.
fn try_create_key_file(path: &Path, key: &str) -> Result<bool, String> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = match options.open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == ErrorKind::AlreadyExists => return Ok(false),
        Err(e) => return Err(format!("Failed to create API key file {:?}: {}", path, e)),
    };
    let written = file
        .write_all(key.as_bytes())
        .and_then(|_| file.sync_all());
    drop(file);
    if let Err(e) = written {
        // Never leave a half-written key behind: the sidecar would read it
        // and the two sides would disagree until someone deletes the file.
        let _ = std::fs::remove_file(path);
        return Err(format!("Failed to write API key file {:?}: {}", path, e));
    }
    Ok(true)
}

/// Return the key in `<data_dir>/api_key`, generating the file when it does
/// not exist yet.
///
/// Creation is exclusive (`create_new`, mode 0600 on Unix), the same
/// create-if-absent the sidecar performs on the same path, so when both race
/// exactly one key survives and the loser reads the winner's file. Writing a
/// temp file and `rename`-ing it over `api_key` would not give that guarantee:
/// `rename` replaces an existing destination on every platform, so it could
/// silently overwrite a key the sidecar had just generated and already loaded.
pub fn ensure_api_key_file(data_dir: &Path) -> Result<String, String> {
    std::fs::create_dir_all(data_dir)
        .map_err(|e| format!("Failed to create data dir {:?}: {}", data_dir, e))?;
    let path = api_key_file(data_dir);
    if let Some(key) = read_key_file(&path)? {
        return Ok(key);
    }
    let key = generate_key();
    if try_create_key_file(&path, &key)? {
        println!("Created API key file at {:?}", path);
        return Ok(key);
    }
    // Lost the creation race: the other creator may still be writing.
    for _ in 0..RACE_READ_ATTEMPTS {
        if let Some(key) = read_key_file(&path)? {
            return Ok(key);
        }
        std::thread::sleep(RACE_READ_DELAY);
    }
    Err(format!(
        "API key file {:?} exists but is empty; delete it and restart Voicebox",
        path
    ))
}

/// Resolve the key this app instance must present to the backend.
///
/// Debug builds honor `VOICEBOX_API_KEY_FILE` (what `just dev` exports, so the
/// shell and the uvicorn it started share `<repo>/data/api_key`); that file
/// must already exist, it is never created here. `VOICEBOX_API_KEY` mirrors
/// the sidecar's own env override: the sidecar inherits the shell's
/// environment, so when it is set the sidecar uses that key and creates no
/// local key file, and the shell has to present the same value. Otherwise the
/// key lives in `<data_dir>/api_key`.
pub fn resolve_api_key(data_dir: &Path) -> Result<String, String> {
    if cfg!(debug_assertions) {
        if let Some(path) = std::env::var_os("VOICEBOX_API_KEY_FILE") {
            let path = PathBuf::from(path);
            return read_key_file(&path)?.ok_or_else(|| {
                format!(
                    "VOICEBOX_API_KEY_FILE points at {:?}, which is missing or empty (run `just api-key`)",
                    path
                )
            });
        }
    }
    if let Ok(key) = std::env::var("VOICEBOX_API_KEY") {
        let key = key.trim();
        if !key.is_empty() {
            return Ok(key.to_string());
        }
    }
    ensure_api_key_file(data_dir)
}
