use std::path::{Path, PathBuf};
use anyhow::{Result, bail};

/// Resolves a model name (e.g. "gemma3:1b") or a direct path to a GGUF blob path.
///
/// Search order for manifests:
///   1. `models_dir` (e.g. `./models` in the project)
///   2. `/usr/share/ollama/.ollama/models` (system Ollama)
///   3. `~/.ollama/models` (user Ollama)
pub fn resolve_model(name_or_path: &str, models_dir: &Path) -> Result<PathBuf> {
    let p = Path::new(name_or_path);
    if p.exists() {
        return Ok(p.to_path_buf());
    }

    let candidates = candidate_roots(models_dir);

    for root in &candidates {
        if let Some(blob) = try_resolve(name_or_path, root) {
            return Ok(blob);
        }
    }

    bail!(
        "Model '{}' not found.\n  Searched: {}\n  Tip: use a full path to a .gguf file, or 'name:tag' matching a manifest.",
        name_or_path,
        candidates.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
    )
}

/// List all available model names from all candidate roots.
pub fn list_models(models_dir: &Path) -> Vec<String> {
    let mut models = Vec::new();
    for root in candidate_roots(models_dir) {
        let manifests_root = root.join("manifests").join("registry.ollama.ai").join("library");
        if let Ok(entries) = std::fs::read_dir(&manifests_root) {
            for entry in entries.flatten() {
                let model_name = entry.file_name().to_string_lossy().to_string();
                let model_path = entry.path();
                if let Ok(tags) = std::fs::read_dir(&model_path) {
                    for tag_entry in tags.flatten() {
                        let tag = tag_entry.file_name().to_string_lossy().to_string();
                        models.push(format!("{}:{}", model_name, tag));
                    }
                }
            }
        }
    }
    models.sort();
    models.dedup();
    models
}

fn candidate_roots(models_dir: &Path) -> Vec<PathBuf> {
    let mut roots = vec![models_dir.to_path_buf()];

    let system = PathBuf::from("/usr/share/ollama/.ollama/models");
    if system != models_dir && system.exists() {
        roots.push(system);
    }

    if let Some(home) = std::env::var_os("HOME") {
        let user = PathBuf::from(home).join(".ollama").join("models");
        if user != models_dir && user.exists() {
            roots.push(user);
        }
    }

    roots
}

fn try_resolve(name: &str, root: &Path) -> Option<PathBuf> {
    let (model_name, tag) = name.split_once(':').unwrap_or((name, "latest"));

    let manifest_path = root
        .join("manifests")
        .join("registry.ollama.ai")
        .join("library")
        .join(model_name)
        .join(tag);

    let data = std::fs::read(&manifest_path).ok()?;
    let manifest: serde_json::Value = serde_json::from_slice(&data).ok()?;

    let digest = manifest["layers"]
        .as_array()?
        .iter()
        .find(|l| l["mediaType"].as_str().map_or(false, |m| m.contains("model")))?
        ["digest"]
        .as_str()?
        .replace("sha256:", "sha256-");

    let blob = root.join("blobs").join(&digest);
    if blob.exists() { Some(blob) } else { None }
}
