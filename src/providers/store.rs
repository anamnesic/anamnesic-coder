use std::collections::HashMap;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use crate::models_dev::Catalog;

/// A single configured provider entry.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ProviderEntry {
    /// The API key — stored in the config file (never logged).
    pub api_key: Option<String>,
    /// Override the default API base URL (e.g. for proxies).
    pub api_base: Option<String>,
    /// Whether this provider is enabled for routing.
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_true() -> bool { true }

/// All configured providers, keyed by provider_id matching models.dev.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ProviderStore {
    #[serde(flatten)]
    pub providers: HashMap<String, ProviderEntry>,
}

impl ProviderStore {
    /// Load from `~/.config/rustcode/providers.toml`.
    /// Returns empty store if file doesn't exist yet.
    pub fn load() -> Self {
        match Self::load_inner() {
            Ok(s) => s,
            Err(_) => Self::default(),
        }
    }

    fn load_inner() -> Result<Self> {
        let path = config_path()?;
        if !path.exists() { return Ok(Self::default()); }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        toml::from_str(&text).context("parsing providers.toml")
    }

    /// Persist to disk with 0600 permissions (owner read/write only).
    pub fn save(&self) -> Result<()> {
        let path = config_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let text = toml::to_string_pretty(self).context("serialising providers")?;
        std::fs::write(&path, &text)
            .with_context(|| format!("writing {}", path.display()))?;
        // Restrict to owner-only
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("chmod 600 {}", path.display()))?;
        Ok(())
    }

    /// Set (or update) an API key for a provider_id.
    pub fn set_key(&mut self, provider_id: &str, api_key: &str) {
        let entry = self.providers.entry(provider_id.to_string()).or_default();
        entry.api_key = Some(api_key.to_string());
        entry.enabled = true;
    }

    /// Set a custom API base URL for a provider.
    pub fn set_base(&mut self, provider_id: &str, api_base: &str) {
        let entry = self.providers.entry(provider_id.to_string()).or_default();
        entry.api_base = Some(api_base.to_string());
    }

    /// Remove a provider (unset key + config).
    pub fn remove(&mut self, provider_id: &str) {
        self.providers.remove(provider_id);
    }

    /// Enable or disable a provider without removing the key.
    pub fn set_enabled(&mut self, provider_id: &str, enabled: bool) {
        if let Some(e) = self.providers.get_mut(provider_id) {
            e.enabled = enabled;
        }
    }

    /// Retrieve the API key for a provider (None if not configured).
    pub fn api_key(&self, provider_id: &str) -> Option<&str> {
        self.providers.get(provider_id)
            .and_then(|e| e.api_key.as_deref())
    }

    /// Configured providers that are enabled and have a key.
    pub fn active_providers(&self) -> Vec<(&str, &ProviderEntry)> {
        self.providers.iter()
            .filter(|(_, e)| e.enabled && e.api_key.is_some())
            .map(|(id, e)| (id.as_str(), e))
            .collect()
    }

    pub fn config_path_display() -> String {
        config_path()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "~/.config/rustcode/providers.toml".into())
    }

    /// Scan the current environment (plus an optional .env file) for known
    /// provider API key vars (from the models.dev catalog) and return those
    /// found but not yet configured in the store.
    pub fn detect_env_keys(catalog: &Catalog) -> Vec<(String, String, String)> {
        // Build a merged map: real env vars + .env file (env takes precedence)
        let env_map = load_env_with_dotenv();

        let store = Self::load();
        let mut found: Vec<(String, String, String)> = Vec::new();

        for (pid, prov) in catalog {
            if store.providers.get(pid).and_then(|e| e.api_key.as_deref()).is_some() {
                continue;
            }
            for env_name in &prov.env {
                if let Some(val) = env_map.get(env_name) {
                    if !val.is_empty() {
                        found.push((pid.clone(), env_name.clone(), val.clone()));
                        break;
                    }
                }
            }
        }
        found
    }

    /// Import keys found in environment (and .env file) into the store and persist.
    pub fn import_from_env(catalog: &Catalog) -> Result<Vec<String>> {
        let detected = Self::detect_env_keys(catalog);
        if detected.is_empty() { return Ok(vec![]); }
        let mut store = Self::load();
        let mut imported = Vec::new();
        for (pid, env_name, key) in detected {
            store.set_key(&pid, &key);
            imported.push(format!("{pid} (from ${env_name})"));
        }
        store.save()?;
        Ok(imported)
    }
}

fn config_path() -> Result<PathBuf> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("XDG_CONFIG_HOME"))
        .context("HOME not set")?;
    Ok(PathBuf::from(home).join(".config").join("rustcode").join("providers.toml"))
}

/// Build a merged env-var map: real process environment + .env file in the
/// current working directory (real env takes precedence over .env values).
fn load_env_with_dotenv() -> HashMap<String, String> {
    let mut map: HashMap<String, String> = std::env::vars().collect();

    // Try project .env in cwd, then parent dirs up to 3 levels
    let candidates = [
        std::env::current_dir().ok().map(|p| p.join(".env")),
        std::env::current_dir().ok().map(|p| p.parent().map(|pp| pp.join(".env"))).flatten(),
    ];
    for maybe_path in candidates.into_iter().flatten() {
        if maybe_path.exists() {
            if let Ok(text) = std::fs::read_to_string(&maybe_path) {
                for line in text.lines() {
                    let line = line.trim();
                    if line.is_empty() || line.starts_with('#') { continue; }
                    // Remove optional `export ` prefix
                    let line = line.strip_prefix("export ").unwrap_or(line);
                    if let Some((k, v)) = line.split_once('=') {
                        let k = k.trim().to_string();
                        let v = v.trim().trim_matches('"').trim_matches('\'').to_string();
                        // real env takes precedence
                        map.entry(k).or_insert(v);
                    }
                }
                break; // use first .env found
            }
        }
    }
    map
}

/// Pretty-print the store for `providers show`.
/// Also shows any keys detectable from environment that aren't yet configured.
pub fn print_store(store: &ProviderStore, catalog: &Catalog) {
    println!("\n  Configured cloud providers");
    println!("  Config: {}", ProviderStore::config_path_display());
    println!("{}", "─".repeat(70));
    println!("  {:<20} {:<8} {:<10} {:<30}", "Provider", "Enabled", "Key", "API base");
    println!("{}", "─".repeat(70));

    // Only show providers that are configured or have keys in env
    let env_detected: Vec<_> = ProviderStore::detect_env_keys(catalog);
    let env_providers: std::collections::HashSet<&str> =
        env_detected.iter().map(|(pid, _, _)| pid.as_str()).collect();

    let mut ids: Vec<String> = store.providers.keys().cloned().collect();
    for (pid, _, _) in &env_detected {
        if !ids.contains(pid) { ids.push(pid.clone()); }
    }
    ids.sort();

    for id in &ids {
        let entry = store.providers.get(id);
        let in_env = env_providers.contains(id.as_str());
        let enabled = entry.map(|e| e.enabled).unwrap_or(in_env);
        let key_status = match entry.and_then(|e| e.api_key.as_deref()) {
            Some(k) => mask_key(k),
            None if in_env => "env (not saved)".into(),
            None => "—".into(),
        };
        let base = entry.and_then(|e| e.api_base.as_deref())
            .or_else(|| catalog.get(id).map(|p| p.api.as_str()))
            .unwrap_or("—");
        let enabled_str = if enabled { "yes" } else { "—" };
        println!("  {:<20} {:<8} {:<15} {:<30}", id, enabled_str, key_status, base);
    }

    if ids.is_empty() {
        println!("  (none — use: rust-agent providers set <id> <api-key>)");
        println!("  (or:   rust-agent providers import   to read from environment)");
    } else if !env_detected.is_empty() {
        let unsaved: Vec<_> = env_detected.iter()
            .filter(|(pid, _, _)| store.providers.get(pid).and_then(|e| e.api_key.as_deref()).is_none())
            .map(|(pid, env, _)| format!("{pid} (${env})"))
            .collect();
        if !unsaved.is_empty() {
            println!("\n  ⚡ Keys found in environment but not yet saved:");
            for s in &unsaved { println!("     {s}"); }
            println!("  Run: rust-agent providers import");
        }
    }
    println!();
}

/// Show only 4 chars of the key then asterisks.
fn mask_key(key: &str) -> String {
    let n = key.len().min(4);
    format!("{}****", &key[..n])
}
