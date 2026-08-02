use anyhow::{bail, Result};
use crate::providers::ProviderEntry;

/// Verify a provider by sending a minimal request to its API.
/// Currently supports: openai-compatible `/models` endpoints.
/// Returns Ok(model_count) on success.
pub fn test_provider(provider_id: &str, entry: &ProviderEntry) -> Result<String> {
    let api_key = match &entry.api_key {
        Some(k) => k.clone(),
        None => bail!("no API key set for provider '{}'", provider_id),
    };

    // Determine the base URL
    let base = entry.api_base.clone()
        .unwrap_or_else(|| default_base(provider_id));

    let models_url = format!("{}/models", base.trim_end_matches('/'));

    let resp = reqwest::blocking::Client::new()
        .get(&models_url)
        .header("Authorization", format!("Bearer {api_key}"))
        .timeout(std::time::Duration::from_secs(10))
        .send();

    match resp {
        Err(e) => bail!("request failed: {e}"),
        Ok(r) => {
            let status = r.status();
            if status == 401 {
                bail!("invalid API key (HTTP 401)");
            } else if status == 403 {
                bail!("forbidden — check key permissions (HTTP 403)");
            } else if !status.is_success() {
                bail!("HTTP {status} from {models_url}");
            }
            // Try to parse as OpenAI-style { data: [...] }
            let body = r.text().unwrap_or_default();
            let count: Option<usize> = serde_json::from_str::<serde_json::Value>(&body)
                .ok()
                .and_then(|v| v.get("data")?.as_array().map(|a| a.len()));
            Ok(count.map(|n| format!("{n} models listed")).unwrap_or_else(|| "OK".into()))
        }
    }
}

/// Default API base URL per provider (used when store/catalog have none).
pub(crate) fn default_base(provider_id: &str) -> String {
    match provider_id {
        "nvidia" | "deepseek" | "minimax" | "z-ai" | "openai" | "anthropic" | "google" | "mistral" | "groq" | "togetherai" | "cohere" | "openrouter" | "perplexity" | "fireworks-ai" | "deepinfra" | "cerebras" | "ollama-cloud" => {
            "https://integrate.api.nvidia.com/v1".into()
        }
        other => format!("https://api.{other}.com/v1"),
    }
}
