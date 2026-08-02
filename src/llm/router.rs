use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use anyhow::{Context, Result};
use crate::llm::client::{LlmClient, ResponseFormat, ToolDef};

/// Default cloud provider id used when none is explicitly selected.
pub const DEFAULT_PROVIDER: &str = "ollama-cloud";

/// Runtime router between the local backend (Ollama / GGUF) and a lazily-built
/// OpenAI-compatible cloud backend (NVIDIA NIM, Ollama Cloud, …).
///
/// The TUI starts with a single backend and lets the user switch cloud
/// providers and models live.  This router decides, per model id, whether a
/// request should go to the local server or to the cloud:
///
/// * provider-qualified ids (`nvidia/…`) → cloud
/// * ids explicitly marked as cloud (picked from a provider's model list) → cloud
/// * everything else → local
///
/// `cloud` and `provider` live behind `Arc<Mutex<…>>` so a clone handed to the
/// agent thread sees provider changes made in the UI thread.
#[derive(Clone)]
pub struct LlmRouter {
    local: LlmClient,
    cloud: Arc<Mutex<Option<LlmClient>>>,
    provider: Arc<Mutex<String>>,
    cloud_models: Arc<Mutex<HashSet<String>>>,
}

impl LlmRouter {
    pub fn new(local: LlmClient) -> Self {
        Self {
            local,
            cloud: Arc::new(Mutex::new(None)),
            provider: Arc::new(Mutex::new(DEFAULT_PROVIDER.to_string())),
            cloud_models: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// The currently active cloud provider id ("" when not configured).
    pub fn provider(&self) -> String {
        self.provider.lock().unwrap().clone()
    }

    /// Build (or rebuild) the cloud backend for `provider` from the models.dev
    /// catalog default base URL + the configured/env API key.  Returns the API
    /// base URL on success.
    pub fn set_provider(&self, provider: &str) -> Result<String> {
        let catalog_client = crate::models_dev::ModelsDevClient::load();
        let (base, key) = crate::providers::ProviderStore::resolve_cloud_credentials(
            provider,
            &catalog_client.catalog,
        )
        .with_context(|| format!("configuring cloud provider '{provider}'"))?;
        *self.cloud.lock().unwrap() = Some(LlmClient::cloud(&base, &key));
        *self.provider.lock().unwrap() = provider.to_string();
        Ok(base)
    }

    /// Whether a cloud client is currently configured.
    pub fn cloud_configured(&self) -> bool {
        self.cloud.lock().unwrap().is_some()
    }

    /// Mark a model id as a cloud model (used by the TUI model picker).
    pub fn mark_cloud(&self, model: &str) {
        self.cloud_models.lock().unwrap().insert(model.to_string());
    }

    /// Unmark a model id so it routes to the local backend again.
    pub fn unmark_cloud(&self, model: &str) {
        self.cloud_models.lock().unwrap().remove(model);
    }

    /// Clear all cloud model markings (e.g. after switching provider).
    pub fn clear_cloud_marks(&self) {
        self.cloud_models.lock().unwrap().clear();
    }

    /// True when `model` should be sent to the cloud backend.
    pub fn is_cloud_model(&self, model: &str) -> bool {
        model.contains('/') || self.cloud_models.lock().unwrap().contains(model)
    }

    /// Pick the concrete client for a model id (cloned; cheap for reqwest).
    pub fn client_for(&self, model: &str) -> Result<LlmClient> {
        if self.is_cloud_model(model) {
            self.cloud
                .lock()
                .unwrap()
                .clone()
                .with_context(|| {
                    format!("model '{model}' is a cloud model but no cloud provider is configured (use /provider)")
                })
        } else {
            Ok(self.local.clone())
        }
    }

    pub async fn generate(
        &self,
        model: &str,
        prompt: &str,
        tools: Option<&Vec<ToolDef>>,
        response_format: Option<&ResponseFormat>,
    ) -> Result<String> {
        self.client_for(model)?.generate(model, prompt, tools, response_format).await
    }

    pub async fn generate_with_retry(
        &self,
        model: &str,
        prompt: &str,
        tools: Option<&Vec<ToolDef>>,
        response_format: Option<&ResponseFormat>,
    ) -> Result<String> {
        self.client_for(model)?
            .generate_with_retry(model, prompt, tools, response_format)
            .await
    }

    pub async fn chat(
        &self,
        model: &str,
        messages: Vec<serde_json::Value>,
        tools: Option<&Vec<ToolDef>>,
        response_format: Option<&ResponseFormat>,
    ) -> Result<String> {
        self.client_for(model)?.chat(model, messages, tools, response_format).await
    }

    pub async fn stream(
        &self,
        model: &str,
        prompt: &str,
        tools: Option<&Vec<ToolDef>>,
        response_format: Option<&ResponseFormat>,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<String> {
        self.client_for(model)?.stream(model, prompt, tools, response_format, on_token).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn router() -> LlmRouter {
        LlmRouter::new(LlmClient::ollama("http://localhost:11434"))
    }

    #[test]
    fn defaults_to_ollama_cloud_provider() {
        assert_eq!(router().provider(), "ollama-cloud");
        assert!(!router().cloud_configured());
    }

    #[test]
    fn provider_qualified_ids_route_to_cloud() {
        let r = router();
        assert!(r.is_cloud_model("nvidia/llama-3.3-nemotron-super-49b-v1.5"));
        assert!(r.is_cloud_model("ollama-cloud/glm-5.1"));
        assert!(!r.is_cloud_model("qwen3:1.7b"));
    }

    #[test]
    fn marked_models_route_to_cloud() {
        let r = router();
        r.mark_cloud("nemotron-3-nano:30b");
        assert!(r.is_cloud_model("nemotron-3-nano:30b"));
        r.unmark_cloud("nemotron-3-nano:30b");
        assert!(!r.is_cloud_model("nemotron-3-nano:30b"));
    }

    #[test]
    fn clear_cloud_marks_resets_marked_set() {
        let r = router();
        r.mark_cloud("glm-5.1");
        r.mark_cloud("kimi-k2.7-code");
        r.clear_cloud_marks();
        assert!(!r.is_cloud_model("glm-5.1"));
        assert!(!r.is_cloud_model("kimi-k2.7-code"));
    }

    #[test]
    fn local_model_uses_local_client() {
        let r = router();
        let client = r.client_for("qwen3:1.7b").unwrap();
        match client {
            LlmClient::Ollama(_) => {}
            _ => panic!("expected local Ollama client"),
        }
    }

    #[test]
    fn unmapped_cloud_model_errors_without_configured_provider() {
        let r = router();
        let err = match r.client_for("nvidia/llama-3.3-nemotron-super-49b-v1.5") {
            Ok(_) => panic!("expected error for unconfigured cloud model"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("no cloud provider is configured"));
    }
}
