use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use anyhow::{Context, Result};
use crate::llm::client::{LlmClient, ResponseFormat, ToolDef};
use crate::models_dev::{ModelsDevClient, base_id};

/// Default cloud provider id used when none is explicitly selected.
pub const DEFAULT_PROVIDER: &str = "ollama-cloud";

/// Runtime router between the local backend (Ollama / GGUF) and a lazily-built
/// OpenAI-compatible cloud backend (NVIDIA NIM, Ollama Cloud, …).
///
/// The TUI starts with a single backend and lets the user switch cloud
/// providers and models live.  This router decides, per model id, whether a
/// request should go to the local server or to the cloud:
///
/// * ids explicitly marked as cloud (picked from a provider's model list) → cloud
/// * ids present in the active provider's catalog (matched by base id, so a
///   single model like `glm-5.2` can be served by several providers) → cloud
/// * provider-qualified ids (`nvidia/…`) → cloud
/// * everything else → local
///
/// The resolved id sent to the API is the active provider's catalog id (e.g.
/// `z-ai/glm-5.2` on NVIDIA NIM vs `glm-5.2` on Ollama Cloud), so one base
/// model works across providers.
///
/// `cloud`, `provider` and `cloud_models` live behind `Arc<Mutex<…>>` so a
/// clone handed to the agent thread sees provider changes made in the UI thread.
#[derive(Clone)]
pub struct LlmRouter {
    local: LlmClient,
    cloud: Arc<Mutex<Option<LlmClient>>>,
    provider: Arc<Mutex<String>>,
    cloud_models: Arc<Mutex<HashSet<String>>>,
    catalog: Arc<ModelsDevClient>,
}

impl LlmRouter {
    pub fn new(local: LlmClient) -> Self {
        Self::with_catalog(local, ModelsDevClient::load())
    }

    /// Build a router against a specific catalog (used by tests).
    pub fn with_catalog(local: LlmClient, catalog: ModelsDevClient) -> Self {
        Self {
            local,
            cloud: Arc::new(Mutex::new(None)),
            provider: Arc::new(Mutex::new(DEFAULT_PROVIDER.to_string())),
            cloud_models: Arc::new(Mutex::new(HashSet::new())),
            catalog: Arc::new(catalog),
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

    /// Resolve a model id against the active provider: `(is_cloud, api_id)`.
    ///
    /// The `api_id` is the provider-specific catalog id when the model is a
    /// cloud model, so one base model (`glm-5.2`) works under any provider
    /// (`z-ai/glm-5.2` on NVIDIA NIM, `glm-5.2` on Ollama Cloud). The catalog
    /// takes precedence over the marked set so a marked base id is still sent
    /// to the API under the active provider's id.
    pub fn resolve(&self, model: &str) -> (bool, String) {
        let provider = self.provider.lock().unwrap().clone();
        if let Some(api_id) = self.catalog.provider_model_api_id(&provider, model) {
            return (true, api_id);
        }
        if self.cloud_models.lock().unwrap().contains(model) {
            return (true, model.to_string());
        }
        if base_id(model) != model {
            return (true, model.to_string());
        }
        (false, model.to_string())
    }

    /// True when `model` should be sent to the cloud backend.
    pub fn is_cloud_model(&self, model: &str) -> bool {
        self.resolve(model).0
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
        let (_, api_id) = self.resolve(model);
        self.client_for(model)?.generate(&api_id, prompt, tools, response_format).await
    }

    pub async fn generate_with_retry(
        &self,
        model: &str,
        prompt: &str,
        tools: Option<&Vec<ToolDef>>,
        response_format: Option<&ResponseFormat>,
    ) -> Result<String> {
        let (_, api_id) = self.resolve(model);
        self.client_for(model)?
            .generate_with_retry(&api_id, prompt, tools, response_format)
            .await
    }

    pub async fn chat(
        &self,
        model: &str,
        messages: Vec<serde_json::Value>,
        tools: Option<&Vec<ToolDef>>,
        response_format: Option<&ResponseFormat>,
    ) -> Result<String> {
        let (_, api_id) = self.resolve(model);
        self.client_for(model)?.chat(&api_id, messages, tools, response_format).await
    }

    pub async fn stream(
        &self,
        model: &str,
        prompt: &str,
        tools: Option<&Vec<ToolDef>>,
        response_format: Option<&ResponseFormat>,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let (_, api_id) = self.resolve(model);
        self.client_for(model)?.stream(&api_id, prompt, tools, response_format, on_token).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models_dev::types::{Catalog, Cost, Limits, ModelInfo, Modalities, Provider};

    fn model(id: &str, tool: bool) -> ModelInfo {
        ModelInfo {
            id: id.into(),
            name: id.into(),
            family: id.into(),
            reasoning: false,
            tool_call: tool,
            temperature: false,
            open_weights: true,
            attachment: false,
            limit: Limits { context: 131_072, output: 4096 },
            cost: Cost { input: 0.0, output: 0.0, cache_read: None, cache_write: None },
            modalities: Modalities { input: vec!["text".into()], output: vec!["text".into()] },
            knowledge: None,
            release_date: None,
        }
    }

    fn provider(id: &str, models: Vec<ModelInfo>) -> Provider {
        let map = models.into_iter().map(|m| (m.id.clone(), m)).collect();
        Provider {
            id: id.into(),
            name: id.into(),
            api: String::new(),
            env: vec![],
            doc: String::new(),
            models: map,
        }
    }

    fn test_catalog() -> Catalog {
        let mut catalog = Catalog::new();
        catalog.insert("ollama-cloud".into(), provider("ollama-cloud", vec![
            model("glm-5.2", true),
            model("nemotron-3-nano:30b", true),
        ]));
        catalog.insert("nvidia".into(), provider("nvidia", vec![
            model("z-ai/glm-5.2", true),
        ]));
        catalog
    }

    fn router() -> LlmRouter {
        LlmRouter::with_catalog(
            LlmClient::ollama("http://localhost:11434"),
            ModelsDevClient { catalog: test_catalog() },
        )
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
    fn single_model_resolves_per_active_provider() {
        let r = router();
        // Default provider is ollama-cloud, which serves glm-5.2 as "glm-5.2".
        assert_eq!(r.resolve("glm-5.2"), (true, "glm-5.2".to_string()));
        // Switch to NVIDIA NIM: same base model, provider-specific id.
        *r.provider.lock().unwrap() = "nvidia".to_string();
        assert_eq!(r.resolve("glm-5.2"), (true, "z-ai/glm-5.2".to_string()));
        assert_eq!(r.resolve("z-ai/glm-5.2"), (true, "z-ai/glm-5.2".to_string()));
        // A model the provider does not serve stays local.
        assert_eq!(r.resolve("qwen3:1.7b"), (false, "qwen3:1.7b".to_string()));
    }

    #[test]
    fn marked_base_model_still_uses_provider_catalog_id() {
        let r = router();
        *r.provider.lock().unwrap() = "nvidia".to_string();
        // Marking a base id must not bypass the catalog's provider-specific id.
        r.mark_cloud("glm-5.2");
        assert_eq!(r.resolve("glm-5.2"), (true, "z-ai/glm-5.2".to_string()));
    }

    #[test]
    fn marked_models_route_to_cloud() {
        let r = router();
        r.mark_cloud("custom-cloud");
        assert!(r.is_cloud_model("custom-cloud"));
        r.unmark_cloud("custom-cloud");
        assert!(!r.is_cloud_model("custom-cloud"));
    }

    #[test]
    fn clear_cloud_marks_resets_marked_set() {
        let r = router();
        r.mark_cloud("custom-a");
        r.mark_cloud("custom-b");
        r.clear_cloud_marks();
        assert!(!r.is_cloud_model("custom-a"));
        assert!(!r.is_cloud_model("custom-b"));
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
