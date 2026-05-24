use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use std::sync::{Arc, Mutex};
use crate::llm::infer::engine::InferenceEngine;

#[derive(Serialize)]
struct GenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(Deserialize)]
struct GenerateResponse {
    response: String,
}

pub struct OllamaClient {
    host: String,
    client: reqwest::Client,
}

impl Clone for OllamaClient {
    fn clone(&self) -> Self {
        Self { host: self.host.clone(), client: self.client.clone() }
    }
}

pub enum LlmClient {
    Ollama(OllamaClient),
    Local(Arc<Mutex<InferenceEngine>>),
}

impl Clone for LlmClient {
    fn clone(&self) -> Self {
        match self {
            LlmClient::Ollama(c) => LlmClient::Ollama(c.clone()),
            LlmClient::Local(e)  => LlmClient::Local(Arc::clone(e)),
        }
    }
}

impl LlmClient {
    pub fn ollama(host: &str) -> Self {
        LlmClient::Ollama(OllamaClient::new(host))
    }

    pub fn local(engine: InferenceEngine) -> Self {
        LlmClient::Local(Arc::new(Mutex::new(engine)))
    }

    pub async fn generate(&self, model: &str, prompt: &str) -> Result<String> {
        match self {
            LlmClient::Ollama(c) => c.generate(model, prompt).await,
            LlmClient::Local(eng) => {
                let mut eng = eng.lock().map_err(|e| anyhow::anyhow!("Lock error: {}", e))?;
                eng.generate(prompt, 512, 0.8, 40).map_err(|e| anyhow::anyhow!("Local inference error: {}", e))
            }
        }
    }

    pub async fn chat(&self, model: &str, messages: Vec<serde_json::Value>) -> Result<String> {
        match self {
            LlmClient::Ollama(c) => c.chat(model, messages).await,
            LlmClient::Local(eng) => {
                // Flatten chat messages into a single prompt for local inference
                let prompt = messages.iter()
                    .filter_map(|m| {
                        let role = m.get("role")?.as_str()?;
                        let content = m.get("content")?.as_str()?;
                        Some(format!("{}: {}", role, content))
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let mut eng = eng.lock().map_err(|e| anyhow::anyhow!("Lock error: {}", e))?;
                eng.generate(&prompt, 512, 0.8, 40).map_err(|e| anyhow::anyhow!("Local inference error: {}", e))
            }
        }
    }
}

impl OllamaClient {
    pub fn new(host: &str) -> Self {
        Self {
            host: host.trim_end_matches('/').to_string(),
            client: reqwest::Client::new(),
        }
    }

    pub async fn generate(&self, model: &str, prompt: &str) -> Result<String> {
        let body = GenerateRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            stream: false,
        };

        let resp = self.client
            .post(format!("{}/api/generate", self.host))
            .json(&body)
            .send()
            .await
            .context("Ollama request failed")?;

        let data: GenerateResponse = resp
            .json()
            .await
            .context("Failed to parse Ollama response")?;

        Ok(data.response)
    }

    pub async fn chat(&self, model: &str, messages: Vec<serde_json::Value>) -> Result<String> {
        #[derive(Serialize)]
        struct ChatRequest {
            model: String,
            messages: Vec<serde_json::Value>,
            stream: bool,
        }

        #[derive(Deserialize)]
        struct ChatResponse {
            message: ChatMessage,
        }

        #[derive(Deserialize)]
        struct ChatMessage {
            content: String,
        }

        let body = ChatRequest {
            model: model.to_string(),
            messages,
            stream: false,
        };

        let resp = self.client
            .post(format!("{}/api/chat", self.host))
            .json(&body)
            .send()
            .await
            .context("Ollama chat request failed")?;

        let data: ChatResponse = resp
            .json()
            .await
            .context("Failed to parse Ollama chat response")?;

        Ok(data.message.content)
    }
}
