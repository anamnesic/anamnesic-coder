use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};

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
