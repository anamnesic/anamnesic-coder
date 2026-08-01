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

/// OpenAI-compatible cloud client (used for NVIDIA NIM and other providers).
pub struct CloudClient {
    base_url: String,
    api_key: String,
    client: reqwest::Client,
}

impl Clone for CloudClient {
    fn clone(&self) -> Self {
        Self {
            base_url: self.base_url.clone(),
            api_key: self.api_key.clone(),
            client: self.client.clone(),
        }
    }
}

pub enum LlmClient {
    Ollama(OllamaClient),
    Local(Arc<Mutex<InferenceEngine>>),
    Cloud(CloudClient),
}

impl Clone for LlmClient {
    fn clone(&self) -> Self {
        match self {
            LlmClient::Ollama(c) => LlmClient::Ollama(c.clone()),
            LlmClient::Local(e)  => LlmClient::Local(Arc::clone(e)),
            LlmClient::Cloud(c)  => LlmClient::Cloud(c.clone()),
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

    /// OpenAI-compatible cloud backend (NVIDIA NIM, etc.).
    pub fn cloud(base_url: &str, api_key: &str) -> Self {
        LlmClient::Cloud(CloudClient::new(base_url, api_key))
    }

    pub async fn generate(&self, model: &str, prompt: &str) -> Result<String> {
        match self {
            LlmClient::Ollama(c) => c.generate(model, prompt).await,
            LlmClient::Cloud(c)  => c.generate(model, prompt).await,
            LlmClient::Local(eng) => {
                let mut eng = eng.lock().map_err(|e| anyhow::anyhow!("Lock error: {}", e))?;
                eng.generate(prompt, 512, 0.8, 40).map_err(|e| anyhow::anyhow!("Local inference error: {}", e))
            }
        }
    }

    /// Generate with bounded retries and exponential backoff on transient failures.
    pub async fn generate_with_retry(&self, model: &str, prompt: &str) -> Result<String> {
        let mut last_err: Option<anyhow::Error> = None;
        for attempt in 0..3 {
            match self.generate(model, prompt).await {
                Ok(text) => return Ok(text),
                Err(e) => {
                    eprintln!("  ⚠ LLM call failed (attempt {}): {e}", attempt + 1);
                    last_err = Some(e);
                    if attempt < 2 {
                        let backoff = std::time::Duration::from_millis(500 * (1 << attempt));
                        tokio::time::sleep(backoff).await;
                    }
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("LLM call failed")))
    }

    /// Stream a response, invoking `on_token` for each text chunk.
    /// Falls back to non-streaming output for local engines.
    pub async fn stream(&self, model: &str, prompt: &str, on_token: &mut dyn FnMut(&str)) -> Result<String> {
        match self {
            LlmClient::Ollama(c) => c.stream_generate(model, prompt, on_token).await,
            LlmClient::Cloud(c) => {
                let messages = vec![serde_json::json!({ "role": "user", "content": prompt })];
                c.stream_chat(model, messages, on_token).await
            }
            LlmClient::Local(eng) => {
                let mut eng = eng.lock().map_err(|e| anyhow::anyhow!("Lock error: {}", e))?;
                let text = eng.generate(prompt, 512, 0.8, 40)
                    .map_err(|e| anyhow::anyhow!("Local inference error: {}", e))?;
                on_token(&text);
                Ok(text)
            }
        }
    }

    pub async fn chat(&self, model: &str, messages: Vec<serde_json::Value>) -> Result<String> {
        match self {
            LlmClient::Ollama(c) => c.chat(model, messages).await,
            LlmClient::Cloud(c)  => c.chat(model, messages).await,
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

    /// Stream a `/api/generate` call (NDJSON lines), feeding text chunks to `on_token`.
    pub async fn stream_generate(&self, model: &str, prompt: &str, on_token: &mut dyn FnMut(&str)) -> Result<String> {
        let body = serde_json::json!({
            "model": model,
            "prompt": prompt,
            "stream": true,
        });

        let mut resp = self.client
            .post(format!("{}/api/generate", self.host))
            .json(&body)
            .send()
            .await
            .context("Ollama stream request failed")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("Ollama stream request failed: HTTP {status} {text}");
        }

        let mut buffer: Vec<u8> = Vec::new();
        let mut full = String::new();
        let mut done = false;
        while !done {
            let Some(chunk) = resp.chunk().await.context("Ollama stream read failed")? else {
                break;
            };
            buffer.extend_from_slice(&chunk);
            while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buffer.drain(..=pos).collect();
                let line = String::from_utf8_lossy(&line).trim().to_string();
                if line.is_empty() {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                    if let Some(t) = v["response"].as_str() {
                        if !t.is_empty() {
                            on_token(t);
                            full.push_str(t);
                        }
                    }
                    if v["done"].as_bool() == Some(true) {
                        done = true;
                        break;
                    }
                }
            }
        }
        Ok(full)
    }
}

impl CloudClient {
    pub fn new(base_url: &str, api_key: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .unwrap_or_default(),
        }
    }

    /// Single-turn text completion (`POST /v1/completions`).
    pub async fn generate(&self, model: &str, prompt: &str) -> Result<String> {
        let body = serde_json::json!({
            "model": model,
            "prompt": prompt,
            "stream": false,
            "max_tokens": 2048,
        });

        let resp = self.client
            .post(format!("{}/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .context("cloud completions request failed")?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            // Newer OpenAI-compatible endpoints (e.g. NVIDIA NIM) are chat-only.
            let messages = vec![serde_json::json!({"role": "user", "content": prompt})];
            return self.chat(model, messages).await;
        }
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("cloud completions request failed: HTTP {status} {text}");
        }

        let data: serde_json::Value = resp
            .json()
            .await
            .context("Failed to parse cloud completions response")?;

        data["choices"][0]["text"].as_str()
            .map(|s| s.to_string())
            .or_else(|| data["choices"][0]["message"]["content"].as_str().map(|s| s.to_string()))
            .context("cloud response missing completion text")
    }

    /// Chat completion (`POST /v1/chat/completions`).
    pub async fn chat(&self, model: &str, messages: Vec<serde_json::Value>) -> Result<String> {
        let body = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": false,
            "max_tokens": 2048,
        });

        let resp = self.client
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .context("cloud chat request failed")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("cloud chat request failed: HTTP {status} {text}");
        }

        let data: serde_json::Value = resp
            .json()
            .await
            .context("Failed to parse cloud chat response")?;

        data["choices"][0]["message"]["content"].as_str()
            .map(|s| s.to_string())
            .context("cloud chat response missing content")
    }

    /// Stream a chat completion (SSE), feeding content deltas to `on_token`.
    pub async fn stream_chat(&self, model: &str, messages: Vec<serde_json::Value>, on_token: &mut dyn FnMut(&str)) -> Result<String> {
        let body = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": true,
            "max_tokens": 2048,
        });

        let mut resp = self.client
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .context("cloud stream request failed")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("cloud stream request failed: HTTP {status} {text}");
        }

        let mut buffer: Vec<u8> = Vec::new();
        let mut full = String::new();
        loop {
            let Some(chunk) = resp.chunk().await.context("cloud stream read failed")? else {
                break;
            };
            buffer.extend_from_slice(&chunk);
            while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buffer.drain(..=pos).collect();
                let line = String::from_utf8_lossy(&line).trim().to_string();
                let Some(data) = line.strip_prefix("data:") else { continue };
                let data = data.trim();
                if data == "[DONE]" {
                    return Ok(full);
                }
                if data.is_empty() {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(t) = v["choices"][0]["delta"]["content"].as_str() {
                        if !t.is_empty() {
                            on_token(t);
                            full.push_str(t);
                        }
                    }
                }
            }
        }
        Ok(full)
    }
}
