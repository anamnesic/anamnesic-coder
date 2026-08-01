use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};
use std::sync::{Arc, Mutex};
use crate::llm::infer::engine::InferenceEngine;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ToolDef {
    pub r#type: String,
    pub function: ToolFunction,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: ToolCallFunction,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ToolCallResult {
    pub tool_call_id: String,
    pub content: String,
}

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

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<serde_json::Value>,
    stream: bool,
    max_tokens: u32,
    tools: Option<Vec<ToolDef>>,
}

#[derive(Deserialize)]
struct ChatResponse {
    message: ChatMessage,
}

#[derive(Deserialize)]
struct ChatMessage {
    content: String,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Serialize)]
struct CloudChatRequest {
    model: String,
    messages: Vec<serde_json::Value>,
    stream: bool,
    max_tokens: u32,
    tools: Option<Vec<ToolDef>>,
}

#[derive(Deserialize)]
struct CloudChatResponse {
    choices: Vec<CloudChoice>,
}

#[derive(Deserialize)]
struct CloudChoice {
    message: CloudMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct CloudMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Serialize)]
struct CloudStreamRequest {
    model: String,
    messages: Vec<serde_json::Value>,
    stream: bool,
    max_tokens: u32,
    tools: Option<Vec<ToolDef>>,
}

#[derive(Deserialize)]
struct CloudStreamDelta {
    choices: Option<Vec<CloudStreamChoice>>,
}

#[derive(Deserialize)]
struct CloudStreamChoice {
    delta: CloudStreamDeltaContent,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct CloudStreamDeltaContent {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
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

    pub async fn generate(&self, model: &str, prompt: &str, tools: Option<&Vec<ToolDef>>) -> Result<String> {
        match self {
            LlmClient::Ollama(c) => c.generate(model, prompt, tools).await,
            LlmClient::Cloud(c)  => c.generate(model, prompt, tools).await,
            LlmClient::Local(eng) => {
                let mut eng = eng.lock().map_err(|e| anyhow::anyhow!("Lock error: {}", e))?;
                eng.generate(prompt, 512, 0.8, 40).map_err(|e| anyhow::anyhow!("Local inference error: {}", e))
            }
        }
    }

    /// Generate with bounded retries and exponential backoff on transient failures.
    pub async fn generate_with_retry(&self, model: &str, prompt: &str, tools: Option<&Vec<ToolDef>>) -> Result<String> {
        let mut last_err: Option<anyhow::Error> = None;
        for attempt in 0..3 {
            match self.generate(model, prompt, tools).await {
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
    pub async fn stream(&self, model: &str, prompt: &str, tools: Option<&Vec<ToolDef>>, on_token: &mut dyn FnMut(&str)) -> Result<String> {
        match self {
            LlmClient::Ollama(c) => c.stream_generate(model, prompt, tools, on_token).await,
            LlmClient::Cloud(c) => {
                let messages = vec![serde_json::json!({ "role": "user", "content": prompt })];
                c.stream_chat(model, messages, tools, on_token).await
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

    pub async fn chat(&self, model: &str, messages: Vec<serde_json::Value>, tools: Option<&Vec<ToolDef>>) -> Result<String> {
        match self {
            LlmClient::Ollama(c) => c.chat(model, messages, tools).await,
            LlmClient::Cloud(c)  => c.chat(model, messages, tools).await,
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

    pub async fn generate(&self, model: &str, prompt: &str, tools: Option<&Vec<ToolDef>>) -> Result<String> {
        let body = serde_json::json!({
            "model": model,
            "prompt": prompt,
            "stream": false,
            "tools": tools.unwrap_or(&vec![]),
        });

        let resp = self.client
            .post(format!("{}/api/generate", self.host))
            .json(&body)
            .send()
            .await
            .context("Ollama request failed")?;

        let data: serde_json::Value = resp
            .json()
            .await
            .context("Failed to parse Ollama response")?;

        // Check if the response contains tool_calls
        if let Some(tool_calls) = data.get("message").and_then(|m| m.get("tool_calls")).and_then(|t| t.as_array()) {
            let mut results = Vec::new();
            for tc in tool_calls {
                let name = tc.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
                let arguments = tc.get("arguments").and_then(|a| a.as_str()).map(|s| s.to_string()).unwrap_or_default();
                results.push(ToolCall {
                    id: tc.get("id").and_then(|i| i.as_str()).unwrap_or("").to_string(),
                    r#type: "function".to_string(),
                    function: ToolCallFunction { name, arguments },
                });
            }
            return Ok(serde_json::to_string(&results)?);
        }

        Ok(data["response"].as_str().unwrap_or("").to_string())
    }

    pub async fn chat(&self, model: &str, messages: Vec<serde_json::Value>, tools: Option<&Vec<ToolDef>>) -> Result<String> {
        let body = ChatRequest {
            model: model.to_string(),
            messages,
            stream: false,
            max_tokens: 2048,
            tools: tools.cloned(),
        };

        let resp = self.client
            .post(format!("{}/api/chat", self.host))
            .json(&body)
            .send()
            .await
            .context("Ollama chat request failed")?;

        let data: serde_json::Value = resp
            .json()
            .await
            .context("Failed to parse Ollama chat response")?;

        // Check for tool_calls in the response
        if let Some(tool_calls) = data.get("message").and_then(|m| m.get("tool_calls")).and_then(|t| t.as_array()) {
            let mut results = Vec::new();
            for tc in tool_calls {
                let name = tc.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
                let arguments = tc.get("arguments").and_then(|a| a.as_str()).map(|s| s.to_string()).unwrap_or_default();
                results.push(ToolCall {
                    id: tc.get("id").and_then(|i| i.as_str()).unwrap_or("").to_string(),
                    r#type: "function".to_string(),
                    function: ToolCallFunction { name, arguments },
                });
            }
            return Ok(serde_json::to_string(&results)?);
        }

        Ok(data["message"]["content"].as_str().unwrap_or("").to_string())
    }

    /// Stream a `/api/generate` call (NDJSON lines), feeding text chunks to `on_token`.
    /// Supports tool calls — when the model emits tool_calls, they are returned
    /// as a JSON array string instead of streaming text.
    pub async fn stream_generate(&self, model: &str, prompt: &str, tools: Option<&Vec<ToolDef>>, on_token: &mut dyn FnMut(&str)) -> Result<String> {
        let body = serde_json::json!({
            "model": model,
            "prompt": prompt,
            "stream": true,
            "tools": tools.unwrap_or(&vec![]),
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
let mut _saw_tool_calls = false;
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
                    if let Some(tool_calls) = v.get("message").and_then(|m| m.get("tool_calls")).and_then(|t| t.as_array()) {
                        _saw_tool_calls = true;
                        for tc in tool_calls {
                            let name = tc.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
                            let arguments = tc.get("arguments").and_then(|a| a.as_str()).map(|s| s.to_string()).unwrap_or_default();
                            let id = tc.get("id").and_then(|i| i.as_str()).unwrap_or("").to_string();
                            on_token(&serde_json::json!({
                                "tool_call": {
                                    "id": id,
                                    "type": "function",
                                    "function": { "name": name, "arguments": arguments }
                                }
                            }).to_string());
                        }
                    }
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
    pub async fn generate(&self, model: &str, prompt: &str, tools: Option<&Vec<ToolDef>>) -> Result<String> {
        let body = serde_json::json!({
            "model": model,
            "prompt": prompt,
            "stream": false,
            "max_tokens": 2048,
            "tools": tools.unwrap_or(&vec![]),
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
            return self.chat(model, messages, tools).await;
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

        // Check for tool_calls in the response
        if let Some(tool_calls) = data["choices"][0]["message"]["tool_calls"].as_array() {
            return Ok(serde_json::to_string(tool_calls)?);
        }

        data["choices"][0]["text"].as_str()
            .map(|s| s.to_string())
            .or_else(|| data["choices"][0]["message"]["content"].as_str().map(|s| s.to_string()))
            .context("cloud response missing completion text")
    }

    /// Chat completion (`POST /v1/chat/completions`).
    pub async fn chat(&self, model: &str, messages: Vec<serde_json::Value>, tools: Option<&Vec<ToolDef>>) -> Result<String> {
        let body = CloudChatRequest {
            model: model.to_string(),
            messages,
            stream: false,
            max_tokens: 2048,
            tools: tools.cloned(),
        };

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

        // Check for tool_calls in the response
        if let Some(tool_calls) = data["choices"][0]["message"]["tool_calls"].as_array() {
            return Ok(serde_json::to_string(tool_calls)?);
        }

        data["choices"][0]["message"]["content"].as_str()
            .map(|s| s.to_string())
            .context("cloud chat response missing content")
    }

    /// Stream a chat completion (SSE), feeding content deltas to `on_token`.
    /// When tool_calls are emitted, they are passed to `on_token` as JSON.
    pub async fn stream_chat(&self, model: &str, messages: Vec<serde_json::Value>, tools: Option<&Vec<ToolDef>>, on_token: &mut dyn FnMut(&str)) -> Result<String> {
        let body = CloudStreamRequest {
            model: model.to_string(),
            messages,
            stream: true,
            max_tokens: 2048,
            tools: tools.cloned(),
        };

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
        let mut _saw_tool_calls = false;
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
                    if let Some(delta) = v["choices"][0]["delta"].as_object() {
                        if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                            if !content.is_empty() {
                                on_token(content);
                                full.push_str(content);
                            }
                        }
                        if let Some(tcs) = delta.get("tool_calls").and_then(|t| t.as_array()) {
                            _saw_tool_calls = true;
                            for tc in tcs {
                                on_token(&serde_json::json!({
                                    "tool_call": tc
                                }).to_string());
                            }
                        }
                    }
                }
            }
        }
        Ok(full)
    }
}
