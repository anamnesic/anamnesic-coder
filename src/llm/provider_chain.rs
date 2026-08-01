//! Módulo de rate-limiting + fallback pro adapter HTTP OpenAI-compatible.
//! Encaixa entre o plan/act/verify loop e o provider real (NIM ou local).
//!
//! Deps sugeridas no Cargo.toml:
//! tokio = { version = "1", features = ["full"] }
//! reqwest = { version = "0.12", features = ["json"] }
//! serde = { version = "1", features = ["derive"] }
//! rand = "0.8"

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use rand::Rng;

// ---------- Token bucket (rate limiter) ----------

/// Token bucket simples. 40 rpm default = 1 token a cada 1.5s, capacidade 40.
pub struct TokenBucket {
    capacity: f64,
    tokens: Mutex<f64>,
    refill_per_sec: f64,
    last_refill: Mutex<Instant>,
}

impl TokenBucket {
    pub fn new(rpm: f64) -> Self {
        Self {
            capacity: rpm,
            tokens: Mutex::new(rpm),
            refill_per_sec: rpm / 60.0,
            last_refill: Mutex::new(Instant::now()),
        }
    }

    /// Bloqueia até ter 1 token disponível.
    pub async fn acquire(&self) {
        loop {
            {
                let mut tokens = self.tokens.lock().await;
                let mut last = self.last_refill.lock().await;
                let elapsed = last.elapsed().as_secs_f64();
                *tokens = (*tokens + elapsed * self.refill_per_sec).min(self.capacity);
                *last = Instant::now();

                if *tokens >= 1.0 {
                    *tokens -= 1.0;
                    return;
                }
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }
}

// ---------- Erros e classificação ----------

#[derive(Debug, Clone)]
pub enum ProviderError {
    RateLimited,       // 429 -> esperar e retry no mesmo provider
    CreditExhausted,   // 402 -> pular pro próximo provider da chain
    Transient(String), // timeout, 5xx -> retry com backoff
    Fatal(String),     // erro que não adianta tentar de novo
}

impl ProviderError {
    fn from_status(status: u16, body: &str) -> Self {
        match status {
            429 => ProviderError::RateLimited,
            402 => ProviderError::CreditExhausted,
            500..=599 => ProviderError::Transient(body.to_string()),
            _ => ProviderError::Fatal(format!("status {status}: {body}")),
        }
    }
}

// ---------- Trait comum pra qualquer backend (NIM, local, etc) ----------

#[async_trait::async_trait]
pub trait CompletionProvider: Send + Sync {
    fn name(&self) -> &str;
    async fn complete(&self, prompt: &str) -> Result<String, ProviderError>;
}

// ---------- Provider NIM com rate limiter embutido ----------

pub struct NimProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    bucket: TokenBucket,
}

impl NimProvider {
    pub fn new(api_key: String, model: String, rpm: f64) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
            bucket: TokenBucket::new(rpm),
        }
    }
}

#[async_trait::async_trait]
impl CompletionProvider for NimProvider {
    fn name(&self) -> &str {
        "nim"
    }

    async fn complete(&self, prompt: &str) -> Result<String, ProviderError> {
        self.bucket.acquire().await;

        let resp = self
            .client
            .post("https://integrate.api.nvidia.com/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&serde_json::json!({
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            }))
            .send()
            .await
            .map_err(|e| ProviderError::Transient(e.to_string()))?;

        let status = resp.status().as_u16();
        if status != 200 {
            let body = resp.text().await.unwrap_or_default();
            return Err(ProviderError::from_status(status, &body));
        }

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| ProviderError::Transient(e.to_string()))?;

        json["choices"][0]["message"]["content"]
            .as_str()
            .map(String::from)
            .ok_or_else(|| ProviderError::Fatal("resposta sem campo content".into()))
    }
}

// ---------- Provider local (Ollama no T1000/1650) ----------

pub struct LocalProvider {
    client: reqwest::Client,
    endpoint: String, // ex: http://localhost:11434
    model: String,    // ex: nemotron-3-nano
}

impl LocalProvider {
    pub fn new(endpoint: String, model: String) -> Self {
        Self { client: reqwest::Client::new(), endpoint, model }
    }
}

#[async_trait::async_trait]
impl CompletionProvider for LocalProvider {
    fn name(&self) -> &str {
        "local"
    }

    async fn complete(&self, prompt: &str) -> Result<String, ProviderError> {
        let resp = self
            .client
            .post(format!("{}/api/generate", self.endpoint))
            .json(&serde_json::json!({
                "model": self.model,
                "prompt": prompt,
                "stream": false,
            }))
            .send()
            .await
            .map_err(|e| ProviderError::Transient(e.to_string()))?;

        let status = resp.status().as_u16();
        if status != 200 {
            let body = resp.text().await.unwrap_or_default();
            return Err(ProviderError::from_status(status, &body));
        }

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| ProviderError::Transient(e.to_string()))?;

        json["response"]
            .as_str()
            .map(String::from)
            .ok_or_else(|| ProviderError::Fatal("resposta sem campo response".into()))
    }
}

// ---------- Fallback chain com retry + backoff exponencial + jitter ----------

pub struct FallbackChain {
    providers: Vec<Arc<dyn CompletionProvider>>,
    max_retries_per_provider: u32,
}

impl FallbackChain {
    pub fn new(providers: Vec<Arc<dyn CompletionProvider>>) -> Self {
        Self { providers, max_retries_per_provider: 3 }
    }

    pub async fn complete(&self, prompt: &str) -> Result<String, String> {
        for provider in &self.providers {
            let mut attempt = 0;

            loop {
                match provider.complete(prompt).await {
                    Ok(text) => return Ok(text),

                    Err(ProviderError::RateLimited) => {
                        attempt += 1;
                        if attempt > self.max_retries_per_provider {
                            break; // esgota tentativas nesse provider, cai pro próximo
                        }
                        backoff_sleep(attempt).await;
                    }

                    Err(ProviderError::Transient(_)) => {
                        attempt += 1;
                        if attempt > self.max_retries_per_provider {
                            break;
                        }
                        backoff_sleep(attempt).await;
                    }

                    Err(ProviderError::CreditExhausted) => {
                        // sem crédito: não adianta insistir, pula direto
                        break;
                    }

                    Err(ProviderError::Fatal(msg)) => {
                        eprintln!("[{}] erro fatal: {}", provider.name(), msg);
                        break;
                    }
                }
            }
        }

        Err("todos os providers da chain falharam".into())
    }
}

async fn backoff_sleep(attempt: u32) {
    let base_ms = 500u64 * 2u64.pow(attempt.min(5));
    let jitter_ms: u64 = rand::thread_rng().gen_range(0..250);
    tokio::time::sleep(Duration::from_millis(base_ms + jitter_ms)).await;
}

// ---------- Exemplo de montagem ----------

pub fn build_default_chain(nim_api_key: String) -> FallbackChain {
    let nim = Arc::new(NimProvider::new(nim_api_key, "meta/llama-3.1-8b-instruct".into(), 40.0));
    let local = Arc::new(LocalProvider::new("http://localhost:11434".into(), "nemotron-3-nano".into()));

    FallbackChain::new(vec![nim, local])
}