// Experimental subsystems (GPU inference kernels, bench, hardware detection) expose
// API that is not yet wired into the CLI — keep them compiling without dead-code noise.
#![allow(dead_code)]

mod agent;
mod llm;
mod tools;
mod memory;
mod types;
mod config;
mod hw_recommend;
mod compressor;
mod bench;
mod ui;
mod models_dev;
mod providers;

use clap::{Parser, Subcommand};
use config::settings::Config;
use agent::state::AgentState;
use agent::r#loop::run_agent_loop;
use llm::client::LlmClient;
use llm::model_resolver;
use llm::infer::engine::InferenceEngine;
use llm::infer::gguf::GgufReader;
use llm::infer::model::Model;
use llm::infer::tokenizer::Tokenizer;
use std::path::PathBuf;
use anyhow::Result;

#[derive(Parser)]
#[command(name = "slowcode", about = "Local coding agent — TinyCoder + llm-on-legacy-gpus fusion")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
    /// Use local GGUF inference instead of Ollama
    #[arg(long)]
    local: bool,
    /// Offload matrix multiplications to OpenCL GPU (requires --local and --features gpu)
    #[arg(long)]
    gpu: bool,
    /// Model name (e.g. gemma3:1b) or path to a .gguf file
    #[arg(long)]
    model: Option<String>,
    #[arg(short, long, default_value = "workspace")]
    dir: String,
    #[arg(long, default_value = "off")]
    caveman: String,
    /// Use a cloud provider (OpenAI-compatible, e.g. NVIDIA NIM) for inference
    #[arg(long)]
    cloud: bool,
    /// Cloud provider id (default: nvidia — NVIDIA NIM)
    #[arg(long, default_value = "nvidia")]
    provider: String,
    /// Cloud model id for inference (overrides planner/coder/summarizer defaults)
    #[arg(long)]
    cloud_model: Option<String>,
    /// Resume a previous session (lists saved sessions to pick from)
    #[arg(long)]
    resume: bool,
    task: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    Check,
    /// Launch the terminal UI
    Tui,
    Repl,
    /// List locally available models
    Models,
    /// List cloud models from models.dev catalog
    Cloud {
        /// Filter by name/family/provider (empty = show all)
        #[arg(default_value = "")]
        query: String,
    },
    /// Configure and manage cloud provider API keys (stored securely at ~/.config/rustcode/providers.toml)
    Providers {
        #[command(subcommand)]
        action: ProvidersAction,
    },
    /// Benchmark all local models and show ranking vs hw_recommend predictions
    Bench {
        /// Category to evaluate against (general, coding, reasoning, chat)
        #[arg(short, long, default_value = "coding")]
        category: String,
        /// Output JSON file for results
        #[arg(short, long, default_value = "bench_results.json")]
        output: String,
        /// Also benchmark cloud models (requires API keys)
        #[arg(long)]
        cloud: bool,
    },
}

/// Sub-actions for `rust-agent providers`
#[derive(Subcommand)]
enum ProvidersAction {
    /// List all providers from the models.dev catalog and their configuration status
    List,
    /// Show currently configured providers and their (masked) API keys
    Show,
    /// Set an API key for a provider  (e.g. rust-agent providers set openai sk-...)
    Set {
        /// Provider ID as in models.dev (e.g. openai, anthropic, groq, mistral)
        provider: String,
        /// API key — read from stdin if omitted (safer: avoids shell history)
        api_key: Option<String>,
        /// Override the API base URL (optional; uses provider default if not set)
        #[arg(long)]
        base: Option<String>,
    },
    /// Remove a provider's API key and config
    Remove {
        provider: String,
    },
    /// Enable or disable a provider without removing its key
    Enable {
        provider: String,
        #[arg(value_parser = clap::value_parser!(bool))]
        enabled: bool,
    },
    /// Test connectivity and API key validity for a provider
    Test {
        provider: String,
    },
    /// Import API keys from environment variables (reads models.dev env var names)
    Import,
}

/// Default model for cloud (NVIDIA NIM) inference.
const DEFAULT_CLOUD_MODEL: &str = "nvidia/llama-3.3-nemotron-super-49b-v1.5";

/// Build the LLM client: cloud provider (`--cloud`), local GGUF (`--local`) or Ollama.
async fn build_client(cli: &Cli, cfg: &mut Config) -> Result<LlmClient> {
    if cli.cloud {
        let catalog_client = tokio::task::spawn_blocking(models_dev::ModelsDevClient::load).await?;
        let (base, key) = providers::ProviderStore::resolve_cloud_credentials(&cli.provider, &catalog_client.catalog)?;
        let model = cli.cloud_model.clone()
            .or_else(|| std::env::var("CLOUD_MODEL").ok())
            .unwrap_or_else(|| DEFAULT_CLOUD_MODEL.to_string());
        cfg.planner_model = std::env::var("PLANNER_MODEL").ok().unwrap_or_else(|| model.clone());
        cfg.coder_model = std::env::var("CODER_MODEL").ok().unwrap_or_else(|| model.clone());
        cfg.summarizer_model = std::env::var("SUMMARIZER_MODEL").ok().unwrap_or_else(|| model.clone());
        println!("Cloud inference: provider='{}' base={} model={}", cli.provider, base, model);
        Ok(LlmClient::cloud(&base, &key))
    } else if cfg.use_local {
        let model_name = cli.model.as_deref().unwrap_or("gemma3:1b");
        let blob_path = model_resolver::resolve_model(model_name, &cfg.models_dir)?;
        println!("Loading {} from {}...", model_name, blob_path.display());
        let model = Model::load(&blob_path.to_string_lossy())?;
        let reader = GgufReader::load(&blob_path.to_string_lossy())?;
        let tokenizer = Tokenizer::load_from_gguf(&reader)?;
        let mut engine = InferenceEngine::new(model, tokenizer, cfg.max_seq_len);
        if cli.gpu {
            engine.init_gpu();
            if !engine.gpu_active() {
                println!("Warning: GPU not available, falling back to CPU.");
            } else {
                println!("GPU acceleration active.");
            }
        }
        Ok(LlmClient::local(engine))
    } else {
        Ok(LlmClient::ollama(&cfg.ollama_host))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    simple_logger::init_with_level(log::Level::Info).ok();
    llm::infer::ops::init_thread_pool();
    let cli = Cli::parse();
    let mut cfg = Config::default();
    cfg.workspace_dir = PathBuf::from(&cli.dir);
    cfg.use_local = cli.local;
    let client = build_client(&cli, &mut cfg).await?;
    let mut state = AgentState::new(cfg)?;
    state.caveman = compressor::caveman::CavemanLevel::from_str(&cli.caveman);
    if cli.resume {
        resume_session(&mut state)?;
    }

    match cli.command {
        Some(Commands::Check) => hw_check().await?,
        Some(Commands::Cloud { query }) => {
            let catalog = models_dev::ModelsDevClient::load();
            catalog.print_list(&query);
        },
        Some(Commands::Providers { action }) => {
            handle_providers(action).await?;
        },
        Some(Commands::Bench { category, output, cloud }) => {
            let results = bench::model_bench::rank_models(&state.config.models_dir, &category);
            bench::model_bench::save_ranking(&results, std::path::Path::new(&output))?;
            bench::display::show_ranking_table(&results)?;
            if cloud {
                let cloud_results = bench::model_bench::rank_cloud_models(&get_cloud_models()).await;
                bench::model_bench::save_ranking(&cloud_results, std::path::Path::new(&output))?;
                bench::display::show_ranking_table(&cloud_results)?;
            }
        },
        Some(Commands::Models) => {
            let models = model_resolver::list_models(&state.config.models_dir);
            if models.is_empty() {
                println!("No models found in {}", state.config.models_dir.display());
            } else {
                println!("Available models:");
                for m in models { println!("  {}", m); }
            }
        },
        Some(Commands::Tui) => {
            ui::run_ui(client, state).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        },
        Some(Commands::Repl) => {
            repl(&client, &mut state).await?;
        }
        None => {
            if let Some(task) = cli.task {
                run_agent_loop(&client, &mut state, &task).await;
            } else {
                ui::run_ui(client, state).map_err(|e| anyhow::anyhow!(e.to_string()))?;
            }
        }
    }
    Ok(())
}

async fn repl(client: &LlmClient, state: &mut AgentState) -> Result<()> {
    use std::io::Write;
    loop {
        let prompt = match state.caveman {
            compressor::caveman::CavemanLevel::Off => "\n[you] ",
            _ => "\n🪨 ",
        };
        print!("{}", prompt);
        std::io::stdout().flush()?;
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim();
        if input.is_empty() { continue; }

        if input.starts_with("/caveman") {
            let rest = input.trim_start_matches("/caveman").trim();
            if rest.is_empty() || rest == "full" {
                state.caveman = compressor::caveman::CavemanLevel::Full;
                println!("🪨 CAVEMAN MODE: full — why use many word when few do trick");
            } else if rest == "lite" {
                state.caveman = compressor::caveman::CavemanLevel::Lite;
                println!("🪨 CAVEMAN MODE: lite");
            } else if rest == "ultra" {
                state.caveman = compressor::caveman::CavemanLevel::Ultra;
                println!("🪨 CAVEMAN MODE: ultra — brain big, mouth small");
            } else if rest == "off" || rest == "stop" || rest == "disable" {
                state.caveman = compressor::caveman::CavemanLevel::Off;
                println!("🗣️ Normal mode restored");
            } else if rest == "stats" {
                let tag = state.caveman.tag();
                println!("═══ Caveman Stats ═══");
                println!("  Mode:      {}", if tag.is_empty() { "off" } else { tag });
                println!("  Sessions:  this session only");
                println!("  Tip:       run /caveman [lite|full|ultra|off]");
            } else {
                println!("Unknown caveman level: '{}'. Use: lite, full, ultra, off, stats", rest);
            }
            continue;
        }

        match input {
            "/exit" | "/quit" => break,
            "/reset" => { state.reset(); println!("Session reset."); }
            "/help" => {
                println!("  /help        Help");
                println!("  /reset       Reset session");
                println!("  /exit        Exit");
                println!("  /check       Hardware check");
                println!("  /caveman     Toggle caveman mode (off/lite/full/ultra)");
                println!("  /caveman stats  Show caveman stats");
            }
            "/check" => hw_check().await?,
            _ => run_agent_loop(client, state, input).await,
        }
    }
    Ok(())
}

/// List saved sessions and (optionally) restore one into the session context.
fn resume_session(state: &mut AgentState) -> Result<()> {
    use std::io::Write;

    let sessions = state.long_memory.get_recent_sessions(10)?;
    if sessions.is_empty() {
        println!("  No saved sessions found.");
        return Ok(());
    }
    println!("\n  Recent sessions (memory.db):");
    println!("{}", "─".repeat(70));
    for (i, (ts, summary)) in sessions.iter().enumerate() {
        let s: String = summary.chars().take(80).collect();
        println!("  [{}] {}  {}", i + 1, ts, s);
    }
    print!("  Resume session # (0 = skip): ");
    std::io::stdout().flush()?;
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let n: usize = input.trim().parse().unwrap_or(0);
    if n == 0 {
        return Ok(());
    }
    if let Some((ts, summary)) = sessions.get(n.saturating_sub(1)) {
        let task = format!("Continue previous session ({ts}): {summary}");
        state.session.add_message("user", &task);
        println!("  ✓ Resumed session from {ts}");
    }
    Ok(())
}

async fn hw_check() -> Result<()> {
    let hw = hw_recommend::detector::detect_hardware();
    hw_recommend::recommender::print_recommendations(&hw, "general");
    println!("═══ Category-specific ═══");
    for cat in &["coding", "reasoning", "chat"] {
        let recs = hw_recommend::recommender::recommend(&hw, cat);
        if let Some(top) = recs.first() {
            println!("  Best for {:>10}: {:<30} ({:.1})", cat, top.model.name, top.score.total);
        }
    }
    Ok(())
}

/// Returns the list of cloud models to benchmark, all via NVIDIA NIM.
fn get_cloud_models() -> Vec<(String, String, String, String, f64)> {
    let api_key = std::env::var("NVIDIA_API_KEY").unwrap_or_default();
    vec![
        ("nvidia/nvidia-nemotron-nano-9b-v2".into(), "nvidia".into(), api_key.clone(), "https://integrate.api.nvidia.com".into(), 40.0),
        ("deepseek-ai/deepseek-v4-flash".into(), "nvidia".into(), api_key.clone(), "https://integrate.api.nvidia.com".into(), 20.0),
        ("nvidia/llama-3.3-nemotron-super-49b-v1.5".into(), "nvidia".into(), api_key.clone(), "https://integrate.api.nvidia.com".into(), 20.0),
        ("minimaxai/minimax-m3".into(), "nvidia".into(), api_key.clone(), "https://integrate.api.nvidia.com".into(), 10.0),
        ("z-ai/glm-5.2".into(), "nvidia".into(), api_key.clone(), "https://integrate.api.nvidia.com".into(), 10.0),
        ("deepseek-ai/deepseek-v4-pro".into(), "nvidia".into(), api_key.clone(), "https://integrate.api.nvidia.com".into(), 5.0),
    ]
}

async fn handle_providers(action: ProvidersAction) -> Result<()> {
    use providers::{ProviderStore, print_store, test_provider};
    let catalog_client = tokio::task::spawn_blocking(models_dev::ModelsDevClient::load).await?;
    let catalog = &catalog_client.catalog;

    match action {
        ProvidersAction::List => {
            let mut ids: Vec<&str> = catalog.keys().map(|s| s.as_str()).collect();
            ids.sort();
            println!("\n  Available providers from models.dev catalog");
            println!("{}", "─".repeat(80));
            println!("  {:<20} {:<40} {:<15}", "Provider ID", "API base", "Env var");
            println!("{}", "─".repeat(80));
            for id in ids {
                let p = &catalog[id];
                let env = p.env.first().map(|s| s.as_str()).unwrap_or("—");
                println!("  {:<20} {:<40} {:<15}", id, p.api, env);
            }
            println!("\n  Use: rust-agent providers set <id> <api-key>");
        }
        ProvidersAction::Show => {
            let store = ProviderStore::load();
            print_store(&store, catalog);
        }
        ProvidersAction::Set { provider, api_key, base } => {
            let key = match api_key {
                Some(k) => k,
                None => {
                    // Read from stdin without echoing
                    eprint!("  API key for '{}' (input hidden): ", provider);
                    rpassword::read_password().unwrap_or_default()
                }
            };
            if key.is_empty() {
                anyhow::bail!("API key cannot be empty");
            }
            let mut store = ProviderStore::load();
            store.set_key(&provider, &key);
            if let Some(b) = base {
                store.set_base(&provider, &b);
            }
            store.save()?;
            println!("  ✓ API key saved for '{}' ({})", provider, ProviderStore::config_path_display());
        }
        ProvidersAction::Remove { provider } => {
            let mut store = ProviderStore::load();
            store.remove(&provider);
            store.save()?;
            println!("  ✓ Removed config for '{}'", provider);
        }
        ProvidersAction::Enable { provider, enabled } => {
            let mut store = ProviderStore::load();
            store.set_enabled(&provider, enabled);
            store.save()?;
            println!("  ✓ '{}' is now {}", provider, if enabled { "enabled" } else { "disabled" });
        }
        ProvidersAction::Test { provider } => {
            let mut store = ProviderStore::load();
            let entry = store.providers.entry(provider.clone()).or_default();
            // fill api_base from catalog if not set
            if entry.api_base.is_none() {
                if let Some(prov) = catalog.get(&provider) {
                    if !prov.api.is_empty() {
                        entry.api_base = Some(prov.api.clone());
                    }
                }
            }
            let entry = entry.clone();
            print!("  Testing '{}' ... ", provider);
            std::io::Write::flush(&mut std::io::stdout()).ok();
            match test_provider(&provider, &entry) {
                Ok(msg) => println!("✓ {msg}"),
                Err(e)  => println!("✗ {e}"),
            }
        }
        ProvidersAction::Import => {
            match ProviderStore::import_from_env(catalog) {
                Ok(imported) if imported.is_empty() => {
                    println!("  No new API keys found in environment.");
                    println!("  Set env vars like OPENAI_API_KEY, GROQ_API_KEY, etc. before running.");
                }
                Ok(imported) => {
                    println!("  ✓ Imported {} provider(s):", imported.len());
                    for s in &imported { println!("    • {s}"); }
                    println!("  Saved to: {}", ProviderStore::config_path_display());
                }
                Err(e) => eprintln!("  ✗ Import failed: {e}"),
            }
        }
    }
    Ok(())
}
