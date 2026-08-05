// Experimental subsystems (GPU inference kernels, bench, hardware detection) expose
// API that is not yet wired into the CLI — keep them compiling without dead-code noise.
#![allow(dead_code)]

mod agent;
mod bench;
mod compressor;
mod config;
mod hw_recommend;
mod llm;
mod mcp;
mod memory;
mod models_dev;
mod providers;
mod repo;
mod terminal;
mod tools;
mod types;
mod ui;

use agent::agent_loop::run_agent_loop;
use agent::state::AgentState;
use anyhow::Result;
use clap::{Parser, Subcommand};
use config::settings::Config;
use llm::client::LlmClient;
use llm::infer::engine::InferenceEngine;
use llm::infer::gguf::GgufReader;
use llm::infer::model::Model;
use llm::infer::tokenizer::Tokenizer;
use llm::model_resolver;
use llm::router::{LlmRouter, DEFAULT_CLOUD_MODEL, DEFAULT_PROVIDER};
use std::fs::{File, OpenOptions};
use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Minimal logger that writes structured lines to `anamnesic.log` instead of
/// stdout. Used by the TUI so `log::warn!` retry storms from the LLM client
/// never interleave with the alternate screen and corrupt the UI.
struct FileLogger {
    file: Arc<Mutex<File>>,
}

impl log::Log for FileLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Info
    }

    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        if let Ok(mut f) = self.file.lock() {
            let _ = writeln!(
                f,
                "{} [{}] {} - {}",
                chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%.3f"),
                record.level(),
                record.target(),
                record.args()
            );
        }
    }

    fn flush(&self) {}
}

fn init_file_logger() -> anyhow::Result<()> {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("anamnesic.log")?;
    let logger = FileLogger {
        file: Arc::new(Mutex::new(file)),
    };
    log::set_boxed_logger(Box::new(logger)).map_err(|e| anyhow::anyhow!("{e}"))?;
    log::set_max_level(log::LevelFilter::Info);
    Ok(())
}

#[derive(Parser)]
#[command(
    name = "slowcode",
    about = "Local coding agent — TinyCoder + llm-on-legacy-gpus fusion"
)]
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
    #[arg(short, long, default_value = ".")]
    dir: String,
    #[arg(long, default_value = "off")]
    caveman: String,
    /// Use a cloud provider (OpenAI-compatible, e.g. NVIDIA NIM) for inference
    #[arg(long)]
    cloud: bool,
    /// Cloud provider id (default: nvidia — NVIDIA NIM)
    #[arg(long, default_value = DEFAULT_PROVIDER)]
    provider: String,
    /// Cloud model id for inference (overrides planner/coder/summarizer defaults)
    #[arg(long)]
    cloud_model: Option<String>,
    /// Resume a previous session (lists saved sessions to pick from)
    #[arg(long)]
    resume: bool,
    /// Continue the most recent session for this workspace without prompting
    #[arg(long, alias = "continue")]
    cont: bool,
    task: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    Check,
    /// Launch the terminal UI
    Tui,
    /// Expose the TUI in the browser via xterm.js over WebSocket
    Serve {
        /// Listen address (default 127.0.0.1)
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        /// Listen port
        #[arg(long, default_value = "7681")]
        port: u16,
    },
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
    Remove { provider: String },
    /// Enable or disable a provider without removing its key
    Enable {
        provider: String,
        #[arg(value_parser = clap::value_parser!(bool))]
        enabled: bool,
    },
    /// Test connectivity and API key validity for a provider
    Test { provider: String },
    /// Import API keys from environment variables (reads models.dev env var names)
    Import,
}

/// Default model for cloud inference.  Plain-name providers (Ollama Cloud)
/// use their own default; provider-qualified ids like NVIDIA's use `nvidia/…`.
const OLLAMA_CLOUD_DEFAULT_MODEL: &str = "nemotron-3-nano:30b";

/// Build the LLM router: a local backend (Ollama or local GGUF) plus, when
/// `--cloud` is given, a cloud backend for the selected provider.
async fn build_router(cli: &Cli, cfg: &mut Config) -> Result<LlmRouter> {
    let local = if cfg.use_local {
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
        LlmClient::local(engine)
    } else {
        LlmClient::ollama(&cfg.ollama_host)
    };
    let router = LlmRouter::new(local);

    if cli.cloud {
        let base = router.set_provider(&cli.provider)?;
        let model = cli
            .cloud_model
            .clone()
            .or_else(|| std::env::var("CLOUD_MODEL").ok())
            .unwrap_or_else(|| {
                if cli.provider == "ollama-cloud" {
                    OLLAMA_CLOUD_DEFAULT_MODEL.to_string()
                } else {
                    DEFAULT_CLOUD_MODEL.to_string()
                }
            });
        cfg.planner_model = std::env::var("PLANNER_MODEL")
            .ok()
            .unwrap_or_else(|| model.clone());
        cfg.coder_model = std::env::var("CODER_MODEL")
            .ok()
            .unwrap_or_else(|| model.clone());
        cfg.summarizer_model = std::env::var("SUMMARIZER_MODEL")
            .ok()
            .unwrap_or_else(|| model.clone());
        // Plain-name cloud models (e.g. Ollama Cloud) have no `/` prefix, so
        // mark them explicitly so the router sends them to the cloud backend.
        router.mark_cloud(&model);
        println!(
            "Cloud inference: provider='{}' base={} model={}",
            cli.provider, base, model
        );
    }
    Ok(router)
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    // The TUI owns the alternate screen, so route all log output to a file
    // instead of stdout. Otherwise retry warnings (HTTP 429/5xx backoff) from
    // the LLM client interleave with the rendered frames and corrupt the UI.
    let tui_mode = matches!(cli.command, Some(Commands::Tui))
        || (cli.command.is_none()
            && cli.task.is_none()
            && std::io::stdin().is_terminal()
            && std::io::stdout().is_terminal());
    if tui_mode {
        init_file_logger()?;
    } else {
        simple_logger::init_with_level(log::Level::Info).ok();
    }
    providers::load_dotenv();
    llm::infer::ops::init_thread_pool();
    let mut cfg = Config {
        workspace_dir: crate::tools::fs::normalize_workspace_path(&PathBuf::from(&cli.dir)),
        use_local: cli.local,
        ..Config::default()
    };
    let client = build_router(&cli, &mut cfg).await?;
    let mut state = AgentState::new(cfg)?;
    state.caveman = compressor::caveman::CavemanLevel::from_str(&cli.caveman);
    if cli.cont || cli.resume {
        continue_latest_session(&mut state)?;
    }

    match cli.command {
        Some(Commands::Check) => hw_check().await?,
        Some(Commands::Cloud { query }) => {
            let catalog = models_dev::ModelsDevClient::load();
            catalog.print_list(&query);
        }
        Some(Commands::Providers { action }) => {
            handle_providers(action).await?;
        }
        Some(Commands::Bench {
            category,
            output,
            cloud,
        }) => {
            let results = bench::model_bench::rank_models(&state.config.models_dir, &category);
            bench::model_bench::save_ranking(&results, std::path::Path::new(&output))?;
            bench::display::show_ranking_table(&results)?;
            if cloud {
                let cloud_results =
                    bench::model_bench::rank_cloud_models(&get_cloud_models()).await;
                bench::model_bench::save_ranking(&cloud_results, std::path::Path::new(&output))?;
                bench::display::show_ranking_table(&cloud_results)?;
            }
        }
        Some(Commands::Models) => {
            let models = model_resolver::list_models(&state.config.models_dir);
            if models.is_empty() {
                println!("No models found in {}", state.config.models_dir.display());
            } else {
                println!("Available models:");
                for m in models {
                    println!("  {}", m);
                }
            }
        }
        Some(Commands::Tui) => {
            // TUI defaults to the GLM-5.2 cloud model when no explicit --cloud
            // flag was given and the nvidia provider is available.
            if !cli.cloud {
                if let Ok(_base) = client.set_provider("nvidia") {
                    let model = DEFAULT_CLOUD_MODEL.to_string();
                    state.config.coder_model = model.clone();
                    state.config.planner_model = model.clone();
                    state.config.summarizer_model = model.clone();
                    client.mark_cloud(&model);
                }
            }
            ui::run_ui(client, state).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        }
        Some(Commands::Repl) => {
            repl(&client, &mut state).await?;
        }
        Some(Commands::Serve { port, host }) => {
            let argv = vec![
                std::env::current_exe()?.to_string_lossy().to_string(),
                "tui".to_string(),
            ];
            terminal::server::serve(&host, port, argv, state.config.workspace_dir.clone()).await?;
        }
        None => {
            if let Some(task) = cli.task {
                run_agent_loop(&client, &mut state, &task).await;
            } else if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
                // Interactive TUI defaults to GLM-5.2 cloud model.
                if !cli.cloud {
                    if let Ok(_base) = client.set_provider("nvidia") {
                        let model = DEFAULT_CLOUD_MODEL.to_string();
                        state.config.coder_model = model.clone();
                        state.config.planner_model = model.clone();
                        state.config.summarizer_model = model.clone();
                        client.mark_cloud(&model);
                    }
                }
                ui::run_ui(client, state).map_err(|e| anyhow::anyhow!(e.to_string()))?;
            } else {
                anyhow::bail!("no task or subcommand supplied; use --help for usage");
            }
        }
    }
    Ok(())
}

async fn repl(client: &LlmRouter, state: &mut AgentState) -> Result<()> {
    use std::io::Write;
    let interactive = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    loop {
        if interactive {
            let prompt = match state.caveman {
                compressor::caveman::CavemanLevel::Off => "\n[you] ",
                _ => "\n🪨 ",
            };
            print!("{}", prompt);
            std::io::stdout().flush()?;
        }
        let mut input = String::new();
        if std::io::stdin().read_line(&mut input)? == 0 {
            if interactive {
                println!();
            }
            break;
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }

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
                println!(
                    "Unknown caveman level: '{}'. Use: lite, full, ultra, off, stats",
                    rest
                );
            }
            continue;
        }

        match input {
            "/exit" | "/quit" => break,
            "/reset" => {
                state.reset();
                println!("Session reset.");
            }
            "/help" => {
                println!("  /help        Help");
                println!("  /reset       Reset session");
                println!("  /exit        Exit");
                println!("  /check       Hardware check");
                println!("  /models      List available models");
                println!("  /caveman     Toggle caveman mode (off/lite/full/ultra)");
                println!("  /caveman stats  Show caveman stats");
            }
            "/check" => hw_check().await?,
            "/models" => {
                let models = model_resolver::list_models(&state.config.models_dir);
                if models.is_empty() {
                    println!("  No models found in {}", state.config.models_dir.display());
                } else {
                    println!("  Available models:");
                    for m in models {
                        println!("    {}", m);
                    }
                }
            }
            _ => run_agent_loop(client, state, input).await,
        }
    }
    Ok(())
}

/// List saved sessions and (optionally) restore one into the session context.
/// Resume/continue the most recent session for this workspace without prompting.
fn resume_session(state: &mut AgentState) -> Result<()> {
    continue_latest_session(state)
}

/// Continue/resume the most recent session for this workspace without prompting.
fn continue_latest_session(state: &mut AgentState) -> Result<()> {
    let workspace = state.config.workspace_dir.display().to_string();
    match state.long_memory.latest_session(&workspace)? {
        Some(id) => {
            let count = state.load_session_into_state(id)?;
            println!("  ✓ Resumed session {id} ({count} messages restored)");
        }
        None => {
            println!("  No previous session found for this workspace.");
        }
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
            println!(
                "  Best for {:>10}: {:<30} ({:.1})",
                cat, top.model.name, top.score.total
            );
        }
    }
    Ok(())
}

/// Returns the list of cloud models to benchmark, all via NVIDIA NIM.
fn get_cloud_models() -> Vec<(String, String, String, String, f64)> {
    let api_key = std::env::var("NVIDIA_API_KEY").unwrap_or_default();
    vec![
        (
            "nvidia/nvidia-nemotron-nano-9b-v2".into(),
            "nvidia".into(),
            api_key.clone(),
            "https://integrate.api.nvidia.com".into(),
            40.0,
        ),
        (
            "deepseek-ai/deepseek-v4-flash".into(),
            "nvidia".into(),
            api_key.clone(),
            "https://integrate.api.nvidia.com".into(),
            20.0,
        ),
        (
            "nvidia/llama-3.3-nemotron-super-49b-v1.5".into(),
            "nvidia".into(),
            api_key.clone(),
            "https://integrate.api.nvidia.com".into(),
            20.0,
        ),
        (
            "minimaxai/minimax-m3".into(),
            "nvidia".into(),
            api_key.clone(),
            "https://integrate.api.nvidia.com".into(),
            10.0,
        ),
        (
            "z-ai/glm-5.2".into(),
            "nvidia".into(),
            api_key.clone(),
            "https://integrate.api.nvidia.com".into(),
            10.0,
        ),
        (
            "deepseek-ai/deepseek-v4-pro".into(),
            "nvidia".into(),
            api_key.clone(),
            "https://integrate.api.nvidia.com".into(),
            5.0,
        ),
    ]
}

async fn handle_providers(action: ProvidersAction) -> Result<()> {
    use providers::{print_store, test_provider, ProviderEntry, ProviderStore};
    let catalog_client = tokio::task::spawn_blocking(models_dev::ModelsDevClient::load).await?;
    let catalog = &catalog_client.catalog;

    match action {
        ProvidersAction::List => {
            let mut ids: Vec<&str> = catalog.keys().map(|s| s.as_str()).collect();
            ids.sort();
            println!("\n  Available providers from models.dev catalog");
            println!("{}", "─".repeat(80));
            println!(
                "  {:<20} {:<40} {:<15}",
                "Provider ID", "API base", "Env var"
            );
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
        ProvidersAction::Set {
            provider,
            api_key,
            base,
        } => {
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
            println!(
                "  ✓ API key saved for '{}' ({})",
                provider,
                ProviderStore::config_path_display()
            );
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
            println!(
                "  ✓ '{}' is now {}",
                provider,
                if enabled { "enabled" } else { "disabled" }
            );
        }
        ProvidersAction::Test { provider } => {
            print!("  Testing '{}' ... ", provider);
            std::io::Write::flush(&mut std::io::stdout()).ok();

            match ProviderStore::resolve_cloud_credentials(&provider, catalog) {
                Ok((base, api_key)) => {
                    let entry = ProviderEntry {
                        api_key: Some(api_key),
                        api_base: Some(base),
                        enabled: true,
                    };
                    match test_provider(&provider, &entry).await {
                        Ok(msg) => println!("✓ {msg}"),
                        Err(e) => println!("✗ {e}"),
                    }
                }
                Err(e) => println!("✗ {e}"),
            }
        }
        ProvidersAction::Import => match ProviderStore::import_from_env(catalog) {
            Ok(imported) if imported.is_empty() => {
                println!("  No new API keys found in environment.");
                println!("  Set env vars like OPENAI_API_KEY, GROQ_API_KEY, etc. before running.");
            }
            Ok(imported) => {
                println!("  ✓ Imported {} provider(s):", imported.len());
                for s in &imported {
                    println!("    • {s}");
                }
                println!("  Saved to: {}", ProviderStore::config_path_display());
            }
            Err(e) => eprintln!("  ✗ Import failed: {e}"),
        },
    }
    Ok(())
}
