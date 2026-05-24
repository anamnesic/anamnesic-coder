mod agent;
mod llm;
mod tools;
mod memory;
mod types;
mod config;
mod hw_recommend;
mod compressor;

use clap::{Parser, Subcommand};
use config::settings::Config;
use agent::state::AgentState;
use agent::r#loop::run_agent_loop;
use llm::client::OllamaClient;
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
    #[arg(long)]
    local: bool,
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(short, long, default_value = "workspace")]
    dir: String,
    #[arg(long, default_value = "off")]
    caveman: String,
    task: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    Check,
    Repl,
}

#[tokio::main]
async fn main() -> Result<()> {
    simple_logger::init_with_level(log::Level::Info).ok();
    let cli = Cli::parse();
    let mut cfg = Config::default();
    cfg.workspace_dir = PathBuf::from(&cli.dir);
    cfg.use_local = cli.local;
    cfg.local_model_path = cli.model;
    let mut state = AgentState::new(cfg)?;
    state.caveman = compressor::caveman::CavemanLevel::from_str(&cli.caveman);

    match cli.command {
        Some(Commands::Check) => hw_check().await?,
        Some(Commands::Repl) | None => {
            if state.config.use_local {
                run_local(&state.config).await?;
            } else {
                let client = OllamaClient::new(&state.config.ollama_host);
                if let Some(task) = cli.task {
                    run_agent_loop(&client, &mut state, &task).await;
                } else {
                    repl(&client, &mut state).await?;
                }
            }
        }
    }
    Ok(())
}

async fn repl(client: &OllamaClient, state: &mut AgentState) -> Result<()> {
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

async fn run_local(cfg: &Config) -> Result<()> {
    let model_path = cfg.local_model_path.as_ref()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "model.gguf".into());

    println!("Loading model from {}...", model_path);
    let model = Model::load(&model_path)?;
    let reader = GgufReader::load(&model_path)?;
    let tokenizer = Tokenizer::load_from_gguf(&reader)?;
    let mut engine = InferenceEngine::new(model, tokenizer, cfg.max_seq_len);

    use std::io::Write;
    loop {
        print!("\n[prompt] ");
        std::io::stdout().flush()?;
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim();
        if input.is_empty() || input == "/exit" || input == "/quit" { break; }
        println!("Generating...");
        let result = engine.generate(input, 256, 0.8, 40)?;
        println!("\n{}", result);
    }
    Ok(())
}
