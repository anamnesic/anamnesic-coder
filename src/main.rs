mod agent;
mod llm;
mod tools;
mod memory;
mod types;
mod config;

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
        print!("\n[you] ");
        std::io::stdout().flush()?;
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim();
        if input.is_empty() { continue; }
        match input {
            "/exit" | "/quit" => break,
            "/reset" => { state.reset(); println!("Session reset."); }
            "/help" => {
                println!("  /help    Help");
                println!("  /reset   Reset session");
                println!("  /exit    Exit");
                println!("  /check   Hardware check");
            }
            "/check" => hw_check().await?,
            _ => run_agent_loop(client, state, input).await,
        }
    }
    Ok(())
}

async fn hw_check() -> Result<()> {
    println!("Hardware: CPU={} OS={}", num_cpus(), std::env::consts::OS);
    println!("\nRecommended: planner=granite3.3:2b coder=qwen3:1.7b");
    println!("  Or use --local --model model.gguf for local inference");
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

fn num_cpus() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}
