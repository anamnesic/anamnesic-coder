use std::path::Path;
use std::time::Instant;
use anyhow::Result;
use serde::{Serialize, Deserialize};

use crate::llm::model_resolver;
use crate::llm::infer::{engine::InferenceEngine, gguf::GgufReader, model::Model, tokenizer::Tokenizer};
use crate::hw_recommend::{detector, recommender, catalog, scoring};
use crate::models_dev::{ModelsDevClient, CloudMatch};

const BENCH_PROMPT: &str = "Write a Python function that computes fibonacci numbers.";
const BENCH_TOKENS: usize = 20;   // enough to measure TPS without hanging for minutes
const BENCH_TEMP: f32 = 0.0;      // greedy for reproducibility
const BENCH_TIMEOUT_SECS: u64 = 120; // skip models that take longer than 2 min

/// Models to skip — not inference models.
const SKIP_MODELS: &[&str] = &["nomic-embed-text:latest", "qwen3-coder:480b-cloud"];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    pub model: String,
    pub load_ms: u64,
    pub gen_ms: u64,
    pub tokens_out: usize,
    pub tps: f32,
    /// Hardware recommender score (0–100), if model found in catalog.
    pub hw_score: Option<f64>,
    /// Predicted TPS from hardware model, if available.
    pub predicted_tps: Option<f64>,
    pub hw_rank: Option<usize>,
    pub output_sample: String,
    pub error: Option<String>,
    /// Nearest cloud equivalent from models.dev.
    pub cloud_match: Option<CloudMatch>,
}

/// Run benchmark for a single model. Returns a BenchResult.
pub fn benchmark_model(name: &str, models_dir: &Path) -> BenchResult {
    let blob_path = match model_resolver::resolve_model(name, models_dir) {
        Ok(p) => p,
        Err(e) => return BenchResult::error(name, format!("resolve: {e}")),
    };

    print!("  [{name}] loading...");
    use std::io::Write;
    std::io::stdout().flush().ok();

    let t0 = Instant::now();
    let model = match Model::load(&blob_path.to_string_lossy()) {
        Ok(m) => m,
        Err(e) => return BenchResult::error(name, format!("model load: {e}")),
    };
    let reader = match GgufReader::load(&blob_path.to_string_lossy()) {
        Ok(r) => r,
        Err(e) => return BenchResult::error(name, format!("gguf reader: {e}")),
    };
    let tokenizer = match Tokenizer::load_from_gguf(&reader) {
        Ok(t) => t,
        Err(e) => return BenchResult::error(name, format!("tokenizer: {e}")),
    };
    let load_ms = t0.elapsed().as_millis() as u64;

    print!(" generating...");
    std::io::stdout().flush().ok();

    let mut engine = InferenceEngine::new(model, tokenizer, 512);
    let t1 = Instant::now();

    // Run generation in a thread with a timeout so slow models don't block forever
    let (output, tokens_out) = {
        use std::sync::mpsc;
        let (tx, rx) = mpsc::channel();
        let prompt = BENCH_PROMPT.to_string();
        std::thread::spawn(move || {
            let res = engine.generate_bench(&prompt, BENCH_TOKENS, BENCH_TEMP, 40);
            let _ = tx.send(res);
        });
        match rx.recv_timeout(std::time::Duration::from_secs(BENCH_TIMEOUT_SECS)) {
            Ok(Ok(r))  => r,
            Ok(Err(e)) => return BenchResult::error(name, format!("generate: {e}")),
            Err(_)     => return BenchResult::error(name, format!("timeout after {BENCH_TIMEOUT_SECS}s")),
        }
    };
    let gen_ms = t1.elapsed().as_millis() as u64;

    let tps = if gen_ms > 0 { tokens_out as f32 / (gen_ms as f32 / 1000.0) } else { 0.0 };
    let sample: String = output.chars().take(80).collect();

    println!(" done ({tokens_out} tok, {tps:.1} tok/s)");

    BenchResult {
        model: name.to_string(),
        load_ms,
        gen_ms,
        tokens_out,
        tps,
        hw_score: None,
        predicted_tps: None,
        hw_rank: None,
        output_sample: sample,
        error: None,
        cloud_match: None,
    }
}

/// Main ranking function.
/// Benchmarks all locally available models, cross-references with hw_recommend,
/// and returns results sorted best → worst by actual TPS.
pub fn rank_models(models_dir: &Path, category: &str) -> Vec<BenchResult> {
    let hw = detector::detect_hardware();
    let hw_recs = recommender::recommend(&hw, category);

    // Load models.dev catalog (uses cache; gracefully empty if offline)
    let cloud = ModelsDevClient::load();

    let available = model_resolver::list_models(models_dir);
    let to_bench: Vec<String> = available
        .into_iter()
        .filter(|m| !SKIP_MODELS.contains(&m.as_str()))
        .collect();

    println!("\n══════════════════════════════════════════════");
    println!("  Hardware: {} | RAM: {}GB | GPU: {}",
        hw.cpu_brand.split_whitespace().last().unwrap_or("CPU"),
        hw.memory_total_gb,
        if hw.has_dedicated_gpu { format!("{} ({}GB VRAM)", hw.gpu_model, hw.gpu_vram_gb) }
        else { format!("{} (integrated)", hw.gpu_model) }
    );
    println!("  Benchmarking {} models — prompt: {} tokens out", to_bench.len(), BENCH_TOKENS);
    println!("══════════════════════════════════════════════\n");

    let mut results: Vec<BenchResult> = to_bench
        .iter()
        .map(|name| {
            let mut r = benchmark_model(name, models_dir);
            // Attach hw_recommend data where available
            if let Some((rank, rec)) = hw_recs.iter().enumerate()
                .find(|(_, rec)| names_match(&rec.model.name, name))
            {
                r.hw_score = Some(rec.score.total);
                r.predicted_tps = Some(estimate_tps_from_catalog(&hw, &rec.model, category));
                r.hw_rank = Some(rank + 1);
            }
            // Attach nearest cloud equivalent from models.dev
            r.cloud_match = cloud.match_local(name);
            r
        })
        .collect();

    // Sort: successful models by TPS desc, errors at the end
    results.sort_by(|a, b| {
        match (a.error.is_none(), b.error.is_none()) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => b.tps.partial_cmp(&a.tps).unwrap_or(std::cmp::Ordering::Equal),
        }
    });

    results
}

/// Persist ranking to JSON.
pub fn save_ranking(results: &[BenchResult], path: &Path) -> Result<()> {
    let json = serde_json::to_string_pretty(results)?;
    std::fs::write(path, json)?;
    println!("\n  Results saved → {}", path.display());
    Ok(())
}

// ── helpers ─────────────────────────────────────────────────────────────────

impl BenchResult {
    fn error(model: &str, msg: String) -> Self {
        println!("  [{model}] ERROR: {msg}");
        BenchResult {
            model: model.to_string(),
            load_ms: 0, gen_ms: 0, tokens_out: 0, tps: 0.0,
            hw_score: None, predicted_tps: None, hw_rank: None,
            output_sample: String::new(),
            error: Some(msg),
            cloud_match: None,
        }
    }
}

/// Fuzzy name match between catalog name (e.g. "llama3.2:3b") and local name (e.g. "llama3.2:3b").
/// Also handles "mistral:7b" ↔ "mistral:latest".
fn names_match(catalog: &str, local: &str) -> bool {
    if catalog == local { return true; }
    // strip tags and compare families
    let cfam = catalog.split(':').next().unwrap_or(catalog);
    let lfam = local.split(':').next().unwrap_or(local);
    if cfam != lfam { return false; }
    // same family — match if tags are compatible
    let ctag = catalog.split(':').nth(1).unwrap_or("latest");
    let ltag = local.split(':').nth(1).unwrap_or("latest");
    ctag == ltag || ltag == "latest" || ctag == "latest"
}

fn estimate_tps_from_catalog(
    hw: &detector::HardwareInfo,
    model: &catalog::ModelEntry,
    category: &str,
) -> f64 {
    let score = scoring::score_model(hw, model, category);
    // Back-compute estimated TPS from speed score × target
    const TARGET_SPEEDS: &[(&str, f64)] = &[
        ("general", 40.0), ("coding", 40.0), ("reasoning", 25.0), ("chat", 50.0),
    ];
    let target = TARGET_SPEEDS.iter()
        .find(|(c, _)| *c == category)
        .map(|(_, s)| *s)
        .unwrap_or(40.0);
    score.speed * target
}
