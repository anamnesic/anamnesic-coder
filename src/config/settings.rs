use std::path::PathBuf;

pub struct Config {
    pub workspace_dir: PathBuf,
    pub memory_dir: PathBuf,
    pub ollama_host: String,
    pub planner_model: String,
    pub coder_model: String,
    pub summarizer_model: String,
    pub max_context_tokens: usize,
    pub max_retries: usize,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub allowed_commands: Vec<String>,
    pub blocked_commands: Vec<String>,
    pub use_local: bool,
    pub local_model_path: Option<PathBuf>,
    pub max_seq_len: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            workspace_dir: PathBuf::from("workspace"),
            memory_dir: PathBuf::from("memory_data"),
            ollama_host: std::env::var("OLLAMA_HOST").unwrap_or("http://localhost:11434".into()),
            planner_model: std::env::var("PLANNER_MODEL").unwrap_or("granite3.3:2b".into()),
            coder_model: std::env::var("CODER_MODEL").unwrap_or("qwen3:1.7b".into()),
            summarizer_model: std::env::var("SUMMARIZER_MODEL").unwrap_or("qwen3:0.6b".into()),
            max_context_tokens: std::env::var("MAX_CONTEXT_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(4096),
            max_retries: std::env::var("MAX_RETRIES").ok().and_then(|v| v.parse().ok()).unwrap_or(3),
            chunk_size: std::env::var("CHUNK_SIZE").ok().and_then(|v| v.parse().ok()).unwrap_or(300),
            chunk_overlap: std::env::var("CHUNK_OVERLAP").ok().and_then(|v| v.parse().ok()).unwrap_or(20),
            allowed_commands: vec![
                "pytest".into(), "python".into(), "npm test".into(), "node".into(),
                "pip".into(), "git".into(), "echo".into(), "cat".into(),
                "ls".into(), "dir".into(), "findstr".into(), "rg".into(),
                "fd".into(), "sg".into(), "tree".into(), "cargo".into(),
            ],
            blocked_commands: vec![
                "rm -rf".into(), "sudo".into(), "reboot".into(), "shutdown".into(),
                "format".into(), "del /f".into(), "rd /s".into(),
            ],
            use_local: false,
            local_model_path: None,
            max_seq_len: 2048,
        }
    }
}
