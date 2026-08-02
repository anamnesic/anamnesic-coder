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
    pub models_dir: PathBuf,
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
            models_dir: std::env::var("MODELS_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("models")),
            max_seq_len: 2048,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_reasonable() {
        let cfg = Config::default();
        assert!(cfg.max_context_tokens > 0);
        assert!(cfg.max_retries > 0);
        assert!(cfg.chunk_overlap < cfg.chunk_size);
        assert!(!cfg.allowed_commands.is_empty());
        assert!(!cfg.blocked_commands.is_empty());
        assert!(cfg.allowed_commands.iter().any(|c| c == "cargo"));
        assert!(cfg.blocked_commands.iter().any(|c| c == "sudo"));
    }

    #[test]
    fn ollama_host_defaults_to_localhost() {
        let prev = std::env::var_os("OLLAMA_HOST");
        std::env::remove_var("OLLAMA_HOST");
        let cfg = Config::default();
        assert_eq!(cfg.ollama_host, "http://localhost:11434");
        match prev {
            Some(v) => std::env::set_var("OLLAMA_HOST", v),
            None => std::env::remove_var("OLLAMA_HOST"),
        }
    }

    #[test]
    fn respects_max_context_env_override() {
        let prev = std::env::var_os("MAX_CONTEXT_TOKENS");
        std::env::set_var("MAX_CONTEXT_TOKENS", "8192");
        let cfg = Config::default();
        assert_eq!(cfg.max_context_tokens, 8192);
        match prev {
            Some(v) => std::env::set_var("MAX_CONTEXT_TOKENS", v),
            None => std::env::remove_var("MAX_CONTEXT_TOKENS"),
        }
    }
}
