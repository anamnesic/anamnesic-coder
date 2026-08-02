use crate::config::settings::Config;
use crate::memory::short_term::ShortTermMemory;
use crate::memory::log::LongTermMemory;
use crate::tools::fs::FileTools;
use crate::tools::git::GitTools;
use crate::compressor::caveman::CavemanLevel;

pub struct AgentState {
    pub retries: usize,
    pub last_test_output: String,
    pub config: Config,
    pub session: ShortTermMemory,
    pub long_memory: LongTermMemory,
    pub files: FileTools,
    pub git: GitTools,
    pub caveman: CavemanLevel,
}

impl AgentState {
    pub fn new(config: Config) -> anyhow::Result<Self> {
        let long_memory = LongTermMemory::new(config.memory_dir.join("memory.db"))?;
        Ok(AgentState {
            retries: 0,
            last_test_output: String::new(),
            session: ShortTermMemory::new(20),
            long_memory,
            files: FileTools::new(config.workspace_dir.clone()),
            git: GitTools::new(config.workspace_dir.to_string_lossy().to_string()),
            config,
            caveman: CavemanLevel::Off,
        })
    }

    pub fn reset(&mut self) {
        self.session.clear();
        self.retries = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_config(tag: &str) -> Config {
        let base = std::env::temp_dir().join(format!("anamnesic-state-{tag}"));
        let _ = fs::remove_dir_all(&base);
        let mut cfg = Config::default();
        cfg.workspace_dir = base.join("workspace");
        cfg.memory_dir = base.join("memory");
        cfg
    }

    #[test]
    fn creates_state_with_empty_session() {
        let cfg = temp_config("new");
        let state = AgentState::new(cfg).unwrap();
        assert!(state.session.history().is_empty());
        assert_eq!(state.retries, 0);
        assert!(!state.caveman.is_active());
        let _ = fs::remove_dir_all(state.config.workspace_dir.parent().unwrap());
    }

    #[test]
    fn reset_clears_session_and_retries() {
        let cfg = temp_config("reset");
        let mut state = AgentState::new(cfg).unwrap();
        state.session.add_message("user", "hello");
        state.session.add_message("assistant", "hi");
        state.retries = 2;
        state.reset();
        assert!(state.session.history().is_empty());
        assert_eq!(state.retries, 0);
        let _ = fs::remove_dir_all(state.config.workspace_dir.parent().unwrap());
    }

    #[test]
    fn files_and_git_point_at_workspace() {
        let cfg = temp_config("tools");
        let state = AgentState::new(cfg).unwrap();
        state.files.write_file("a.txt", "content").unwrap();
        assert_eq!(state.files.read_file("a.txt").as_deref(), Some("content"));
        let _ = fs::remove_dir_all(state.config.workspace_dir.parent().unwrap());
    }
}
