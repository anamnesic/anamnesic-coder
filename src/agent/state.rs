use crate::compressor::caveman::CavemanLevel;
use crate::config::settings::Config;
use crate::memory::log::LongTermMemory;
use crate::memory::short_term::ShortTermMemory;
use crate::tools::fs::FileTools;
use crate::tools::git::GitTools;
use crate::tools::test::{VerificationResult, VerificationStatus};
use crate::tools::transaction::{WorkspaceDiff, WorkspaceTransaction};
use std::collections::BTreeSet;

pub struct AgentState {
    pub retries: usize,
    pub repair_attempt: usize,
    pub last_test_output: String,
    pub verification: Option<VerificationResult>,
    pub changed_files: BTreeSet<String>,
    /// Actions refused by the approval policy during this turn. A turn with
    /// blocked actions must never be reported as a success.
    pub blocked_actions: Vec<String>,
    pub last_diff: WorkspaceDiff,
    pub transaction: Option<WorkspaceTransaction>,
    pub dirty: bool,
    pub config: Config,
    pub session: ShortTermMemory,
    pub long_memory: LongTermMemory,
    pub files: FileTools,
    pub git: GitTools,
    pub caveman: CavemanLevel,
    pub mcp_clients: Vec<crate::mcp::McpClient>,
}

impl AgentState {
    pub fn new(mut config: Config) -> anyhow::Result<Self> {
        config.workspace_dir =
            crate::tools::fs::normalize_workspace_path(&config.workspace_dir);
        let long_memory = LongTermMemory::new(config.memory_dir.join("memory.db"))?;
        Ok(AgentState {
            retries: 0,
            repair_attempt: 0,
            last_test_output: String::new(),
            verification: None,
            changed_files: BTreeSet::new(),
            blocked_actions: Vec::new(),
            last_diff: WorkspaceDiff::default(),
            transaction: None,
            dirty: false,
            session: ShortTermMemory::new(20),
            long_memory,
            files: FileTools::new(config.workspace_dir.clone()),
            git: GitTools::new(config.workspace_dir.to_string_lossy().to_string()),
            config,
            caveman: CavemanLevel::Off,
            mcp_clients: Vec::new(),
        })
    }

    pub fn start_turn(&mut self) -> anyhow::Result<()> {
        self.retries = 0;
        self.repair_attempt = 0;
        self.last_test_output.clear();
        self.verification = None;
        self.changed_files.clear();
        self.blocked_actions.clear();
        self.last_diff = WorkspaceDiff::default();
        self.dirty = false;
        self.transaction = Some(WorkspaceTransaction::begin(
            self.config.workspace_dir.clone(),
            self.config.transaction_max_bytes,
        )?);
        Ok(())
    }

    pub fn refresh_workspace_diff(&mut self) -> anyhow::Result<WorkspaceDiff> {
        let diff = self
            .transaction
            .as_ref()
            .map(WorkspaceTransaction::diff)
            .transpose()?
            .unwrap_or_default();
        self.changed_files = diff.paths().into_iter().collect();
        self.dirty = !diff.is_empty();
        self.last_diff = diff.clone();
        Ok(diff)
    }

    pub fn keep_changes(&mut self) -> anyhow::Result<WorkspaceDiff> {
        let diff = self.refresh_workspace_diff()?;
        self.transaction = None;
        Ok(diff)
    }

    pub fn rollback_changes(&mut self) -> anyhow::Result<WorkspaceDiff> {
        let diff = match self.transaction.take() {
            Some(transaction) => transaction.rollback()?,
            None => WorkspaceDiff::default(),
        };
        self.changed_files.clear();
        self.dirty = false;
        self.verification = None;
        self.last_test_output.clear();
        self.last_diff = diff.clone();
        Ok(diff)
    }

    pub fn mark_changed(&mut self, path: impl Into<String>) {
        self.changed_files.insert(path.into());
        self.dirty = true;
        self.verification = None;
        self.last_test_output.clear();
    }

    pub fn record_verification(&mut self, result: VerificationResult) {
        self.last_test_output = result.output.clone();
        self.verification = Some(result);
    }

    pub fn verification_failed(&self) -> bool {
        self.verification
            .as_ref()
            .map(|result| result.status == VerificationStatus::Failed)
            .unwrap_or(false)
    }

    pub fn reset(&mut self) {
        self.session.clear();
        self.retries = 0;
        self.repair_attempt = 0;
        self.last_test_output.clear();
        self.verification = None;
        self.changed_files.clear();
        self.blocked_actions.clear();
        self.last_diff = WorkspaceDiff::default();
        self.transaction = None;
        self.dirty = false;
    }

    pub fn record_blocked_action(&mut self, action: impl Into<String>) {
        self.blocked_actions.push(action.into());
    }
}

impl Clone for AgentState {
    fn clone(&self) -> Self {
        Self {
            retries: 0,
            repair_attempt: 0,
            last_test_output: String::new(),
            verification: None,
            changed_files: BTreeSet::new(),
            blocked_actions: Vec::new(),
            last_diff: WorkspaceDiff::default(),
            transaction: None,
            dirty: false,
            config: self.config.clone(),
            session: ShortTermMemory::new(self.config.max_context_tokens),
            long_memory: self.long_memory.clone(),
            files: self.files.clone(),
            git: self.git.clone(),
            caveman: self.caveman,
            mcp_clients: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_config(tag: &str) -> Config {
        let base = std::env::temp_dir().join(format!("anamnesic-state-{tag}"));
        let _ = fs::remove_dir_all(&base);
        Config {
            workspace_dir: base.join("workspace"),
            memory_dir: base.join("memory"),
            ..Config::default()
        }
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
        state.mark_changed("src/lib.rs");
        state.last_test_output = "stale failure".into();
        state.reset();
        assert!(state.session.history().is_empty());
        assert_eq!(state.retries, 0);
        assert!(state.last_test_output.is_empty());
        assert!(state.changed_files.is_empty());
        assert!(!state.dirty);
        assert!(state.verification.is_none());
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
