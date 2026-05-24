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
