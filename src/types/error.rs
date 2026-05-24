use std::fmt;

#[derive(Debug)]
pub enum AgentError {
    LlmError(String),
    ToolError(String),
    PlanError(String),
    ConfigError(String),
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentError::LlmError(msg) => write!(f, "LLM error: {}", msg),
            AgentError::ToolError(msg) => write!(f, "Tool error: {}", msg),
            AgentError::PlanError(msg) => write!(f, "Plan error: {}", msg),
            AgentError::ConfigError(msg) => write!(f, "Config error: {}", msg),
        }
    }
}

impl std::error::Error for AgentError {}
