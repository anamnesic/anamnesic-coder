use crate::llm::client::LlmClient;
use crate::llm::provider_chain::FallbackChain;
use crate::llm::prompt::PlannerPrompt;
use crate::types::plan::{Plan, PlanStep};
use crate::compressor::caveman::CavemanLevel;
use anyhow::Result;

pub async fn plan_task(client: &LlmClient, model: &str, task: &str, context: &str, caveman: &CavemanLevel) -> Result<Plan> {
    let system = PlannerPrompt::with_caveman(caveman);
    let prompt = format!("{}\n\nContext:\n{}\n\nTask:\n{}", system, context, task);
    let response = client.generate_with_retry(model, &prompt, None).await?;
    parse_plan(response, task)
}

pub async fn plan_task_with_chain(chain: &FallbackChain, _model: &str, task: &str, context: &str, caveman: &CavemanLevel) -> Result<Plan> {
    let system = PlannerPrompt::with_caveman(caveman);
    let prompt = format!("{}\n\nContext:\n{}\n\nTask:\n{}", system, context, task);
    let response = chain.complete(&prompt).await.map_err(|e| anyhow::anyhow!("{e}"))?;
    parse_plan(response, task)
}

fn parse_plan(response: String, task: &str) -> Result<Plan> {
    if let Some(json_start) = response.find('{') {
        if let Some(json_end) = response.rfind('}') {
            let json_str = &response[json_start..=json_end];
            if let Ok(plan) = serde_json::from_str::<Plan>(json_str) {
                return Ok(plan);
            }
        }
    }

    Ok(Plan {
        steps: vec![PlanStep {
            step_type: "answer".into(),
            description: task.to_string(),
            filename: None,
            pattern: None,
            command: None,
        }],
    })
}
