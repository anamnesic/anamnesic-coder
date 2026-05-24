use crate::agent::state::AgentState;
use crate::agent::planner;
use crate::agent::executor;
use crate::llm::client::LlmClient;

pub async fn run_agent_loop(client: &LlmClient, state: &mut AgentState, task: &str) {
    let caveman_tag = state.caveman.tag();
    if !caveman_tag.is_empty() {
        println!("[{}]", caveman_tag);
    }
    println!("[Planning] {}", task);
    state.session.add_message("user", task);

    let context = state.session.get_context();
    let plan = planner::plan_task(client, &state.config.planner_model, task, &context, &state.caveman).await
        .unwrap_or_else(|e| {
            eprintln!("Planner error: {}. Falling back to direct execution.", e);
            crate::types::plan::Plan {
                steps: vec![crate::types::plan::PlanStep {
                    step_type: "answer".into(),
                    description: task.into(),
                    filename: None, pattern: None, command: None,
                }],
            }
        });

    let steps = plan.steps;
    if steps.is_empty() {
        eprintln!("Planner returned no steps.");
        return;
    }

    let total = steps.len();
    for (i, step) in steps.iter().enumerate() {
        println!("\n[{}/{}] [{}]: {}", i + 1, total, step.step_type, step.description);
        executor::execute_step(client, state, step).await;
    }

    let summary: Vec<&str> = steps.iter().map(|s| s.description.as_str()).collect();
    let summary_str = summary.join("; ");
    state.session.add_action(&summary_str);
    state.long_memory.save_session(task, &summary_str).ok();

    if state.retries < state.config.max_retries && needs_fix(state) {
        state.retries += 1;
        let fix_task = format!("Fix issues in: {}", task);
        Box::pin(run_agent_loop(client, state, &fix_task)).await;
    }
}

fn needs_fix(state: &AgentState) -> bool {
    let out = state.last_test_output.to_lowercase();
    if out.is_empty() { return false; }
    if out.contains("passed") && !out.contains("failed") { return false; }
    out.contains("error") || out.contains("failed")
}
