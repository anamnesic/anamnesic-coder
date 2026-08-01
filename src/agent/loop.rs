use crate::agent::state::AgentState;
use crate::agent::planner;
use crate::agent::executor;
use crate::llm::client::LlmClient;
use crate::tools::test;

/// Compact the session history when it exceeds a rough token budget.
const CONTEXT_COMPACT_THRESHOLD: usize = 2500;

/// Cap on verification output fed back into the fix loop (protects context).
const FIX_FEEDBACK_CAP: usize = 2000;

async fn maybe_compact(client: &LlmClient, state: &mut AgentState) {
    if state.session.estimated_tokens() < CONTEXT_COMPACT_THRESHOLD {
        return;
    }
    let transcript = state.session.transcript();
    let prompt = format!(
        "Summarize this coding-session conversation into a concise summary (max ~150 tokens). Keep key decisions, files touched, and open issues.\n\n{}",
        transcript
    );
    if let Ok(summary) = client.generate_with_retry(&state.config.summarizer_model, &prompt).await {
        let summary = summary.trim();
        if !summary.is_empty() {
            state.session.compact(summary.to_string());
            eprintln!("  [context compacted — history summarized]");
        }
    }
}

pub async fn run_agent_loop(client: &LlmClient, state: &mut AgentState, task: &str) {
    let caveman_tag = state.caveman.tag();
    if !caveman_tag.is_empty() {
        println!("[{}]", caveman_tag);
    }
    println!("[Planning] {}", task);
    state.session.add_message("user", task);

    maybe_compact(client, state).await;

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

    // Stop-hook gate: if the plan never ran a test suite but the workspace is a
    // Cargo project, verify with `cargo test` before declaring the work done.
    if state.last_test_output.is_empty() && state.config.workspace_dir.join("Cargo.toml").exists() {
        eprintln!("  [verify] running cargo test...");
        state.last_test_output = test::run_tests("cargo test", &state.config);
    }

    if state.retries < state.config.max_retries && needs_fix(state) {
        state.retries += 1;
        let feedback = if state.last_test_output.is_empty() {
            String::new()
        } else {
            let cap: String = state.last_test_output.chars().take(FIX_FEEDBACK_CAP).collect();
            format!("\n\nVerification output from your last attempt (fix failures; do NOT weaken existing tests to make them pass):\n```\n{}\n```", cap)
        };
        let fix_task = format!("Fix issues in: {}{}", task, feedback);
        Box::pin(run_agent_loop(client, state, &fix_task)).await;
    }
}

fn needs_fix(state: &AgentState) -> bool {
    let out = &state.last_test_output;
    if out.is_empty() { return false; }
    if out.contains("test result: ok") { return false; }
    if out.contains("test result: FAILED") { return true; }
    let low = out.to_lowercase();
    if low.contains("passed") && !low.contains("failed") { return false; }
    low.contains("error") || low.contains("failed") || low.contains("failures:")
}
