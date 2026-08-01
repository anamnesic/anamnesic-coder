use crate::agent::state::AgentState;
use crate::agent::planner;
use crate::agent::executor;
use crate::llm::client::LlmClient;
use crate::tools::test;
use anyhow::Result;

/// Compact the session history when it exceeds a rough token budget.
const CONTEXT_COMPACT_THRESHOLD: usize = 2500;

/// Cap on verification output fed back into the fix loop (protects context).
const FIX_FEEDBACK_CAP: usize = 2000;

/// Maximum tool-use iterations per agent loop turn.
const MAX_TOOL_ITERATIONS: usize = 5;

async fn maybe_compact(client: &LlmClient, state: &mut AgentState) {
    if state.session.estimated_tokens() < CONTEXT_COMPACT_THRESHOLD {
        return;
    }
    let transcript = state.session.transcript();
    let prompt = format!(
        "Summarize this coding-session conversation into a concise summary (max ~150 tokens). Keep key decisions, files touched, and open issues.\n\n{}",
        transcript
    );
    if let Ok(summary) = client.generate_with_retry(&state.config.summarizer_model, &prompt, None).await {
        let summary = summary.trim();
        if !summary.is_empty() {
            state.session.compact(summary.to_string());
            eprintln!("  [context compacted — history summarized]");
        }
    }
}

/// Run a tool-use iteration: send prompt with tools, execute any tool calls,
/// feed results back, and repeat until the model produces final text.
/// Returns the final text response.
async fn run_tool_use_iteration(
    client: &LlmClient,
    model: &str,
    prompt: &str,
    tools: &Vec<crate::llm::client::ToolDef>,
) -> Result<String> {
    let mut conversation: Vec<serde_json::Value> = vec![
        serde_json::json!({"role": "user", "content": prompt}),
    ];

    for _iteration in 0..MAX_TOOL_ITERATIONS {
        let response = client.chat(model, conversation.clone(), Some(tools)).await?;

        // Try to parse the response as tool calls
        if let Ok(tool_calls) = serde_json::from_str::<Vec<crate::llm::client::ToolCall>>(&response) {
            if !tool_calls.is_empty() {
                let mut tool_results = Vec::new();
                for tc in &tool_calls {
                    let result = execute_tool(&tc);
                    tool_results.push(serde_json::json!({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }));
                }
                // Add assistant message with tool calls and tool results
                conversation.push(serde_json::json!({
                    "role": "assistant",
                    "tool_calls": tool_calls,
                }));
                for tr in tool_results {
                    conversation.push(tr);
                }
                continue;
            }
        }

        // No tool calls — this is the final response
        return Ok(response);
    }

    Ok("Max tool iterations reached.".to_string())
}

/// Execute a single tool call and return the result as a string.
fn execute_tool(tc: &crate::llm::client::ToolCall) -> String {
    match tc.function.name.as_str() {
        "read_file" => {
            // The arguments contain the filename; we return a placeholder
            // since the actual file tools are handled by the executor
            format!("Tool read_file called with: {}", tc.function.arguments)
        }
        "run_command" => {
            format!("Tool run_command called with: {}", tc.function.arguments)
        }
        "search_code" => {
            format!("Tool search_code called with: {}", tc.function.arguments)
        }
        _ => format!("Unknown tool: {}", tc.function.name),
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
