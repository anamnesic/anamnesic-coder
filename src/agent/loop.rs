use crate::agent::state::AgentState;
use crate::agent::planner;
use crate::agent::executor;
use crate::llm::router::LlmRouter;
use crate::llm::provider_chain::FallbackChain;
use crate::tools::test;
use crate::tools::shell;
use anyhow::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Progress events emitted by the agent loop, consumed by the TUI to show
/// live tool calls, plan steps and final results (mirrors exec-cell streaming
/// in 2026-era coding-agent TUIs).
#[derive(Debug, Clone)]
pub enum AgentEvent {
    Status(String),
    ToolCall { name: String, summary: String },
    PlanStep { index: usize, total: usize, description: String },
    Done { message: String },
    Interrupted,
}

/// Optional UI hooks for the agent loop: a progress sink and a cancellation
/// flag. When `on_event` is set, the loop reports through it instead of
/// printing to stdout (the TUI owns the screen); otherwise it prints as before.
#[derive(Default, Clone)]
pub struct AgentHooks {
    pub on_event: Option<Arc<dyn Fn(AgentEvent) + Send + Sync>>,
    pub interrupt: Option<Arc<AtomicBool>>,
}

impl AgentHooks {
    pub fn emit(&self, event: AgentEvent) {
        if let Some(f) = &self.on_event {
            f(event);
        }
    }

    pub fn note(&self, text: &str) {
        if self.on_event.is_some() {
            self.emit(AgentEvent::Status(text.to_string()));
        } else {
            println!("{text}");
        }
    }

    pub fn warn(&self, text: &str) {
        if self.on_event.is_some() {
            self.emit(AgentEvent::Status(text.to_string()));
        } else {
            eprintln!("{text}");
        }
    }

    pub fn interrupted(&self) -> bool {
        self.interrupt
            .as_ref()
            .map(|f| f.load(Ordering::Relaxed))
            .unwrap_or(false)
    }
}

/// Compact the session history when it exceeds a rough token budget.
const CONTEXT_COMPACT_THRESHOLD: usize = 2500;

/// Cap on verification output fed back into the fix loop (protects context).
const FIX_FEEDBACK_CAP: usize = 2000;

/// Maximum tool-use iterations per agent loop turn.
const MAX_TOOL_ITERATIONS: usize = 5;

async fn maybe_compact(client: &LlmRouter, state: &mut AgentState) {
    if state.session.estimated_tokens() < CONTEXT_COMPACT_THRESHOLD {
        return;
    }
    let transcript = state.session.transcript();
    let prompt = format!(
        "Summarize this coding-session conversation into a concise summary (max ~150 tokens). Keep key decisions, files touched, and open issues.\n\n{}",
        transcript
    );
    if let Ok(summary) = client.generate_with_retry(&state.config.summarizer_model, &prompt, None, None).await {
        let summary = summary.trim();
        if !summary.is_empty() {
            state.session.compact(summary.to_string());
            eprintln!("  [context compacted — history summarized]");
        }
    }
}

async fn maybe_compact_chain(chain: &FallbackChain, state: &mut AgentState) {
    if state.session.estimated_tokens() < CONTEXT_COMPACT_THRESHOLD {
        return;
    }
    let transcript = state.session.transcript();
    let prompt = format!(
        "Summarize this coding-session conversation into a concise summary (max ~150 tokens). Keep key decisions, files touched, and open issues.\n\n{}",
        transcript
    );
    if let Ok(summary) = chain.complete(&prompt).await {
        let summary = summary.trim();
        if !summary.is_empty() {
            state.session.compact(summary.to_string());
            eprintln!("  [context compacted — history summarized]");
        }
    }
}

/// Run a tool-use iteration: send prompt with tools, execute any tool calls,
/// feed results back, and repeat until the model produces final text.
async fn run_tool_use_iteration(
    client: &LlmRouter,
    state: &mut AgentState,
    model: &str,
    prompt: &str,
    tools: &[crate::llm::client::ToolDef],
    hooks: &AgentHooks,
) -> Result<(bool, String)> {
    let mut conversation: Vec<serde_json::Value> = vec![
        serde_json::json!({"role": "system", "content": "You are a coding agent. Use tools to inspect, edit, and verify the workspace. Call tools whenever an action is needed. Never claim a command ran unless you received its tool result. When finished, respond briefly with what changed and verification status."}),
        serde_json::json!({"role": "user", "content": prompt}),
    ];
    let mut used_tools = false;

    for iteration in 0..MAX_TOOL_ITERATIONS {
        if hooks.interrupted() {
            return Ok((used_tools, "[interrupted by user]".to_string()));
        }
        let response = client.chat(model, conversation.clone(), Some(&tools.to_vec()), None).await?;

        if let Ok(tool_calls) = serde_json::from_str::<Vec<crate::llm::client::ToolCall>>(&response) {
            if !tool_calls.is_empty() {
                used_tools = true;
                hooks.note(&format!("  [tools] iteration {}: {} call(s)", iteration + 1, tool_calls.len()));
                let mut tool_results = Vec::new();
                for tc in &tool_calls {
                    let result = execute_tool(state, tc);
                    hooks.note(&format!("    → {}: {}", tc.function.name, truncate_tool_output(&result)));
                    hooks.emit(AgentEvent::ToolCall {
                        name: tc.function.name.clone(),
                        summary: truncate_tool_output(&result),
                    });
                    tool_results.push(serde_json::json!({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }));
                }
                conversation.push(serde_json::json!({
                    "role": "assistant",
                    "content": null,
                    "tool_calls": tool_calls,
                }));
                for tr in tool_results {
                    conversation.push(tr);
                }
                continue;
            }
        }

        let final_text = if used_tools && response.trim().is_empty() {
            "Tool calls completed.".to_string()
        } else {
            response
        };
        return Ok((used_tools, final_text));
    }

    Ok((used_tools, "Maximum tool iterations reached before a final response.".to_string()))
}

fn execute_tool(state: &mut AgentState, tc: &crate::llm::client::ToolCall) -> String {
    let args: serde_json::Value = match serde_json::from_str(&tc.function.arguments) {
        Ok(args) => args,
        Err(e) => return format!("invalid JSON arguments: {e}"),
    };
    let string_arg = |name: &str| args.get(name).and_then(|v| v.as_str());
    match tc.function.name.as_str() {
        "read_file" => match string_arg("path") {
            Some(path) => state.files.read_file(path)
                .map(|content| truncate_tool_output(&content))
                .unwrap_or_else(|| "file not found or path is outside workspace".into()),
            None => "missing required argument: path".into(),
        },
        "write_file" => match (string_arg("path"), string_arg("content")) {
            (Some(path), Some(content)) => match state.files.write_file(path, content) {
                Ok(()) => {
                    state.session.add_file(path);
                    format!("wrote {} bytes to {path}", content.len())
                }
                Err(e) => format!("write failed: {e}"),
            },
            _ => "missing required arguments: path, content".into(),
        },
        "run_command" => match string_arg("command") {
            Some(command) => truncate_tool_output(&shell::run_command(command, &state.config)),
            None => "missing required argument: command".into(),
        },
        "search_code" => match string_arg("pattern") {
            Some(pattern) => truncate_tool_output(&search_workspace(state, pattern)),
            None => "missing required argument: pattern".into(),
        },
        _ => format!("Unknown tool: {}", tc.function.name),
    }
}

fn truncate_tool_output(value: &str) -> String {
    const LIMIT: usize = 8_000;
    if value.len() <= LIMIT { value.to_string() }
    else { format!("{}\n...[truncated]", &value[..LIMIT]) }
}

fn search_workspace(state: &AgentState, pattern: &str) -> String {
    match std::process::Command::new("rg")
        .args(["-n", "--max-count", "20", pattern])
        .current_dir(&state.config.workspace_dir)
        .output()
    {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).to_string(),
        Ok(_) => format!("No matches for: {pattern}"),
        Err(e) => format!("search failed: {e}"),
    }
}

fn coding_tools() -> Vec<crate::llm::client::ToolDef> {
    use crate::llm::client::{ToolDef, ToolFunction};
    let object = |properties, required| serde_json::json!({
        "type": "object", "properties": properties, "required": required, "additionalProperties": false
    });
    vec![
        ToolDef { r#type: "function".into(), function: ToolFunction { name: "read_file".into(), description: "Read a UTF-8 file relative to the workspace.".into(), parameters: object(serde_json::json!({"path":{"type":"string"}}), serde_json::json!(["path"])) } },
        ToolDef { r#type: "function".into(), function: ToolFunction { name: "write_file".into(), description: "Create or replace a UTF-8 file relative to the workspace.".into(), parameters: object(serde_json::json!({"path":{"type":"string"},"content":{"type":"string"}}), serde_json::json!(["path", "content"])) } },
        ToolDef { r#type: "function".into(), function: ToolFunction { name: "search_code".into(), description: "Search workspace files with a ripgrep pattern.".into(), parameters: object(serde_json::json!({"pattern":{"type":"string"}}), serde_json::json!(["pattern"])) } },
        ToolDef { r#type: "function".into(), function: ToolFunction { name: "run_command".into(), description: "Run an allowlisted command in the workspace and return stdout/stderr. Use it for checks and tests.".into(), parameters: object(serde_json::json!({"command":{"type":"string"}}), serde_json::json!(["command"])) } },
    ]
}

pub async fn run_agent_loop(client: &LlmRouter, state: &mut AgentState, task: &str) {
    run_agent_loop_with_hooks(client, state, task, &AgentHooks::default()).await;
}

/// Agent loop with optional progress stream and interrupt support for the TUI.
pub async fn run_agent_loop_with_hooks(
    client: &LlmRouter,
    state: &mut AgentState,
    task: &str,
    hooks: &AgentHooks,
) {
    let caveman_tag = state.caveman.tag();
    if !caveman_tag.is_empty() {
        hooks.note(&format!("[{}]", caveman_tag));
    }
    hooks.note(&format!("[Planning] {task}"));
    state.session.add_message("user", task);

    maybe_compact(client, state).await;

    if hooks.interrupted() {
        hooks.emit(AgentEvent::Interrupted);
        return;
    }

    // Tool calling is the primary coding-agent loop. The legacy planner is
    // retained below for models that choose not to emit tool calls.
    let tools = coding_tools();
    match run_tool_use_iteration(client, state, &state.config.coder_model.clone(), task, &tools, hooks).await {
        Ok((true, final_text)) => {
            if !final_text.trim().is_empty() {
                hooks.note(&format!("\n[agent] {}", final_text.trim()));
            }
            state.session.add_message("assistant", &final_text);
            state.session.add_action("tool-use turn completed");
            state.long_memory.save_session(task, &final_text).ok();
            if hooks.interrupted() {
                hooks.emit(AgentEvent::Interrupted);
            } else {
                hooks.emit(AgentEvent::Done { message: final_text.trim().to_string() });
            }
            return;
        }
        Ok((false, _)) => hooks.warn("  [tools] model did not request tools; using planner fallback"),
        Err(e) => hooks.warn(&format!("  [tools] unavailable ({e}); using planner fallback")),
    }

    if hooks.interrupted() {
        hooks.emit(AgentEvent::Interrupted);
        return;
    }

    let context = state.session.get_context();
    let fallback_plan = || crate::types::plan::Plan {
        steps: vec![crate::types::plan::PlanStep {
            step_type: "answer".into(),
            description: task.into(),
            filename: None, pattern: None, command: None,
        }],
    };
    let plan = planner::plan_task(client, &state.config.planner_model, task, &context, &state.caveman).await
        .unwrap_or_else(|e| {
            hooks.warn(&format!("Planner error: {}. Falling back to direct execution.", e));
            fallback_plan()
        });

    let steps = plan.steps;
    hooks.warn(&format!("  [plan] {} step(s)", steps.len()));
    if steps.is_empty() {
        hooks.warn("Planner returned no steps.");
        return;
    }

    // Fail fast with a clear message if the selected coder model's backend is
    // not available (e.g. a cloud model with no configured provider).
    if let Err(e) = client.client_for(&state.config.coder_model) {
        hooks.warn(&format!("Coder backend unavailable: {e}"));
        return;
    }

    let total = steps.len();
    for (i, step) in steps.iter().enumerate() {
        if hooks.interrupted() {
            hooks.emit(AgentEvent::Interrupted);
            return;
        }
        hooks.note(&format!("\n[{}/{}] [{}]: {}", i + 1, total, step.step_type, step.description));
        hooks.emit(AgentEvent::PlanStep {
            index: i + 1,
            total,
            description: step.description.clone(),
        });
        executor::execute_step(client, state, step).await;
    }

    let summary: Vec<&str> = steps.iter().map(|s| s.description.as_str()).collect();
    let summary_str = summary.join("; ");
    state.session.add_message("assistant", &format!("Completed: {summary_str}"));
    state.session.add_action(&summary_str);
    state.long_memory.save_session(task, &summary_str).ok();

    if state.last_test_output.is_empty() && state.config.workspace_dir.join("Cargo.toml").exists() {
        hooks.warn("  [verify] running cargo test...");
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
        Box::pin(run_agent_loop_with_hooks(client, state, &fix_task, hooks)).await;
    }

    hooks.emit(AgentEvent::Done {
        message: format!("Completed: {summary_str}"),
    });
}

/// Run the agent loop with a FallbackChain for cloud provider fallback.
pub async fn run_agent_loop_with_fallback(
    chain: &FallbackChain,
    state: &mut AgentState,
    task: &str,
) -> Result<()> {
    let caveman_tag = state.caveman.tag();
    if !caveman_tag.is_empty() {
        println!("[{}]", caveman_tag);
    }
    println!("[Planning] {}", task);
    state.session.add_message("user", task);

    maybe_compact_chain(chain, state).await;

    let context = state.session.get_context();
    let plan = planner::plan_task_with_chain(chain, &state.config.planner_model, task, &context, &state.caveman).await
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
        return Ok(());
    }

    let total = steps.len();
    for (i, step) in steps.iter().enumerate() {
        println!("\n[{}/{}] [{}]: {}", i + 1, total, step.step_type, step.description);
        executor::execute_step_with_chain(chain, state, step).await;
    }

    let summary: Vec<&str> = steps.iter().map(|s| s.description.as_str()).collect();
    let summary_str = summary.join("; ");
    state.session.add_action(&summary_str);
    state.long_memory.save_session(task, &summary_str).ok();

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
        Box::pin(run_agent_loop_with_fallback(chain, state, &fix_task)).await?;
    }

    Ok(())
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
