use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers, MouseButton, MouseEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Terminal,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::{
    error::Error,
    io,
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use crate::agent::agent_loop::{AgentEvent, AgentHooks, ApprovalDecision, ApprovalRequest};
use crate::agent::state::AgentState;
use crate::config::settings::ApprovalPolicy;
use crate::llm::router::{LlmRouter, DEFAULT_PROVIDER};

mod diff_render;
mod file_search;
mod line_truncation;
mod live_wrap;
mod pager_overlay;
mod width;

use unicode_segmentation::UnicodeSegmentation;

use file_search::FileMatch;
use width::display_width;

const SPINNER: [char; 8] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧'];

/// Available slash commands, shown in the interactive picker (modern-harness style).
const SLASH_COMMANDS: &[(&str, &str)] = &[
    ("/info", "Show workspace info (context, tokens, models, todo, files)"),
    ("/help", "Show help"),
    ("/status", "Show model, provider, directory, context tokens"),
    (
        "/model",
        "Select the active model (no arg = pick from best models for NVIDIA)",
    ),
    (
        "/provider",
        "Select cloud model provider (no arg = pick from list)",
    ),
    ("/reset", "Reset session"),
    ("/resume", "Resume a saved session (picker)"),
    ("/continue", "Continue the most recent session"),
];

#[derive(PartialEq)]
pub enum Focus {
    Sidebar,
    Messages,
    Input,
    Editor,
}

/// Agent execution mode: Agent uses tool-use iteration (interactive),
/// Plan generates a plan first then executes steps sequentially.
#[derive(PartialEq, Clone, Copy)]
pub enum AgentMode {
    Agent,
    Plan,
}

#[derive(Debug, Clone)]
pub struct TokenBreakdown {
    pub input: usize,
    pub output: usize,
    pub reasoning: usize,
    pub cache_read: usize,
    pub cache_write: usize,
    pub cache_rate: f64,
    pub generation_speed: f64,
}

#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub provider: String,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct TodoItem {
    pub text: String,
    pub done: bool,
}

#[derive(Debug, Clone)]
pub struct InfoPanelSection {
    pub title: &'static str,
    pub open: bool,
    pub lines: Vec<Span<'static>>,
}

impl InfoPanelSection {
    pub fn new(title: &'static str) -> Self {
        Self {
            title,
            open: false,
            lines: Vec::new(),
        }
    }

    pub fn closed(title: &'static str) -> Self {
        Self::new(title)
    }

    pub fn open(title: &'static str, lines: Vec<Span<'static>>) -> Self {
        Self { title, open: true, lines }
    }
}

pub struct App {
    pub messages: Vec<(String, String)>, // (role, content)
    pub input: String,
    pub cursor_position: usize,
    pub loading: bool,
    pub sidebar_items: Vec<String>,
    pub sidebar_selected: usize,
    pub focus: Focus,
    pub agent_mode: AgentMode,
    pub editor_file: Option<String>,
    pub editor_lines: Vec<String>,
    pub editor_row: usize,
    pub editor_col: usize,
    pub editor_dirty: bool,
    pub editor_scroll: usize,
    pub input_history: Vec<String>,
    pub history_index: Option<usize>,
    pub status: String,
    pub scroll_offset: usize,
    pub follow: bool,
    pub model: String,
    pub auto_model: bool,
    pub caveman: String,
    pub dir: String,
    pub git_branch: String,
    pub tokens: usize,
    pub elapsed: Duration,
    pub spinner_frame: usize,
    pub command_menu: bool,
    pub command_items: Vec<(String, String)>,
    pub command_selected: usize,
    pub model_selector: bool,
    pub model_items: Vec<String>,
    pub model_selected: usize,
    pub provider: String,
    pub provider_selector: bool,
    pub provider_items: Vec<String>,
    pub provider_selected: usize,
    /// Pending approval prompt (`ask` policy): the worker blocks until the
    /// user answers, but rendering and input keep running.
    pub pending_approval: Option<ApprovalRequest>,
    pub quit_pending: bool,
    pub token_breakdown: TokenBreakdown,
    pub max_context_tokens: usize,
    pub context_cost: f64,
    pub models: Vec<ModelEntry>,
    pub todo_items: Vec<TodoItem>,
    pub modified_files: Vec<String>,
    pub memory_enabled: bool,
    pub indexing_enabled: bool,
    pub info_sections: Vec<InfoPanelSection>,
    pub info_selected: usize,
    /// Resume session picker state.
    pub resume_selector: bool,
    pub resume_items: Vec<String>,
    pub resume_ids: Vec<i64>,
    pub resume_selected: usize,
    /// Tracks the in-flight streaming tool-call delta so chunks of the same
    /// call accumulate on one chat line instead of stacking new lines.
    pub last_delta_line: Option<(usize, String)>,
    /// True while the model is streaming assistant text; the last message is
    /// the live, still-growing response. Reset on Done/Failed/Interrupted.
    pub streaming_assistant: bool,
    /// Unified diff content for modified files (+green / -red).
    pub diff_content: Vec<String>,
    /// Whether the reasoning panel is expanded (showing full content)
    /// or collapsed (showing only a summary).
    pub reasoning_expanded: bool,
    /// Whether tool call details are expanded (showing full output)
    /// or collapsed (showing rollup summary). Toggled with Ctrl+O or Ctrl+E.
    pub tool_calls_expanded: bool,
    /// Fuzzy file-search overlay state (Ctrl+P): query, ranking and selection.
    pub file_search: bool,
    pub file_search_query: String,
    pub file_search_selected: usize,
    pub file_search_results: Vec<FileMatch>,
    pub file_search_paths: Vec<String>,
    /// /info overlay: full-screen view of the former left sidebar sections.
    pub info_popup: bool,
    /// Accumulated reasoning content for thinking models (GLM-5.2, deepseek-r1).
    pub reasoning: String,
    /// Active top tab: 0 = Session, 1 = Issues, 2 = Pull Requests, 3 = Gists
    pub active_tab: usize,
}

/// Byte offset of the `char_index`-th character of `s`. Clamped to `s.len()`
/// when the index is at/past the end, so it is safe to use with `insert`.
fn char_index_to_byte_index(s: &str, char_index: usize) -> usize {
    s.char_indices()
        .nth(char_index)
        .map(|(byte, _)| byte)
        .unwrap_or(s.len())
}

impl App {
    pub fn new(model: &str, caveman: &str) -> Self {
        Self {
            messages: Vec::new(),
            input: String::new(),
            cursor_position: 0,
            loading: false,
            sidebar_items: Vec::new(),
            sidebar_selected: 0,
            focus: Focus::Input,
            agent_mode: AgentMode::Agent,
            editor_file: None,
            editor_lines: Vec::new(),
            editor_row: 0,
            editor_col: 0,
            editor_dirty: false,
            editor_scroll: 0,
            input_history: Vec::new(),
            history_index: None,
            status: "Ready · F1: Session · F2: Issues · F3: PRs · F4: Gists · Enter to send · Esc interrupt"
                .into(),
            scroll_offset: 0,
            follow: true,
model: model.to_string(),
            auto_model: false,
            caveman: caveman.to_string(),
            dir: String::new(),
            git_branch: String::new(),
            tokens: 0,
            elapsed: Duration::ZERO,
            spinner_frame: 0,
            command_menu: false,
            command_items: Vec::new(),
            command_selected: 0,
            model_selector: false,
            model_items: Vec::new(),
            model_selected: 0,
            provider: DEFAULT_PROVIDER.to_string(),
            provider_selector: false,
            provider_items: Vec::new(),
            provider_selected: 0,
            pending_approval: None,
            quit_pending: false,
            token_breakdown: TokenBreakdown {
                input: 0,
                output: 0,
                reasoning: 0,
                cache_read: 0,
                cache_write: 0,
                cache_rate: 0.0,
                generation_speed: 0.0,
            },
            max_context_tokens: 0,
            context_cost: 0.0,
            models: Vec::new(),
            todo_items: Vec::new(),
            modified_files: Vec::new(),
            memory_enabled: false,
            indexing_enabled: false,
            info_sections: vec![
                InfoPanelSection::open("Context", vec![Span::styled("Loading…", Style::default())]),
                InfoPanelSection::closed("Token Usage"),
                InfoPanelSection::closed("Models"),
                InfoPanelSection::closed("Code Indexing"),
                InfoPanelSection::closed("Todo"),
                InfoPanelSection::closed("Modified Files"),
                InfoPanelSection::closed("Memory"),
            ],
            info_selected: 0,
            resume_selector: false,
            resume_items: Vec::new(),
            resume_ids: Vec::new(),
            resume_selected: 0,
            last_delta_line: None,
            streaming_assistant: false,
            diff_content: Vec::new(),
            reasoning_expanded: true,
            tool_calls_expanded: false,
            file_search: false,
            file_search_query: String::new(),
            file_search_selected: 0,
            file_search_results: Vec::new(),
            file_search_paths: Vec::new(),
            info_popup: false,
            reasoning: String::new(),
            active_tab: 0,
        }
    }

    pub fn add_message(&mut self, role: &str, content: &str) {
        self.messages.push((role.to_string(), content.to_string()));
    }

    pub fn feed_tool_delta(&mut self, index: usize, name: &str, args_delta: &str) {
        const MAX_DELTA_CHARS: usize = 240;
        let mut text;
        if let Some((last_index, ref last_text)) = self.last_delta_line {
            if last_index == index {
                if let Some((_, last)) = self.messages.last_mut() {
                    if last.starts_with("Δ ") {
                        text = last_text.clone();
                        if text.chars().count() < MAX_DELTA_CHARS {
                            text.push_str(args_delta);
                        }
                        *last = text.clone();
                        self.last_delta_line = Some((index, text));
                        return;
                    }
                }
            }
        }
        let tag = format!("Δ {name}[{index}]");
        text = format!("{tag} {args_delta}");
        self.messages.push(("Tool".to_string(), text.clone()));
        self.last_delta_line = Some((index, text));
    }

    /// Append streamed assistant text to a single live message. Creates the
    /// message on the first delta of a turn, then accumulates onto it until a
    /// terminal event (Done/Failed/Interrupted) calls `end_streaming`.
    pub fn feed_text_delta(&mut self, text: &str) {
        const MAX_STREAM_CHARS: usize = 20_000;
        if self.streaming_assistant {
            if let Some((role, last)) = self.messages.last_mut() {
                if role == "Assistant" && last.chars().count() < MAX_STREAM_CHARS {
                    last.push_str(text);
                    return;
                }
            }
            self.streaming_assistant = false;
        }
        self.messages.push(("Assistant".to_string(), text.to_string()));
        self.streaming_assistant = true;
    }

    /// Accumulate reasoning content deltas into a single live message.
    /// Unlike text deltas, reasoning content is displayed separately
    /// (italic, subdued) and does not count as the assistant's response.
    /// Flushes to the transcript periodically so thinking appears live.
    pub fn feed_reasoning_delta(&mut self, text: &str) {
        const MAX_REASONING_CHARS: usize = 8_000;
        const FLUSH_THRESHOLD: usize = 120;
        if self.reasoning.len() < MAX_REASONING_CHARS {
            self.reasoning.push_str(text);
        }
        if self.reasoning.len() >= FLUSH_THRESHOLD {
            self.messages.push(("Thinking".to_string(), self.reasoning.clone()));
            self.reasoning.clear();
        }
    }

    /// Reset reasoning accumulation between tool iterations.
    /// Call at the start of each new assistant response to avoid
    /// concatenating reasoning from multiple iterations without a separator.
    /// Any remaining tail (< 120 chars) is flushed to the transcript first.
    pub fn reset_reasoning(&mut self) {
        if !self.reasoning.is_empty() {
            self.messages.push(("Thinking".to_string(), self.reasoning.clone()));
            self.reasoning.clear();
        }
    }

    /// Stop the live assistant message. If `final_content` is provided, the
    /// last streaming Assistant partial is replaced with the final text so the
    /// transcript shows exactly one copy of the response; any trailing
    /// Thinking flushes are kept before it (Thinking → Assistant). Open
    /// reasoning tails are flushed first. Returns true if a streaming message
    /// was finalized (callers should not push the final message again).
    pub fn end_streaming(&mut self, final_content: Option<&str>) -> bool {
        // Close any open reasoning accumulation before finalizing content.
        // Insert remaining reasoning before the Assistant message so the
        // transcript reads Thinking → Assistant, not Assistant → Thinking.
        if !self.reasoning.is_empty() {
            let reasoning = std::mem::take(&mut self.reasoning);
            match self.messages.last_mut() {
                Some((role, _)) if role == "Assistant" => {
                    let idx = self.messages.len().saturating_sub(1);
                    self.messages.insert(idx, ("Thinking".to_string(), reasoning));
                }
                Some((_, last)) => {
                    last.push_str(&reasoning);
                }
                None => {
                    self.messages.push(("Thinking".to_string(), reasoning));
                }
            }
        }
        if !self.streaming_assistant {
            return false;
        }
        if let Some(content) = final_content {
            // Replace the last streaming Assistant partial with the final
            // content and keep any trailing Thinking flushes (one or several,
            // e.g. interleaved reasoning) BEFORE it, so the transcript always
            // reads ... Thinking → Assistant(final).
            if let Some(idx) = self.messages.iter().rposition(|(r, _)| r == "Assistant") {
                let mut tail = self.messages.split_off(idx);
                let mut assistant = tail.remove(0);
                assistant.1 = content.to_string();
                self.messages.extend(tail);
                self.messages.push(assistant);
            } else {
                self.messages.push(("Assistant".to_string(), content.to_string()));
            }
        }
        self.streaming_assistant = false;
        true
    }

    /// Open the Ctrl+P overlay seeded with the workspace file list.
    pub fn open_file_search(&mut self, paths: Vec<String>) {
        self.file_search = true;
        self.file_search_query.clear();
        self.file_search_selected = 0;
        self.file_search_paths = paths;
        self.refresh_file_search();
    }

    /// Re-rank the file-search results for the current query.
    pub fn refresh_file_search(&mut self) {
        let limit = 100;
        self.file_search_results =
            file_search::search_files(&self.file_search_paths, &self.file_search_query, limit);
        self.file_search_selected = self
            .file_search_selected
            .min(self.file_search_results.len().saturating_sub(1));
    }

    pub fn move_file_search_selection(&mut self, up: bool) {
        let len = self.file_search_results.len();
        if len == 0 {
            return;
        }
        if up {
            self.file_search_selected = self.file_search_selected.saturating_sub(1);
        } else if self.file_search_selected + 1 < len {
            self.file_search_selected += 1;
        }
    }

    /// Load `path` (workspace-relative) into the built-in editor.
    pub fn open_editor(&mut self, path: &str, content: Option<&str>) {
        let lines: Vec<String> = content
            .map(|c| c.lines().map(str::to_string).collect())
            .unwrap_or_default();
        self.editor_file = Some(path.to_string());
        self.editor_lines = lines;
        self.editor_row = 0;
        self.editor_col = 0;
        self.editor_dirty = false;
        self.editor_scroll = 0;
        self.focus = Focus::Editor;
        self.status = format!("Editing {path} — Ctrl+S save · Esc close");
    }

    pub fn clear_input(&mut self) {
        self.input.clear();
        self.cursor_position = 0;
    }
    fn previous_input(&mut self) {
        if self.input_history.is_empty() {
            return;
        }
        let index = self
            .history_index
            .unwrap_or(self.input_history.len())
            .saturating_sub(1);
        self.input = self.input_history[index].clone();
        self.cursor_position = self.input.chars().count();
        self.history_index = Some(index);
    }

    fn next_input(&mut self) {
        let Some(index) = self.history_index else {
            return;
        };
        if index + 1 < self.input_history.len() {
            self.input = self.input_history[index + 1].clone();
            self.cursor_position = self.input.chars().count();
            self.history_index = Some(index + 1);
        } else {
            self.clear_input();
            self.history_index = None;
        }
    }
}

/// Handle TUI slash commands (e.g. /model). Returns true if the command was
/// handled locally and should not be sent to the agent.
fn handle_slash_command(
    input: &str,
    app: &mut App,
    state: &Arc<Mutex<AgentState>>,
    router: &LlmRouter,
) -> bool {
    let cmd = input.split_whitespace().next().unwrap_or("");
    match cmd {
        "/reset" => {
            state.lock().unwrap().reset();
            app.messages.clear();
            app.add_message("System", "Session reset.");
            app.status = "Ready · Enter to send · ↑/↓ history · PgUp/PgDn scroll · mouse wheel · Esc interrupt".into();
            true
        }
        "/status" => {
            let st = state.lock().unwrap();
            let out = format!(
                "model: {}\ncaveman: {}\nprovider: {}\ndir: {}\ncontext tokens: {}",
                app.model,
                app.caveman,
                app.provider,
                st.config.workspace_dir.display(),
                st.session.estimated_tokens()
            );
            app.add_message("System", &out);
            true
        }
        "/info" => {
            let st = state.lock().unwrap();
            update_info_sections(app, &st);
            drop(st);
            app.info_popup = true;
            app.status = "Workspace info — ↑/↓ navigate · Enter toggle · Esc close".into();
            true
        }
        "/provider" => {
            let arg = input.trim_start_matches("/provider").trim().to_lowercase();
            let catalog = crate::models_dev::ModelsDevClient::load();
            let mut provs: Vec<(String, String, usize)> = catalog
                .catalog
                .iter()
                .map(|(pid, p)| {
                    let n = p
                        .models
                        .values()
                        .filter(|m| m.tool_call && m.modalities.output.iter().any(|o| o == "text"))
                        .count();
                    (pid.clone(), p.name.clone(), n)
                })
                .filter(|(_, _, n)| *n > 0)
                .collect();
            provs.sort_by_key(|item| item.1.to_lowercase());
            if !arg.is_empty() {
                provs.retain(|(pid, name, _)| {
                    pid.to_lowercase().contains(&arg) || name.to_lowercase().contains(&arg)
                });
            }
            if provs.is_empty() {
                let msg = if arg.is_empty() {
                    "No cloud providers with tool-capable models found in the models.dev catalog (offline?).".to_string()
                } else {
                    format!("No provider matches \"{arg}\".")
                };
                app.add_message("System", &msg);
            } else if provs.len() == 1 && !arg.is_empty() {
                set_active_provider(app, state, router, &provs[0].0);
            } else {
                app.provider_items = provs
                    .iter()
                    .map(|(pid, name, n)| format!("{} — {} ({n} models)", pid, name))
                    .collect();
                app.provider_selected = app
                    .provider_items
                    .iter()
                    .position(|s| s.starts_with(&format!("{} —", app.provider)))
                    .unwrap_or(0);
                app.provider_selector = true;
            }
            true
        }
        "/model" => {
            let arg = input.trim_start_matches("/model").trim();
            if arg.is_empty() {
                let st = state.lock().unwrap();
                let local = crate::llm::model_resolver::list_models(&st.config.models_dir);
                let dir = st.config.models_dir.clone();
                drop(st);
                // Cloud models come from the models.dev catalog for the active provider.
let provider = app.provider.clone();
                let catalog = crate::models_dev::ModelsDevClient::load();
                let mut cloud_ranked: Vec<(usize, String)> = catalog
                    .provider_models(&provider)
                    .into_iter()
                    .filter(|m| m.tool_call && m.modalities.output.iter().any(|o| o == "text"))
                    .filter(|m| {
                        if provider == "nvidia" {
                            let base = crate::models_dev::base_id(&m.id);
                            matches!(
                                base.as_str(),
                                "glm-5.2"
                                    | "qwen3.5-397b-a17b"
                                    | "deepseek-v4-pro"
                                    | "kimi-k2.6"
                                    | "minimax-m3"
                                    | "nemotron-3-ultra-550b-a55b"
                            )
                        } else {
                            match &m.release_date {
                                Some(d) if d.starts_with("2026") => m.open_weights,
                                _ => false,
                            }
                        }
                    })
.map(|m| {
                         let base = crate::models_dev::base_id(&m.id);
                         let display = if provider == "nvidia" {
                             ranked_model_name(&base)
                         } else {
                             base.clone()
                         };
                         (ranked_model_order(&base), display)
                     })
                    .collect();
cloud_ranked.sort_by_key(|(rank, _)| *rank);
                let cloud: Vec<String> = cloud_ranked
                    .into_iter()
                    .map(|(_, name)| format!("{} [cloud]", name))
                    .collect();
                // Add "auto" option for NVIDIA provider.
                let mut items: Vec<String> = local.clone();
                if provider == "nvidia" {
                    items.push("auto".into());
                }
                items.extend(unique_model_ids(cloud));
                items.dedup();
                if items.is_empty() {
                    app.add_message(
                        "System",
                        &format!("No models found in {}. models.dev catalog also empty (offline?). Use /model <name> to set one anyway.", dir.display()),
                    );
                    app.status =
                        "Ready · Enter to send · ↑/↓ history · PgUp/PgDn scroll · mouse wheel · Esc interrupt".into();
                } else {
                    app.model_items = items;
                    app.model_selected = app
                        .model_items
                        .iter()
                        .position(|m| {
                            let trimmed = m.trim_end_matches(" [cloud]");
                            trimmed.eq_ignore_ascii_case(&app.model)
                        })
                        .unwrap_or(0);
                    app.model_selector = true;
                }
            } else {
                set_active_model(app, state, router, arg);
            }
            true
        }
        "/resume" | "/continue" => {
            let mut st = state.lock().unwrap();
            let workspace = st.config.workspace_dir.display().to_string();
            match st.long_memory.latest_session(&workspace) {
                Ok(Some(id)) => {
                    match st.load_session_into_state(id) {
                        Ok(count) => {
                            drop(st);
                            app.messages.clear();
                            let s = state.lock().unwrap();
                            for (role, content) in s.session.history() {
                                app.add_message(&display_role(&role), &content);
                            }
                            app.add_message("System", &format!("✓ Resumed session {id} ({count} messages restored)"));
                        }
                        Err(e) => app.add_message("Error", &format!("Failed to load session: {e}")),
                    }
                }
                Ok(None) => app.add_message("System", "No previous session found for this workspace."),
                Err(e) => app.add_message("Error", &format!("Failed to query sessions: {e}")),
            }
            true
        }
        "/help" => {
            let cmds: Vec<&str> = SLASH_COMMANDS.iter().map(|(c, _)| *c).collect();
            app.add_message(
                "System",
                &format!(
                    "Commands: {}\nKeys: PgUp/PgDn scroll · ↑/↓ 3-line scroll · mouse wheel · Ctrl+L clear · Esc interrupt/quit",
                    cmds.join(" · ")
                ),
            );
            true
        }
        _ => false,
    }
}

/// Dedupe a model id list while preserving order (used by the model pickers).
fn unique_model_ids(items: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    items
        .into_iter()
        .filter(|m| seen.insert(m.clone()))
        .collect()
}

/// Return the ranked display name for a model base ID, or the ID itself
/// if not in the top list.  Used for the NVIDIA provider's best-model picker.
fn ranked_model_name(base_id: &str) -> String {
    match base_id {
        "glm-5.2" => "GLM-5.2".into(),
        "qwen3.5-397b-a17b" => "Qwen3.5-397B-A17B".into(),
        "deepseek-v4-pro" => "DeepSeek-V4-Pro".into(),
        "kimi-k2.6" => "Kimi-K2.6".into(),
        "minimax-m3" => "MiniMax-M3".into(),
        "nemotron-3-ultra-550b-a55b" => "Nemotron-3-Ultra-550B-A55B".into(),
        _ => base_id.to_string(),
    }
}

/// Return the rank position of a model base ID in the NVIDIA best-model list.
/// Models not in the list sort last.
fn ranked_model_order(base_id: &str) -> usize {
    match base_id {
        "glm-5.2" => 0,
        "qwen3.5-397b-a17b" => 1,
        "deepseek-v4-pro" => 2,
        "kimi-k2.6" => 3,
        "minimax-m3" => 4,
        "nemotron-3-ultra-550b-a55b" => 5,
        _ => usize::MAX,
    }
}

/// Set the active coder model for subsequent agent turns.  Strips the
/// " [cloud]" picker suffix and tells the router whether the model is a cloud
/// model so requests go to the right backend.
fn set_active_model(app: &mut App, state: &Arc<Mutex<AgentState>>, router: &LlmRouter, name: &str) {
    let clean = name.trim_end_matches(" [cloud]").to_string();
    if clean == "auto" {
        app.auto_model = true;
        let best = "glm-5.2";
        {
            let mut st = state.lock().unwrap();
            st.config.coder_model = best.to_string();
            st.config.planner_model = best.to_string();
            st.config.summarizer_model = best.to_string();
        }
        app.model = best.to_string();
        router.set_model(best);
        router.mark_cloud(best);
        app.add_message("System", "Auto model enabled — GLM-5.2 (will fallback on failure)");
        app.status = format!("Ready · model: {best} (auto) · Esc interrupt");
        return;
    }
    app.auto_model = false;
    let mut is_cloud = name.ends_with(" [cloud]");
    if !is_cloud {
        // Typed names: resolve against the active provider's catalog so plain
        // cloud ids (e.g. Ollama Cloud "glm-5.2") still route to the cloud.
        let provider = app.provider.clone();
        let catalog = crate::models_dev::ModelsDevClient::load();
        is_cloud = catalog.provider_model_api_id(&provider, &clean).is_some();
    }
    if is_cloud {
        router.mark_cloud(&clean);
    } else {
        router.unmark_cloud(&clean);
    }
    {
        let mut st = state.lock().unwrap();
        // The selected model drives the whole agent lifecycle (mirrors
        // `--model`, which sets planner/coder/summarizer).  Keeping the
        // planner and summarizer on separate local default models while the
        // coder runs on a cloud model would silently send planning and
        // compaction requests to Ollama.
        st.config.coder_model = clean.clone();
        st.config.planner_model = clean.clone();
        st.config.summarizer_model = clean.clone();
    }
    app.model = clean.clone();
    router.set_model(&clean);
    app.add_message(
        "System",
        &format!(
            "Model set to {clean}{}",
            if is_cloud { " (cloud)" } else { "" }
        ),
    );
    app.status = format!("Ready · model: {clean} · Esc interrupt");
}

/// Set the active cloud provider: rebuilds the router's cloud backend using
/// the models.dev catalog base URL + the configured/env API key.
fn set_active_provider(
    app: &mut App,
    _state: &Arc<Mutex<AgentState>>,
    router: &LlmRouter,
    name: &str,
) {
    match router.set_provider(name) {
        Ok(base) => {
            router.clear_cloud_marks();
            app.provider = name.to_string();
            app.add_message("System", &format!("Cloud provider set to {name} ({base})"));
            app.status =
                format!("Ready · provider: {name} · run /model to list its models");
        }
        Err(e) => {
            app.add_message("Error", &format!("Provider {name} not configured: {e}"));
            app.status = format!("Provider {name} failed · see error above");
        }
    }
}

/// Current git branch of the workspace, or "no git" when not a repository.
fn git_branch(dir: &std::path::Path) -> String {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .current_dir(dir)
        .output();
    match out {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout);
            let s = s.trim();
            if s.is_empty() {
                "detached".into()
            } else {
                s.to_string()
            }
        }
        _ => "no git".into(),
    }
}

/// Open/refresh the slash-command picker as the user types. Modern harnesses
/// (Claude Code, Codex) promote commands as soon as the prompt starts with `/`.
fn refresh_command_menu(app: &mut App) {
    app.command_menu = app.input.starts_with('/') && app.focus == Focus::Input;
    if app.command_menu {
        app.command_items = SLASH_COMMANDS
            .iter()
            .filter(|(cmd, _)| cmd.starts_with(&app.input))
            .map(|(c, d)| (c.to_string(), d.to_string()))
            .collect();
        if app.command_selected >= app.command_items.len() {
            app.command_selected = app.command_items.len().saturating_sub(1);
        }
    }
}

/// Submit a user input line: run slash commands locally or spawn the agent turn.
#[allow(clippy::too_many_arguments)]
fn run_input(
    app: &mut App,
    state: &Arc<Mutex<AgentState>>,
    client: &LlmRouter,
    interrupt: &Arc<AtomicBool>,
    agent_tx: &mpsc::Sender<AgentEvent>,
    approval_tx: &mpsc::Sender<ApprovalRequest>,
    approval_rx: &Arc<Mutex<mpsc::Receiver<ApprovalDecision>>>,
    start: &mut Instant,
    input: &str,
) {
    if input.starts_with('/') && handle_slash_command(input, app, state, client) {
        app.clear_input();
        return;
    }
    app.add_message("User", input);
    app.input_history.push(input.to_string());
    app.history_index = None;
    app.loading = true;
    app.follow = true;
    app.status = "Working… (Esc to interrupt)".into();
    interrupt.store(false, Ordering::Relaxed);
    *start = Instant::now();
    app.elapsed = Duration::ZERO;
    let input_clone = input.to_string();
    let client_clone = client.clone();
    let state_clone = Arc::clone(state);
    let interrupt_clone = Arc::clone(interrupt);
    let agent_tx_clone = agent_tx.clone();
    let agent_tx_text = agent_tx.clone();
    let approval_tx_clone = approval_tx.clone();
    let approval_rx_clone = Arc::clone(approval_rx);
    let agent_mode = app.agent_mode;
    thread::spawn(move || {
        let hooks = AgentHooks {
            on_event: Some(Arc::new(move |ev| {
                let _ = agent_tx_clone.send(ev);
            })),
            on_tool_call_delta: None,
            on_text_delta: Some(Arc::new(move |text| {
                let _ = agent_tx_text.send(AgentEvent::TextDelta {
                    text: text.to_string(),
                });
            })),
            on_approval: Some(Arc::new(move |request| {
                if approval_tx_clone.send(request).is_err() {
                    return ApprovalDecision::Deny;
                }
                approval_rx_clone
                    .lock()
                    .map(|rx| rx.recv().unwrap_or(ApprovalDecision::Deny))
                    .unwrap_or(ApprovalDecision::Deny)
            })),
            interrupt: Some(interrupt_clone),
        };
        let rt = tokio::runtime::Runtime::new().unwrap();
        let mut st = state_clone.lock().unwrap();
        rt.block_on(crate::agent::agent_loop::run_agent_loop_with_hooks(
            &client_clone,
            &mut st,
            &input_clone,
            &hooks,
            agent_mode,
        ));
    });
    app.clear_input();
}

pub fn run_ui(client: LlmRouter, state: AgentState) -> Result<(), Box<dyn Error>> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app and event channels
    let model_name = state.config.coder_model.clone();
    let caveman_tag = state.caveman.tag();
    let mut initial_app = App::new(
        &model_name,
        if caveman_tag.is_empty() {
            "off"
        } else {
            caveman_tag
        },
    );
    initial_app.dir = state.config.workspace_dir.display().to_string();
    initial_app.git_branch = git_branch(&state.config.workspace_dir);
    // Bring the default cloud provider online (e.g. nvidia with
    // NVIDIA_API_KEY from .env) so cloud models work without the --cloud flag.
    let provider_msg = match client.set_provider(&initial_app.provider) {
        Ok(base) => format!("Cloud provider ready: {} ({base})", initial_app.provider),
        Err(e) => format!(
            "Cloud provider {} not configured: {e}",
            initial_app.provider
        ),
    };
    initial_app.add_message("System", &provider_msg);
    for (role, content) in state.session.history() {
        initial_app.add_message(&display_role(&role), &content);
    }
    let app = Arc::new(Mutex::new(initial_app));
    let state = Arc::new(Mutex::new(state));
    // populate sidebar with workspace files
    {
        let st = state.lock().unwrap();
        let files = st.files.list_files("");
        let root = st.files.workspace_root().to_path_buf();
        let mut a = app.lock().unwrap();
        a.sidebar_items = files;
        a.file_search_paths = file_search::walk_files(&root);
        update_info_sections(&mut a, &st);
    }
    let (tx, rx) = mpsc::channel();
    // Agent progress stream + interrupt flag (Esc while loading cancels the turn).
    let (agent_tx, agent_rx) = mpsc::channel::<AgentEvent>();
    // Approval broker: worker → UI requests, UI → worker decisions.
    let (approval_tx, approval_rx) = mpsc::channel::<ApprovalRequest>();
    let (decision_tx, decision_rx) = mpsc::channel::<ApprovalDecision>();
    let decision_rx = Arc::new(Mutex::new(decision_rx));
    let interrupt = Arc::new(AtomicBool::new(false));
    let mut start = Instant::now();

    // Spawn input handling thread
    thread::spawn(move || {
        let tx = tx.clone();
        loop {
            if let Ok(event) = event::read() {
                if tx.send(event).is_err() {
                    break;
                }
            }
        }
    });

    // App loop
    {
        let mut init = app.lock().unwrap();
        init.add_message("System", "Anamnesic is ready. Type '/' for commands (e.g. /model), ask for a change, or inspect files.");
    }

    loop {
        // Drain agent progress events (tool calls, plan steps, final result).
        let agent_events: Vec<AgentEvent> = agent_rx.try_iter().collect();
        {
            let mut a = app.lock().unwrap();
            for ev in agent_events {
                match ev {
                    AgentEvent::Status(text) => a.status = text,
                    AgentEvent::ToolCall { name, summary } => {
                        let s: String = summary.chars().take(120).collect();
                        a.add_message("Tool", &format!("{name} — {s}"));
                    }
                    AgentEvent::ToolCallDelta { index, name, args_delta } => {
                        let prefix = name.as_deref().unwrap_or("?");
                        a.feed_tool_delta(index, prefix, &args_delta);
                    }
                    AgentEvent::PlanStep {
                        index,
                        total,
                        description,
                    } => {
                        a.add_message("Plan", &format!("[{index}/{total}] {description}"));
                    }
                    AgentEvent::FileChanged { path } => {
                        a.add_message("File", &format!("changed {path}"));
                    }
                    AgentEvent::Verification {
                        status,
                        command,
                        summary,
                    } => {
                        let command = command.unwrap_or_else(|| "auto-detect".into());
                        a.add_message("Verify", &format!("[{status}] {command} — {summary}"));
                    }
                    AgentEvent::ReasoningDelta { text } => {
                        a.feed_reasoning_delta(&text);
                    }
                    AgentEvent::ResetReasoning => {
                        a.reset_reasoning();
                    }
                    AgentEvent::Done { message } => {
                        if !a.end_streaming(Some(&message)) {
                            a.add_message("Assistant", &message);
                        }
                        a.loading = false;
                        a.status =
                            "Ready · Enter to send · ↑/↓ history · PgUp/PgDn scroll · mouse wheel · Esc interrupt"
                                .into();
                    }
                    AgentEvent::Transaction { action, summary } => {
                        a.add_message("Workspace", &format!("[{action}] {summary}"));
                    }
                    AgentEvent::Failed { message } => {
                        a.end_streaming(None);
                        a.add_message("Error", &message);
                        a.loading = false;
                        a.status = "Failed · Enter to retry · Esc interrupt".into();
                    }
                    AgentEvent::Interrupted => {
                        a.end_streaming(None);
                        a.add_message("System", "Turn interrupted by user.");
                        a.loading = false;
                        a.status = "Interrupted · Enter to send · Esc interrupt".into();
                    }
                    AgentEvent::TextDelta { text } => {
                        a.feed_text_delta(&text);
                    }
                    AgentEvent::TokenUsage {
                        prompt_tokens,
                        completion_tokens,
                        reasoning_tokens,
                        total_tokens,
                    } => {
                        a.token_breakdown.input = prompt_tokens;
                        a.token_breakdown.output = completion_tokens;
                        a.token_breakdown.reasoning = reasoning_tokens;
                        a.tokens = total_tokens;
                        a.context_cost = a.token_breakdown.input as f64 * 0.000003
                            + a.token_breakdown.output as f64 * 0.000012
                            + a.token_breakdown.reasoning as f64 * 0.000001;
                    }
                }
            }
        }
        // Surface any pending approval request from the worker.
        if let Ok(request) = approval_rx.try_recv() {
            let mut a = app.lock().unwrap();
            a.status = format!("Approval required: {} · a/s/d", request.tool);
            a.pending_approval = Some(request);
        }
        // Refresh context budget + workspace diff + elapsed + spinner from shared state.
        // Do not block rendering while the worker owns AgentState for a turn.
        {
            let mut a = app.lock().unwrap();
            if let Ok(st) = state.try_lock() {
                a.tokens = st.session.estimated_tokens();
                let diff = st.last_diff.paths();
                a.modified_files = if diff.is_empty() {
                    vec!["No changes".into()]
                } else {
                    diff
                };
                a.diff_content = st.last_diff.diff_content.clone();
                update_info_sections(&mut a, &st);
            }
            if a.loading {
                a.elapsed = start.elapsed();
                a.spinner_frame = (a.spinner_frame + 1) % SPINNER.len();
            }
        }
        {
            let guard = app.lock().unwrap();
            draw(&mut terminal, &guard)?;
        }

        match rx.recv_timeout(Duration::from_millis(80)) {
            Ok(Event::Key(key)) => {
                // On Windows, crossterm emits both Press and Release events.
                // Only handle Press (and Repeat for held keys) to avoid double input.
                if key.kind == KeyEventKind::Release {
                    continue;
                }
                let mut guard = app.lock().unwrap();
                // The approval modal only captures explicit decisions.
                if guard.pending_approval.is_some() {
                    let decision = match key.code {
                        KeyCode::Char('a') => Some(ApprovalDecision::AllowOnce),
                        KeyCode::Char('s') => Some(ApprovalDecision::AllowSession),
                        KeyCode::Char('d') | KeyCode::Esc => Some(ApprovalDecision::Deny),
                        _ => None,
                    };
                    if let Some(decision) = decision {
                        let request = guard.pending_approval.take();
                        let label = match decision {
                            ApprovalDecision::AllowOnce => "allowed once",
                            ApprovalDecision::AllowSession => "allowed for session",
                            ApprovalDecision::Deny => "denied",
                        };
                        if let Some(request) = request {
                            guard.add_message("Approval", &format!("{} — {label}", request.tool));
                        }
                        guard.status = "Working…".into();
                        let _ = decision_tx.send(decision);
                        continue;
                    }
                }
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('l') {
                    guard.messages.clear();
                    guard.status = "Chat cleared (session memory is preserved).".into();
                    continue;
                }
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                    if guard.loading {
                        interrupt.store(true, Ordering::Relaxed);
                        guard.status = "Interrupting…".into();
                    } else if guard.quit_pending {
                        break;
                    } else {
                        guard.quit_pending = true;
                        guard.status = "Ctrl+C again to quit".into();
                    }
                    continue;
                }
                // Ctrl+P: toggle the fuzzy file-search overlay.
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('p') {
                    if guard.file_search {
                        guard.file_search = false;
                    } else {
                        let paths = guard.file_search_paths.clone();
                        guard.open_file_search(paths);
                    }
                    continue;
                }
                // Ctrl+T: toggle reasoning panel expand/collapse.
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('t') {
                    guard.reasoning_expanded = !guard.reasoning_expanded;
                    continue;
                }
                // Ctrl+O: toggle tool call details expand/collapse.
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('o') {
                    guard.tool_calls_expanded = !guard.tool_calls_expanded;
                    continue;
                }
                // Ctrl+E: toggle tool call timeline expand/collapse.
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('e') {
                    guard.tool_calls_expanded = !guard.tool_calls_expanded;
                    continue;
                }
                // Ctrl+R: toggle compact/expanded view.
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('r') {
                    guard.tool_calls_expanded = !guard.tool_calls_expanded;
                    guard.reasoning_expanded = !guard.reasoning_expanded;
                    continue;
                }
                // The file-search overlay captures all keystrokes while open.
                if guard.file_search {
                    let mut handled = true;
                    match key.code {
                        KeyCode::Char(c) => {
                            guard.file_search_query.push(c);
                            guard.refresh_file_search();
                        }
                        KeyCode::Backspace => {
                            guard.file_search_query.pop();
                            guard.refresh_file_search();
                        }
                        KeyCode::Up => guard.move_file_search_selection(true),
                        KeyCode::Down => guard.move_file_search_selection(false),
                        KeyCode::Enter => {
                            if !guard.loading {
                                if let Some(m) = guard
                                    .file_search_results
                                    .get(guard.file_search_selected)
                                    .cloned()
                                {
                                    let path = m.path.clone();
                                    guard.file_search = false;
                                    let content =
                                        state.lock().unwrap().files.read_file(&path);
                                    guard.open_editor(&path, content.as_deref());
                                }
                            }
                        }
                        KeyCode::Esc => guard.file_search = false,
                        _ => handled = false,
                    }
                    if handled {
                        continue;
                    }
                }
                // Modal pickers (slash-command menu / model selector) take over
                // Up/Down/Enter/Esc while open.
                if guard.command_menu || guard.model_selector || guard.provider_selector || guard.resume_selector {
                    let mut handled = true;
                    match key.code {
                        KeyCode::Up => {
                            if guard.command_menu {
                                guard.command_selected = guard.command_selected.saturating_sub(1);
                            } else if guard.model_selector {
                                guard.model_selected = guard.model_selected.saturating_sub(1);
                            } else if guard.resume_selector {
                                guard.resume_selected = guard.resume_selected.saturating_sub(1);
                            } else {
                                guard.provider_selected = guard.provider_selected.saturating_sub(1);
                            }
                        }
                        KeyCode::Down => {
                            if guard.command_menu {
                                if guard.command_selected + 1 < guard.command_items.len() {
                                    guard.command_selected += 1;
                                }
                            } else if guard.model_selector {
                                if guard.model_selected + 1 < guard.model_items.len() {
                                    guard.model_selected += 1;
                                }
                            } else if guard.resume_selector {
                                if guard.resume_selected + 1 < guard.resume_items.len() {
                                    guard.resume_selected += 1;
                                }
                            } else if guard.provider_selected + 1 < guard.provider_items.len() {
                                guard.provider_selected += 1;
                            }
                        }
                        KeyCode::Enter => {
                            if guard.command_menu {
                                let sel = guard
                                    .command_selected
                                    .min(guard.command_items.len().saturating_sub(1));
                                let text = guard
                                    .command_items
                                    .get(sel)
                                    .map(|(c, _)| c.clone())
                                    .unwrap_or_else(|| guard.input.trim().to_string());
                                guard.command_menu = false;
                                if !guard.loading {
                                    run_input(
                                        &mut guard,
                                        &state,
                                        &client,
                                        &interrupt,
                                        &agent_tx,
                                        &approval_tx,
                                        &decision_rx,
                                        &mut start,
                                        &text,
                                    );
                                }
                            } else if guard.model_selector {
                                let name = guard
                                    .model_items
                                    .get(guard.model_selected)
                                    .cloned()
                                    .unwrap_or_default();
                                guard.model_selector = false;
                                if !name.is_empty() {
                                    set_active_model(&mut guard, &state, &client, &name);
                                }
                            } else if guard.provider_selector {
                                let name = guard
                                    .provider_items
                                    .get(guard.provider_selected)
                                    .cloned()
                                    .unwrap_or_default();
                                guard.provider_selector = false;
                                if !name.is_empty() {
                                    let id = name.split(" — ").next().unwrap_or(&name).to_string();
                                    set_active_provider(&mut guard, &state, &client, &id);
                                }
                            } else if guard.resume_selector {
                                if let Some(&session_id) = guard.resume_ids.get(guard.resume_selected) {
                                    guard.resume_selector = false;
                                    let mut st = state.lock().unwrap();
                                    match st.load_session_into_state(session_id) {
                                        Ok(count) => {
                                            drop(st);
                                            guard.messages.clear();
                                            let s = state.lock().unwrap();
                                            for (role, content) in s.session.history() {
                                                guard.add_message(&display_role(&role), &content);
                                            }
                                            guard.add_message("System", &format!("✓ Resumed session {session_id} ({count} messages)"));
                                        }
                                        Err(e) => guard.add_message("Error", &format!("Failed: {e}")),
                                    }
                                }
                            }
                        }
                        KeyCode::Esc => {
                            guard.command_menu = false;
                            guard.model_selector = false;
                            guard.provider_selector = false;
                            guard.resume_selector = false;
                        }
                        _ => handled = false,
                    }
                    if handled || guard.model_selector || guard.provider_selector || guard.resume_selector {
                        continue;
                    }
                }
                // /info overlay captures navigation keys while open.
                if guard.info_popup {
                    match key.code {
                        KeyCode::Esc => guard.info_popup = false,
                        KeyCode::Up => {
                            if guard.info_selected > 0 {
                                guard.info_selected -= 1;
                            }
                        }
                        KeyCode::Down => {
                            if guard.info_selected + 1 < guard.info_sections.len() {
                                guard.info_selected += 1;
                            }
                        }
                        KeyCode::Enter => {
                            let idx = guard.info_selected;
                            if let Some(section) = guard.info_sections.get_mut(idx) {
                                section.open = !section.open;
                            }
                        }
                        _ => {}
                    }
                    continue;
                }
                match key.code {
                    KeyCode::PageUp => {
                        guard.scroll_offset = guard.scroll_offset.saturating_sub(10);
                        guard.follow = false;
                    }
                    KeyCode::PageDown => {
                        guard.scroll_offset += 10;
                    }
                    KeyCode::Home => {
                        guard.scroll_offset = 0;
                        guard.follow = false;
                    }
                    KeyCode::End => {
                        guard.follow = true;
                    }
                    KeyCode::Tab => {
                        guard.agent_mode = match guard.agent_mode {
                            AgentMode::Agent => AgentMode::Plan,
                            AgentMode::Plan => AgentMode::Agent,
                        };
                        let mode_str = match guard.agent_mode {
                            AgentMode::Agent => "agent (tool-use)",
                            AgentMode::Plan => "plan (planner-first)",
                        };
                        guard.add_message("System", &format!("Mode switched to {}", mode_str));
                        guard.status = format!(
                            "Ready · mode: {} · Esc interrupt",
                            match guard.agent_mode {
                                AgentMode::Agent => "agent",
                                AgentMode::Plan => "plan",
                            }
                        );
                    }
                    KeyCode::Up => {
                        if let Focus::Sidebar = guard.focus {
                            if guard.info_selected > 0 {
                                guard.info_selected -= 1;
                            }
                        } else if guard.focus == Focus::Editor {
                            if guard.editor_row > 0 {
                                guard.editor_row -= 1;
                            }
                            let line_len = guard
                                .editor_lines
                                .get(guard.editor_row)
                                .map(|l| l.len())
                                .unwrap_or(0);
                            if guard.editor_col > line_len {
                                guard.editor_col = line_len;
                            }
                            // adjust scroll
                            if guard.editor_row < guard.editor_scroll {
                                guard.editor_scroll = guard.editor_row;
                            }
                        } else if guard.focus == Focus::Input && !guard.loading {
                            guard.previous_input();
                        } else if !guard.loading {
                            guard.scroll_offset = guard.scroll_offset.saturating_sub(3);
                            guard.follow = false;
                        }
                    }
                    KeyCode::Down => {
                        if let Focus::Sidebar = guard.focus {
                            if guard.info_selected + 1 < guard.info_sections.len() {
                                guard.info_selected += 1;
                            }
                        } else if guard.focus == Focus::Editor {
                            if guard.editor_row + 1 < guard.editor_lines.len() {
                                guard.editor_row += 1;
                            }
                            let line_len = guard
                                .editor_lines
                                .get(guard.editor_row)
                                .map(|l| l.len())
                                .unwrap_or(0);
                            if guard.editor_col > line_len {
                                guard.editor_col = line_len;
                            }
                        } else if guard.focus == Focus::Input && !guard.loading {
                            guard.next_input();
                        } else if !guard.loading {
                            guard.scroll_offset += 3;
                        }
                    }
                    KeyCode::Enter => {
                        match guard.focus {
                            Focus::Input | Focus::Messages => {
                                if guard.loading {
                                    continue;
                                }
                                let input = guard.input.trim().to_string();
                                if !input.is_empty() {
                                    guard.command_menu = false;
                                    guard.focus = Focus::Input;
                                    run_input(
                                        &mut guard,
                                        &state,
                                        &client,
                                        &interrupt,
                                        &agent_tx,
                                        &approval_tx,
                                        &decision_rx,
                                        &mut start,
                                        &input,
                                    );
                                }
                            }
                            Focus::Sidebar => {
                                let idx = guard.info_selected;
                                if let Some(section) = guard.info_sections.get_mut(idx) {
                                    section.open = !section.open;
                                }
                            }
                            Focus::Editor
                                // insert newline at cursor
                                if guard.editor_row <= guard.editor_lines.len() => {
                                    let line = if guard.editor_row < guard.editor_lines.len() {
                                        guard.editor_lines[guard.editor_row].clone()
                                    } else {
                                        String::new()
                                    };
                                    let byte = char_index_to_byte_index(&line, guard.editor_col);
                                    let (left, right) = line.split_at(byte);
                                    let row = guard.editor_row;
                                    if row < guard.editor_lines.len() {
                                        guard.editor_lines[row] = left.to_string();
                                        guard.editor_lines.insert(row + 1, right.to_string());
                                    } else {
                                        guard.editor_lines.push(left.to_string());
                                        guard.editor_lines.push(right.to_string());
                                    }
                                    guard.editor_row += 1;
                                    guard.editor_col = 0;
                                    guard.editor_dirty = true;
                                }
                            _ => {}
                        }
                    }
                    KeyCode::Char(c) => {
                        if guard.loading {
                            continue;
                        }
                        if guard.focus == Focus::Editor {
                            // insert char into editor at cursor
                            if guard.editor_row >= guard.editor_lines.len() {
                                guard.editor_lines.push(String::new());
                            }
                            let row = guard.editor_row;
                            let col_char = guard.editor_col;
                            let line = &mut guard.editor_lines[row];
                            let col = char_index_to_byte_index(line, col_char);
                            if col_char <= line.chars().count() {
                                line.insert(col, c);
                            } else {
                                line.push(c);
                            }
                            guard.editor_col += 1;
                            guard.editor_dirty = true;
                        } else {
                            let pos = char_index_to_byte_index(&guard.input, guard.cursor_position);
                            guard.input.insert(pos, c);
                            guard.cursor_position += 1;
                            refresh_command_menu(&mut guard);
                        }
                    }
                    KeyCode::Backspace => {
                        if guard.loading {
                            continue;
                        }
                        if guard.focus == Focus::Editor {
                            let row = guard.editor_row;
                            let col = guard.editor_col;
                            if row < guard.editor_lines.len() {
                                if col > 0 {
                                    let byte =
                                        char_index_to_byte_index(&guard.editor_lines[row], col - 1);
                                    guard.editor_lines[row].remove(byte);
                                    guard.editor_col -= 1;
                                } else if row > 0 {
                                    // join with previous line
                                    let prev_len = guard.editor_lines[row - 1].chars().count();
                                    let cur = guard.editor_lines.remove(row);
                                    guard.editor_row -= 1;
                                    guard.editor_col = prev_len;
                                    guard.editor_lines[row - 1].push_str(&cur);
                                }
                            }
                            guard.editor_dirty = true;
                        } else {
                            if guard.cursor_position > 0 {
                                let pos = char_index_to_byte_index(
                                    &guard.input,
                                    guard.cursor_position - 1,
                                );
                                guard.input.remove(pos);
                                guard.cursor_position -= 1;
                                refresh_command_menu(&mut guard);
                            }
                        }
                    }
                    KeyCode::Left => {
                        if guard.focus == Focus::Editor {
                            if guard.editor_col > 0 {
                                guard.editor_col -= 1;
                            } else if guard.editor_row > 0 {
                                guard.editor_row -= 1;
                                let row = guard.editor_row;
                                guard.editor_col = guard.editor_lines[row].chars().count();
                            }
                        } else if guard.cursor_position > 0 {
                            guard.cursor_position -= 1;
                        }
                    }
                    KeyCode::Right => {
                        if guard.focus == Focus::Editor {
                            let row = guard.editor_row;
                            if row < guard.editor_lines.len() {
                            let len = guard.editor_lines[row].chars().count();
                                if guard.editor_col < len {
                                    guard.editor_col += 1;
                                } else if row + 1 < guard.editor_lines.len() {
                                    guard.editor_row += 1;
                                    guard.editor_col = 0;
                                }
                            }
                        } else if guard.cursor_position < guard.input.chars().count() {
                            guard.cursor_position += 1;
                        }
                    }
                    KeyCode::Esc => {
                        // If editing, close editor; else interrupt running turn
                        if guard.focus == Focus::Editor {
                            guard.editor_file = None;
                            guard.editor_lines.clear();
                            guard.focus = Focus::Input;
                        } else if guard.loading {
                            interrupt.store(true, Ordering::Relaxed);
                            guard.status = "Interrupting…".into();
                        } else if guard.focus != Focus::Input {
                            guard.focus = Focus::Input;
                        }
                        guard.quit_pending = false;
                    }
                    _ => {}
                }
                // handle Ctrl-S save
                if key.modifiers.contains(KeyModifiers::CONTROL) {
                    if let KeyCode::Char('s') = key.code {
                        if guard.focus == Focus::Editor {
                            if let Some(fname) = &guard.editor_file {
                                let content = guard.editor_lines.join("\n");
                                let mut st = state.lock().unwrap();
                                match st.config.write_tool_policy {
                                    ApprovalPolicy::Deny => {
                                        guard.add_message(
                                            "Error",
                                            "Save denied: write policy is set to deny",
                                        );
                                    }
                                    _ => match st.files.write_file(fname, &content) {
                                        Ok(_) => {
                                            let path = fname.clone();
                                            st.mark_changed(path);
                                            guard.add_message("Info", "File saved");
                                            guard.editor_dirty = false;
                                        }
                                        Err(e) => {
                                            guard.add_message(
                                                "Error",
                                                &format!("Save failed: {}", e),
                                            );
                                        }
                                    },
                                }
                            }
                        }
                    }
                }
            }
            Ok(Event::Mouse(mouse)) => {
                let mut guard = app.lock().unwrap();
                if matches!(mouse.kind, MouseEventKind::Down(MouseButton::Left))
                    && !guard.loading
                {
                    guard.focus = Focus::Messages;
                }
                match mouse.kind {
                    MouseEventKind::ScrollUp => {
                        guard.scroll_offset = guard.scroll_offset.saturating_sub(5);
                        guard.follow = false;
                    }
                    MouseEventKind::ScrollDown => {
                        guard.scroll_offset += 5;
                    }
                    MouseEventKind::Down(MouseButton::Left) => {
                        // Toggle tool call rollup expansion on click.
                        let flattened = flatten_messages(&guard);
                        let msg_area_top = 2; // margin(1) + header(1)
                        let clicked_row = mouse.row as usize;
                        if clicked_row > msg_area_top {
                            let offset = if guard.follow {
                                flattened
                                    .len()
                                    .saturating_sub(guard.scroll_offset)
                            } else {
                                guard.scroll_offset
                            };
                            let line_idx = clicked_row - msg_area_top + offset;
                            if line_idx < flattened.len() {
                                let line_text = flattened[line_idx].to_string();
                                if line_text.starts_with("●")
                                    || line_text.starts_with("Δ")
                                    || line_text.contains(" tool use")
                                {
                                    guard.tool_calls_expanded =
                                        !guard.tool_calls_expanded;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(_) => {}
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    Ok(())
}

fn update_info_sections(app: &mut App, _state: &AgentState) {
    let used = app.tokens;
    let max = app.max_context_tokens;
    let pct = if max > 0 { (used as f64 / max as f64) * 100.0 } else { 0.0 };
    let cost = app.context_cost;

    app.info_sections[0] = InfoPanelSection::open("Context", vec![
        Span::styled(format!("{} tokens", used), Style::default()),
        Span::styled(format!("{:.0}% used", pct), Style::default()),
        Span::styled(format!("${:.4} spent", cost), Style::default()),
    ]);

    let tb = &app.token_breakdown;
    app.info_sections[1] = InfoPanelSection::open("Token Usage", vec![
        Span::styled(format!("Input: {}", tb.input), Style::default()),
        Span::styled(format!("Output: {}", tb.output), Style::default()),
        Span::styled(format!("Reasoning: {}", tb.reasoning), Style::default()),
        Span::styled(format!("Cache read: {}", tb.cache_read), Style::default()),
        Span::styled(format!("Cache write: {}", tb.cache_write), Style::default()),
        Span::styled(format!("Cache rate: {:.1}%", tb.cache_rate), Style::default()),
        Span::styled(format!("Speed: {:.1} tok/s", tb.generation_speed), Style::default()),
        Span::styled(format!("Cost: ${:.4}", cost), Style::default()),
    ]);

    app.info_sections[2] = InfoPanelSection::open("Models", {
        let mut lines = app
            .models
            .iter()
            .map(|m| Span::styled(format!("{} — {}", m.provider, m.name), Style::default()))
            .collect::<Vec<_>>();
        if lines.is_empty() {
            lines.push(Span::styled("No models loaded", Style::default()));
        }
        lines
    });

    app.info_sections[3] = InfoPanelSection::open("Code Indexing", vec![
        Span::styled(
            if app.indexing_enabled { "Enabled" } else { "Disabled" },
            Style::default(),
        ),
    ]);

    app.info_sections[4] = InfoPanelSection::open("Todo", {
        let mut lines = Vec::new();
        if app.todo_items.is_empty() {
            lines.push(Span::styled("No active tasks", Style::default()));
        } else {
            for item in &app.todo_items {
                let mark = if item.done { "[x]" } else { "[ ]" };
                lines.push(Span::styled(format!("{} {}", mark, item.text), Style::default()));
            }
        }
        lines
    });

    app.info_sections[5] = InfoPanelSection::open("Modified Files", {
        if app.diff_content.is_empty() {
            if app.modified_files.is_empty() {
                vec![Span::styled("No changes", Style::default())]
            } else {
                app.modified_files
                    .iter()
                    .map(|f| Span::styled(f.clone(), Style::default().fg(Color::Gray)))
                    .collect()
            }
        } else {
            app.diff_content
                .iter()
                .map(|line| {
                    let style = if line.starts_with('+') {
                        Style::default().fg(Color::Green)
                    } else if line.starts_with('-') {
                        Style::default().fg(Color::Red)
                    } else if line.starts_with("@@") {
                        Style::default().fg(Color::Cyan)
                    } else {
                        Style::default().fg(Color::Gray)
                    };
                    Span::styled(line.clone(), style)
                })
                .collect()
        }
    });

    app.info_sections[6] = InfoPanelSection::open("Memory", vec![Span::styled(
        if app.memory_enabled { "Enabled" } else { "Disabled" },
        Style::default(),
    )]);
}

fn draw<B: ratatui::backend::Backend>(
    f: &mut Terminal<B>,
    app: &App,
) -> Result<(), Box<dyn Error>> {
    f.draw(|f| {
        let size = f.area();
        let page = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints(
                [
                    Constraint::Length(1),
                    Constraint::Min(0),
                    Constraint::Length(1),
                    Constraint::Length(1),
                    Constraint::Length(1),
                ]
                .as_ref(),
            )
            .split(size);
        // Compact single-line header (modern harness style): title badge + context,
        // token counter right-aligned. No box — full-bleed like Claude Code/Codex.
        let header_w = size.width.saturating_sub(2) as usize;
        let left_txt = format!(
            "  {}  ·  {}  ·  {}  ·  {}  ·  {}",
            truncate_str(&app.model, 22),
            app.caveman,
            truncate_str(&app.provider, 12),
            app.git_branch,
            match app.agent_mode {
                AgentMode::Agent => "agent",
                AgentMode::Plan => "plan",
            },
        );
        let right_txt = format!("ctx: {} tok ", app.tokens);
        let pad = header_w.saturating_sub(left_txt.chars().count() + right_txt.chars().count());
        let header = Paragraph::new(Line::from(vec![
            Span::styled(
                " ANAMNESIC ",
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(left_txt, Style::default().fg(Color::Gray)),
            Span::styled(" ".repeat(pad), Style::default()),
            Span::styled(right_txt, Style::default().fg(Color::Gray)),
        ]));
        f.render_widget(header, page[0]);
        // Fixed status line: shows current status text (warnings, planning, retries)
        // truncated to terminal width. Never wraps — keeps layout stable.
        let status_w = size.width as usize;
        let status_text = truncate_str(&sanitize_status(&app.status), status_w.saturating_sub(2));
        let status_line = Paragraph::new(Line::from(vec![Span::styled(
            format!(" {} ", status_text),
            Style::default().fg(Color::Yellow),
        )]));
        f.render_widget(status_line, page[2]);
        // Input bar: bare prompt at the bottom (modern harness style, no box).
        let prompt = if app.loading { "▍" } else { "❯" };
        let input_line = Line::from(vec![
            Span::styled(
                format!("{prompt} "),
                Style::default()
                    .fg(if app.loading {
                        Color::Gray
                    } else {
                        Color::Green
                    })
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                app.input.clone(),
                Style::default().fg(if app.loading {
                    Color::Gray
                } else {
                    Color::White
                }),
            ),
        ]);
        let input = Paragraph::new(input_line).wrap(Wrap { trim: true });
        f.render_widget(input, page[3]);
        // Set cursor position next to the typed text.
        if !app.loading {
            f.set_cursor_position((page[3].x + 2 + app.cursor_position as u16, page[3].y));
        }
        // Thin bottom bar: dir · branch on the left, spinner + elapsed on the right.
        let bottom_left = format!(" {} · {}", truncate_str(&app.dir, 48), app.git_branch);
        let bottom_right = if app.loading {
            format!(
                " {} {:<5}  (Ctrl+C cancel) ",
                SPINNER[app.spinner_frame],
                format_elapsed(app.elapsed)
            )
        } else if app.pending_approval.is_some() {
            " a: allow once · s: allow session · d: deny ".into()
        } else {
            " Esc interrupt · Ctrl+O tool calls · Ctrl+E tool calls · Ctrl+R view ".into()
        };
        let bottom_pad = (size.width as usize)
            .saturating_sub(bottom_left.chars().count() + bottom_right.chars().count());
        let bottom = Paragraph::new(Line::from(vec![
            Span::styled(
                bottom_left,
                Style::default().fg(Color::Gray),
            ),
            Span::styled(" ".repeat(bottom_pad), Style::default()),
            Span::styled(
                bottom_right,
                Style::default().fg(Color::Gray),
            ),
        ]));
        f.render_widget(bottom, page[4]);

        // Full-width transcript (Codex-style single column — no sidebar).
        if let Some(editor_file) = app.editor_file.as_ref() {
            let title = format!(
                "Editor - {}{}",
                editor_file,
                if app.editor_dirty { " *" } else { "" }
            );
            // compute visible lines based on scroll and area height
            let area_height = page[1].height as usize - 2; // leave room for borders
            let total_lines = app.editor_lines.len();
            let scroll = if app.editor_scroll + area_height > total_lines {
                total_lines.saturating_sub(area_height)
            } else {
                app.editor_scroll
            };
            let end = std::cmp::min(scroll + area_height, total_lines);
            let line_num_width = total_lines.to_string().len();
            let visible: Vec<Line<'static>> = app.editor_lines[scroll..end]
                .iter()
                .enumerate()
                .map(|(i, line)| {
                    let num = scroll + i + 1;
                    let prefix = format!("{:>width$} │ ", num, width = line_num_width);
                    Line::from(vec![
                        Span::styled(prefix, Style::default().fg(Color::Gray)),
                        Span::styled(line.clone(), Style::default()),
                    ])
                })
                .collect();
            let editor = Paragraph::new(visible)
                .block(Block::default().title(title).borders(Borders::ALL));
            f.render_widget(editor, page[1]);
            // set cursor in editor area relative to scroll
            let r = (app.editor_row.saturating_sub(scroll)) as u16;
            let _c = app.editor_col as u16 + line_num_width as u16 + 3; // offset for line numbers
            f.set_cursor_position((page[1].x + _c + 1, page[1].y + r + 1));
        } else {
        let messages_lines = flatten_messages(app);
        let view_height = page[1].height as usize;
        let widget = Paragraph::new(messages_lines).wrap(Wrap { trim: true });
        // Ratatui's own WordWrapper drives both the render and the scroll
        // math, so the visible window always matches the wrapped layout.
        let total_wrapped = widget.line_count(page[1].width);
        let offset = if app.follow {
            total_wrapped.saturating_sub(view_height)
        } else {
            app.scroll_offset.min(total_wrapped.saturating_sub(view_height))
        };
        f.render_widget(widget.scroll((offset as u16, 0)), page[1]);
        }

        // Overlays: slash-command picker / model selector (modal, like modern harness TUIs).
        if app.command_menu || app.model_selector || app.provider_selector || app.resume_selector {
            let (items, title, selected) = if app.command_menu {
                let items = app
                    .command_items
                    .iter()
                    .map(|(cmd, desc)| {
                        ListItem::new(Line::from(vec![
                            Span::styled(
                                cmd.clone(),
                                Style::default()
                                    .fg(Color::Cyan)
                                    .add_modifier(Modifier::BOLD),
                            ),
                            Span::styled(format!("  {desc}"), Style::default().fg(Color::Gray)),
                        ]))
                    })
                    .collect::<Vec<_>>();
                let selected = app.command_selected.min(items.len().saturating_sub(1));
                (items, " Slash commands ".to_string(), selected)
            } else if app.model_selector {
                let items = app
                    .model_items
                    .iter()
                    .map(|m| {
                        ListItem::new(Span::styled(m.clone(), Style::default().fg(Color::Cyan)))
                    })
                    .collect::<Vec<_>>();
                let selected = app.model_selected.min(items.len().saturating_sub(1));
                (
                    items,
                    " Select model — ↑/↓ · Enter set · Esc cancel ".to_string(),
                    selected,
                )
            } else if app.resume_selector {
                let items = app
                    .resume_items
                    .iter()
                    .map(|s| {
                        ListItem::new(Span::styled(
                            s.clone(),
                            Style::default().fg(Color::LightGreen),
                        ))
                    })
                    .collect::<Vec<_>>();
                let selected = app.resume_selected.min(items.len().saturating_sub(1));
                (
                    items,
                    " Resume session — ↑/↓ · Enter resume · Esc cancel ".to_string(),
                    selected,
                )
            } else {
                let items = app
                    .provider_items
                    .iter()
                    .map(|p| {
                        ListItem::new(Span::styled(
                            p.clone(),
                            Style::default().fg(Color::LightBlue),
                        ))
                    })
                    .collect::<Vec<_>>();
                let selected = app.provider_selected.min(items.len().saturating_sub(1));
                (
                    items,
                    " Select provider — ↑/↓ · Enter set · Esc cancel ".to_string(),
                    selected,
                )
            };
            let popup_w = 64.min(size.width.saturating_sub(4));
            let popup_h = (items.len() as u16 + 2).min(size.height.saturating_sub(4));
            let x = size.x + (size.width.saturating_sub(popup_w)) / 2;
            let y = size.y + (size.height.saturating_sub(popup_h)) / 3;
            let area = Rect {
                x,
                y,
                width: popup_w,
                height: popup_h,
            };
            let mut list_state = ratatui::widgets::ListState::default();
            list_state.select(Some(selected));
            let list = List::new(items)
                .block(Block::default().title(title).borders(Borders::ALL))
                .highlight_style(Style::default().bg(Color::Gray))
                .highlight_symbol("▶ ");
            f.render_stateful_widget(list, area, &mut list_state);
        }

        // /info overlay: former left-sidebar sections, rendered full-screen.
        if app.info_popup {
            let popup_w = (size.width.saturating_sub(6)).max(30);
            let popup_h = (size.height.saturating_sub(4)).max(10);
            let x = size.x + (size.width.saturating_sub(popup_w)) / 2;
            let y = size.y + 2;
            let area = Rect { x, y, width: popup_w, height: popup_h };
            let mut panel_lines: Vec<Line> = Vec::new();
            for (idx, section) in app.info_sections.iter().enumerate() {
                let arrow = if section.open { "▼" } else { "▶" };
                let selected = idx == app.info_selected;
                let style = if selected {
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::Cyan)
                };
                let mut spans = Vec::new();
                if selected {
                    spans.push(Span::styled(
                        "▶ ",
                        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                    ));
                } else {
                    spans.push(Span::styled("  ", Style::default()));
                }
                spans.push(Span::styled(format!("{} {}", arrow, section.title), style));
                panel_lines.push(Line::from(spans));
                if section.open {
                    for span in &section.lines {
                        panel_lines.push(Line::from(vec![
                            Span::styled("    ", Style::default()),
                            Span::styled(&*span.content, Style::default().fg(Color::Gray)),
                        ]));
                    }
                }
            }
            let panel_block = Block::default()
                .title(" Workspace info — ↑/↓ navigate · Enter toggle · Esc close ")
                .borders(Borders::ALL);
            let panel = Paragraph::new(panel_lines).block(panel_block).wrap(Wrap { trim: false });
            f.render_widget(panel, area);
        }

        // Ctrl+P fuzzy file-search overlay.
        if app.file_search {
            let popup_w = 72.min(size.width.saturating_sub(4));
            let max_h = size.height.saturating_sub(2).max(4);
            let popup_h = (app.file_search_results.len() as u16 + 4).min(max_h);
            let x = size.x + (size.width.saturating_sub(popup_w)) / 2;
            let y = size.y + (size.height.saturating_sub(popup_h)) / 3;
            let area = Rect { x, y, width: popup_w, height: popup_h };
            let title = format!(
                " Open file · {} result{} · ↑/↓ · Enter open · Esc cancel ",
                app.file_search_results.len(),
                if app.file_search_results.len() == 1 { "" } else { "s" }
            );
            let mut lines: Vec<Line> = Vec::new();
            lines.push(Line::from(vec![
                Span::styled(
                    "❯ ",
                    Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    app.file_search_query.clone(),
                    Style::default().fg(Color::White),
                ),
            ]));
            lines.push(Line::from(""));
            if app.file_search_results.is_empty() {
                let hint = if app.file_search_query.is_empty() {
                    "type to filter workspace files…"
                } else {
                    "no matches"
                };
                lines.push(Line::from(Span::styled(
                    hint,
                    Style::default().fg(Color::Gray),
                )));
            } else {
                for (i, m) in app.file_search_results.iter().enumerate() {
                    let selected = i == app.file_search_selected;
                    let mut spans: Vec<Span> = vec![Span::styled(
                        if selected { "▶ " } else { "  " },
                        Style::default().fg(Color::Yellow),
                    )];
                    for (start, end, matched) in file_search::highlight_segments(&m.path, &m.indices) {
                        let style = if matched {
                            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
                        } else if selected {
                            Style::default().fg(Color::White)
                        } else {
                            Style::default().fg(Color::Gray)
                        };
                        spans.push(Span::styled(m.path[start..end].to_string(), style));
                    }
                    lines.push(Line::from(spans));
                }
            }
            let para = Paragraph::new(lines)
                .block(Block::default().title(title).borders(Borders::ALL))
                .wrap(Wrap { trim: true });
            f.render_widget(para, area);
        }

        // Approval modal (write/command policy set to `ask`).
        if let Some(request) = &app.pending_approval {
            let popup_w = 72.min(size.width.saturating_sub(4));
            let popup_h = 8.min(size.height.saturating_sub(4));
            let area = Rect {
                x: size.x + (size.width.saturating_sub(popup_w)) / 2,
                y: size.y + (size.height.saturating_sub(popup_h)) / 3,
                width: popup_w,
                height: popup_h,
            };
            let body = vec![
                Line::from(Span::styled(
                    request.tool.clone(),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                )),
                Line::from(Span::raw(request.summary.clone())),
                Line::from(Span::styled(
                    request.risk.clone(),
                    Style::default().fg(Color::Gray),
                )),
                Line::from(Span::styled(
                    "a: allow once   s: allow for session   d/Esc: deny",
                    Style::default().fg(Color::Cyan),
                )),
            ];
            let paragraph = Paragraph::new(body)
                .block(
                    Block::default()
                        .title(" Approval required ")
                        .borders(Borders::ALL),
                )
                .wrap(Wrap { trim: true });
            f.render_widget(paragraph, area);
        }
    })?;
    Ok(())
}

fn display_role(role: &str) -> String {
    match role {
        "user" => "User".into(),
        "assistant" => "Assistant".into(),
        _ => role.to_string(),
    }
}

/// Count the number of visual lines produced when wrapping `lines` to
/// `width` cells. Used for scroll-offset math so that `scroll_offset`
/// tracks visual rows, not logical `Line` entries.
/// Number of visual rows a set of lines occupies once wrapped at `width`.
/// Delegates to ratatui's own `Paragraph::line_count` so the count always
/// matches how the widget will actually render (WordWrapper, trim, spans
/// concatenated without separators).
fn count_wrapped_lines(lines: &[Line<'static>], width: usize) -> usize {
    if width == 0 {
        return lines.len();
    }
    Paragraph::new(lines.to_vec())
        .wrap(Wrap { trim: true })
        .line_count(width as u16)
}

/// Flatten chat messages into renderable transcript lines (Codex-style):
/// user messages use a bold-dim `›` prefix, assistant messages render markdown
/// with a dim `•` bullet, and status/tool messages stay compact and subdued.
/// Flush accumulated tool calls into a rollup summary or individual lines.
fn flush_tool_rollup(
    app: &App,
    count: &mut usize,
    buffer: &mut Vec<String>,
    lines: &mut Vec<Line<'static>>,
) {
    if *count == 0 {
        return;
    }
    let expanded = app.tool_calls_expanded;
    if expanded {
        for content in buffer.drain(..) {
            lines.extend(status_message_lines("tool", &content));
            lines.push(Line::from(""));
        }
    } else {
        let summary = format_tool_rollup(buffer);
        lines.extend(status_message_lines("tool", &summary));
    }
    *count = 0;
    buffer.clear();
}

/// Build a rollup summary string from tool call contents.
/// Groups by tool name and shows counts, e.g. "Read file, Glob, Bash ×5".
/// Active (streaming) tool calls are marked with a ● prefix.
fn format_tool_rollup(buffer: &[String]) -> String {
    let mut counts: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    let mut active_tools: Vec<String> = Vec::new();
    for content in buffer {
        let name = extract_tool_name(content);
        if content.starts_with("Δ ") {
            if !active_tools.contains(&name) {
                active_tools.push(name.clone());
            }
        }
        *counts.entry(name).or_insert(0) += 1;
    }
    let parts: Vec<String> = counts
        .iter()
        .map(|(name, &count)| {
            let is_active = active_tools.contains(name);
            let prefix = if is_active { "● " } else { "" };
            if count > 1 {
                format!("{prefix}{name} ×{count}")
            } else {
                format!("{prefix}{name}")
            }
        })
        .collect();
    let total = buffer.len();
    let plural = if total == 1 { "" } else { "s" };
    format!("{} tool use{}: {}", total, plural, parts.join(", "))
}

/// Extract the tool name from a tool call content string.
/// Handles both "Δ name[idx] ..." (delta) and "name — summary" (completed) formats.
fn extract_tool_name(content: &str) -> String {
    if let Some(rest) = content.strip_prefix("Δ ") {
        let end = rest.find(['[', ' ']).unwrap_or(rest.len());
        rest[..end].to_string()
    } else if let Some(pos) = content.find(' ') {
        content[..pos].to_string()
    } else {
        content.to_string()
    }
}

fn flatten_messages(app: &App) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let mut tool_count = 0usize;
    let mut tool_buffer = Vec::new();
    let mut in_plan = false;
    for (role, content) in &app.messages {
        match role.to_ascii_lowercase().as_str() {
            "user" => {
                in_plan = false;
                flush_tool_rollup(
                    app,
                    &mut tool_count,
                    &mut tool_buffer,
                    &mut lines,
                );
                lines.extend(user_message_lines(content));
            }
            "assistant" => {
                in_plan = false;
                flush_tool_rollup(
                    app,
                    &mut tool_count,
                    &mut tool_buffer,
                    &mut lines,
                );
                lines.extend(assistant_message_lines(content));
            }
            "thinking" => {
                in_plan = false;
                flush_tool_rollup(
                    app,
                    &mut tool_count,
                    &mut tool_buffer,
                    &mut lines,
                );
                if app.reasoning_expanded {
                    lines.extend(status_message_lines(role, content));
                } else {
                    let summary = format!("Thinking ({} chars)", content.len());
                    lines.extend(status_message_lines("thinking", &summary));
                }
            }
            "plan" => {
                in_plan = true;
                flush_tool_rollup(
                    app,
                    &mut tool_count,
                    &mut tool_buffer,
                    &mut lines,
                );
                lines.extend(status_message_lines(role, content));
            }
            "tool" => {
                if app.tool_calls_expanded {
                    flush_tool_rollup(
                        app,
                        &mut tool_count,
                        &mut tool_buffer,
                        &mut lines,
                    );
                    if in_plan {
                        lines.extend(plan_tool_message_lines(content));
                    } else {
                        lines.extend(status_message_lines(role, content));
                    }
                } else {
                    tool_count += 1;
                    tool_buffer.push(content.clone());
                }
            }
            _ => {
                in_plan = false;
                flush_tool_rollup(
                    app,
                    &mut tool_count,
                    &mut tool_buffer,
                    &mut lines,
                );
                lines.extend(status_message_lines(role, content));
            }
        }
        lines.push(Line::from(""));
    }
    flush_tool_rollup(
        app,
        &mut tool_count,
        &mut tool_buffer,
        &mut lines,
    );
    lines
}

/// Render a tool message inside a plan step with ⎿ child indentation.
fn plan_tool_message_lines(content: &str) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    for line in content.lines() {
        let mut spans = Vec::new();
        spans.push(Span::styled("⎿ ", Style::default().fg(Color::Gray)));
        spans.push(Span::styled(line.to_string(), Style::default().fg(Color::Gray)));
        out.push(Line::from(spans));
    }
    out
}

/// Codex-style user cell: `› ` (bold dim) on the first line, `  ` continuation.
fn user_message_lines(content: &str) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    let style = Style::default().fg(Color::LightBlue);
    let mut first = true;
    for content_line in content.lines() {
        let prefix = if first {
            Span::styled(
                "› ",
                Style::default()
                    .add_modifier(Modifier::BOLD)
                    .add_modifier(Modifier::DIM),
            )
        } else {
            Span::styled("  ", Style::default())
        };
        first = false;
        out.push(Line::from(vec![prefix, Span::styled(content_line.to_string(), style)]));
    }
    if first {
        // Empty message: render the prompt prefix so the cell stays visible.
        out.push(Line::from(vec![Span::styled(
            "› ",
            Style::default()
                .add_modifier(Modifier::BOLD)
                .add_modifier(Modifier::DIM),
        )]));
    }
    out
}

/// Codex-style assistant cell: markdown rendered, first line prefixed with
/// `• ` (dim), continuation lines with two spaces.
fn assistant_message_lines(content: &str) -> Vec<Line<'static>> {
    let raw_spans = match render_markdown_lines(content) {
        Some(spans) => spans,
        None => vec![Span::styled(content.to_string(), Style::default().fg(Color::Gray))],
    };

    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut current_line_spans: Vec<Span<'static>> = Vec::new();
    let mut is_first_line = true;

    for span in raw_spans {
        let text = span.content.as_ref();
        if text.contains('\n') {
            let parts: Vec<&str> = text.split('\n').collect();
            for (idx, part) in parts.iter().enumerate() {
                if !part.is_empty() {
                    let mut s = span.clone();
                    s.content = (*part).to_string().into();
                    current_line_spans.push(s);
                }
                if idx < parts.len() - 1 {
                    let prefix = if is_first_line {
                        is_first_line = false;
                        Span::styled("• ", Style::default().add_modifier(Modifier::DIM))
                    } else {
                        Span::styled("  ", Style::default())
                    };
                    let mut line_spans = vec![prefix];
                    line_spans.append(&mut current_line_spans);
                    lines.push(Line::from(line_spans));
                }
            }
        } else {
            current_line_spans.push(span);
        }
    }

    if !current_line_spans.is_empty() || lines.is_empty() {
        let prefix = if is_first_line {
            Span::styled("• ", Style::default().add_modifier(Modifier::DIM))
        } else {
            Span::styled("  ", Style::default())
        };
        let mut line_spans = vec![prefix];
        line_spans.append(&mut current_line_spans);
        lines.push(Line::from(line_spans));
    }

    lines
}

/// Merge runs of consecutive blank spans (a lone `\n` splits into two empty
/// pieces) so paragraphs are separated by exactly one blank line.
fn collapse_consecutive_blanks(spans: Vec<Span<'static>>) -> Vec<Span<'static>> {
    let mut out: Vec<Span<'static>> = Vec::with_capacity(spans.len());
    for span in spans {
        let blank = span.content.trim().is_empty();
        if blank && out.last().is_some_and(|last: &Span<'static>| last.content.trim().is_empty()) {
            continue;
        }
        out.push(span);
    }
    out
}

/// Compact subdued cell for system/tool/plan/error/verify/workspace messages.
fn status_message_lines(role: &str, content: &str) -> Vec<Line<'static>> {
    let lower = role.to_ascii_lowercase();
    let (prefix, prefix_style, content_style, italic) = match lower.as_str() {
        "error" => (
            "✗ ",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            Style::default().fg(Color::Red),
            false,
        ),
        "plan" => (
            "⏺ ",
            Style::default()
                .fg(Color::LightMagenta)
                .add_modifier(Modifier::BOLD),
            Style::default().fg(Color::LightMagenta),
            false,
        ),
        "tool" => {
            let is_active = content.starts_with('Δ');
            let is_rollup = content.contains("tool use");
            let icon = if is_rollup {
                "↳ "
            } else if content.contains("list_dir") || content.contains("search") || content.contains("grep") {
                "🔍 "
            } else if content.contains("read") {
                "📖 "
            } else if content.contains("write") || content.contains("replace") || content.contains("edit") {
                "✏️ "
            } else if content.contains("run") || content.contains("exec") || content.contains("cargo") {
                "⚡ "
            } else if content.contains("verify") || content.contains("test") {
                "🧪 "
            } else if is_active {
                "⏳ "
            } else {
                "↳ "
            };
            let prefix_style = if is_active {
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray)
            };
            let content_style = if is_active {
                Style::default().fg(Color::Cyan)
            } else {
                Style::default().fg(Color::Gray)
            };
            (icon, prefix_style, content_style, false)
        }
        "file" => (
            "↳ ",
            Style::default().fg(Color::Cyan),
            Style::default().fg(Color::Cyan),
            false,
        ),
        "verify" => (
            "",
            Style::default(),
            Style::default().fg(Color::Gray),
            false,
        ),
        "workspace" => (
            "",
            Style::default(),
            Style::default().fg(Color::Gray),
            false,
        ),
        "approval" => (
            "",
            Style::default(),
            Style::default().fg(Color::Green),
            false,
        ),
        "info" => ("", Style::default(), Style::default().fg(Color::Cyan), false),
        _ => (
            "",
            Style::default(),
            Style::default().fg(Color::Gray),
            true,
        ),
    };
    let mut out = Vec::new();
    let mut first = true;
    for content_line in content.lines() {
        let mut spans = Vec::new();
        if first {
            if !prefix.is_empty() {
                spans.push(Span::styled(prefix, prefix_style));
            }
        } else {
            spans.push(Span::styled("  ", Style::default()));
        }
        first = false;
        let style = if italic {
            content_style.add_modifier(Modifier::ITALIC)
        } else {
            content_style
        };
        spans.push(Span::styled(content_line.to_string(), style));
        out.push(Line::from(spans));
    }
    if first {
        out.push(Line::from(""));
    }
    out
}

fn render_markdown_lines(text: &str) -> Option<Vec<Span<'static>>> {
    use pulldown_cmark::{Event, Parser, Tag, TagEnd, Options};
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_TABLES);
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_TASKLISTS);
    let parser = Parser::new_ext(text, opts);
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut in_code = false;
    let mut in_bold = false;
    let mut in_italic = false;
    let mut in_strike = false;
    let mut code_text = String::new();
    for event in parser {
        match event {
            Event::Text(t) => {
                if in_code {
                    code_text.push_str(t.as_ref());
                } else if in_bold || in_italic || in_strike {
                    spans.push(Span::styled(
                        t.as_ref().to_string(),
                        Style::default()
                            .add_modifier(if in_bold { Modifier::BOLD } else { Modifier::empty() })
                            .add_modifier(if in_italic { Modifier::ITALIC } else { Modifier::empty() })
                            .add_modifier(if in_strike { Modifier::DIM } else { Modifier::empty() }),
                    ));
                } else {
                    spans.push(Span::raw(t.as_ref().to_string()));
                }
            }
            Event::Code(t) => {
                spans.push(Span::styled(
                    t.as_ref().to_string(),
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                ));
            }
            Event::SoftBreak | Event::HardBreak => {
                if in_code {
                    code_text.push('\n');
                } else {
                    spans.push(Span::raw("\n".to_string()));
                }
            }
            Event::Start(tag) => match tag {
                Tag::Paragraph => {
                    if !spans.is_empty() && !in_code {
                        spans.push(Span::raw("\n\n".to_string()));
                    }
                }
                Tag::CodeBlock(_) => {
                    in_code = true;
                    code_text.clear();
                }
                Tag::Strong => in_bold = true,
                Tag::Emphasis => in_italic = true,
                Tag::Strikethrough => in_strike = true,
                Tag::Link { .. } => {}
                _ => {}
            },
            Event::End(tag) => match tag {
                TagEnd::CodeBlock => {
                    spans.push(Span::styled(
                        code_text.clone(),
                        Style::default()
                            .fg(Color::Yellow)
                            .bg(Color::Rgb(30, 30, 30))
                            .add_modifier(Modifier::BOLD),
                    ));
                    in_code = false;
                    code_text.clear();
                }
                TagEnd::Strong => in_bold = false,
                TagEnd::Emphasis => in_italic = false,
                TagEnd::Strikethrough => in_strike = false,
                _ => {}
            },
            Event::Rule => {
                spans.push(Span::styled("─".repeat(40), Style::default().fg(Color::Gray)));
            }
            _ => {}
        }
    }
    if !spans.is_empty() {
        Some(spans)
    } else {
        None
    }
}

/// Split any span that embeds newlines (e.g. a fenced code block) into one
/// span per display row, so each entry in the returned list occupies exactly
/// one terminal line. Keeps vertical scroll math exact in `flatten_messages`.
fn expand_multiline_spans(spans: Vec<Span<'static>>) -> Vec<Span<'static>> {
    let mut out = Vec::new();
    for span in spans {
        if span.content.contains('\n') {
            for piece in span.content.split('\n') {
                let mut line = span.clone();
                line.content = piece.to_string().into();
                out.push(line);
            }
        } else {
            out.push(span);
        }
    }
    out
}

/// Truncate a string to at most `max` display columns, keeping the head and
/// appending an ellipsis on overflow. Operates on grapheme clusters so
/// combining marks and wide (CJK) characters never split mid-glyph.
fn truncate_str(s: &str, max: usize) -> String {
    if max == 0 || display_width(s) <= max {
        return s.to_string();
    }
    let target = max.saturating_sub(1);
    let mut used = 0usize;
    let mut end = 0usize;
    for (idx, grapheme) in s.grapheme_indices(true) {
        let grapheme_width = display_width(grapheme);
        if used + grapheme_width > target {
            break;
        }
        used += grapheme_width;
        end = idx + grapheme.len();
    }
    let mut out: String = s[..end].to_string();
    out.push('…');
    out
}

/// Collapse a status message to a single display line: newlines/carriage
/// returns become spaces and repeated whitespace is squeezed, so the fixed
/// status row can never render as stacked lines (e.g. fallback error chains).
fn sanitize_status(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_ws = false;
    for ch in s.chars() {
        if ch == '\n' || ch == '\r' || ch == '\t' || ch == ' ' {
            if !prev_ws {
                out.push(' ');
                prev_ws = true;
            }
        } else {
            out.push(ch);
            prev_ws = false;
        }
    }
    out
}

fn format_elapsed(d: Duration) -> String {
    let s = d.as_secs();
    if s < 60 {
        format!("{s}s")
    } else if s < 3600 {
        format!("{}m {:02}s", s / 60, s % 60)
    } else {
        format!("{}h {:02}m", s / 3600, (s % 3600) / 60)
    }
}

fn enable_raw_mode() -> Result<(), Box<dyn Error>> {
    crossterm::terminal::enable_raw_mode()?;
    Ok(())
}

fn disable_raw_mode() -> Result<(), Box<dyn Error>> {
    crossterm::terminal::disable_raw_mode()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_keeps_short_strings() {
        assert_eq!(truncate_str("short", 22), "short");
    }

    #[test]
    fn truncate_keeps_head_and_appends_ellipsis() {
        let out = truncate_str("abcdefghijklmnop", 5);
        assert!(out.ends_with('…'));
        assert_eq!(out.chars().count(), 5);
        assert!(out.starts_with("abcd"));
    }

    #[test]
    fn truncate_handles_unicode() {
        let out = truncate_str("olá mundo feliz", 6);
        assert_eq!(out.chars().count(), 6);
        assert!(out.starts_with("olá m"));
    }

    #[test]
    fn truncate_respects_display_width_of_wide_chars() {
        // "界" occupies 2 columns: budget of 4 fits "a界" plus ellipsis.
        let out = truncate_str("a界bc", 4);
        assert_eq!(out, "a界…");
        assert_eq!(display_width(&out), 4);
    }

    #[test]
    fn format_elapsed_seconds() {
        assert_eq!(format_elapsed(Duration::from_secs(42)), "42s");
    }

    #[test]
    fn format_elapsed_minutes() {
        assert_eq!(format_elapsed(Duration::from_secs(125)), "2m 05s");
    }

    #[test]
    fn format_elapsed_hours() {
        assert_eq!(format_elapsed(Duration::from_secs(7200)), "2h 00m");
    }

    #[test]
    fn display_role_maps_user_and_assistant() {
        assert_eq!(display_role("user"), "User");
        assert_eq!(display_role("assistant"), "Assistant");
        assert_eq!(display_role("system"), "system");
    }

    #[test]
    fn sanitize_status_removes_newlines() {
        assert_eq!(sanitize_status("a\nb"), "a b");
        assert_eq!(sanitize_status("a\r\nb"), "a b");
        assert_eq!(sanitize_status("a\tb"), "a b");
        assert_eq!(sanitize_status("a\n\nb"), "a b");
        assert_eq!(sanitize_status(" a b "), " a b ");
    }

    #[test]
    fn expand_multiline_spans_splits_code_blocks() {
        let spans = vec![
            Span::raw("before"),
            Span::raw("let x = 1;\nlet y = 2;\nlet z = 3;"),
            Span::raw("after"),
        ];
        let out = expand_multiline_spans(spans);
        let texts: Vec<&str> = out.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(
            texts,
            vec!["before", "let x = 1;", "let y = 2;", "let z = 3;", "after"]
        );
        assert!(out.iter().all(|s| !s.content.contains('\n')));
    }

    #[test]
    fn feed_tool_delta_accumulates_on_one_line() {
        let mut app = App::new("model", "off");
        app.feed_tool_delta(0, "edit_file", "{\"path\":");
        app.feed_tool_delta(0, "?", "\"a.rs\",");
        app.feed_tool_delta(0, "?", "\"new\":1}");
        assert_eq!(app.messages.len(), 1);
        assert_eq!(app.messages[0].0, "Tool");
        assert_eq!(app.messages[0].1, "Δ edit_file[0] {\"path\":\"a.rs\",\"new\":1}");
    }

    #[test]
    fn feed_tool_delta_starts_new_line_for_new_index() {
        let mut app = App::new("model", "off");
        app.feed_tool_delta(0, "read_file", "a");
        app.feed_tool_delta(1, "edit_file", "b");
        assert_eq!(app.messages.len(), 2);
        assert!(app.messages[0].1.starts_with("Δ read_file[0]"));
        assert!(app.messages[1].1.starts_with("Δ edit_file[1]"));
    }

    #[test]
    fn feed_text_delta_accumulates_one_live_message() {
        let mut app = App::new("model", "off");
        app.feed_text_delta("Hello ");
        app.feed_text_delta("world");
        assert_eq!(app.messages.len(), 1);
        assert_eq!(app.messages[0].0, "Assistant");
        assert_eq!(app.messages[0].1, "Hello world");
        assert!(app.streaming_assistant);
    }

    #[test]
    fn feed_text_delta_recovers_from_stale_stream_flag() {
        let mut app = App::new("model", "off");
        app.streaming_assistant = true;
        app.feed_text_delta("fresh");
        assert_eq!(app.messages.len(), 1);
        assert_eq!(app.messages[0].1, "fresh");
    }

    #[test]
    fn end_streaming_replaces_live_message_once() {
        let mut app = App::new("model", "off");
        app.feed_text_delta("partial ");
        app.feed_text_delta("text");
        let replaced = app.end_streaming(Some("final answer"));
        assert!(replaced);
        assert_eq!(app.messages.len(), 1);
        assert_eq!(app.messages[0].1, "final answer");
        assert!(!app.streaming_assistant);
        assert!(!app.end_streaming(Some("again")));
    }

    #[test]
    fn end_streaming_without_stream_noops() {
        let mut app = App::new("model", "off");
        assert!(!app.end_streaming(Some("x")));
        assert!(app.messages.is_empty());
    }

    #[test]
    fn file_search_ranks_and_selects() {
        let mut app = App::new("model", "off");
        app.open_file_search(vec![
            "src/main.rs".to_string(),
            "tests/main_test.rs".to_string(),
            "docs/README.md".to_string(),
        ]);
        assert!(app.file_search);
        app.file_search_query.push_str("s");
        app.refresh_file_search();
        assert_eq!(app.file_search_results.len(), 3);
        assert_eq!(app.file_search_results[0].path, "src/main.rs");
        app.move_file_search_selection(false);
        assert_eq!(app.file_search_selected, 1);
        app.move_file_search_selection(false);
        assert_eq!(app.file_search_selected, 2);
        app.move_file_search_selection(false);
        assert_eq!(app.file_search_selected, 2);
        app.move_file_search_selection(true);
        assert_eq!(app.file_search_selected, 1);
    }

    #[test]
    fn open_editor_loads_lines_and_focus() {
        let mut app = App::new("model", "off");
        app.open_editor("src/main.rs", Some("fn main() {}\n// end"));
        assert!(app.focus == Focus::Editor);
        assert_eq!(app.editor_file.as_deref(), Some("src/main.rs"));
        assert_eq!(app.editor_lines, vec!["fn main() {}", "// end"]);
        assert_eq!(app.editor_row, 0);
        assert_eq!(app.editor_col, 0);
        assert!(!app.editor_dirty);
    }

    #[test]
    fn open_editor_handles_missing_content() {
        let mut app = App::new("model", "off");
        app.open_editor("empty.txt", None);
        assert!(app.focus == Focus::Editor);
        assert!(app.editor_lines.is_empty());
    }

    #[test]
    fn git_branch_reports_no_git_outside_repo() {
        let dir = std::env::temp_dir();
        assert_eq!(git_branch(&dir), "no git");
    }

    fn test_app_state_router() -> (App, Arc<Mutex<AgentState>>, LlmRouter) {
        let mut cfg = crate::config::settings::Config::default();
        let base = std::env::temp_dir().join(format!("anamnesic-ui-{}", std::process::id()));
        cfg.workspace_dir = base.join("workspace");
        cfg.memory_dir = base.join("memory");
        let state = Arc::new(Mutex::new(
            crate::agent::state::AgentState::new(cfg).unwrap(),
        ));
        let app = App::new(&state.lock().unwrap().config.coder_model, "off");
        let router = LlmRouter::new(crate::llm::client::LlmClient::ollama(
            "http://localhost:11434",
        ));
        (app, state, router)
    }

    #[test]
    fn flatten_messages_user_prefix_and_assistant_bullet() {
        let mut app = App::new("model", "off");
        app.add_message("User", "fix the bug");
        app.add_message("Assistant", "Done.");
        let lines = flatten_messages(&app);
        let texts: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
        assert!(texts[0].starts_with("› fix the bug"));
        assert_eq!(texts[1], "");
        assert!(texts[2].starts_with("• Done."));
    }

    #[test]
    fn flatten_messages_status_cells_are_subdued() {
        let mut app = App::new("model", "off");
        app.add_message("System", "ready");
        app.add_message("Error", "boom");
        app.add_message("Tool", "edit_file — changed src/main.rs");
        app.tool_calls_expanded = true;
        let lines = flatten_messages(&app);
        let texts: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
        assert_eq!(texts[0], "ready");
        assert!(texts[2].starts_with("✗ boom"));
        assert!(texts[4].contains("edit_file — changed src/main.rs"));
    }

    #[test]
    fn flatten_messages_collapses_tool_calls_when_expanded_is_false() {
        let mut app = App::new("model", "off");
        app.tool_calls_expanded = false;
        app.add_message("Tool", "edit_file — changed src/main.rs");
        app.add_message("Tool", "run_tests — all passed");
        app.add_message("Tool", "bash — exit 0");
        let lines = flatten_messages(&app);
        let texts: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
        // All 3 tool calls should be collapsed into a single rollup line.
        let tool_lines: Vec<_> = texts.iter().filter(|t| t.starts_with("↳")).collect();
        assert_eq!(tool_lines.len(), 1);
        assert!(tool_lines[0].contains("3 tool uses"));
        assert!(tool_lines[0].contains("edit_file"));
    }

    #[test]
    fn extract_tool_name_handles_delta_and_completed_formats() {
        assert_eq!(extract_tool_name("Δ edit_file[0] {\"path\": \"a.rs\"}"), "edit_file");
        assert_eq!(extract_tool_name("Δ run_tests[3] all passed"), "run_tests");
        assert_eq!(extract_tool_name("read_file — read src/main.rs"), "read_file");
        assert_eq!(extract_tool_name("bash"), "bash");
    }

    #[test]
    fn flatten_messages_rollup_with_streaming_delta_does_not_panic() {
        let mut app = App::new("model", "off");
        app.tool_calls_expanded = false;
        app.add_message("Tool", "read_file — read src/main.rs");
        app.feed_tool_delta(0, "edit_file", "{\"path\":\"a.rs\"}");
        let lines = flatten_messages(&app);
        let texts: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
        let tool_lines: Vec<_> = texts.iter().filter(|t| t.starts_with("↳")).collect();
        assert_eq!(tool_lines.len(), 1);
        assert!(tool_lines[0].contains("2 tool uses"));
    }

    #[test]
    fn markdown_paragraphs_split_into_separate_lines() {
        let mut app = App::new("model", "off");
        app.add_message("Assistant", "first para\n\nsecond para");
        let lines = flatten_messages(&app);
        let texts: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
        assert_eq!(texts[0], "• first para");
        assert!(texts[1].trim().is_empty());
        assert_eq!(texts[2], "  second para");
    }

    #[test]
    fn info_command_opens_popup() {
        let (mut app, state, router) = test_app_state_router();
        let handled = handle_slash_command("/info", &mut app, &state, &router);
        assert!(handled);
        assert!(app.info_popup);
        assert!(!app.info_sections.is_empty());
    }

    #[test]
    fn selecting_cloud_model_drives_entire_lifecycle_to_cloud() {
        let (mut app, state, router) = test_app_state_router();
        set_active_model(&mut app, &state, &router, "z-ai/glm-5.2 [cloud]");
        let st = state.lock().unwrap();
        assert_eq!(st.config.coder_model, "z-ai/glm-5.2");
        assert_eq!(st.config.planner_model, "z-ai/glm-5.2");
        assert_eq!(st.config.summarizer_model, "z-ai/glm-5.2");
        assert_eq!(app.model, "z-ai/glm-5.2");
        assert!(router.is_cloud_model("z-ai/glm-5.2"));
        match router.client_for("z-ai/glm-5.2") {
            Ok(_) => panic!("expected cloud model to fail without configured provider"),
            Err(e) => assert!(e.to_string().contains("no cloud provider is configured")),
        }
    }

    #[test]
    fn selecting_local_model_keeps_lifecycle_on_ollama() {
        let (mut app, state, router) = test_app_state_router();
        set_active_model(&mut app, &state, &router, "qwen3:1.7b");
        let st = state.lock().unwrap();
        assert_eq!(st.config.coder_model, "qwen3:1.7b");
        assert_eq!(st.config.planner_model, "qwen3:1.7b");
        assert_eq!(st.config.summarizer_model, "qwen3:1.7b");
        assert!(!router.is_cloud_model("qwen3:1.7b"));
        match router.client_for("qwen3:1.7b").unwrap() {
            crate::llm::client::LlmClient::Ollama(_) => {}
            _ => panic!("expected local Ollama client for local model"),
        }
    }

    #[test]
    fn planner_summarizer_default_to_local_models() {
        let (mut app, state, router) = test_app_state_router();
        set_active_model(&mut app, &state, &router, "z-ai/glm-5.2 [cloud]");
        // Switching back to a local model must also move planner/summarizer.
        set_active_model(&mut app, &state, &router, "granite3.3:2b");
        let st = state.lock().unwrap();
        assert_eq!(st.config.planner_model, "granite3.3:2b");
        assert_eq!(st.config.coder_model, "granite3.3:2b");
        assert_eq!(st.config.summarizer_model, "granite3.3:2b");
        assert!(!router.is_cloud_model("granite3.3:2b"));
    }

    #[test]
    fn tool_call_delta_event_is_handled() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&events);
        let hooks = AgentHooks {
            on_event: Some(Arc::new(move |ev| captured.lock().unwrap().push(ev))),
            on_tool_call_delta: None,
            on_text_delta: None,
            on_approval: None,
            interrupt: None,
        };
        hooks.emit(AgentEvent::ToolCallDelta {
            index: 0,
            name: Some("read_file".into()),
            args_delta: "{\"path\":\"/tmp".into(),
        });
        let captured = events.lock().unwrap();
        assert_eq!(captured.len(), 1);
    }

    #[test]
    fn feed_reasoning_delta_flushes_at_threshold() {
        let mut app = App::new("model", "off");
        let chunk = "x".repeat(120);
        app.feed_reasoning_delta(&chunk);
        assert_eq!(app.reasoning.len(), 0);
        assert_eq!(app.messages.len(), 1);
        assert_eq!(app.messages[0].0, "Thinking");
        assert_eq!(app.messages[0].1, chunk);
    }

    #[test]
    fn feed_reasoning_delta_accumulates_below_threshold() {
        let mut app = App::new("model", "off");
        app.feed_reasoning_delta("hello ");
        app.feed_reasoning_delta("world");
        assert_eq!(app.reasoning, "hello world");
        assert_eq!(app.messages.len(), 0);
    }

    #[test]
    fn end_streaming_inserts_reasoning_tail_before_assistant() {
        let mut app = App::new("model", "off");
        app.streaming_assistant = true;
        app.messages.push(("Assistant".to_string(), "streamed partial".to_string()));
        app.reasoning.push_str("tail reasoning");
        let finalized = app.end_streaming(Some("final content"));
        assert!(finalized);
        assert_eq!(app.messages.len(), 2);
        assert_eq!(app.messages[0].0, "Thinking");
        assert_eq!(app.messages[0].1, "tail reasoning");
        assert_eq!(app.messages[1].0, "Assistant");
        assert_eq!(app.messages[1].1, "final content");
    }

    #[test]
    fn end_streaming_appends_to_existing_thinking() {
        let mut app = App::new("model", "off");
        app.streaming_assistant = true;
        app.messages.push(("Assistant".to_string(), "streamed partial".to_string()));
        app.messages.push(("Thinking".to_string(), "existing thinking".to_string()));
        app.reasoning.push_str(" more tail");
        let finalized = app.end_streaming(Some("final content"));
        assert!(finalized);
        assert_eq!(app.messages.len(), 2);
        assert_eq!(app.messages[0].0, "Thinking");
        assert_eq!(app.messages[0].1, "existing thinking more tail");
        assert_eq!(app.messages[1].0, "Assistant");
        assert_eq!(app.messages[1].1, "final content");
    }

    #[test]
    fn end_streaming_no_content_only_flushes_reasoning() {
        let mut app = App::new("model", "off");
        app.streaming_assistant = true;
        app.messages.push(("Assistant".to_string(), "streamed partial".to_string()));
        app.reasoning.push_str("tail reasoning");
        let finalized = app.end_streaming(None);
        assert!(finalized);
        assert_eq!(app.messages.len(), 2);
        assert_eq!(app.messages[0].0, "Thinking");
        assert_eq!(app.messages[0].1, "tail reasoning");
        assert_eq!(app.messages[1].0, "Assistant");
        assert_eq!(app.messages[1].1, "streamed partial");
    }

    #[test]
    fn reset_reasoning_flushes_tail() {
        let mut app = App::new("model", "off");
        app.reasoning.push_str("small tail");
        app.reset_reasoning();
        assert!(app.reasoning.is_empty());
        assert_eq!(app.messages.len(), 1);
        assert_eq!(app.messages[0].0, "Thinking");
        assert_eq!(app.messages[0].1, "small tail");
    }

    #[test]
    fn reset_reasoning_no_op_when_empty() {
        let mut app = App::new("model", "off");
        app.reset_reasoning();
        assert!(app.reasoning.is_empty());
        assert_eq!(app.messages.len(), 0);
    }

    #[test]
    fn end_streaming_early_return_does_not_lose_reasoning() {
        let mut app = App::new("model", "off");
        app.streaming_assistant = false;
        app.reasoning.push_str("orphaned tail");
        let finalized = app.end_streaming(Some("final content"));
        assert!(!finalized);
        assert_eq!(app.messages.len(), 1);
        assert_eq!(app.messages[0].0, "Thinking");
        assert_eq!(app.messages[0].1, "orphaned tail");
    }

    #[test]
    fn end_streaming_handles_multiple_thinking_flushes_after_partial() {
        let mut app = App::new("model", "off");
        app.streaming_assistant = true;
        app.messages.push(("Assistant".to_string(), "streamed partial".to_string()));
        app.messages.push(("Thinking".to_string(), "flush A".to_string()));
        app.messages.push(("Thinking".to_string(), "flush B".to_string()));
        let finalized = app.end_streaming(Some("final content"));
        assert!(finalized);
        assert_eq!(app.messages.len(), 3);
        assert_eq!(app.messages[0].0, "Thinking");
        assert_eq!(app.messages[0].1, "flush A");
        assert_eq!(app.messages[1].0, "Thinking");
        assert_eq!(app.messages[1].1, "flush B");
        assert_eq!(app.messages[2].0, "Assistant");
        assert_eq!(app.messages[2].1, "final content");
    }

    #[test]
    fn end_streaming_without_assistant_pushes_final_content() {
        let mut app = App::new("model", "off");
        app.streaming_assistant = true;
        app.messages.push(("Thinking".to_string(), "reasoning only".to_string()));
        let finalized = app.end_streaming(Some("final content"));
        assert!(finalized);
        assert_eq!(app.messages.len(), 2);
        assert_eq!(app.messages[0].0, "Thinking");
        assert_eq!(app.messages[0].1, "reasoning only");
        assert_eq!(app.messages[1].0, "Assistant");
        assert_eq!(app.messages[1].1, "final content");
    }

    #[test]
    fn char_index_to_byte_index_maps_char_to_byte_offsets() {
        let s = "o que é";
        assert_eq!(char_index_to_byte_index(s, 0), 0);
        assert_eq!(char_index_to_byte_index(s, 6), 6);
        assert_eq!(char_index_to_byte_index(s, 7), s.len());
        assert_eq!(char_index_to_byte_index("", 0), 0);
    }

    #[test]
    fn insert_char_after_multibyte_uses_byte_offset() {
        let mut input = String::new();
        let mut cursor = 0usize;
        for c in "o que é".chars() {
            let pos = char_index_to_byte_index(&input, cursor);
            input.insert(pos, c);
            cursor += 1;
        }
        assert_eq!(input, "o que é");
        assert_eq!(cursor, input.chars().count());

        let pos = char_index_to_byte_index(&input, cursor);
        input.insert(pos, '?');
        assert_eq!(input, "o que é?");
    }

    #[test]
    fn backspace_after_multibyte_removes_char_not_byte() {
        let mut input = "o que é".to_string();
        let mut cursor = input.chars().count();
        let pos = char_index_to_byte_index(&input, cursor - 1);
        input.remove(pos);
        cursor -= 1;
        assert_eq!(input, "o que ");
        assert_eq!(cursor, 6);
    }

    #[test]
    fn previous_input_cursor_at_char_count_for_multibyte() {
        let mut app = App::new("model", "off");
        app.input_history.push("o que é".to_string());
        app.previous_input();
        assert_eq!(app.input, "o que é");
        assert_eq!(app.cursor_position, "o que é".chars().count());
    }

    #[test]
    fn count_wrapped_lines_counts_visual_rows() {
        use ratatui::text::Line;
        use ratatui::text::Span;
        // A 100-char unbroken run wraps to 2 visual rows at width 80
        // (80 chars on the first row, the remaining 20 on the second).
        let lines = vec![Line::from(Span::raw("x".repeat(100)))];
        assert_eq!(count_wrapped_lines(&lines, 80), 2);
    }

    #[test]
    fn count_wrapped_lines_matches_ratatui_for_prefixed_lines() {
        use ratatui::text::Line;
        use ratatui::text::Span;
        // Prefix span + content span are concatenated WITHOUT a separator,
        // so "• " + 78 chars is exactly one visual row at width 80. A
        // hand-rolled counter inserting a virtual space would count 2.
        let exact = Line::from(vec![Span::raw("• "), Span::raw("x".repeat(78))]);
        assert_eq!(count_wrapped_lines(&[exact], 80), 1);

        // Multi-word content wraps by words, not by whole-span units.
        let words: String = "hello world ".repeat(20);
        let content = words.trim_end().to_string();
        let line = Line::from(vec![Span::raw("• "), Span::raw(content)]);
        let lines = vec![line];
        let width = 80;
        let expected = Paragraph::new(lines.clone())
            .wrap(Wrap { trim: true })
            .line_count(width as u16);
        assert_eq!(count_wrapped_lines(&lines, width), expected);
        assert!(expected >= 3);
    }

    #[test]
    fn count_wrapped_lines_counts_multiple_lines() {
        use ratatui::text::Line;
        use ratatui::text::Span;
        let lines = vec![
            Line::from(Span::raw("x".repeat(80))),
            Line::from(Span::raw("y".repeat(80))),
        ];
        assert_eq!(count_wrapped_lines(&lines, 80), 2);
    }

    #[test]
    fn reflow_renders_long_line_within_width() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;
        let mut app = App::new("model", "off");
        app.streaming_assistant = false;
        app.messages.push(("Assistant".to_string(), "x".repeat(200)));
        let backend = TestBackend::new(40, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|f| {
                let area = f.area();
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Min(1), Constraint::Length(3)])
                    .split(area);
                let messages_lines = flatten_messages(&app);
                let widget = Paragraph::new(messages_lines).wrap(Wrap { trim: true });
                f.render_widget(widget, chunks[0]);
            })
            .unwrap();
        let buf = terminal.backend().buffer();
        // The long line should be reflowed, not cut off at column 40.
        // At least one row beyond the first should contain 'x'.
        let mut found_x_beyond_row0 = false;
        for y in 1..10 {
            for x in 0..40 {
                if buf.cell((x, y)).is_some_and(|c| c.symbol() == "x") {
                    found_x_beyond_row0 = true;
                    break;
                }
            }
            if found_x_beyond_row0 {
                break;
            }
        }
        assert!(found_x_beyond_row0, "reflow did not wrap long line");
    }
}
