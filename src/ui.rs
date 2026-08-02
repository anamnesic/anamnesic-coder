use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers, MouseButton, MouseEventKind},
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

const SPINNER: [char; 4] = ['|', '/', '-', '\\'];

/// Available slash commands, shown in the interactive picker (modern-harness style).
const SLASH_COMMANDS: &[(&str, &str)] = &[
    ("/help", "Show help"),
    ("/status", "Show model, provider, directory, context tokens"),
    (
        "/model",
        "Select the active model (no arg = pick from list)",
    ),
    ("/models", "List available models"),
    (
        "/provider",
        "Select cloud model provider (no arg = pick from list)",
    ),
    ("/reset", "Reset session"),
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
            status: "Ready · Enter to send · Tab: mode · ↑/↓ history · PgUp/PgDn scroll · mouse wheel · Esc quit"
                .into(),
            scroll_offset: 0,
            follow: true,
            model: model.to_string(),
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
        }
    }

    pub fn add_message(&mut self, role: &str, content: &str) {
        self.messages.push((role.to_string(), content.to_string()));
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
        self.cursor_position = self.input.len();
        self.history_index = Some(index);
    }

    fn next_input(&mut self) {
        let Some(index) = self.history_index else {
            return;
        };
        if index + 1 < self.input_history.len() {
            self.input = self.input_history[index + 1].clone();
            self.cursor_position = self.input.len();
            self.history_index = Some(index + 1);
        } else {
            self.clear_input();
            self.history_index = None;
        }
    }
}

/// Handle TUI slash commands (e.g. /models). Returns true if the command was
/// handled locally and should not be sent to the agent.
fn handle_slash_command(
    input: &str,
    app: &mut App,
    state: &Arc<Mutex<AgentState>>,
    router: &LlmRouter,
) -> bool {
    let cmd = input.split_whitespace().next().unwrap_or("");
    match cmd {
        "/models" => {
            let st = state.lock().unwrap();
            let local = crate::llm::model_resolver::list_models(&st.config.models_dir);
            let dir = st.config.models_dir.clone();
            drop(st);
            let provider = app.provider.clone();
            let cloud: Vec<String> = crate::models_dev::ModelsDevClient::load()
                .provider_models(&provider)
                .into_iter()
                .filter(|m| m.tool_call && m.modalities.output.iter().any(|o| o == "text"))
                .map(|m| format!("{} [cloud]", crate::models_dev::base_id(&m.id)))
                .collect();
            let cloud = unique_model_ids(cloud);
            if local.is_empty() && cloud.is_empty() {
                app.add_message(
                    "System",
                    &format!(
                        "No models found in {} or for provider {} in the models.dev catalog.",
                        dir.display(),
                        provider
                    ),
                );
            } else {
                let mut out = String::from("Local models:");
                if local.is_empty() {
                    out.push_str(" (none)");
                }
                for m in &local {
                    out.push_str("\n  ");
                    out.push_str(m);
                }
                out.push_str(&format!("\nCloud models ({provider}):"));
                if cloud.is_empty() {
                    out.push_str(" (none)");
                }
                for m in &cloud {
                    out.push_str("\n  ");
                    out.push_str(m);
                }
                app.add_message("System", &out);
            }
            app.status = "Ready · Enter to send · ↑/↓ history · PgUp/PgDn scroll · mouse wheel · Esc quit".into();
            true
        }
        "/reset" => {
            state.lock().unwrap().reset();
            app.messages.clear();
            app.add_message("System", "Session reset.");
            app.status = "Ready · Enter to send · ↑/↓ history · PgUp/PgDn scroll · mouse wheel · Esc quit".into();
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
                let cloud: Vec<String> = crate::models_dev::ModelsDevClient::load()
                    .provider_models(&provider)
                    .into_iter()
                    .filter(|m| m.tool_call && m.modalities.output.iter().any(|o| o == "text"))
                    .map(|m| format!("{} [cloud]", crate::models_dev::base_id(&m.id)))
                    .collect();
                let mut items: Vec<String> = local.clone();
                items.extend(unique_model_ids(cloud));
                items.dedup();
                if items.is_empty() {
                    app.add_message(
                        "System",
                        &format!("No models found in {}. models.dev catalog also empty (offline?). Use /model <name> to set one anyway.", dir.display()),
                    );
                    app.status =
                        "Ready · Enter to send · ↑/↓ history · PgUp/PgDn scroll · mouse wheel · Esc quit".into();
                } else {
                    app.model_items = items;
                    app.model_selected = app
                        .model_items
                        .iter()
                        .position(|m| m.trim_end_matches(" [cloud]") == app.model)
                        .unwrap_or(0);
                    app.model_selector = true;
                }
            } else {
                set_active_model(app, state, router, arg);
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

/// Set the active coder model for subsequent agent turns.  Strips the
/// " [cloud]" picker suffix and tells the router whether the model is a cloud
/// model so requests go to the right backend.
fn set_active_model(app: &mut App, state: &Arc<Mutex<AgentState>>, router: &LlmRouter, name: &str) {
    let clean = name.trim_end_matches(" [cloud]").to_string();
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
    app.status = format!("Ready · model: {clean} · Esc quit");
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
                format!("Ready · provider: {name} · run /models or /model to list its models");
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
    let approval_tx_clone = approval_tx.clone();
    let approval_rx_clone = Arc::clone(approval_rx);
    let agent_mode = app.agent_mode;
    thread::spawn(move || {
        let hooks = AgentHooks {
            on_event: Some(Arc::new(move |ev| {
                let _ = agent_tx_clone.send(ev);
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
        let files = state.lock().unwrap().files.list_files("");
        let mut a = app.lock().unwrap();
        a.sidebar_items = files;
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
                    AgentEvent::Status(text) => a.add_message("System", &text),
                    AgentEvent::ToolCall { name, summary } => {
                        let s: String = summary.chars().take(120).collect();
                        a.add_message("Tool", &format!("{name} — {s}"));
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
                    AgentEvent::Done { message } => {
                        a.add_message("Assistant", &message);
                        a.loading = false;
                        a.status =
                            "Ready · Enter to send · ↑/↓ history · PgUp/PgDn scroll · mouse wheel · Esc quit"
                                .into();
                    }
                    AgentEvent::Transaction { action, summary } => {
                        a.add_message("Workspace", &format!("[{action}] {summary}"));
                    }
                    AgentEvent::Failed { message } => {
                        a.add_message("Error", &message);
                        a.loading = false;
                        a.status = "Failed · Enter to retry · Esc quit".into();
                    }
                    AgentEvent::Interrupted => {
                        a.add_message("System", "Turn interrupted by user.");
                        a.loading = false;
                        a.status = "Interrupted · Enter to send · Esc quit".into();
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
        // Refresh context budget + elapsed + spinner from shared state.
        // Do not block rendering while the worker owns AgentState for a turn.
        if let Ok(st) = state.try_lock() {
            let mut a = app.lock().unwrap();
            a.tokens = st.session.estimated_tokens();
        }
        {
            let mut a = app.lock().unwrap();
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
                let mut guard = app.lock().unwrap();
                // The approval modal owns input until the user decides.
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
                        guard.status = "Working… (Esc to interrupt)".into();
                        let _ = decision_tx.send(decision);
                    }
                    continue;
                }
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('l') {
                    guard.messages.clear();
                    guard.status = "Chat cleared (session memory is preserved).".into();
                    continue;
                }
                // Modal pickers (slash-command menu / model selector) take over
                // Up/Down/Enter/Esc while open.
                if guard.command_menu || guard.model_selector || guard.provider_selector {
                    let mut handled = true;
                    match key.code {
                        KeyCode::Up => {
                            if guard.command_menu {
                                guard.command_selected = guard.command_selected.saturating_sub(1);
                            } else if guard.model_selector {
                                guard.model_selected = guard.model_selected.saturating_sub(1);
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
                            }
                        }
                        KeyCode::Esc => {
                            guard.command_menu = false;
                            guard.model_selector = false;
                            guard.provider_selector = false;
                        }
                        _ => handled = false,
                    }
                    if handled || guard.model_selector || guard.provider_selector {
                        continue;
                    }
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
                            "Ready · mode: {} · Esc quit",
                            match guard.agent_mode {
                                AgentMode::Agent => "agent",
                                AgentMode::Plan => "plan",
                            }
                        );
                    }
                    KeyCode::Up => {
                        if let Focus::Sidebar = guard.focus {
                            if guard.sidebar_selected > 0 {
                                guard.sidebar_selected -= 1;
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
                            if guard.sidebar_selected + 1 < guard.sidebar_items.len() {
                                guard.sidebar_selected += 1;
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
                            Focus::Input => {
                                if guard.loading {
                                    continue;
                                }
                                let input = guard.input.trim().to_string();
                                if !input.is_empty() {
                                    guard.command_menu = false;
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
                                if let Some(fname) = guard.sidebar_items.get(guard.sidebar_selected)
                                {
                                    let st = state.lock().unwrap();
                                    if let Some(data) = st.files.read_file(fname) {
                                        // open editor with file contents
                                        guard.editor_file = Some(fname.clone());
                                        guard.editor_lines =
                                            data.lines().map(|s| s.to_string()).collect();
                                        if guard.editor_lines.is_empty() {
                                            guard.editor_lines.push(String::new());
                                        }
                                        guard.editor_row = 0;
                                        guard.editor_col = guard.editor_lines[0].len();
                                        guard.editor_dirty = false;
                                        guard.focus = Focus::Editor;
                                    } else {
                                        guard.add_message("Error", "Failed to read file");
                                    }
                                }
                            }
                            Focus::Editor => {
                                // insert newline at cursor
                                if guard.editor_row <= guard.editor_lines.len() {
                                    let line = if guard.editor_row < guard.editor_lines.len() {
                                        guard.editor_lines[guard.editor_row].clone()
                                    } else {
                                        String::new()
                                    };
                                    let (left, right) =
                                        line.split_at(guard.editor_col.min(line.len()));
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
                            let col = guard.editor_col;
                            let line = &mut guard.editor_lines[row];
                            if col <= line.len() {
                                line.insert(col, c);
                            } else {
                                line.push(c);
                            }
                            guard.editor_col += 1;
                            guard.editor_dirty = true;
                        } else {
                            let pos = guard.cursor_position;
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
                                    guard.editor_lines[row].remove(col - 1);
                                    guard.editor_col -= 1;
                                } else if row > 0 {
                                    // join with previous line
                                    let prev_len = guard.editor_lines[row - 1].len();
                                    let cur = guard.editor_lines.remove(row);
                                    guard.editor_row -= 1;
                                    guard.editor_col = prev_len;
                                    guard.editor_lines[row - 1].push_str(&cur);
                                }
                            }
                            guard.editor_dirty = true;
                        } else {
                            let pos = guard.cursor_position;
                            if pos > 0 {
                                guard.input.remove(pos - 1);
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
                                guard.editor_col = guard.editor_lines[row].len();
                            }
                        } else if guard.cursor_position > 0 {
                            guard.cursor_position -= 1;
                        }
                    }
                    KeyCode::Right => {
                        if guard.focus == Focus::Editor {
                            let row = guard.editor_row;
                            if row < guard.editor_lines.len() {
                                let len = guard.editor_lines[row].len();
                                if guard.editor_col < len {
                                    guard.editor_col += 1;
                                } else if row + 1 < guard.editor_lines.len() {
                                    guard.editor_row += 1;
                                    guard.editor_col = 0;
                                }
                            }
                        } else if guard.cursor_position < guard.input.len() {
                            guard.cursor_position += 1;
                        }
                    }
                    KeyCode::Esc => {
                        // If editing, close editor; else interrupt running turn or exit
                        if guard.focus == Focus::Editor {
                            guard.editor_file = None;
                            guard.editor_lines.clear();
                            guard.focus = Focus::Input;
                        } else if guard.loading {
                            interrupt.store(true, Ordering::Relaxed);
                            guard.status = "Interrupting… (Esc again to force quit)".into();
                        } else {
                            break;
                        }
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
                if matches!(mouse.kind, MouseEventKind::Down(MouseButton::Left)) {
                    if !guard.loading {
                        guard.focus = Focus::Messages;
                    }
                }
                match mouse.kind {
                    MouseEventKind::ScrollUp => {
                        guard.scroll_offset = guard.scroll_offset.saturating_sub(5);
                        guard.follow = false;
                    }
                    MouseEventKind::ScrollDown => {
                        guard.scroll_offset += 5;
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
            Span::styled(left_txt, Style::default().fg(Color::DarkGray)),
            Span::styled(" ".repeat(pad), Style::default()),
            Span::styled(right_txt, Style::default().fg(Color::DarkGray)),
        ]));
        f.render_widget(header, page[0]);
        // Thin status bar at the very bottom: dir · branch on the left,
        // spinner + elapsed on the right while a turn is running.
        let status_left = format!(" {} · {}", truncate_str(&app.dir, 48), app.git_branch);
        let status_right = if app.loading {
            format!(
                " {} {:<5}  (Esc interrupt) ",
                SPINNER[app.spinner_frame],
                format_elapsed(app.elapsed)
            )
        } else {
            " Esc quit ".into()
        };
        let status_pad = (size.width as usize)
            .saturating_sub(status_left.chars().count() + status_right.chars().count());
        let status = Paragraph::new(Line::from(vec![
            Span::styled(
                status_left,
                Style::default().fg(if app.loading {
                    Color::Yellow
                } else {
                    Color::DarkGray
                }),
            ),
            Span::styled(" ".repeat(status_pad), Style::default()),
            Span::styled(
                status_right,
                Style::default().fg(if app.loading {
                    Color::Yellow
                } else {
                    Color::DarkGray
                }),
            ),
        ]));
        f.render_widget(status, page[3]);

        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(28), Constraint::Percentage(72)].as_ref())
            .split(page[1]);

        // Sidebar (files)
        let sidebar_items: Vec<ListItem> = app
            .sidebar_items
            .iter()
            .map(|s| ListItem::new(Span::raw(s.clone())))
            .collect();
        let sidebar = List::new(sidebar_items)
            .block(
                Block::default()
                    .title(" Workspace files ")
                    .borders(Borders::ALL),
            )
            .highlight_style(Style::default().bg(Color::DarkGray))
            .highlight_symbol("▶ ");
        let mut list_state = ratatui::widgets::ListState::default();
        if !app.sidebar_items.is_empty() {
            list_state.select(Some(app.sidebar_selected));
        }
        f.render_stateful_widget(sidebar, cols[0], &mut list_state);

        // Right column: messages or editor (full height — input lives at the bottom bar).
        if app.editor_file.is_some() {
            let title = format!(
                "Editor - {}{}",
                app.editor_file.as_ref().unwrap(),
                if app.editor_dirty { " *" } else { "" }
            );
            // compute visible lines based on scroll and area height
            let area_height = cols[1].height as usize - 2; // leave room for borders
            let total_lines = app.editor_lines.len();
            let scroll = if app.editor_scroll + area_height > total_lines {
                if total_lines > area_height {
                    total_lines - area_height
                } else {
                    0
                }
            } else {
                app.editor_scroll
            };
            let end = std::cmp::min(scroll + area_height, total_lines);
            let visible = app.editor_lines[scroll..end].join("\n");
            let editor =
                Paragraph::new(visible).block(Block::default().title(title).borders(Borders::ALL));
            f.render_widget(editor, cols[1]);
            // set cursor in editor area relative to scroll
            let r = (app.editor_row.saturating_sub(scroll)) as u16;
            let c = app.editor_col as u16;
            f.set_cursor_position((cols[1].x + c + 1, cols[1].y + r + 1));
        } else {
            let messages_lines = flatten_messages(app);
            let view_height = (cols[1].height as usize).saturating_sub(2); // borders
            let total = messages_lines.len();
            let offset = if app.follow {
                total.saturating_sub(view_height)
            } else {
                let max = total.saturating_sub(1);
                app.scroll_offset.min(max)
            };
            let scroll_offset = offset.min(u16::MAX as usize) as u16;
            let messages_block = Block::default().title(" Chat ").borders(Borders::ALL);
            let messages_widget = Paragraph::new(messages_lines)
                .block(messages_block)
                .scroll((0, scroll_offset));
            f.render_widget(messages_widget, cols[1]);
        }

        // Input bar: bare prompt at the bottom (modern harness style, no box).
        let prompt = if app.loading { "▍" } else { "❯" };
        let input_line = Line::from(vec![
            Span::styled(
                format!("{prompt} "),
                Style::default()
                    .fg(if app.loading {
                        Color::DarkGray
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
        f.render_widget(input, page[2]);
        // Set cursor position next to the typed text.
        if !app.loading {
            f.set_cursor_position((page[2].x + 2 + app.cursor_position as u16, page[2].y));
        }

        // Overlays: slash-command picker / model selector (modal, like modern harness TUIs).
        if app.command_menu || app.model_selector || app.provider_selector {
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
                            Span::styled(format!("  {desc}"), Style::default().fg(Color::DarkGray)),
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
                .highlight_style(Style::default().bg(Color::DarkGray))
                .highlight_symbol("▶ ");
            f.render_stateful_widget(list, area, &mut list_state);
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
                    Style::default().fg(Color::DarkGray),
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

/// Flatten chat messages into renderable lines (handles multi-line content and
/// adds role labels). Mirrors the HistoryCell transcript rendering of modern
/// coding-agent TUIs.
fn flatten_messages(app: &App) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for (role, content) in &app.messages {
        let style = match role.as_str() {
            "User" => Style::default().fg(Color::Green),
            "System" => Style::default().fg(Color::Yellow),
            "Tool" => Style::default().fg(Color::Cyan),
            "Plan" => Style::default().fg(Color::LightMagenta),
            "Error" => Style::default().fg(Color::Red),
            "File" => Style::default().fg(Color::Cyan),
            _ => Style::default().fg(Color::Gray),
        };
        let label = format!("{role}: ");
        let mut first = true;
        for content_line in content.lines() {
            if first {
                lines.push(Line::from(vec![
                    Span::styled(label.clone(), Style::default().add_modifier(Modifier::BOLD)),
                    Span::styled(content_line.to_string(), style),
                ]));
                first = false;
            } else {
                lines.push(Line::from(vec![
                    Span::styled("       ", style),
                    Span::styled(content_line.to_string(), style),
                ]));
            }
        }
        lines.push(Line::from(""));
    }
    lines
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().skip(s.chars().count() - max).collect();
        out = format!("…{out}");
        out
    }
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
    fn truncate_keeps_tail_and_prefixes_ellipsis() {
        let out = truncate_str("abcdefghijklmnop", 5);
        assert!(out.starts_with('…'));
        assert_eq!(out.chars().count(), 6);
        assert!(out.ends_with("mnop"));
    }

    #[test]
    fn truncate_handles_unicode() {
        let out = truncate_str("olá mundo feliz", 6);
        assert_eq!(out.chars().count(), 7);
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
}
