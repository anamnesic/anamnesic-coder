use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Frame, Terminal,
};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, DisableLineWrap, EnableLineWrap},
};
use std::{
    error::Error,
    io,
    sync::mpsc,
    thread,
    time::Duration,
};
use std::sync::{Arc, Mutex};

use crate::llm::client::LlmClient;
use crate::agent::state::AgentState;

#[derive(PartialEq)]
pub enum Focus {
    Sidebar,
    Messages,
    Input,
    Editor,
}

pub struct App {
    pub messages: Vec<(String, String)>, // (role, content)
    pub input: String,
    pub cursor_position: usize,
    pub loading: bool,
    pub sidebar_items: Vec<String>,
    pub sidebar_selected: usize,
    pub focus: Focus,
    pub editor_file: Option<String>,
    pub editor_lines: Vec<String>,
    pub editor_row: usize,
    pub editor_col: usize,
    pub editor_dirty: bool,
    pub editor_scroll: usize,
}

impl App {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            input: String::new(),
            cursor_position: 0,
            loading: false,
            sidebar_items: Vec::new(),
            sidebar_selected: 0,
            focus: Focus::Input,
            editor_file: None,
            editor_lines: Vec::new(),
            editor_row: 0,
            editor_col: 0,
            editor_dirty: false,
            editor_scroll: 0,
        }
    }

    pub fn add_message(&mut self, role: &str, content: &str) {
        self.messages.push((role.to_string(), content.to_string()));
    }

    pub fn clear_input(&mut self) {
        self.input.clear();
        self.cursor_position = 0;
    }
}

pub fn run_ui(client: LlmClient, state: AgentState) -> Result<(), Box<dyn Error>> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app and event channels
    let app = Arc::new(Mutex::new(App::new()));
    let app_clone = app.clone();
    let state = Arc::new(Mutex::new(state));
    // populate sidebar with workspace files
    {
        let files = state.lock().unwrap().files.list_files("");
        let mut a = app.lock().unwrap();
        a.sidebar_items = files;
    }
    let (tx, rx) = mpsc::channel();

    // Spawn input handling thread
    thread::spawn(move || {
        let tx = tx.clone();
        loop {
            if let Ok(event) = event::read() {
                if let Err(_) = tx.send(event) {
                    break;
                }
            }
        }
    });

    // App loop
    {
        let mut init = app.lock().unwrap();
        init.add_message("System", "Welcome to the Ratatui UI. Type your message and press Enter.");
    }

    loop {
        {
            let guard = app.lock().unwrap();
            draw(&mut terminal, &guard)?;
        }

        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(Event::Key(key)) => {
                let mut guard = app.lock().unwrap();
                match key.code {
                    KeyCode::Tab => {
                        guard.focus = match guard.focus {
                            Focus::Sidebar => Focus::Messages,
                            Focus::Messages => Focus::Input,
                            Focus::Input => Focus::Sidebar,
                            Focus::Editor => Focus::Input,
                        };
                    }
                    KeyCode::Up => {
                        if let Focus::Sidebar = guard.focus {
                            if guard.sidebar_selected > 0 { guard.sidebar_selected -= 1; }
                        } else if guard.focus == Focus::Editor {
                            if guard.editor_row > 0 { guard.editor_row -= 1; }
                            let line_len = guard.editor_lines.get(guard.editor_row).map(|l| l.len()).unwrap_or(0);
                            if guard.editor_col > line_len { guard.editor_col = line_len; }
                            // adjust scroll
                            if guard.editor_row < guard.editor_scroll { guard.editor_scroll = guard.editor_row; }
                        }
                    }
                    KeyCode::Down => {
                        if let Focus::Sidebar = guard.focus {
                            if guard.sidebar_selected + 1 < guard.sidebar_items.len() { guard.sidebar_selected += 1; }
                        } else if guard.focus == Focus::Editor {
                            if guard.editor_row + 1 < guard.editor_lines.len() { guard.editor_row += 1; }
                            let line_len = guard.editor_lines.get(guard.editor_row).map(|l| l.len()).unwrap_or(0);
                            if guard.editor_col > line_len { guard.editor_col = line_len; }
                        }
                    }
                    KeyCode::Enter => {
                        match guard.focus {
                            Focus::Input => {
                                let input = guard.input.trim().to_string();
                                if !input.is_empty() {
                                    guard.add_message("User", &input);
                                    guard.loading = true;
                                    let input_clone = input.clone();
                                    let client_clone = client.clone();
                                    let state_clone = Arc::clone(&state);
                                    let app_clone = app.clone();
                                    thread::spawn(move || {
                                        let rt = tokio::runtime::Runtime::new().unwrap();
                                        let mut st = state_clone.lock().unwrap();
                                        rt.block_on(crate::agent::r#loop::run_agent_loop(
                                            &client_clone, &mut st, &input_clone,
                                        ));
                                        let mut a = app_clone.lock().unwrap();
                                        a.loading = false;
                                    });
                                    guard.clear_input();
                                }
                            }
                            Focus::Sidebar => {
                                if let Some(fname) = guard.sidebar_items.get(guard.sidebar_selected) {
                                    let st = state.lock().unwrap();
                                    if let Some(data) = st.files.read_file(fname) {
                                        // open editor with file contents
                                        guard.editor_file = Some(fname.clone());
                                        guard.editor_lines = data.lines().map(|s| s.to_string()).collect();
                                        if guard.editor_lines.is_empty() { guard.editor_lines.push(String::new()); }
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
                                    let (left, right) = line.split_at(guard.editor_col.min(line.len()));
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
                        }
                    }
                    KeyCode::Backspace => {
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
                            }
                        }
                    }
                    KeyCode::Left => {
                        if guard.focus == Focus::Editor {
                            if guard.editor_col > 0 { guard.editor_col -= 1; }
                            else if guard.editor_row > 0 {
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
                                if guard.editor_col < len { guard.editor_col += 1; }
                                else if row + 1 < guard.editor_lines.len() {
                                    guard.editor_row += 1;
                                    guard.editor_col = 0;
                                }
                            }
                        } else if guard.cursor_position < guard.input.len() {
                            guard.cursor_position += 1;
                        }
                    }
                    KeyCode::Esc => {
                        // if editing, close editor; else exit
                        if guard.focus == Focus::Editor {
                            guard.editor_file = None;
                            guard.editor_lines.clear();
                            guard.focus = Focus::Input;
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
                                match st.files.write_file(fname, &content) {
                                    Ok(_) => {
                                        guard.add_message("Info", "File saved");
                                        guard.editor_dirty = false;
                                    }
                                    Err(e) => { guard.add_message("Error", &format!("Save failed: {}", e)); }
                                }
                            }
                        }
                    }
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

fn draw<B: ratatui::backend::Backend>(f: &mut Terminal<B>, app: &App) -> Result<(), Box<dyn Error>> {
    f.draw(|mut f| {
        let size = f.size();
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .margin(1)
            .constraints([Constraint::Percentage(28), Constraint::Percentage(72)].as_ref())
            .split(size);

        // Sidebar (files)
        let sidebar_items: Vec<ListItem> = app
            .sidebar_items
            .iter()
            .map(|s| ListItem::new(Span::raw(s.clone())))
            .collect();
        let sidebar = List::new(sidebar_items)
            .block(Block::default().title("Files").borders(Borders::ALL))
            .highlight_style(Style::default().bg(Color::DarkGray))
            .highlight_symbol("▶ ");
        let mut list_state = ratatui::widgets::ListState::default();
        if !app.sidebar_items.is_empty() {
            list_state.select(Some(app.sidebar_selected));
        }
        f.render_stateful_widget(sidebar, cols[0], &mut list_state);

        // Right column: messages + input
        let right_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(0), Constraint::Length(3)].as_ref())
            .split(cols[1]);

        // Messages area or editor
            if app.editor_file.is_some() {
            let title = format!("Editor - {}{}", app.editor_file.as_ref().unwrap(), if app.editor_dirty { " *" } else { "" });
            // compute visible lines based on scroll and area height
            let area_height = right_chunks[0].height as usize - 2; // leave room for borders
            let total_lines = app.editor_lines.len();
            let scroll = if app.editor_scroll + area_height > total_lines {
                if total_lines > area_height { total_lines - area_height } else { 0 }
            } else {
                app.editor_scroll
            };
            let end = std::cmp::min(scroll + area_height, total_lines);
            let visible = app.editor_lines[scroll..end].join("\n");
            let editor = Paragraph::new(visible).block(Block::default().title(title).borders(Borders::ALL));
            f.render_widget(editor, right_chunks[0]);
            // set cursor in editor area relative to scroll
            let r = (app.editor_row.saturating_sub(scroll)) as u16;
            let c = app.editor_col as u16;
            f.set_cursor(right_chunks[0].x + c + 1, right_chunks[0].y + r + 1);
        } else {
            let messages: Vec<ListItem> = app
                .messages
                .iter()
                .map(|(role, content)| {
                    let style = if role == "User" {
                        Style::default().fg(Color::Green)
                    } else if role == "System" {
                        Style::default().fg(Color::Yellow)
                    } else if role == "File" {
                        Style::default().fg(Color::Cyan)
                    } else {
                        Style::default()
                    };
                    let content = Line::from(vec![
                        Span::styled(format!("{}: ", role), Style::default().add_modifier(Modifier::BOLD)),
                        Span::styled(content.clone(), style),
                    ]);
                    ListItem::new(content)
                })
                .collect();
            let messages_block = Block::default().title("Messages").borders(Borders::ALL);
            let messages_widget = List::new(messages)
                .block(messages_block)
                .highlight_style(Style::default().bg(Color::DarkGray))
                .highlight_symbol(">> ");
            f.render_widget(messages_widget, right_chunks[0]);
        }

        // Input area
        let input = Paragraph::new(app.input.as_str())
            .block(Block::default().title("Input").borders(Borders::ALL))
            .style(if app.loading { Style::default().fg(Color::Gray) } else { Style::default() })
            .wrap(Wrap { trim: true });
        f.render_widget(input, right_chunks[1]);
        // Set cursor position
        if !app.loading {
            f.set_cursor(right_chunks[1].x + app.cursor_position as u16 + 1, right_chunks[1].y + 1);
        }
    })?;
    Ok(())
}

fn enable_raw_mode() -> Result<(), Box<dyn Error>> {
    crossterm::terminal::enable_raw_mode()?;
    execute!(io::stdout(), EnterAlternateScreen, EnableMouseCapture)?;
    Ok(())
}

fn disable_raw_mode() -> Result<(), Box<dyn Error>> {
    crossterm::terminal::disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture)?;
    Ok(())
}