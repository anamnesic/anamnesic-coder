use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Frame, Terminal,
};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
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

pub struct App {
    pub messages: Vec<(String, String)>, // (role, content)
    pub input: String,
    pub cursor_position: usize,
    pub loading: bool,
}

impl App {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            input: String::new(),
            cursor_position: 0,
            loading: false,
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
                    KeyCode::Enter => {
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
                    KeyCode::Char(c) => {
                        let pos = guard.cursor_position;
                        guard.input.insert(pos, c);
                        guard.cursor_position += 1;
                    }
                    KeyCode::Backspace => {
                        let pos = guard.cursor_position;
                        if pos > 0 {
                            guard.input.remove(pos - 1);
                            guard.cursor_position -= 1;
                        }
                    }
                    KeyCode::Left => {
                        if guard.cursor_position > 0 {
                            guard.cursor_position -= 1;
                        }
                    }
                    KeyCode::Right => {
                        if guard.cursor_position < guard.input.len() {
                            guard.cursor_position += 1;
                        }
                    }
                    KeyCode::Esc => {
                        break;
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

fn draw<B: ratatui::backend::Backend>(f: &mut Terminal<B>, app: &App) -> Result<(), Box<dyn Error>> {
    f.draw(|mut f| {
        let size = f.size();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints(
                [
                    Constraint::Min(0),
                    Constraint::Length(3),
                ]
                .as_ref(),
            )
            .split(size);

        // Messages area
        let messages: Vec<ListItem> = app
            .messages
            .iter()
            .map(|(role, content)| {
                let style = if role == "User" {
                    Style::default().fg(Color::Green)
                } else if role == "System" {
                    Style::default().fg(Color::Yellow)
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
        let messages_block = Block::default()
            .title("Messages")
            .borders(Borders::ALL);
        let messages_widget = List::new(messages)
            .block(messages_block)
            .highlight_style(Style::default().bg(Color::DarkGray))
            .highlight_symbol(">> ");
        f.render_widget(messages_widget, chunks[0]);

        // Input area
        let input = Paragraph::new(app.input.as_str())
            .block(Block::default().title("Input").borders(Borders::ALL))
            .style(if app.loading {
                Style::default().fg(Color::Gray)
            } else {
                Style::default()
            })
            .wrap(Wrap { trim: true });
        f.render_widget(input, chunks[1]);
        // Set cursor position
        if !app.loading {
            f.set_cursor(
                // Put cursor past the end of the input text
                chunks[1].x + app.cursor_position as u16 + 1,
                chunks[1].y + 1,
            );
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