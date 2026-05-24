use crate::types::plan::PlanStep;
use crate::agent::state::AgentState;
use crate::llm::client::LlmClient;
use crate::llm::prompt::CoderPrompt;
use crate::tools::shell;
use crate::tools::test;
use crate::compressor::layer1;
use crate::compressor::caveman::CavemanLevel;

async fn coder_generate(client: &LlmClient, model: &str, task: &str, context: &str, caveman: &CavemanLevel) -> String {
    let system = CoderPrompt::with_caveman(caveman);
    let prompt = format!("{}\n\nContext:\n{}\n\nTask:\n{}", system, context, task);
    client.generate(model, &prompt).await.unwrap_or_default()
}

fn compress_output(output: &str, label: &str) -> String {
    if output.len() < 200 {
        return output.to_string();
    }
    let result = layer1::compress(output);
    if result.compressed_lines < result.original_lines {
        let saved = result.original_lines.saturating_sub(result.compressed_lines);
        let pct = if result.original_lines > 0 {
            (saved as f64 / result.original_lines as f64 * 100.0) as u32
        } else {
            0
        };
        if label == "command" {
            eprintln!("  [NTK-L1: {} lines ({}% saved)]", result.output.lines().count(), pct);
        }
    }
    result.output
}

pub async fn execute_step(client: &LlmClient, state: &mut AgentState, step: &PlanStep) {
    let caveman = state.caveman;
    match step.step_type.as_str() {
        "create_file" => {
            if let Some(filename) = &step.filename {
                println!("  Generating [{}]...", filename);
                let context = grep_context(state);
                let content = coder_generate(client, &state.config.coder_model,
                    &format!("Create file '{}': {}", filename, step.description), &context, &caveman).await;
                if !content.is_empty() {
                    if let Some(code_start) = content.find("```") {
                        let after = &content[code_start + 3..];
                        if let Some(code_end) = after.find("```") {
                            let code = after[..code_end].trim();
                            let code = code.split('\n').skip(1).collect::<Vec<&str>>().join("\n");
                            state.files.write_file(filename, &code).ok();
                            println!("  Created {}", filename);
                        }
                    } else {
                        state.files.write_file(filename, &content).ok();
                        println!("  Created {}", filename);
                    }
                }
            }
        },
        "edit_file" => {
            if let Some(filename) = &step.filename {
                let content = state.files.read_file(filename).unwrap_or_default();
                if content.is_empty() {
                    println!("  File {} not found", filename);
                    return;
                }
                let system = CoderPrompt::with_caveman(&caveman);
                let prompt = format!("{}\n\nFile: {}\n\nContent:\n```\n{}\n```\n\nInstruction: {}\nReturn only the modified file content.",
                    system, filename, content, step.description);
                let edited = client.generate(&state.config.coder_model, &prompt).await.unwrap_or_default();
                if !edited.is_empty() && edited != content {
                    state.files.write_file(filename, &edited).ok();
                    println!("  Edited {}", filename);
                } else {
                    println!("  No changes to {}", filename);
                }
            }
        },
        "read_file" => {
            if let Some(filename) = &step.filename {
                match state.files.read_file(filename) {
                    Some(content) => {
                        let compressed = compress_output(&content, "file");
                        let truncated: String = compressed.chars().take(3000).collect();
                        println!("{}", truncated);
                        if content.len() > 3000 {
                            println!("  ...({} more chars)", content.len() - 3000);
                        }
                    },
                    None => println!("  File {} not found", filename),
                }
            } else {
                let results = search_code(state, &step.description);
                println!("{}", compress_output(&results, "search"));
            }
        },
        "search_code" => {
            let pattern = step.pattern.as_deref().unwrap_or(&step.description);
            let results = search_code(state, pattern);
            let compressed = compress_output(&results, "search");
            println!("{}", if compressed.is_empty() { "  No results".into() } else { compressed });
        },
        "run_command" => {
            let cmd = step.command.as_deref().unwrap_or("");
            if !cmd.is_empty() {
                println!("  Running: {}", cmd);
                let result = shell::run_command(cmd, &state.config);
                let compressed = compress_output(&result, "command");
                let truncated: String = compressed.chars().take(2000).collect();
                println!("{}", truncated);
            }
        },
        "run_tests" => {
            let output = test::run_tests(&step.description, &state.config);
            state.last_test_output = output.clone();
            let compressed = compress_output(&output, "test");
            let truncated: String = compressed.chars().take(2000).collect();
            println!("{}", truncated);
        },
        "answer" => {
            let context = grep_context(state);
            let answer = coder_generate(client, &state.config.coder_model, &step.description, &context, &caveman).await;
            println!("{}", answer);
        },
        "git_init" => {
            if !state.git.is_git_repo() {
                state.git.init();
                state.git.branch("main");
                state.git.add(".");
                println!("  Git repo initialized");
            } else {
                println!("  Git repo already exists");
            }
        },
        "git_commit" => {
            state.git.add(".");
            let result = state.git.commit(&step.description);
            println!("  {}", result);
        },
        "git_status" => {
            println!("{}", state.git.status());
        },
        "done" => {
            println!("  Done: {}", step.description);
        },
        _ => println!("  Unknown step type: {}", step.step_type),
    }
}

fn search_code(state: &AgentState, pattern: &str) -> String {
    let result = std::process::Command::new("rg")
        .args(&["-n", "--max-count", "10", pattern])
        .current_dir(&state.config.workspace_dir)
        .output();
    match result {
        Ok(out) => {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if s.is_empty() { format!("No matches for: {}", pattern) } else { s }
        },
        Err(_) => "ripgrep not installed".into(),
    }
}

fn grep_context(state: &AgentState) -> String {
    let task = state.session.get_context();
    let result = std::process::Command::new("rg")
        .args(&["-n", "--max-count", "5", &task])
        .current_dir(&state.config.workspace_dir)
        .output();
    match result {
        Ok(out) => {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if s.is_empty() { "No relevant context found.".into() } else { s }
        },
        Err(_) => "No relevant context found.".into(),
    }
}
