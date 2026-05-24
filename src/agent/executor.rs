use crate::types::plan::PlanStep;
use crate::agent::state::AgentState;
use crate::llm::client::OllamaClient;
use crate::llm::prompt::CoderPrompt;
use crate::tools::shell;
use crate::tools::test;

async fn coder_generate(client: &OllamaClient, model: &str, task: &str, context: &str) -> String {
    let prompt = format!("{}\n\nContext:\n{}\n\nTask:\n{}", CoderPrompt::system(), context, task);
    client.generate(model, &prompt).await.unwrap_or_default()
}

pub async fn execute_step(client: &OllamaClient, state: &mut AgentState, step: &PlanStep) {
    match step.step_type.as_str() {
        "create_file" => {
            if let Some(filename) = &step.filename {
                println!("  Generating [{}]...", filename);
                let context = grep_context(state);
                let content = coder_generate(client, &state.config.coder_model,
                    &format!("Create file '{}': {}", filename, step.description), &context).await;
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
                let prompt = format!("{}\n\nFile: {}\n\nContent:\n```\n{}\n```\n\nInstruction: {}\nReturn only the modified file content.",
                    CoderPrompt::system(), filename, content, step.description);
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
                        let truncated: String = content.chars().take(3000).collect();
                        println!("{}", truncated);
                        if content.len() > 3000 {
                            println!("  ...({} more chars)", content.len() - 3000);
                        }
                    },
                    None => println!("  File {} not found", filename),
                }
            } else {
                let results = search_code(state, &step.description);
                println!("{}", results);
            }
        },
        "search_code" => {
            let pattern = step.pattern.as_deref().unwrap_or(&step.description);
            let results = search_code(state, pattern);
            println!("{}", if results.is_empty() { "  No results".into() } else { results });
        },
        "run_command" => {
            let cmd = step.command.as_deref().unwrap_or("");
            if !cmd.is_empty() {
                println!("  Running: {}", cmd);
                let result = shell::run_command(cmd, &state.config);
                let truncated: String = result.chars().take(2000).collect();
                println!("{}", truncated);
            }
        },
        "run_tests" => {
            let output = test::run_tests(&step.description, &state.config);
            state.last_test_output = output.clone();
            let truncated: String = output.chars().take(2000).collect();
            println!("{}", truncated);
        },
        "answer" => {
            let context = grep_context(state);
            let answer = coder_generate(client, &state.config.coder_model, &step.description, &context).await;
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
