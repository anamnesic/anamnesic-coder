use crate::types::plan::PlanStep;
use crate::agent::state::AgentState;
use crate::llm::router::LlmRouter;
use crate::llm::provider_chain::FallbackChain;
use crate::llm::prompt::CoderPrompt;
use crate::tools::shell;
use crate::tools::test;
use crate::compressor::layer1;
use std::io::Write;

/// How many fix rounds to run after a `.rs` file fails `cargo check`.
const MAX_FILE_FIX_ATTEMPTS: usize = 2;

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

/// Known fenced-block language identifiers. Used to strip the leading tag line
/// of a markdown code block without accidentally dropping real code.
fn is_language_tag(line: &str) -> bool {
    let t = line.trim().to_lowercase();
    matches!(t.as_str(),
        "rust" | "rs" | "python" | "py" | "javascript" | "js" | "typescript" | "ts"
        | "tsx" | "jsx" | "bash" | "sh" | "shell" | "zsh" | "go" | "golang" | "c"
        | "cpp" | "c++" | "h" | "hpp" | "java" | "kotlin" | "kt" | "swift" | "ruby"
        | "rb" | "php" | "perl" | "lua" | "sql" | "html" | "css" | "scss" | "json"
        | "yaml" | "yml" | "xml" | "toml" | "ini" | "markdown" | "md" | "text" | "txt"
        | "diff" | "patch" | "dockerfile" | "makefile" | "plaintext" | "plain" | "console"
        | "fish" | "powershell" | "ps1" | "dart" | "elixir" | "ex" | "exs" | "haskell"
        | "hs" | "clojure" | "clj" | "scala" | "groovy" | "csharp" | "cs" | "fsharp" | "fs"
        | "asm" | "cmake" | "gradle" | "proto" | "graphql" | "gql" | "svg" | "csv"
        | "svelte" | "vue" | "astro" | "cargo,ignore"
    )
}

/// Extract the contents of the first fenced code block from a model response.
/// Falls back to the trimmed response if no fence is found.
fn extract_code_block(content: &str) -> String {
    let content = content.trim();
    let Some(start) = content.find("```") else { return content.to_string(); };
    let after = &content[start + 3..];
    let Some(end) = after.find("```") else { return content.to_string(); };
    let block = after[..end].trim();
    let mut lines = block.lines();
    if let Some(first) = lines.next() {
        if is_language_tag(first) {
            let rest: Vec<&str> = lines.collect();
            let code = rest.join("\n").trim().to_string();
            if !code.is_empty() {
                return code;
            }
        }
    }
    block.to_string()
}

/// Run `cargo check` in the workspace. Returns Some(errors) on failure, None when green.
/// Skips the gate when the workspace isn't a Cargo project.
fn verify_cargo(state: &AgentState) -> Option<String> {
    if !state.config.workspace_dir.join("Cargo.toml").exists() {
        return None;
    }
    let out = shell::run_command_raw("cargo check --message-format short", &state.config);
    if out.code == Some(0) {
        None
    } else {
        Some(out.combined())
    }
}

/// Generate file content via the coder model, write it, and verify `.rs` files
/// with `cargo check`, feeding errors back until it passes (bounded retries).
async fn write_with_verification<F>(
    client: &LlmRouter,
    state: &mut AgentState,
    filename: &str,
    make_prompt: F,
    verbose_label: &str,
) where
    F: Fn(&str) -> String,
{
    let mut extra = String::new();
    for attempt in 0..=MAX_FILE_FIX_ATTEMPTS {
        let prompt = make_prompt(&extra);
        let content = match client.generate_with_retry(&state.config.coder_model, &prompt, None, None).await {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  ✗ LLM call failed for {}: {e}", filename);
                return;
            }
        };
        let code = extract_code_block(&content);
        if code.trim().is_empty() {
            eprintln!("  ✗ model returned empty content for {}", filename);
            return;
        }
        if let Err(e) = state.files.write_file(filename, &code) {
            eprintln!("  ✗ write failed for {}: {e}", filename);
            return;
        }
        println!("  {} {}", verbose_label, filename);

        if filename.ends_with(".rs") {
            match verify_cargo(state) {
                None => {
                    println!("  ✓ cargo check passed");
                    return;
                }
                Some(err) if attempt < MAX_FILE_FIX_ATTEMPTS => {
                    eprintln!("  cargo check failed (attempt {}); fixing...", attempt + 1);
                    extra = format!(
                        "Your previous output failed `cargo check`. Fix the errors below and return the COMPLETE corrected file in a single code block:\n```\n{}\n```",
                        err
                    );
                }
                Some(err) => {
                    eprintln!("  ✗ cargo check still failing after retries:\n{}", err);
                    return;
                }
            }
        } else {
            return;
        }
    }
}

/// Heuristic: extract a likely file path from a step description, e.g.
/// "create src/calc.rs with a factorial function" -> "src/calc.rs".
/// Used as a fallback when the planner omits the `filename` field.
fn extract_path(description: &str) -> Option<String> {
    let known_ext = [
        "rs", "py", "js", "ts", "tsx", "jsx", "go", "c", "h", "cpp", "hpp", "java", "rb", "php",
        "sh", "toml", "json", "yaml", "yml", "md", "html", "css", "sql", "kt", "swift", "dart",
        "lua", "r", "zig", "ex", "exs", "fs", "cs", "vue", "svelte", "astro", "prisma", "lock",
    ];
    let toks: Vec<&str> = description.split_whitespace().collect();
    for tok in &toks {
        let clean = tok.trim_matches(|c: char| !c.is_alphanumeric() && c != '.' && c != '/' && c != '-' && c != '_' && c != '\\');
        if clean.contains('/') || clean.contains('\\') {
            return Some(clean.to_string());
        }
        if let Some(ext) = clean.rsplit('.').next() {
            if known_ext.contains(&ext) && clean != "." {
                return Some(clean.to_string());
            }
        }
    }
    None
}

pub async fn execute_step(client: &LlmRouter, state: &mut AgentState, step: &PlanStep) {
    let caveman = state.caveman;
    execute_step_inner(client, state, step, caveman).await
}

/// Execute a step using a FallbackChain for cloud provider fallback.
pub async fn execute_step_with_chain(chain: &FallbackChain, state: &mut AgentState, step: &PlanStep) {
    let caveman = state.caveman;
    execute_step_inner_chain(chain, state, step, caveman).await
}

async fn execute_step_inner(client: &LlmRouter, state: &mut AgentState, step: &PlanStep, caveman: crate::compressor::caveman::CavemanLevel) {
    match step.step_type.as_str() {
        "create_file" => {
            if let Some(filename) = step.filename.clone().or_else(|| extract_path(&step.description)) {
                println!("  Generating [{}]...", filename);
                let context = grep_context(state);
                let fname = filename.clone();
                let description = step.description.clone();
                write_with_verification(client, state, &fname, |extra| {
                    format!(
                        "{}\n\nContext:\n{}\n\nTask:\nCreate file '{}': {}\nReturn only the COMPLETE file content inside a single code block.\n{}",
                        CoderPrompt::with_caveman(&caveman),
                        context,
                        fname,
                        description,
                        extra
                    )
                }, "Created").await;
            } else {
                println!("  ✗ create_file step is missing a filename");
            }
        },
        "edit_file" => {
            if let Some(filename) = step.filename.clone().or_else(|| extract_path(&step.description)) {
                let content = state.files.read_file(&filename).unwrap_or_default();
                if content.is_empty() {
                    println!("  File {} not found", filename);
                    return;
                }
                println!("  Editing [{}]...", filename);
                let fname = filename.clone();
                let file_content = content;
                let description = step.description.clone();
                write_with_verification(client, state, &fname, |extra| {
                    format!(
                        "{}\n\nFile: {}\n\nContent:\n```\n{}\n```\n\nInstruction: {}\n{}\nReturn only the COMPLETE modified file content inside a single code block.",
                        CoderPrompt::with_caveman(&caveman),
                        fname,
                        file_content,
                        description,
                        extra
                    )
                }, "Edited").await;
            }
        },
        "read_file" => {
            if let Some(filename) = step.filename.clone().or_else(|| extract_path(&step.description)) {
                match state.files.read_file(&filename) {
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
            let cmd = step.command.clone().unwrap_or_else(|| step.description.clone());
            if !cmd.is_empty() {
                println!("  Running: {}", cmd);
                let result = shell::run_command(&cmd, &state.config);
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
            let system = CoderPrompt::with_caveman(&caveman);
            let prompt = format!("{}\n\nContext:\n{}\n\nTask:\n{}", system, context, step.description);
            let mut out = std::io::stdout();
            let result = client.stream(&state.config.coder_model, &prompt, None, None, &mut |tok| {
                let _ = write!(out, "{}", tok);
                let _ = out.flush();
            }).await;
            println!();
            if let Err(e) = result {
                eprintln!("  ✗ answer failed: {e}");
            }
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

async fn execute_step_inner_chain(chain: &FallbackChain, state: &mut AgentState, step: &PlanStep, caveman: crate::compressor::caveman::CavemanLevel) {
    match step.step_type.as_str() {
        "answer" => {
            let context = grep_context(state);
            let system = CoderPrompt::with_caveman(&caveman);
            let prompt = format!("{}\n\nContext:\n{}\n\nTask:\n{}", system, context, step.description);
            match chain.complete(&prompt).await {
                Ok(text) => {
                    let mut out = std::io::stdout();
                    for tok in text.chars() {
                        let _ = write!(out, "{}", tok);
                        let _ = out.flush();
                    }
                    println!();
                }
                Err(e) => eprintln!("  ✗ answer failed: {e}"),
            }
        },
        _ => {
            // For non-answer steps, fall back to direct execution without a client
            println!("  [fallback] step type '{}' requires a client, skipping", step.step_type);
        }
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

#[cfg(test)]
mod tests {
    use super::extract_code_block;

    #[test]
    fn extracts_fenced_block() {
        let out = extract_code_block("Here is the code:\n```rust\nfn main() {}\n```\nDone");
        assert_eq!(out, "fn main() {}");
    }

    #[test]
    fn handles_block_without_language_tag() {
        let out = extract_code_block("```\nhello\nworld\n```");
        assert_eq!(out, "hello\nworld");
    }

    #[test]
    fn falls_back_without_fence() {
        let out = extract_code_block("just plain text");
        assert_eq!(out, "just plain text");
    }

    #[test]
    fn keeps_single_line_code() {
        let out = extract_code_block("```rust\nfn main() {}\n```");
        assert_eq!(out, "fn main() {}");
    }
}
