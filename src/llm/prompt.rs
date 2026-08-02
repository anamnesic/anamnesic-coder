use crate::compressor::caveman::CavemanLevel;

pub struct PlannerPrompt;

impl PlannerPrompt {
    pub fn system() -> &'static str {
        r#"You are a coding task planner. Given a task and context, output a JSON plan.
Each step has a type and description.

Step types: read_file, edit_file, create_file, search_code, run_command, run_tests, answer, git_init, git_commit, git_status, done

Rules:
- create_file, edit_file and read_file MUST include "filename" (relative path, e.g. "src/calc.rs").
- run_command MUST include "command" (the exact shell command).
- search_code MUST include "pattern" (regex to search).
- run_tests MUST include the test filter in "description" (use "cargo test" for Rust projects).
- For features or bug fixes, follow TDD: include a step that writes/runs the tests FIRST (RED), then implementation steps (GREEN), then re-run tests.
- NEVER modify or weaken existing tests to make them pass; fix the implementation instead.

Output JSON format:
{
  "steps": [
    {"type": "search_code", "description": "find the factorial function", "pattern": "fn factorial"},
    {"type": "read_file", "description": "inspect existing module", "filename": "src/calc.rs"},
    {"type": "create_file", "description": "add factorial module with unit test", "filename": "src/calc.rs"},
    {"type": "edit_file", "description": "call factorial from main", "filename": "src/main.rs"},
    {"type": "run_command", "description": "compile check", "command": "cargo check"},
    {"type": "run_tests", "description": "cargo test"},
    {"type": "done", "description": "factorial implemented and tested"}
  ]
}

Keep plans minimal: 1-6 steps. Only include necessary steps. Output ONLY the JSON, nothing else."#
    }

    pub fn with_caveman(level: &CavemanLevel) -> String {
        let base = Self::system();
        let suffix = level.system_prompt_suffix();
        if suffix.is_empty() {
            base.to_string()
        } else {
            format!("{}{}", base, suffix)
        }
    }
}

pub struct CoderPrompt;

impl CoderPrompt {
    pub fn system() -> &'static str {
        include_str!("../../prompts/coder.txt").trim()
    }

    pub fn load_project_context(workspace: &std::path::Path) -> String {
        let candidates = ["AGENTS.md", "CLAUDE.md", ".cursorrules", "CONTEXT.md"];
        let mut loaded = Vec::new();
        for name in candidates {
            let path = workspace.join(name);
            if path.is_file() {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    let trimmed = content.trim();
                    if !trimmed.is_empty() {
                        let capped: String = trimmed.chars().take(4000).collect();
                        loaded.push(format!("### Project Instructions ({name})\n{capped}"));
                    }
                }
            }
        }
        loaded.join("\n\n")
    }

    pub fn with_context(level: &CavemanLevel, project_context: &str) -> String {
        let base = Self::system();
        let suffix = level.system_prompt_suffix();
        let mut prompt = if suffix.is_empty() {
            base.to_string()
        } else {
            format!("{}{}", base, suffix)
        };
        if !project_context.trim().is_empty() {
            prompt.push_str("\n\nProject Instructions:\n");
            prompt.push_str(project_context.trim());
        }
        prompt
    }

    pub fn with_caveman(level: &CavemanLevel) -> String {
        Self::with_context(level, "")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_project_context_from_workspace() {
        let temp = std::env::temp_dir().join(format!("test_agents_md_{}", std::process::id()));
        std::fs::create_dir_all(&temp).unwrap();
        std::fs::write(temp.join("AGENTS.md"), "# Rules\nRule 1: Always check tests").unwrap();

        let ctx = CoderPrompt::load_project_context(&temp);
        assert!(ctx.contains("AGENTS.md"));
        assert!(ctx.contains("Rule 1"));

        let full_prompt = CoderPrompt::with_context(&CavemanLevel::Off, &ctx);
        assert!(full_prompt.contains("Project Instructions:"));
        assert!(full_prompt.contains("Rule 1"));

        std::fs::remove_dir_all(&temp).ok();
    }
}
