pub struct PlannerPrompt;

impl PlannerPrompt {
    pub fn system() -> &'static str {
        r#"You are a coding task planner. Given a task and context, output a JSON plan.
Each step has a type and description.

Step types: read_file, edit_file, create_file, search_code, run_command, run_tests, answer, git_init, git_commit, git_status, done

Output JSON format:
{
  "steps": [
    {"type": "search_code", "description": "find the login route"},
    {"type": "read_file", "description": "inspect auth middleware", "filename": "src/auth.rs"},
    {"type": "edit_file", "description": "modify token validation", "filename": "src/auth.rs"},
    {"type": "done", "description": "token validation updated"}
  ]
}

Keep plans minimal: 1-5 steps. Only include necessary steps."#
    }
}

pub struct CoderPrompt;

impl CoderPrompt {
    pub fn system() -> &'static str {
        r#"You are a code generation assistant. You write clean, idiomatic code.
Given context and a task, produce the code changes or answer.

Rules:
- Return only the code/content requested
- For files, use code blocks
- Keep explanations minimal
- Follow existing code style"#
    }

    pub fn generate_file_system() -> &'static str {
        "Return ONLY the file content inside a code block. Do NOT include the filename or extra explanation."
    }
}
