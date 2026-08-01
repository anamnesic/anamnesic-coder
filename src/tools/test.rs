use std::process::Command;
use crate::config::settings::Config;

/// Run the workspace test suite. Detects a Cargo project and runs `cargo test`
/// (optionally filtered by `path`), otherwise falls back to `python -m pytest`.
pub fn run_tests(path: &str, config: &Config) -> String {
    if config.workspace_dir.join("Cargo.toml").exists() {
        return run_cargo_tests(path, config);
    }
    let actual_path = if path.is_empty() { "tests" } else { path };
    match Command::new("python")
        .args(&["-m", "pytest", actual_path, "-v"])
        .current_dir(&config.workspace_dir)
        .output()
    {
        Ok(out) => {
            let mut output = String::from_utf8_lossy(&out.stdout).to_string();
            if !out.stderr.is_empty() {
                output.push_str(&format!("\nSTDERR:\n{}", String::from_utf8_lossy(&out.stderr)));
            }
            if output.trim().is_empty() { "(no output)".into() } else { output }
        },
        Err(e) => format!("Error running tests: {}", e),
    }
}

fn run_cargo_tests(filter: &str, config: &Config) -> String {
    let filter = filter
        .trim()
        .trim_start_matches("cargo test")
        .trim_start_matches("cargo")
        .trim();
    let mut cmd = Command::new("cargo");
    cmd.arg("test").current_dir(&config.workspace_dir);
    if !filter.is_empty() {
        cmd.arg("--").arg(filter);
    }
    match cmd.output() {
        Ok(out) => {
            let mut output = String::from_utf8_lossy(&out.stdout).to_string();
            if !out.stderr.is_empty() {
                output.push_str(&format!("\nSTDERR:\n{}", String::from_utf8_lossy(&out.stderr)));
            }
            if output.trim().is_empty() { "(no output)".into() } else { output }
        },
        Err(e) => format!("Error running tests: {}", e),
    }
}
