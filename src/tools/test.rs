use std::process::Command;
use crate::config::settings::Config;

pub fn run_tests(path: &str, config: &Config) -> String {
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
