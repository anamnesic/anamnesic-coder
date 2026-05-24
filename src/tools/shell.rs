use std::process::Command;
use crate::config::settings::Config;

pub fn run_command(cmd: &str, config: &Config) -> String {
    let cmd_lower = cmd.to_lowercase();
    for blocked in &config.blocked_commands {
        if cmd_lower.contains(blocked) {
            return format!("Command blocked: {}", cmd);
        }
    }
    let allowed = config.allowed_commands.iter().any(|a| cmd_lower.starts_with(a));
    if !allowed {
        return format!("Command not in allowed list: {}", cmd);
    }

    match Command::new("cmd").args(&["/C", cmd]).current_dir(&config.workspace_dir).output() {
        Ok(out) => {
            let mut result = String::from_utf8_lossy(&out.stdout).to_string();
            if !out.stderr.is_empty() {
                result.push_str(&format!("\nSTDERR:\n{}", String::from_utf8_lossy(&out.stderr)));
            }
            if result.trim().is_empty() { "(no output)".into() } else { result }
        },
        Err(e) => format!("Error: {}", e),
    }
}
