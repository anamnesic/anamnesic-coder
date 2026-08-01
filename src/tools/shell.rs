use std::process::Command;
use crate::config::settings::Config;

/// Result of running a shell command.
pub struct CommandOutput {
    pub code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

impl CommandOutput {
    pub fn combined(&self) -> String {
        let mut result = self.stdout.clone();
        if !self.stderr.is_empty() {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str("STDERR:\n");
            result.push_str(&self.stderr);
        }
        if let Some(code) = self.code {
            if code != 0 {
                if !result.is_empty() {
                    result.push('\n');
                }
                result.push_str(&format!("exit code: {code}"));
            }
        }
        if result.trim().is_empty() {
            "(no output)".into()
        } else {
            result
        }
    }
}

/// Spawn the platform's default shell: `sh -c` on Unix, `cmd /C` on Windows.
fn shell_command(cmd: &str) -> Command {
    #[cfg(target_family = "unix")]
    {
        let mut c = Command::new("sh");
        c.arg("-c").arg(cmd);
        c
    }
    #[cfg(target_family = "windows")]
    {
        let mut c = Command::new("cmd");
        c.arg("/C").arg(cmd);
        c
    }
}

/// Check whether a command is allowed by the allow/block lists.
pub fn is_allowed(cmd: &str, config: &Config) -> bool {
    let trimmed = cmd.trim_start();
    if trimmed.is_empty() {
        return false;
    }
    let cmd_lower = trimmed.to_lowercase();
    for blocked in &config.blocked_commands {
        if cmd_lower.contains(&blocked.to_lowercase()) {
            return false;
        }
    }
    config.allowed_commands.iter().any(|a| cmd_lower.starts_with(&a.to_lowercase()))
}

/// Run a command through the platform shell and return combined output.
pub fn run_command(cmd: &str, config: &Config) -> String {
    if !is_allowed(cmd, config) {
        return format!("Command not in allowed list: {}", cmd);
    }

    let child = shell_command(cmd)
        .current_dir(&config.workspace_dir)
        .output();

    match child {
        Ok(out) => {
            CommandOutput {
                code: out.status.code(),
                stdout: String::from_utf8_lossy(&out.stdout).to_string(),
                stderr: String::from_utf8_lossy(&out.stderr).to_string(),
            }
            .combined()
        },
        Err(e) => format!("Error: {}", e),
    }
}

/// Run a command and return the raw `CommandOutput` (used by the verification gate).
pub fn run_command_raw(cmd: &str, config: &Config) -> CommandOutput {
    match shell_command(cmd).current_dir(&config.workspace_dir).output() {
        Ok(out) => CommandOutput {
            code: out.status.code(),
            stdout: String::from_utf8_lossy(&out.stdout).to_string(),
            stderr: String::from_utf8_lossy(&out.stderr).to_string(),
        },
        Err(e) => CommandOutput {
            code: None,
            stdout: String::new(),
            stderr: format!("Error: {e}"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::settings::Config;

    fn test_config() -> Config {
        let mut cfg = Config::default();
        cfg.workspace_dir = std::env::temp_dir();
        cfg
    }

    #[test]
    fn blocks_dangerous_commands() {
        let cfg = test_config();
        assert!(!is_allowed("rm -rf /", &cfg));
        assert!(!is_allowed("sudo apt install", &cfg));
    }

    #[test]
    fn allows_listed_commands() {
        let cfg = test_config();
        assert!(is_allowed("cargo check", &cfg));
        assert!(is_allowed("echo hello", &cfg));
    }

    #[cfg(target_family = "unix")]
    #[test]
    fn runs_via_sh_on_unix() {
        let cfg = test_config();
        let out = run_command("printf 'ok'", &cfg);
        assert!(out.contains("ok"), "got: {out}");
    }
}
