//! Long-running detached commands (C9 background tasks).
//!
//! Lets the agent kick off a slow build/test command, keep working, and poll the
//! task later for status and captured output. Re-uses the same allow/block +
//! workspace-containment gates as the synchronous `run_command` so policy still
//! applies to backgrounded processes.

use std::collections::BTreeMap;
use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::config::settings::Config;
use crate::tools::shell;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskStatus {
    Running,
    Done { exit_code: Option<i32>, timed_out: bool },
}

#[derive(Debug)]
struct Inner {
    id: String,
    command: String,
    started_at: Instant,
    stdout: String,
    stderr: String,
    status: TaskStatus,
    child: Option<Child>,
}

#[derive(Debug, Clone)]
pub struct BackgroundTaskManager {
    inner: Arc<Mutex<BTreeMap<String, Arc<Mutex<Inner>>>>>,
}

impl BackgroundTaskManager {
    pub fn new() -> Self {
        Self { inner: Arc::new(Mutex::new(BTreeMap::new())) }
    }

    /// Spawn `command` detached. Returns the new task id, or an error message
    /// when the allow/block or workspace-containment gate rejects it.
    pub fn spawn(&self, command: &str, config: &Config) -> Result<String, String> {
        if command.trim().is_empty() {
            return Err("empty command".into());
        }
        if shell::escapes_workspace(command, config) {
            return Err(format!(
                "command rejected: it mutates a path outside the workspace (and outside PATH_ALLOWLIST): {command}"
            ));
        }
        if !shell::is_allowed(command, config) {
            return Err(format!(
                "Command not in allowed list or contains blocked operation: {command}"
            ));
        }

        #[cfg(target_family = "windows")]
        let mut cmd = {
            let mut c = Command::new("cmd.exe");
            c.args(["/C", command]);
            c
        };
        #[cfg(not(target_family = "windows"))]
        let mut cmd = {
            let mut c = Command::new("sh");
            c.args(["-c", command]);
            c
        };
        cmd.current_dir(&config.workspace_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| format!("spawn failed: {e}"))?;
        let stdout = child.stdout.take().expect("stdout piped");
        let stderr = child.stderr.take().expect("stderr piped");
        let id = format!("bg{}", self.next_id());
        let task = Arc::new(Mutex::new(Inner {
            id: id.clone(),
            command: command.to_string(),
            started_at: Instant::now(),
            stdout: String::new(),
            stderr: String::new(),
            status: TaskStatus::Running,
            child: Some(child),
        }));
        self.inner.lock().unwrap().insert(id.clone(), task.clone());

        // Reader thread: drains both pipes, updates the shared buffer, and
        // waits the child periodically to detect completion/timeout.
        let timeout = Duration::from_secs(config.command_timeout_secs);
        thread::spawn(move || {
            let stdout = thread::spawn(move || read_pipe(stdout));
            let stderr = thread::spawn(move || read_pipe(stderr));
            let deadline = Instant::now() + timeout;
            loop {
                {
                    let mut guard = task.lock().unwrap();
                    if !matches!(guard.status, TaskStatus::Running) {
                        break;
                    }
                    if let Some(child) = guard.child.as_mut() {
                        match child.try_wait() {
                            Ok(Some(status)) => {
                                guard.status = TaskStatus::Done {
                                    exit_code: status.code(),
                                    timed_out: false,
                                };
                            }
                            Ok(None) => {
                                if Instant::now() >= deadline {
                                    let _ = child.kill();
                                    guard.status = TaskStatus::Done {
                                        exit_code: None,
                                        timed_out: true,
                                    };
                                }
                            }
                            Err(_) => {
                                guard.status = TaskStatus::Done {
                                    exit_code: None,
                                    timed_out: false,
                                };
                            }
                        }
                    }
                    if !matches!(guard.status, TaskStatus::Running) {
                        break;
                    }
                }
                thread::sleep(Duration::from_millis(20));
            }
            // Flush captured pipes once the process has terminated.
            let out = stdout.join().unwrap_or_default();
            let err = stderr.join().unwrap_or_default();
            {
                let mut guard = task.lock().unwrap();
                guard.stdout.push_str(&out);
                guard.stderr.push_str(&err);
            }
        });

        Ok(id)
    }

    fn next_id(&self) -> usize {
        self.inner.lock().unwrap().len() + 1
    }

    /// Most recent captured output (stdout + stderr tail) and current status.
    pub fn status(&self, id: &str) -> Option<(TaskStatus, String, Duration)> {
        let map = self.inner.lock().unwrap();
        let task = map.get(id)?;
        let guard = task.lock().unwrap();
        Some((guard.status.clone(), render_output(&guard), guard.started_at.elapsed()))
    }

    pub fn list(&self) -> Vec<(String, String, String)> {
        let map = self.inner.lock().unwrap();
        map.iter()
            .map(|(id, task)| {
                let g = task.lock().unwrap();
                (
                    id.clone(),
                    g.command.clone(),
                    match g.status {
                        TaskStatus::Running => "running".into(),
                        TaskStatus::Done { exit_code, timed_out } => {
                            if let Some(code) = exit_code {
                                format!("done (exit {code})")
                            } else if timed_out {
                                "done (timed out)".into()
                            } else {
                                "done".into()
                            }
                        }
                    },
                )
            })
            .collect()
    }

    /// Stop a running task, or no-op when already done.
    pub fn kill(&self, id: &str) -> bool {
        let map = self.inner.lock().unwrap();
        let Some(task) = map.get(id) else {
            return false;
        };
        let mut guard = task.lock().unwrap();
        if let Some(child) = guard.child.as_mut() {
            let _ = child.kill();
        }
        guard.status = TaskStatus::Done { exit_code: None, timed_out: false };
        true
    }
}

impl Default for BackgroundTaskManager {
    fn default() -> Self {
        Self::new()
    }
}

fn read_pipe<R: Read>(mut pipe: R) -> String {
    let mut bytes = Vec::new();
    let _ = pipe.read_to_end(&mut bytes);
    String::from_utf8_lossy(&bytes).into_owned()
}

fn render_output(inner: &Inner) -> String {    let mut out = inner.stdout.clone();
    if !inner.stderr.is_empty() {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str("STDERR:\n");
        out.push_str(&inner.stderr);
    }
    if out.trim().is_empty() {
        "(no output yet)".into()
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn config() -> Config {
        Config {
            workspace_dir: env::temp_dir(),
            command_timeout_secs: 30,
            ..Config::default()
        }
    }

    #[test]
    fn spawn_status_and_list() {
        let cfg = config();
        let mgr = BackgroundTaskManager::new();
        let id = mgr.spawn("echo hello-bg", &cfg).expect("spawn");
        thread::sleep(Duration::from_millis(300));
        let (status, output, _) = mgr.status(&id).expect("status");
        assert!(
            matches!(status, TaskStatus::Done { .. }),
            "status: {status:?}"
        );
        assert!(output.contains("hello-bg"), "output: {output}");
        let list = mgr.list();
        assert!(list.iter().any(|(i, _, _)| i == &id), "list: {list:?}");
    }

    #[test]
    fn rejected_command_returns_error_not_task() {
        let cfg = config();
        let mgr = BackgroundTaskManager::new();
        let err = mgr.spawn("sudo whoami", &cfg).unwrap_err();
        assert!(err.contains("not in allowed list") || err.contains("blocked"));
    }
}
