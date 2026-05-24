use std::process::Command;

pub struct GitTools {
    repo: String,
}

impl GitTools {
    pub fn new(repo: String) -> Self { GitTools { repo } }

    fn git(&self, args: &[&str]) -> String {
        match Command::new("git").args(args).current_dir(&self.repo).output() {
            Ok(out) => {
                let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if s.is_empty() { String::from_utf8_lossy(&out.stderr).trim().to_string() } else { s }
            },
            Err(e) => format!("Git error: {}", e),
        }
    }

    pub fn is_git_repo(&self) -> bool { !self.git(&["rev-parse", "--git-dir"]).contains("error") }
    pub fn init(&self) -> String { self.git(&["init"]) }
    pub fn add(&self, path: &str) -> String { self.git(&["add", path]) }
    pub fn branch(&self, name: &str) -> String { self.git(&["checkout", "-b", name]) }
    pub fn commit(&self, msg: &str) -> String { self.git(&["commit", "-m", msg]) }
    pub fn diff(&self, target: &str) -> String { self.git(&["diff", target]) }
    pub fn restore(&self, path: &str) -> String { self.git(&["restore", path]) }
    pub fn status(&self) -> String { self.git(&["status"]) }
    pub fn log(&self, count: usize) -> String { self.git(&["log", &format!("--oneline"), &format!("-{}", count)]) }
}
