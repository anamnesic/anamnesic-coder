use std::path::PathBuf;
use std::path::Component;
use std::fs;

pub struct FileTools {
    workspace: PathBuf,
}

#[cfg(test)]
mod tests {
    use super::FileTools;
    use std::fs;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    fn temp_workspace() -> std::path::PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "anamnesic-file-tools-test-{}-{n}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn rejects_parent_directory_paths() {
        let tools = FileTools::new(temp_workspace());
        assert!(tools.write_file("../outside.txt", "nope").is_err());
    }

    #[test]
    fn rejects_absolute_and_root_paths() {
        let tools = FileTools::new(temp_workspace());
        assert!(tools.write_file("/etc/passwd", "nope").is_err());
        assert!(tools.append_file("/etc/hosts", "nope").is_err());
        assert!(tools.read_file("/etc/hosts").is_none());
    }

    #[test]
    fn write_then_read_roundtrip() {
        let tools = FileTools::new(temp_workspace());
        tools.write_file("src/main.rs", "fn main() {}\n").unwrap();
        assert_eq!(tools.read_file("src/main.rs").unwrap(), "fn main() {}\n");
    }

    #[test]
    fn append_adds_to_existing_file() {
        let tools = FileTools::new(temp_workspace());
        tools.write_file("log.txt", "line1\n").unwrap();
        tools.append_file("log.txt", "line2\n").unwrap();
        assert_eq!(tools.read_file("log.txt").unwrap(), "line1\nline2\n");
    }

    #[test]
    fn append_creates_file_when_missing() {
        let tools = FileTools::new(temp_workspace());
        tools.append_file("fresh.txt", "hi").unwrap();
        assert_eq!(tools.read_file("fresh.txt").unwrap(), "hi");
    }

    #[test]
    fn read_missing_file_returns_none() {
        let tools = FileTools::new(temp_workspace());
        assert!(tools.read_file("nope.txt").is_none());
    }

    #[test]
    fn list_files_returns_files_within_workspace() {
        let workspace = temp_workspace();
        let tools = FileTools::new(workspace.clone());
        tools.write_file("a.txt", "a").unwrap();
        tools.write_file("sub/b.txt", "b").unwrap();
        assert_eq!(tools.list_files(""), vec!["a.txt"]);
        assert_eq!(tools.list_files("sub"), vec!["sub/b.txt"]);
    }
}

impl FileTools {
    pub fn new(workspace: PathBuf) -> Self {
        fs::create_dir_all(&workspace).ok();
        FileTools { workspace }
    }

    fn resolve(&self, path: &str) -> Option<PathBuf> {
        let p = PathBuf::from(path);
        if p.is_absolute() || p.components().any(|c| matches!(c, Component::ParentDir | Component::RootDir | Component::Prefix(_))) {
            return None;
        }
        Some(self.workspace.join(p))
    }

    pub fn read_file(&self, path: &str) -> Option<String> {
        let p = self.resolve(path)?;
        fs::read_to_string(&p).ok()
    }

    pub fn write_file(&self, path: &str, content: &str) -> anyhow::Result<()> {
        let p = self.resolve(path).ok_or_else(|| anyhow::anyhow!("path must be relative to the workspace"))?;
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&p, content)?;
        Ok(())
    }

    pub fn append_file(&self, path: &str, content: &str) -> anyhow::Result<()> {
        let p = self.resolve(path).ok_or_else(|| anyhow::anyhow!("path must be relative to the workspace"))?;
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = fs::OpenOptions::new().append(true).create(true).open(&p)?;
        use std::io::Write;
        file.write_all(content.as_bytes())?;
        Ok(())
    }

    pub fn list_files(&self, path: &str) -> Vec<String> {
        let mut files = Vec::new();
        let Some(p) = (if path.is_empty() { Some(self.workspace.clone()) } else { self.resolve(path) }) else {
            return files;
        };
        if let Ok(entries) = fs::read_dir(&p) {
            for entry in entries.flatten() {
                if entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                    if let Ok(rel) = entry.path().strip_prefix(&self.workspace) {
                        files.push(rel.to_string_lossy().to_string());
                    }
                }
            }
        }
        files
    }
}
