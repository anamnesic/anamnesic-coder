use std::path::{Path, PathBuf};
use std::fs;

pub struct FileTools {
    workspace: PathBuf,
}

impl FileTools {
    pub fn new(workspace: PathBuf) -> Self {
        fs::create_dir_all(&workspace).ok();
        FileTools { workspace }
    }

    fn resolve(&self, path: &str) -> PathBuf {
        let p = PathBuf::from(path);
        if p.is_absolute() { p } else { self.workspace.join(&p) }
    }

    pub fn read_file(&self, path: &str) -> Option<String> {
        let p = self.resolve(path);
        fs::read_to_string(&p).ok()
    }

    pub fn write_file(&self, path: &str, content: &str) -> anyhow::Result<()> {
        let p = self.resolve(path);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&p, content)?;
        Ok(())
    }

    pub fn append_file(&self, path: &str, content: &str) -> anyhow::Result<()> {
        let p = self.resolve(path);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = fs::OpenOptions::new().append(true).create(true).open(&p)?;
        use std::io::Write;
        file.write_all(content.as_bytes())?;
        Ok(())
    }

    pub fn list_files(&self, path: &str) -> Vec<String> {
        let p = if path.is_empty() { self.workspace.clone() } else { self.resolve(path) };
        let mut files = Vec::new();
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
