use std::fs;
use std::path::Component;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Lexically normalize a path: collapse `.` and resolve `..` without touching the filesystem.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

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
    fn allows_absolute_paths_within_workspace() {
        let workspace = temp_workspace();
        let tools = FileTools::new(workspace.clone());
        let abs = workspace.join("sub/abs.txt");
        tools
            .write_file(abs.to_str().unwrap(), "via absolute\n")
            .unwrap();
        assert_eq!(tools.read_file("sub/abs.txt").unwrap(), "via absolute\n");
        assert_eq!(
            tools.read_file(abs.to_str().unwrap()).unwrap(),
            "via absolute\n"
        );
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_escape_outside_workspace() {
        let workspace = temp_workspace();
        std::fs::create_dir_all(workspace.join("link")).ok();
        let target = workspace.join("link").join("evil");
        std::os::unix::fs::symlink(std::path::Path::new("/etc"), &target).unwrap();
        let tools = FileTools::new(workspace.clone());
        assert!(tools.read_file("link/evil/passwd").is_none());
        assert!(tools.write_file("link/evil/newfile", "x").is_err());
        assert!(!std::path::Path::new("/etc/newfile").exists());
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
    fn list_files_returns_files_and_dirs_within_workspace() {
        let workspace = temp_workspace();
        let tools = FileTools::new(workspace.clone());
        tools.write_file("a.txt", "a").unwrap();
        tools.write_file("sub/b.txt", "b").unwrap();
        let root = tools.list_files("");
        assert!(root.contains(&"a.txt".to_string()));
        assert!(root.contains(&"sub/".to_string()), "directories should appear with trailing /");
        assert_eq!(tools.list_files("sub"), vec!["sub/b.txt"]);
    }
}

impl FileTools {
    pub fn new(workspace: PathBuf) -> Self {
        fs::create_dir_all(&workspace).ok();
        FileTools { workspace }
    }

    fn resolve(&self, path: &str) -> Option<PathBuf> {
        let raw = PathBuf::from(path);
        let joined = if raw.is_absolute() {
            raw
        } else {
            self.workspace.join(raw)
        };
        let normalized = normalize(&joined);
        let ws = self
            .workspace
            .canonicalize()
            .unwrap_or_else(|_| normalize(&self.workspace));

        let mut current = normalized.clone();
        let mut suffix: Vec<std::ffi::OsString> = Vec::new();
        while !current.exists() {
            match current.parent() {
                Some(parent) => {
                    suffix.push(current.file_name()?.to_os_string());
                    current = parent.to_path_buf();
                }
                None => return None,
            }
        }
        let mut real = current.canonicalize().ok()?;
        for part in suffix.iter().rev() {
            real.push(part);
        }
        if real.starts_with(&ws) {
            Some(real)
        } else {
            None
        }
    }

    pub fn read_file(&self, path: &str) -> Option<String> {
        let p = self.resolve(path)?;
        fs::read_to_string(p).ok()
    }

    pub fn write_file(&self, path: &str, content: &str) -> anyhow::Result<()> {
        let p = self
            .resolve(path)
            .ok_or_else(|| anyhow::anyhow!("path must be inside the workspace"))?;
        self.atomic_write(&p, content)
    }

    pub fn append_file(&self, path: &str, content: &str) -> anyhow::Result<()> {
        let p = self
            .resolve(path)
            .ok_or_else(|| anyhow::anyhow!("path must be relative to the workspace"))?;
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = fs::OpenOptions::new().append(true).create(true).open(&p)?;
        use std::io::Write;
        file.write_all(content.as_bytes())?;
        Ok(())
    }

    pub fn list_files(&self, path: &str) -> Vec<String> {
        let mut entries_out = Vec::new();
        let Some(p) = (if path.is_empty() {
            Some(self.workspace.clone())
        } else {
            self.resolve(path)
        }) else {
            return entries_out;
        };
        let ws = self.workspace.canonicalize().unwrap_or_else(|_| self.workspace.clone());
        let real_p = p.canonicalize().unwrap_or(p);
        if let Ok(entries) = fs::read_dir(&real_p) {
            for entry in entries.flatten() {
                let entry_path = entry.path().canonicalize().unwrap_or_else(|_| entry.path());
                if let Ok(rel) = entry_path.strip_prefix(&ws) {
                    let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
                    let mut name = rel.to_string_lossy().replace('\\', "/");
                    if is_dir && !name.ends_with('/') {
                        name.push('/');
                    }
                    entries_out.push(name);
                }
            }
        }
        entries_out.sort();
        entries_out
    }

    pub fn read_file_range(
        &self,
        path: &str,
        start_line: usize,
        end_line: usize,
    ) -> anyhow::Result<String> {
        if start_line == 0 || end_line < start_line {
            anyhow::bail!("line range must be 1-based and end_line >= start_line");
        }
        let content = self
            .read_file(path)
            .ok_or_else(|| anyhow::anyhow!("file not found or path is outside workspace"))?;
        let lines: Vec<&str> = content.lines().collect();
        let selected = lines
            .iter()
            .enumerate()
            .skip(start_line - 1)
            .take(end_line - start_line + 1)
            .map(|(index, line)| format!("{:>6}  {}", index + 1, line))
            .collect::<Vec<_>>()
            .join("\n");
        Ok(format!(
            "[lines {}-{} of {}]\n{}",
            start_line,
            end_line.min(lines.len()),
            lines.len(),
            selected
        ))
    }

    pub fn list_tree(
        &self,
        path: &str,
        max_depth: usize,
        max_entries: usize,
    ) -> anyhow::Result<String> {
        let root = if path.is_empty() {
            self.workspace.clone()
        } else {
            self.resolve(path)
                .ok_or_else(|| anyhow::anyhow!("path is outside workspace"))?
        };
        if !root.is_dir() {
            anyhow::bail!("tree path is not a directory");
        }
        let mut pending = vec![(root, 0usize)];
        let mut output = Vec::new();
        while let Some((directory, depth)) = pending.pop() {
            if depth > max_depth || output.len() >= max_entries {
                continue;
            }
            let mut entries = fs::read_dir(&directory)?
                .filter_map(Result::ok)
                .collect::<Vec<_>>();
            entries.sort_by_key(|entry| entry.file_name());
            for entry in entries.into_iter().rev() {
                if output.len() >= max_entries {
                    break;
                }
                let file_type = entry.file_type()?;
                if file_type.is_symlink() {
                    continue;
                }
                let relative = entry
                    .path()
                    .strip_prefix(&self.workspace)
                    .unwrap_or(&entry.path())
                    .to_string_lossy()
                    .replace('\\', "/");
                if file_type.is_dir() {
                    output.push(format!("{relative}/"));
                    if depth < max_depth {
                        pending.push((entry.path(), depth + 1));
                    }
                } else if file_type.is_file() {
                    output.push(relative);
                }
            }
        }
        output.sort();
        if output.len() >= max_entries {
            output.push(format!("...[truncated at {max_entries} entries]"));
        }
        Ok(output.join("\n"))
    }

    pub fn replace_exact(&self, path: &str, old: &str, new: &str) -> anyhow::Result<()> {
        if old.is_empty() {
            anyhow::bail!("old text must not be empty");
        }
        let target = self
            .resolve(path)
            .ok_or_else(|| anyhow::anyhow!("path is outside workspace"))?;
        let content = fs::read_to_string(&target)?;
        let matches = content.match_indices(old).count();
        if matches != 1 {
            anyhow::bail!("expected exactly one match, found {matches}; file was not changed");
        }
        self.atomic_write(&target, &content.replacen(old, new, 1))
    }

    fn atomic_write(&self, path: &Path, content: &str) -> anyhow::Result<()> {
        let parent = path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("file has no parent directory"))?;
        fs::create_dir_all(parent)?;
        let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let temporary = parent.join(format!(".anamnesic-{}-{counter}.tmp", std::process::id()));
        fs::write(&temporary, content)?;
        if let Err(error) = fs::rename(&temporary, path) {
            let _ = fs::remove_file(&temporary);
            return Err(error.into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod transactional_tests {
    use super::FileTools;
    use std::fs;

    fn workspace(name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "anamnesic-fs-transaction-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn replace_exact_changes_one_match() {
        let root = workspace("one");
        let tools = FileTools::new(root.clone());
        tools.write_file("a.txt", "before unique after").unwrap();
        tools.replace_exact("a.txt", "unique", "changed").unwrap();
        assert_eq!(tools.read_file("a.txt").unwrap(), "before changed after");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn replace_exact_leaves_stale_or_ambiguous_files_untouched() {
        let root = workspace("conflict");
        let tools = FileTools::new(root.clone());
        tools.write_file("a.txt", "same same").unwrap();
        assert!(tools.replace_exact("a.txt", "missing", "x").is_err());
        assert!(tools.replace_exact("a.txt", "same", "x").is_err());
        assert_eq!(tools.read_file("a.txt").unwrap(), "same same");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn ranged_read_and_tree_are_bounded() {
        let root = workspace("discovery");
        let tools = FileTools::new(root.clone());
        tools.write_file("src/lib.rs", "one\ntwo\nthree\n").unwrap();
        let range = tools.read_file_range("src/lib.rs", 2, 3).unwrap();
        assert!(range.contains("2  two"));
        assert!(!range.contains("one"));
        let tree = tools.list_tree("", 2, 20).unwrap();
        assert!(tree.contains("src/lib.rs"));
        let _ = fs::remove_dir_all(root);
    }
}
