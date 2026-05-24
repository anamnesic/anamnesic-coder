use std::collections::VecDeque;

pub struct ShortTermMemory {
    messages: VecDeque<(String, String)>,
    actions: Vec<String>,
    files: Vec<String>,
    max_messages: usize,
}

impl ShortTermMemory {
    pub fn new(max_messages: usize) -> Self {
        ShortTermMemory { messages: VecDeque::new(), actions: Vec::new(), files: Vec::new(), max_messages }
    }

    pub fn add_message(&mut self, role: &str, content: &str) {
        self.messages.push_back((role.to_string(), content.to_string()));
        if self.messages.len() > self.max_messages { self.messages.pop_front(); }
    }

    pub fn add_action(&mut self, action: &str) { self.actions.push(action.to_string()); }

    pub fn add_file(&mut self, filepath: &str) {
        if !self.files.contains(&filepath.to_string()) {
            self.files.push(filepath.to_string());
        }
    }

    pub fn get_context(&self) -> String {
        let mut lines = Vec::new();
        if let Some(last) = self.messages.back() {
            lines.push(format!("Last message ({}): {}", last.0, &last.1[..last.1.len().min(200)]));
        }
        if !self.actions.is_empty() {
            let recent: Vec<&str> = self.actions.iter().rev().take(5).map(|s| s.as_str()).collect();
            lines.push(format!("Recent actions: {}", recent.join(", ")));
        }
        if !self.files.is_empty() {
            let recent: Vec<&str> = self.files.iter().rev().take(10).map(|s| s.as_str()).collect();
            lines.push(format!("Files: {}", recent.join(", ")));
        }
        lines.join("\n")
    }

    pub fn clear(&mut self) { self.messages.clear(); self.actions.clear(); self.files.clear(); }
}
