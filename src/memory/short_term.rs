use std::collections::VecDeque;

pub struct ShortTermMemory {
    messages: VecDeque<(String, String)>,
    actions: Vec<String>,
    files: Vec<String>,
    summary: Option<String>,
    max_messages: usize,
}

/// Rough token estimate — code-heavy text is denser than plain prose.
pub fn estimate_tokens(text: &str) -> usize {
    let chars = text.chars().count();
    let non_ascii = text.chars().filter(|c| !c.is_ascii()).count();
    chars / 3 + non_ascii
}

impl ShortTermMemory {
    pub fn new(max_messages: usize) -> Self {
        ShortTermMemory { messages: VecDeque::new(), actions: Vec::new(), files: Vec::new(), summary: None, max_messages }
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

    /// Full conversation transcript (messages + summary prefix).
    pub fn transcript(&self) -> String {
        let mut lines = Vec::new();
        if let Some(summary) = &self.summary {
            lines.push(format!("[Session summary so far] {}", summary));
        }
        for (role, content) in &self.messages {
            lines.push(format!("{role}: {content}"));
        }
        lines.join("\n")
    }

    /// Estimated token count of the full transcript.
    pub fn estimated_tokens(&self) -> usize {
        estimate_tokens(&self.transcript())
    }

    /// Collapse old messages into a summary, keeping the most recent message.
    pub fn compact(&mut self, summary: String) {
        let kept = self.messages.back().cloned();
        self.messages.clear();
        if let Some(msg) = kept {
            self.messages.push_back(msg);
        }
        self.summary = Some(summary);
    }

    pub fn get_context(&self) -> String {
        let mut lines = Vec::new();
        if let Some(summary) = &self.summary {
            lines.push(format!("Session summary: {}", summary));
        }
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

    /// Messages in display order, for the interactive chat UI.
    pub fn history(&self) -> Vec<(String, String)> {
        self.messages.iter().cloned().collect()
    }

    pub fn last_message(&self) -> Option<(String, String)> {
        self.messages.back().cloned()
    }

    pub fn clear(&mut self) { self.messages.clear(); self.actions.clear(); self.files.clear(); self.summary = None; }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_estimate_is_positive() {
        assert!(estimate_tokens("fn main() { println!(\"hello\"); }") > 0);
    }

    #[test]
    fn compact_keeps_last_message_and_summary() {
        let mut m = ShortTermMemory::new(20);
        m.add_message("user", "task one");
        m.add_message("assistant", "did one");
        m.add_message("user", "task two");
        m.compact("summarized".into());
        let ctx = m.get_context();
        assert!(ctx.contains("Session summary: summarized"));
        assert!(ctx.contains("task two"));
    }
}
