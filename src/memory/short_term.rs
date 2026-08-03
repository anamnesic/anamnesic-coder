use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct ShortTermMemory {
    messages: VecDeque<(String, String)>,
    actions: Vec<String>,
    files: Vec<String>,
    summary: Option<String>,
    max_messages: usize,
}

/// Calibrated BPE token estimator for code, prose, and JSON structures.
pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    let mut tokens = 0usize;
    let mut current_word_len = 0usize;

    for c in text.chars() {
        if c.is_ascii_whitespace() {
            if current_word_len > 0 {
                tokens += (current_word_len + 3) / 4;
                current_word_len = 0;
            }
            tokens += 1;
        } else if c.is_ascii_punctuation() {
            if current_word_len > 0 {
                tokens += (current_word_len + 3) / 4;
                current_word_len = 0;
            }
            tokens += 1;
        } else if !c.is_ascii() {
            if current_word_len > 0 {
                tokens += (current_word_len + 3) / 4;
                current_word_len = 0;
            }
            tokens += 2;
        } else {
            current_word_len += 1;
        }
    }
    if current_word_len > 0 {
        tokens += (current_word_len + 3) / 4;
    }
    tokens.max(1)
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
    fn token_estimate_counts_non_ascii_extra() {
        let ascii = estimate_tokens("hello world");
        let non_ascii = estimate_tokens("olá mundo");
        assert!(non_ascii >= ascii);
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

    #[test]
    fn prunes_oldest_messages_over_capacity() {
        let mut m = ShortTermMemory::new(2);
        m.add_message("user", "1");
        m.add_message("user", "2");
        m.add_message("user", "3");
        let hist = m.history();
        assert_eq!(hist.len(), 2);
        assert_eq!(hist[0].1, "2");
        assert_eq!(hist[1].1, "3");
    }

    #[test]
    fn add_file_deduplicates() {
        let mut m = ShortTermMemory::new(10);
        m.add_file("src/main.rs");
        m.add_file("src/main.rs");
        assert_eq!(m.files.len(), 1);
    }

    #[test]
    fn transcript_contains_messages_in_order() {
        let mut m = ShortTermMemory::new(10);
        m.add_message("user", "hi");
        m.add_message("assistant", "yo");
        let t = m.transcript();
        assert!(t.contains("user: hi"));
        assert!(t.contains("assistant: yo"));
        assert!(t.find("user: hi").unwrap() < t.find("assistant: yo").unwrap());
    }

    #[test]
    fn last_message_returns_most_recent() {
        let mut m = ShortTermMemory::new(10);
        m.add_message("user", "first");
        m.add_message("user", "last");
        assert_eq!(m.last_message().unwrap().1, "last");
        assert!(ShortTermMemory::new(5).last_message().is_none());
    }

    #[test]
    fn get_context_truncates_long_last_message() {
        let mut m = ShortTermMemory::new(10);
        m.add_message("user", &"x".repeat(500));
        let ctx = m.get_context();
        assert!(ctx.len() < 400);
    }

    #[test]
    fn clear_resets_all_state() {
        let mut m = ShortTermMemory::new(10);
        m.add_message("user", "hi");
        m.add_action("act");
        m.add_file("f.txt");
        m.compact("sum".into());
        m.clear();
        assert!(m.history().is_empty());
        assert!(m.get_context().is_empty());
        assert!(m.last_message().is_none());
    }
}
