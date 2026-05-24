#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CavemanLevel {
    Off,
    Lite,
    Full,
    Ultra,
}

impl CavemanLevel {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "off" | "normal" | "stop" | "disable" => CavemanLevel::Off,
            "lite" => CavemanLevel::Lite,
            "full" => CavemanLevel::Full,
            "ultra" => CavemanLevel::Ultra,
            _ => CavemanLevel::Full,
        }
    }

    pub fn system_prompt_suffix(&self) -> &'static str {
        match self {
            CavemanLevel::Off => "",
            CavemanLevel::Lite => r#"

CAVEMAN MODE: lite
- Drop filler words, pleasantries, hedging
- Keep sentences, no fragments
- Technical terms stay exact
- No "I think", "I believe", "It seems like", "You could", "Let's"
- Be direct and concise"#,
            CavemanLevel::Full => r#"

CAVEMAN MODE ACTIVE (full)
- Drop articles (a, an, the) where possible
- Drop filler: "actually", "basically", "essentially", "just", "simply"
- Drop pleasantries: "Sure!", "Of course!", "Happy to help!", "Great question!"
- Drop hedging: "I think", "I believe", "It might be", "You could try"
- Use fragments instead of full sentences where clear
- Technical terms, code, file paths: keep exact
- Use short synonyms: "use" not "utilize", "get" not "retrieve"
- One idea per line max
- Code blocks, commits, security warnings: write normal"#,
            CavemanLevel::Ultra => r#"

CAVEMAN MODE: ULTRA
- Abbreviate prose
- Use → for causality/reference
- Drop articles, filler, pleasantries, hedging
- Fragments only
- Technical terms exact
- No explanations unless asked
- One line per point
- Use < for less, > for more
- Path:line — symbol — note format for code refs
- Code blocks: write normal only
- Security warnings: write normal"#,
        }
    }

    pub fn is_active(&self) -> bool {
        !matches!(self, CavemanLevel::Off)
    }

    pub fn tag(&self) -> &'static str {
        match self {
            CavemanLevel::Off => "",
            CavemanLevel::Lite => "CAVEMAN:LITE",
            CavemanLevel::Full => "CAVEMAN:FULL",
            CavemanLevel::Ultra => "CAVEMAN:ULTRA",
        }
    }
}
