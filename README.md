1. Project Type and Tech Stack
Language: Rust (2021 edition)  
Build System: Cargo (Cargo.toml / Cargo.lock)  
Binary: anamnesic (entry point: src/main.rs)  
Description: "Local coding agent — plan → act → verify. Fusion of TinyCoder + llm-on-legacy-gpus"
Key Dependencies
Category	Crates
TUI	ratatui (0.29, with unstable-rendered-line-info), crossterm (0.28)
Async runtime	tokio (1, full features)
LLM / HTTP	reqwest (0.12), serde/serde_json, axum (0.8, with ws)
Local inference	ocl (optional, GPU), bytemuck, rand
Database	rusqlite (0.31, bundled)
Terminal/PTY	portable-pty (0.8) — ConPTY on Windows, PTY on Unix
Search/Regex	regex (1), ignore (0.4)
Diff/Text	diffy (0.4), pulldown-cmark (0.12), unicode-segmentation, unicode-width
CLI	clap (4, derive)
Testing	wiremock (0.6, dev-dependency)
2. Directory Structure Overview
C:\Users\luann\Documents\GitHub\anamnesic-coder/
├── Cargo.toml
├── Cargo.lock
├── src/
│   ├── main.rs                    # Entry point, CLI parsing, REPL, session resume
│   ├── agent/
│   │   ├── mod.rs
│   │   ├── agent_loop.rs          # Core agent loop, tool execution, MCP integration, search_code
│   │   ├── executor.rs            # Step execution, search_code + grep_context helpers
│   │   ├── planner.rs
│   │   └── state.rs               # AgentState (session, MCP clients, todos, transactions)
│   ├── bench/                     # Model benchmarking (local + cloud)
│   ├── compressor/                # L1/L2 output compression
│   ├── config/                    # Settings, global settings, allow/deny lists
│   ├── hw_recommend/              # Hardware detection + model recommendations
│   ├── llm/                       # LLM router, client, provider chain, GGUF inference, embedder
│   │   └── infer/                 # GGUF reading, tokenizer, engine, GPU kernels
│   ├── mcp/                       # MCP client (JSON-RPC stdio subprocess)
│   ├── memory/                    # Short-term + long-term (SQLite) memory
│   ├── models_dev/                # models.dev catalog client
│   ├── providers/                 # Provider store, verification
│   ├── repo/                      # Repo scanner, SymbolIndex, RepoMapGenerator, context
│   ├── skills/                    # Skills system (SKILL.md packs)
│   ├── terminal/                  # PTY + WebSocket server for browser TUI
│   │   ├── pty.rs                 # TerminalSession (portable-pty wrapper)
│   │   ├── session.rs             # TerminalSessionManager
│   │   ├── server.rs              # Axum HTTP + WS server
│   │   ├── shell.rs               # Default shell detection (Windows/Unix)
│   │   └── websocket.rs
│   ├── tools/                     # Agent tool implementations
│   │   ├── fs.rs                  # FileTools (path-scoped, symlink, allow/deny)
│   │   ├── transaction.rs         # WorkspaceTransaction, diff, checksums
│   │   ├── shell.rs               # Shell command execution (cross-platform)
│   │   ├── git.rs                 # Git tools
│   │   ├── test.rs                # Verification gates (cargo test, pytest, npm)
│   │   ├── background.rs          # Background task manager (C9)
│   │   └── web.rs                 # Web search + fetch
│   ├── types/                     # Shared types (error, action, plan)
│   └── ui/                        # Ratatui TUI
│       ├── mod.rs                 # Main TUI (~4023 lines)
│       ├── file_search.rs         # Fuzzy file search (Ctrl+P)
│       ├── diff_render.rs
│       ├── live_wrap.rs
│       ├── pager_overlay.rs
│       └── line_truncation.rs
├── docs/
│   ├── adr/                       # 15 ADRs (0001–0015)
│   └── gap-analysis-2026-08.md
├── prompts/                       # System/coder/planner/fixer prompt files
├── sandbox/                       # Dockerfile + runner.sh
└── memory_data/memory.db          # SQLite long-term memory
3. Existing Files Related to Backlog Items
A. Symlink Handling
Key files:
- src/tools/fs.rs — FileTools with workspace containment logic
- Lines 27–50: normalize_workspace_path() with Windows verbatim (\\?\) prefix support
- Line 134: Unix-only test rejects_symlink_escape_outside_workspace — creates a symlink to /etc and verifies the agent cannot read/write through it
- Lines 518+: file_type.is_symlink() check in list_files/list_tree traversal
- Lines 193–221: Windows-only tests verbatim_prefix_workspace_is_supported and absolute_verbatim_path_inside_workspace_resolves
- src/tools/transaction.rs — WorkspaceTransaction
- Line 275: if path.is_symlink() { continue; } — symlinks are skipped during workspace snapshot
- src/agent/agent_loop.rs
- Line 1642: if file_type.is_symlink() { continue; } in search_workspace_without_rg()
- docs/adr/0008-robustness-and-floating-point-safety.md — Documents path traversal, symlink escape, and injection safety fixes
Windows-specific gap: Only the Unix symlink escape test exists (#[cfg(unix)]). There is no Windows junction/reparse-point symlink test. The Windows FileTools tests focus on verbatim path prefixes instead.
B. Search/Lookup Functionality (Regex-based)
Key files:
- src/agent/agent_loop.rs
- Lines 1622–1686: search_workspace_without_rg() — fallback regex search when rg is not installed. Uses regex::Regex::new(pattern) with literal fallback, scans files skipping symlinks, .git, target, node_modules, memory_data
- Lines 850: "search_code" tool dispatch
- Lines 1793–1794: Tool definition: "search_code" — "Search workspace text using a regex or literal pattern."
- src/agent/executor.rs
- Lines 302–370: search_code(state, pattern) function — prefers rg (ripgrep), falls back to search_workspace_without_rg()
- Lines 426–485: grep_context() helper
- src/repo/scanner.rs — RepoMapGenerator + SymbolIndex
- Lines 130–194: Regex-based symbol extraction for Rust (fn/struct/enum/trait), Python (def/class), JS/TS (function/class)
- Lines 197–241: Tests for repo map generation and symbol index search
- src/ui/file_search.rs — Fuzzy file search for Ctrl+P
- walk_files(), search_files(), fuzzy_match() (subsequence matching with scoring)
- 8 inline tests
- src/compressor/layer1.rs and layer2.rs — Regex-based output compression (timestamps, UUIDs, hashes, JWT, etc.)
- src/llm/router.rs — capability_lookup_reflects_the_active_provider_catalog() test
C. MCP Server Management
Key files:
- src/mcp/mod.rs — Complete MCP client implementation (235 lines)
- McpServerConfig struct: command, args, env
- McpClient::connect() — spawns subprocess, pipes stdin/stdout
- JSON-RPC 2.0 protocol: initialize, tools/list, tools/call
- list_tools() → converts to ToolDef format
- has_tool() / call_tool() for dispatching
- Tests: config creation/equality, fake_mcp_server_process (spawns self as fake MCP server via ANAMNESIC_FAKE_MCP_SERVER=1)
- src/agent/agent_loop.rs
- Lines 675–683: connect_mcp_clients() — connects all configured MCP servers at startup
- Lines 685–717: try_mcp_tool() — dispatches tool calls to MCP clients with approval policy
- Lines 1962–1964: MCP tools are enumerated and merged with native tools
- Lines 2406–2454: MCP tool tests (no clients, ask policy, allow policy, run policy)
- src/config/settings.rs
- Line 123: pub mcp_servers: Vec<crate::mcp::McpServerConfig> in Config
- src/agent/state.rs
- Line 40: pub mcp_clients: Vec<crate::mcp::McpClient> in AgentState
- docs/adr/0012-mcp-client.md — ADR for MCP stdio client
D. Session Management
Key files:
- src/agent/state.rs
- AgentState struct fields: session (ShortTermMemory), session_id (Option<i64>), session_persist, last_persisted_seq, last_persisted_summary
- ensure_session() — lazily creates persistent session record
- persist_session() — writes transcript to DB with compaction summary
- load_session_into_state(id) — restores session from DB, injects summary if present
- reset() — clears session, sets session_id = None for fresh start
- start_turn() — begins WorkspaceTransaction
- src/memory/short_term.rs — ShortTermMemory (in-context conversation history with token budget)
- src/memory/log.rs — LongTermMemory (SQLite-backed, session CRUD, vector store for memory_search)
- src/main.rs
- Lines 111–116: --resume / --cont CLI flags
- Lines 298–300: continue_latest_session() on startup
- Lines 441–460: resume_session() / continue_latest_session() — finds latest session for workspace and restores it
E. Terminal/TUI Components
Key files:
- src/ui/mod.rs — Main TUI (~4023 lines)
- Ratatui + Crossterm backend
- Focus modes: Sidebar, Messages, Input, Editor
- Slash commands: /info, /help, /status, /model, /provider, /reset, /resume, /continue
- Approval modal (a allow once, s allow session, d/Esc deny)
- File search overlay (Ctrl+P)
- src/terminal/ — Web-based terminal via WebSocket
- pty.rs: TerminalSession using portable-pty (ConPTY on Windows)
- session.rs: TerminalSessionManager (create/get/remove/resize sessions)
- server.rs: Axum HTTP + WS server, serves xterm.js SPA
- shell.rs: Platform-aware default shell detection
- src/main.rs
- Commands::Tui — launches ui::run_ui()
- Commands::Serve — launches terminal::server::serve() for browser access
- Auto-detects TUI mode when stdin/stdout are terminals
4. Key Entry Points and Main Modules
Entry Point	Location
main()	src/main.rs:259
run_agent_loop()	src/agent/agent_loop.rs
ui::run_ui()	src/ui/mod.rs
terminal::server::serve()	src/terminal/server.rs
repl()	src/main.rs:384
build_router()	src/main.rs:202
5. Testing Infrastructure
Framework: Rust built-in #[test] (no external test framework like rstest or proptest)  
Test count: 298 test functions across the codebase  
Dev dependency: wiremock (0.6) — available but not heavily used in the current inline tests
Test distribution by file:
File
src/agent/agent_loop.rs
src/tools/fs.rs
src/tools/shell.rs
src/llm/router.rs
src/memory/short_term.rs
src/config/settings.rs
src/compressor/layer1.rs
src/ui/file_search.rs
src/tools/transaction.rs
src/agent/state.rs
Test patterns:
- Most tests use temp directories (std::env::temp_dir()) with cleanup
- MCP tests use a clever self-spawning fake server (ANAMNESIC_FAKE_MCP_SERVER=1)
- Platform-gated tests: #[cfg(unix)] for symlink tests, #[cfg(windows)] for verbatim path tests
- Agent loop tests use a test_state() helper that creates isolated temp workspaces
6. Competitive Backlog References (C1, C2, C8)
Primary reference: docs/adr/0015-competitive-backlog.md
C1 — Path-scoped permissions ✅ Implemented
- Files: src/config/settings.rs, src/tools/fs.rs, src/tools/shell.rs
- path_allowlist, path_denylist, block_workspace_escape config
- FileTools::resolve() enforces workspace containment
- Symlink escape prevention (Unix tested, Windows not explicitly tested for junctions)
C2 — Symbol / LSP search ✅ Implemented (regex-based, not full LSP)
- Files: src/repo/scanner.rs, src/agent/agent_loop.rs
- SymbolIndex::build() extracts symbols via regex (Rust, Python, JS/TS)
- SymbolIndex::search() and search_type()
- RepoMapGenerator::generate_map() produces compact repo map
- TODO.md notes: "Melhora futura: símbolos via LSP/tree-sitter (ver C2 no backlog competitivo)" — indicates the regex approach is a stepping stone to full LSP
C8 — Notebook editing / timers / image gen ❌ Deferred
- ADR states: "Requires heavy dependencies (tree-sitter, external APIs). Not in scope."
- TODO.md: "Notebook editing (Claude/Cursor), timers/cron (Antigravity), image gen (Antigravity). Avaliar público-alvo."
Other C-items (for context):
Item	Status
C3 Parallel sub-agents	✅
C4 Skills system	✅
C5 Checksums/change-tracking	✅
C6 Extended thinking	✅
C7 Adversarial verification	✅
C9 Background tasks	✅
7. Notable Gaps / Observations
1. Windows symlink testing is absent. The only symlink escape test is #[cfg(unix)]. On Windows, NTFS junctions/reparse points are not tested for escape scenarios.
2. Search is regex + ripgrep fallback, not LSP. search_code uses rg or a manual directory walk with regex::Regex. C2 was "implemented" as regex symbol extraction, but the TODO explicitly flags LSP/tree-sitter as the future direction.
3. No integration test harness with mock LLM providers. The dev-dependencies include wiremock but the inline tests use a self-spawning MCP fake server pattern instead. There is no mock Ollama/cloud provider for end-to-end agent loop testing.
4. Testing is all inline #[test] modules. No separate tests/ directory or integration test binaries (except the self-spawning MCP server trick).
5. C8 is explicitly deferred — notebook editing, timers, and image generation are out of scope per ADR 0015.