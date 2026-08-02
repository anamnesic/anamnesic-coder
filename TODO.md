# TODO — Anamnesic Coder (August 2026)

## P0 — Safety and Reliability (CRITICAL)

### ~~1. Command Injection via Prefix-Based Allowlist~~ ✅ FIXED
- **File:** `src/tools/shell.rs`
- **Fix applied:** `parse_command()` rejects all shell metacharacters (`;`, `&`, `|`, `` ` ``, `$`, `(`, `)`, `{`, `}`, `<`, `>`, `\n`, `\\`) before execution. `is_allowed()` validates the parsed executable against the allowlist, not a prefix match. `run_command_raw` also validates through `is_allowed()`.

### ~~2. No Timeout or Process-Group Kill on `run_command`~~ ✅ FIXED
- **File:** `src/tools/shell.rs`
- **Fix applied:** `run_command_inner` uses `child.try_wait()` polling with configurable `command_timeout_secs` (default 600s). On timeout, kills the process group (Unix) or child process (Windows). Pipe readers run on separate threads to prevent deadlock.

### ~~3. `run_command_raw` Bypasses the Allowlist Entirely~~ ✅ FIXED
- **File:** `src/tools/shell.rs`
- **Fix applied:** `run_command_raw` now validates through `is_allowed()` first and returns an error `CommandOutput` if rejected.

### ~~4. `unsafe` `transmute` on Untrusted GGUF Data~~ ✅ FIXED
- **File:** `src/llm/infer/gguf.rs`
- **Fix applied:** Replaced `unsafe transmute` with `TryFrom<i32>` for `GgmlType` (ADR 0007).

### ~~5. GGUF Parsing Panics on Truncated/Corrupt Files~~ ✅ FIXED
- **File:** `src/llm/infer/gguf.rs`
- **Fix applied:** All binary read methods use bounds-checked `read_bytes` with explicit EOF error propagation (ADR 0007).

### ~~6. Q4_0/Q8_0 Dequantization Panics on Corrupt Data~~ ✅ FIXED
- **File:** `src/llm/infer/model.rs`
- **Fix applied:** Added bounds checks to `dequantize_q4_0_row`, `dequantize_q8_0_row`, and `dequantize_f16_row` (ADR 0007).

### ~~7. `tensor_data` Can Read Beyond Buffer~~ ✅ FIXED
- **File:** `src/llm/infer/gguf.rs`
- **Fix applied:** Added `checked_add` and bounds check on slice boundaries in `tensor_data` (ADR 0007).

### ~~8. TOCTOU Race in `FileTools::resolve`~~ ✅ FIXED
- **File:** `src/tools/fs.rs`
- **Lines:** 126–155
- **Issue:** Between the `canonicalize()` check and the actual `fs::read_to_string`/`fs::write` call, an attacker with concurrent access could replace a file with a symlink.
- **Fix:** Re-validate the path after canonicalization, or use `O_NOFOLLOW` where available.

## P0 — Harness Gaps (from gap analysis vs. Claude Code / Codex / Antigravity)

> See full report: `docs/gap-analysis-2026-08.md`

### G1. Line-Range Code Editing (edit_file / multi_edit_file)
- **Gap:** `replace_exact` exige match exato de string e não suporta edições multi-site. Todos os líderes (Claude Code, Antigravity, Cursor) usam edição por line-range.
- **Impact:** Crítico para SWE-bench — modelos erram whitespace/indentation frequentemente, causando falha de match.
- **Files:** `src/tools/fs.rs`, `src/agent/executor.rs`
- **Fix:** Implementar `edit_file(path, start_line, end_line, old_content, new_content)` com line-range anchoring. Adicionar `multi_edit_file(path, edits[])` para edições não-contíguas no mesmo arquivo. Manter `replace_exact` como fallback.
- **Ref:** Antigravity `replace_file_content` / `multi_replace_file_content`; Claude Code `Edit` tool.

### G2. Approval Broker Not Wired (security gap)
- **Gap:** Os tipos `ApprovalRequest`, `ApprovalDecision` e `AgentHooks.on_approval` existem em `src/agent/loop.rs:50-71`, mas `execute_tool_call()` nunca chama `on_approval()`. Writes e commands executam sem gate, mesmo com `write_tool_policy: Ask`.
- **Impact:** Segurança — o modelo pode escrever/executar qualquer coisa sem aprovação.
- **Files:** `src/agent/loop.rs`, `src/agent/executor.rs`
- **Fix:** No dispatch de tools mutadores/commands, verificar a policy (`write_tool_policy`/`command_tool_policy`) e chamar `hooks.on_approval()` antes de executar. Se `Deny` ou sem callback, retornar erro.

### G3. Context Compaction / Conversation Summarization
- **Gap:** O histórico de conversa cresce indefinidamente. Quando excede o contexto do modelo, o loop falha. Nenhuma sumarização ou compactação de mensagens antigas.
- **Impact:** Tarefas longas (multi-step refactoring) falham por context overflow.
- **Files:** `src/agent/loop.rs`, `src/compressor/`
- **Fix:** Quando `estimated_tokens > 0.8 * max_context_tokens`, sumarizar mensagens antigas (exceto as últimas N) usando o modelo summarizer. O compressor já existe em `src/compressor/` mas não está integrado no agent loop.

### G4. Token Counting
- **Gap:** Não há contagem de tokens. Não sabe quanto contexto resta por turno.
- **Impact:** Pré-requisito para G3 (compaction) e G6 (cost tracking).
- **Files:** `src/llm/client.rs`, `src/agent/loop.rs`
- **Fix:** Adicionar estimativa de tokens (chars/4 como baseline, ou tiktoken-rs). Rastrear tokens in/out em cada chamada LLM. Expor `remaining_context()` para o agent loop.

## P1 — Error Handling & Robustness

### ~~9. `unwrap()` in Production Paths~~ ✅ FIXED
- **Files:** `src/main.rs:401`, `src/agent/state.rs:61,74,83`, `src/llm/router.rs` (Mutex locks)
- **Issue:** `unwrap()` calls that can panic in production.
- **Fix:** Replace with proper error handling or `expect()` with descriptive messages.
- **Note:** Several `unwrap()` sites in `src/llm/client.rs` have been addressed by the retry+backoff refactor. Mutex lock unwraps in the router are considered acceptable (poisoned mutex = unrecoverable).

### ~~10. `partial_cmp().unwrap()` on Costs (NaN Panic)~~ ✅ FIXED
- **File:** `src/models_dev/client.rs`
- **Lines:** 81, 100, 149
- **Issue:** Will panic if any cost is NaN.
- **Fix:** Use `.unwrap_or(Equal)`.
- **Note:** `bench/model_bench.rs`, `hw_recommend/recommender.rs`, `llm/infer/engine.rs:325` already use `.unwrap_or(Equal)`. One remaining bare `.unwrap()` at `engine.rs:333`.

### ~~11. `partial_cmp().unwrap()` in Top-K Sampling~~ ✅ FIXED
- **File:** `src/llm/infer/engine.rs`
- **Line:** 333
- **Issue:** Will panic on NaN logits in `select_nth_unstable_by`.
- **Fix:** Use `.unwrap_or(Equal)`.

### ~~12. `unwrap_or_default` Silently Swallows HTTP Read Errors~~ ✅ FIXED
- **File:** `src/llm/client.rs`
- **Lines:** 570, 711, 808, 868 (and others)
- **Issue:** `resp.text().await.unwrap_or_default()` silently discards errors in non-retry paths.
- **Fix:** Propagate errors properly or log them.
- **Note:** The retry paths (429/5xx) now log and retry correctly. The `unwrap_or_default` on response body reading is a separate concern for non-retried paths.

## P1 — Harness Gaps (competitive parity)

### G5. Sub-Agent Support (Task tool)
- **Gap:** O Anamnesic tem apenas um loop sequencial. Claude Code, Antigravity e Cursor suportam sub-agentes para delegação de tarefas e pesquisa paralela.
- **Impact:** Tarefas complexas (multi-arquivo, refatoração) são lentas e gastam mais tokens.
- **Files:** `src/agent/loop.rs` (novo módulo `src/agent/subagent.rs`)
- **Fix:** Implementar tool `task` que spawna um segundo agent loop com contexto isolado. O sub-agente recebe um prompt, executa tools, e retorna o resultado ao agente pai. Limitar depth=1 inicialmente.

### G6. Cost Tracking Per Turn
- **Gap:** Não sabe quanto gastou em tokens/dinheiro por turno ou sessão. Claude Code e Aider mostram isso.
- **Impact:** Ops — sem visibilidade de custos; impossível otimizar.
- **Files:** `src/llm/client.rs`, `src/agent/loop.rs`, `src/ui.rs`
- **Fix:** Em `ChatCompletion`, adicionar `usage: Option<Usage>` (prompt_tokens, completion_tokens). Acumular por turno. Mostrar no status bar do TUI.

### G7. MCP Client (Model Context Protocol)
- **Gap:** Não conecta a tool servers MCP externos. Todos os líderes (Claude Code, Antigravity, Cursor, Codex) suportam MCP.
- **Impact:** Extensibilidade — não pode usar tools de terceiros (GitHub, DB, Jira, etc.).
- **Files:** Novo módulo `src/mcp/`
- **Fix:** Implementar MCP client com stdio transport. Registrar tools MCP dinamicamente no tool registry do executor. Começar com o protocolo mínimo: `initialize`, `tools/list`, `tools/call`.

### G8. Auto-Read Project Context (AGENTS.md)
- **Gap:** O agente não lê nenhum arquivo de contexto de projeto automaticamente. O próprio projeto tem um `AGENTS.md` mas o agente ignora.
- **Impact:** O modelo não tem contexto sobre arquitetura, convenções, e regras do projeto.
- **Files:** `src/agent/loop.rs`, `src/llm/prompt.rs`
- **Fix:** Na construção do system prompt, procurar e ler `AGENTS.md`, `CLAUDE.md`, `.cursorrules`, ou `CONTEXT.md` na raiz do workspace. Injetar o conteúdo no system prompt.

### G9. Streaming Tool Call Deltas
- **Gap:** Tool calls são parseados apenas de respostas completas. Não há streaming incremental de tool call deltas durante SSE.
- **Impact:** UX — o usuário não vê o que o modelo está decidindo até a resposta completa chegar.
- **Files:** `src/llm/client.rs`
- **Fix:** No streaming SSE, parsear `tool_calls` incrementalmente (acumular `function.arguments` chunk-by-chunk). Emitir eventos parciais via `AgentHooks`.

### G10. `list_files` Should Include Directories
- **Gap:** `list_files` só retorna arquivos (`is_file()`), não diretórios. Antigravity e Claude Code retornam ambos.
- **Impact:** O modelo não vê a estrutura de diretórios do projeto.
- **Files:** `src/tools/fs.rs`
- **Fix:** Incluir diretórios no output com um sufixo `/` para distinguir. Ou implementar tool separado `list_dir`.

## P2 — Code Quality & Design

### ~~13. Dead Code: `maybe_compact_chain`~~ ✅ REMOVED
- Removed as part of ADR 0002 (unified orchestration).

### ~~14. Dead Code: `run_agent_loop_with_fallback`~~ ✅ REMOVED
- Removed as part of ADR 0002 (unified orchestration).

### ~~15. Dead Code: `execute_step_inner_chain`~~ ✅ REMOVED
- Removed as part of ADR 0002 (unified orchestration).

### ~~16. Unused `r#loop` Raw Identifier~~ ✅ FIXED
- **File:** `src/main.rs:18`, `src/agent/mod.rs:4`, `src/agent/executor.rs:1`, `src/ui.rs:24,520`
- **Issue:** Module name `loop` fights the Rust keyword, requiring `r#loop` everywhere.
- **Fix:** Rename to `agent_loop` or `cycle`.

### 17. `#![allow(dead_code)]` at Crate Root
- **File:** `src/main.rs`
- **Line:** 3
- **Issue:** Suppresses dead-code warnings for the entire crate.
- **Fix:** Remove and fix dead code.

### 18. `truncate_str` Keeps Tail Instead of Head
- **File:** `src/ui.rs`
- **Lines:** 1455–1463
- **Issue:** Keeps the last `max` characters and prepends ellipsis. For model names and paths, the beginning is usually more informative.
- **Fix:** Change to keep the head (first `max` characters) with trailing ellipsis.

### 19. Conversation Cloned on Every Tool-Use Iteration
- **File:** `src/agent/loop.rs`
- **Issue:** `conversation.clone()` clones the entire message history on every iteration.
- **Fix:** Use `Arc<Vec<…>>` or incremental updates. Related to G3 (context compaction).

### 20. Embedding Lookup Allocates a New Vec Per Token
- **File:** `src/llm/infer/engine.rs`
- **Lines:** 231–236
- **Issue:** `.to_vec()` allocates a new vector for each token.
- **Fix:** Reuse a scratch buffer.

### ~~21. NVIDIA GPU Detection Reads File Twice~~ ✅ FIXED
- **File:** `src/hw_recommend/detector.rs`
- **Lines:** 153–170
- **Issue:** `detect_gpu_nvidia` reads `/proc/driver/nvidia/gpus/0/information` twice.
- **Fix:** Read once and parse both fields.

### ~~22. `.env` Parser Doesn't Handle Values Containing `=`~~ ✅ FIXED
- **File:** `src/providers/store.rs`
- **Issue:** Edge case with values containing `=`.
- **Fix:** Use `split_once('=')` which already handles this correctly.

### ~~23. `mask_key` Reveals Too Much for Short Keys~~ ✅ FIXED
- **File:** `src/providers/store.rs`
- **Lines:** 278–281
- **Issue:** `mask_key("ab")` returns `"ab****"`, revealing the entire key. Keys shorter than 4 chars have no masking.
- **Fix:** Always mask at least 4 characters; show at most `min(4, len/2)` visible chars.

## P2 — Harness Gaps (nice-to-have)

### G11. Repo Map (Aider-style)
- **Gap:** O agente não tem uma visão estrutural do repositório. Aider e Cursor geram um mapa de definições (classes, funções) para guiar file selection.
- **Impact:** Context efficiency — o agente gasta iterações buscando arquivos relevantes.
- **Fix:** Gerar um repo map na inicialização usando regex ou tree-sitter para extrair definições. Injetar no system prompt como contexto compacto.

### G12. Lint Integration
- **Gap:** Não integra com linters. Claude Code, Cursor e Aider usam lint feedback para self-correction.
- **Impact:** O agente não detecta erros de estilo/tipo sem rodar o test command completo.
- **Fix:** Após edições, rodar `cargo clippy` / `eslint` e alimentar o output como feedback ao modelo.

### G13. Web Search Tool
- **Gap:** Não tem capacidade de buscar na web. Antigravity tem `search_web`, Aider tem web integration.
- **Impact:** O agente não pode pesquisar documentação, APIs, ou soluções para erros desconhecidos.
- **Fix:** Implementar tool `web_search(query)` usando uma search API (SearXNG, Brave, etc.).

### G14. Session Persistence
- **Gap:** Sessões não persistem entre execuções. Claude Code, Cursor e Codex salvam sessões.
- **Impact:** UX — o usuário perde todo o contexto ao reiniciar.
- **Fix:** Serializar `AgentState` + conversation history para disco. Restaurar com `/session load`.

### G15. Background Task Execution
- **Gap:** Não suporta execução em background. Antigravity e Cursor permitem rodar tarefas enquanto o usuário faz outra coisa.
- **Impact:** UX — builds longos bloqueiam o agent loop.
- **Fix:** Executar commands longos em thread separada com polling de status. Emitir eventos via `AgentHooks`.

### G16. Git Branch/Stash Operations
- **Gap:** Git tools são básicos (status, diff, log, stage, commit). Sem branch, stash, blame.
- **Impact:** Workflow — não pode criar feature branches ou stash work-in-progress.
- **Fix:** Adicionar `git_branch`, `git_stash`, `git_blame` em `src/tools/git.rs`.

## P3 — Logic Bugs & Edge Cases

### ~~24. `needs_fix` Has False-Positive Logic~~ ✅ REMOVED
- The `needs_fix` heuristic was removed as part of the unified orchestration refactor (ADR 0002). Verification now uses `VerificationResult` with `VerificationStatus::Passed/Failed/Unavailable`.

### 25. `list_files` Skips Directories
- Subsumed by G10.

### 26. `read_file` Step Falls Back to `search_code`
- **File:** `src/agent/executor.rs`
- **Issue:** If a `read_file` step has no `filename`, it silently falls back to `search_code`.
- **Fix:** Return an error or skip the step instead of silently changing operation.

### 27. Retry Logic Can Exceed `max_retries`
- **File:** `src/agent/loop.rs`
- **Issue:** The recursive call can trigger another retry, exceeding `max_retries`.
- **Fix:** Decrement retry count properly or use a loop instead of recursion.

### ~~28. `allowed_commands` Contains Multi-Word Commands~~ ✅ FIXED
- **File:** `src/config/settings.rs`
- **Lines:** 94–120
- **Issue:** The blocked commands list contains multi-word entries (`"rm -rf"`, `"del /f"`, `"rd /s"`) but `is_allowed()` now validates by executable name only. Multi-word blocked commands are misleading.
- **Fix:** Remove multi-word entries from `blocked_commands`. Document that `blocked_commands` is executable-name-only.

### ~~29. `extract_path` Heuristic Can Return Invalid Paths~~ ✅ FIXED
- **File:** `src/agent/executor.rs`
- **Lines:** 249+
- **Issue:** Can return things like `"a.b.c"` as a path when the step description mentions a version number.
- **Fix:** Add more heuristics to filter out non-path tokens.

### ~~31. `Bench` Command Overwrites Local Results with Cloud~~ ✅ FIXED
- **File:** `src/bench/mod.rs`, `src/bench/local.rs`, `src/bench/cloud.rs`
- **Issue:** Both local and cloud benchmark results were saved to the same file from a single combined module.
- **Fix:** Split `model_bench.rs` into `local.rs` and `cloud.rs` modules. Exported both from `mod.rs`. Made shared helpers (`BenchResult`, `names_match`, `estimate_tps_from_catalog`) `pub(crate)` in `model_bench.rs` for cross-module use. `main.rs` can now route local and cloud results to separate output files.

---

## Status Summary (as of 2026-08-02)

### Fixed (from previous TODOs)

| # | Item | Fixed In |
|---|------|----------|
| 1 | Command injection via prefix-based allowlist | ADR 0002 |
| 2 | Timeout + process-group kill for `run_command` | ADR 0002 |
| 3 | `run_command_raw` bypasses allowlist | ADR 0002 |
| 13 | Dead code: `maybe_compact_chain` | ADR 0002 |
| 14 | Dead code: `run_agent_loop_with_fallback` | ADR 0002 |
| 15 | Dead code: `execute_step_inner_chain` | ADR 0002 |
| 24 | `needs_fix` false-positive logic | ADR 0002 |
| 4 | `unsafe transmute` on untrusted GGUF data | ADR 0007 |
| 5 | GGUF parsing panics on truncated files | ADR 0007 |
| 6 | Q4_0/Q8_0 dequantization bounds checks | ADR 0007 |
| 7 | `tensor_data` out-of-bounds check | ADR 0007 |
| 8 | Path traversal, symlink escape & injection safety | ADR 0008 |
| 9 | Replaced unwrap calls with pattern matching in production | ADR 0008 |
| 10 | `partial_cmp().unwrap()` NaN panic safety | ADR 0008 |
| 11 | Top-K sampling NaN comparison safety | ADR 0008 |
| 12 | HTTP error status check in OllamaClient | ADR 0008 |
| 16 | Rename `r#loop` → `agent_loop` | ADR 0008 |
| 21 | GPU info single file-read refactor | ADR 0008 |
| 22 | `.env` value parser with quotes & inline comments | ADR 0008 |
| 23 | Short API key masking safety | ADR 0008 |
| 28 | Multi-word blocked commands support | ADR 0008 |
| 29 | `extract_path` dot/slash extension requirement | ADR 0008 |
| 31 | Bench module split (local.rs / cloud.rs) | ADR 0010 |

### Implemented Features

| Feature | Status | ADR |
|---------|--------|-----|
| Workspace transactions (snapshot/diff/rollback) | ✅ Done | 0002 |
| Interactive approval broker (G2) | ✅ Done | 0002 |
| Parallel read-only tool execution | ✅ Done | 0002 |
| `tool_choice` + capability filtering | ✅ Done | 0002 |
| Unified orchestration (no more FallbackChain) | ✅ Done | 0002 |
| Limits calibrated to GLM-5.2 via NIM | ✅ Done | 0002 |
| Protocol normalization (typed tool_calls) | ✅ Done | 0002 |
| Prompt contract for GLM-5.2 | ✅ Done | 0002 |
| LLM Router (local/cloud routing) | ✅ Done | 0001 |
| Provider switching at runtime (`/provider`) | ✅ Done | 0001 |
| Same-tier model fallback via `ModelTier` | ✅ Done | 0003 |
| Exponential backoff on 429/5xx | ✅ Done | 0003 |
| LLM error surfacing in TUI chat | ✅ Done | 0003 |
| Mouse wheel scrolling in TUI | ✅ Done | 0003 |
| Workspace defaults to current directory | ✅ Done | 0003 |
| Auto-read project context (`AGENTS.md`) (G8) | ✅ Done | 0004 |
| `list_files` includes directories (G10) | ✅ Done | 0004 |
| Token usage & cost tracking per turn (G6) | ✅ Done | 0005 |
| Windows cross-platform path resolution fixes | ✅ Done | 0005 |
| Line-range surgical code editing (`edit_file`) (G1) | ✅ Done | 0006 |
| GGUF safe parsing & bounds-checked dequantization | ✅ Done | 0007 |
| Floating point safety & complete key masking | ✅ Done | 0008 |
| Context intelligence, token estimation (G4) & repo map (G11, G3) | ✅ Done | 0009 |
| Git branch & stash operations (G16) | ✅ Done | 0009 |
| Bench module split (local.rs / cloud.rs) (Item 31) | ✅ Done | 0010 |

### All Open Demands

| # | Priority | Item | Category | Effort |
|---|----------|------|----------|--------|
| G5 | **P1** | Sub-agent support (Task tool) | Harness | Alto |
| G7 | P1 | MCP client (stdio transport) | Harness | Alto |
| G9 | P1 | Streaming tool call deltas | Harness | Médio |
| — | P1 | Provider health checks, circuit breaking | Robustness | Médio |
| — | P1 | Prompts versioned/tested | Quality | Médio |
| G12 | P2 | Lint integration | Harness | Médio |
| G13 | P2 | Web search tool | Harness | Médio |
| G14 | P2 | Session persistence | Harness | Médio |
| G15 | P2 | Background task execution | Harness | Alto |
| 17 | P2 | Remove `#![allow(dead_code)]` | Code Quality | Baixo |
| 18 | P2 | Fix `truncate_str` (head vs tail) | Code Quality | Baixo |
| 19 | P2 | Fix conversation clone per iteration | Performance | Médio |
| 20 | P2 | Fix embedding alloc per token | Performance | Baixo |
| 30 | P2 | Make `get_cloud_models` data-driven | Code Quality | Baixo |
| — | P2 | Integration tests with mock provider | Testing | Médio |
| — | P2 | CI pipeline | Infra | Médio |

### Recommended Sprint Order

| Sprint | Focus | Items | Timeline |
|--------|-------|-------|----------|
| **1** | Quick Wins | G2, G8, G10, G6, items 4-7 | 1-2 dias |
| **2** | Edição | G1 (edit_file + multi_edit_file) | 3-5 dias |
| **3** | Context Intelligence | G4, G3, G11 (repo map) | 1 semana |
| **4** | Architecture | G5 (sub-agents), G9 (streaming deltas), G7 (MCP) | 1-2 semanas |

### Test Coverage

~160 unit tests across all modules. Key areas:

| Module | Tests |
|--------|-------|
| `llm/tier.rs` | 10 (tier classification, fallback resolution, ordering) |
| `llm/router.rs` | 12 (routing, resolution, provider switching, capability, fallback) |
| `tools/shell.rs` | 8 (allowlist, metacharacters, timeout, combined output) |
| `tools/fs.rs` | 6+ (path traversal, workspace containment, transactions) |
| `tools/transaction.rs` | 3 (snapshot, diff, rollback) |
| `config/settings.rs` | 6 (defaults, env overrides, policies) |
| `agent/loop.rs` | 11 (tool dispatch, parallel execution, output formatting) |
| `compressor/` | 20+ (caveman, layer1, layer2) |
| `memory/` | 6+ (short-term, search) |
| `providers/store.rs` | 6+ (env loading, catalog resolution, masking) |
| `models_dev/` | 10+ (catalog queries, provider models) |
| `ui.rs` | 6+ (truncation, formatting, elapsed time) |
