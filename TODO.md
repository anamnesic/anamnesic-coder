# TODO — P0 Safety & Reliability Fixes (August 2026)

## P0 — Safety and Reliability (CRITICAL)

### 1. Command Injection via Prefix-Based Allowlist
- **File:** `src/tools/shell.rs`
- **Lines:** 54–66 (original)
- **Issue:** `is_allowed()` checks whether the command string *starts with* an allowed command, then checks whether it *contains* a blocked substring. Trivially bypassed by appending shell metacharacters after a safe prefix (e.g., `echo hello; rm -rf /`).
- **Fix:** Replace prefix-based allowlist with parsed executable + args. Reject shell metacharacters (`;`, `&`, `|`, `` ` ``, `$`, `(`, `)`, `{`, `}`, `<`, `>`, `\n`, `\\`) by default.

### 2. No Timeout or Process-Group Kill on `run_command`
- **File:** `src/tools/shell.rs`
- **Lines:** 69–105 (original)
- **Issue:** Both `run_command` and `run_command_raw` use `Command::output()` with no timeout. A hung or malicious subprocess blocks the agent loop indefinitely.
- **Fix:** Add `command_timeout_secs` timeout using `child.wait_with_timeout()`. Kill the process group on timeout.

### 3. `run_command_raw` Bypasses the Allowlist Entirely
- **File:** `src/tools/shell.rs`
- **Lines:** 92–105 (original)
- **Issue:** `run_command_raw` does **not** call `is_allowed()` before executing. Used by `verify_cargo` in `src/agent/executor.rs:78`. If the allowlist is a security boundary, bypassing it here is a gap.
- **Fix:** Add `is_allowed()` check to `run_command_raw`.

### 4. `unsafe` `transmute` on Untrusted GGUF Data
- **File:** `src/llm/infer/gguf.rs`
- **Line:** 138
- **Issue:** `std::mem::transmute::<i32, GgmlType>(ty_i)` converts a raw integer from a GGUF file into an enum. If the file is crafted or corrupted with an out-of-range integer, this is **undefined behavior**.
- **Fix:** Use `TryFrom<i32>` with explicit validation and return an error for invalid values.

### 5. GGUF Parsing Panics on Truncated/Corrupt Files
- **File:** `src/llm/infer/gguf.rs`
- **Lines:** 159–168
- **Issue:** All `read_u8`, `read_u16`, `read_u32`, `read_u64`, `read_f32`, `read_f64` methods use `try_into().unwrap()` and slice indexing without bounds checks. A truncated GGUF will panic rather than return an error.
- **Fix:** Replace `unwrap()` with proper error propagation. Add bounds checks before slicing.

### 6. Q4_0/Q8_0 Dequantization Panics on Corrupt Data
- **File:** `src/llm/infer/model.rs`
- **Lines:** 24, 40, 53
- **Issue:** `u16::from_le_bytes(block[...].try_into().unwrap())` will panic if the tensor data is shorter than expected (corrupt GGUF).
- **Fix:** Replace `unwrap()` with proper error handling and bounds checking.

### 7. `tensor_data` Can Read Beyond Buffer
- **File:** `src/llm/infer/gguf.rs`
- **Lines:** 149–157
- **Issue:** `&self.data[start..start + nbytes]` does not check that `start + nbytes <= self.data.len()`. A malformed GGUF with a bogus offset or size will panic.
- **Fix:** Add bounds check before slicing, return error if out of bounds.

### 8. TOCTOU Race in `FileTools::resolve`
- **File:** `src/tools/fs.rs`
- **Lines:** 126–155
- **Issue:** Between the `canonicalize()` check and the actual `fs::read_to_string`/`fs::write` call, an attacker with concurrent access could replace a file with a symlink.
- **Fix:** Re-validate the path after canonicalization, or use `O_NOFOLLOW` where available.

## P1 — Error Handling & Robustness

### 9. `unwrap()` in Production Paths
- **Files:** `src/llm/client.rs:472`, `src/main.rs:401`, `src/agent/executor.rs:200`, `src/ui.rs:442`, `src/agent/state.rs:61,74,83`, `src/llm/router.rs` (multiple)
- **Issue:** `unwrap()` calls that can panic in production.
- **Fix:** Replace with proper error handling or `expect()` with descriptive messages.

### 10. `partial_cmp().unwrap()` on Costs (NaN Panic)
- **File:** `src/models_dev/client.rs`
- **Lines:** 81, 100, 149
- **Issue:** Will panic if any cost is NaN.
- **Fix:** Use `.unwrap_or(Equal)`.

### 11. `partial_cmp().unwrap()` in Top-K Sampling
- **File:** `src/llm/infer/engine.rs`
- **Line:** 333
- **Issue:** Will panic on NaN logits.
- **Fix:** Use `.unwrap_or(Equal)`.

### 12. `unwrap_or_default` Silently Swallows HTTP Read Errors
- **File:** `src/llm/client.rs`
- **Lines:** 412, 517, 568, 622
- **Issue:** `resp.text().await.unwrap_or_default()` silently discards errors.
- **Fix:** Propagate errors properly or log them.

## P2 — Code Quality & Design

### 13. Dead Code: `maybe_compact_chain`
- **File:** `src/agent/loop.rs`
- **Lines:** 89–105
- **Issue:** Defined but never called.
- **Fix:** Remove or integrate into the primary loop.

### 14. Dead Code: `run_agent_loop_with_fallback`
- **File:** `src/agent/loop.rs`
- **Lines:** 408–470
- **Issue:** Defined but never called. TODO.md flags this.
- **Fix:** Remove or integrate into the primary loop.

### 15. Dead Code: `execute_step_inner_chain`
- **File:** `src/agent/executor.rs`
- **Lines:** 301–324
- **Issue:** Only handles `"answer"` step type; all others fall through to a `println!` fallback.
- **Fix:** Remove or implement properly.

### 16. Unused `r#loop` Raw Identifier
- **File:** `src/main.rs`
- **Line:** 21
- **Issue:** Module name `loop` fights the Rust keyword.
- **Fix:** Rename to `agent_loop` or `cycle`.

### 17. `#![allow(dead_code)]` at Crate Root
- **File:** `src/main.rs`
- **Line:** 3
- **Issue:** Suppresses dead-code warnings for the entire crate.
- **Fix:** Remove and fix dead code.

### 18. `truncate_str` Keeps Tail Instead of Head
- **File:** `src/ui.rs`
- **Lines:** 1086–1094
- **Issue:** Keeps the last `max` characters and prepends ellipsis. For model names and paths, the beginning is usually more informative.
- **Fix:** Change to keep the head (first `max` characters) with trailing ellipsis.

### 19. Conversation Cloned on Every Tool-Use Iteration
- **File:** `src/agent/loop.rs`
- **Line:** 127
- **Issue:** `conversation.clone()` clones the entire message history on every iteration.
- **Fix:** Use `Arc<Vec<...>>` or incremental updates.

### 20. Embedding Lookup Allocates a New Vec Per Token
- **File:** `src/llm/infer/engine.rs`
- **Lines:** 231–236
- **Issue:** `.to_vec()` allocates a new vector for each token.
- **Fix:** Reuse a scratch buffer.

### 21. NVIDIA GPU Detection Reads File Twice
- **File:** `src/hw_recommend/detector.rs`
- **Lines:** 153–170
- **Issue:** `detect_gpu_nvidia` reads `/proc/driver/nvidia/gpus/0/information` twice.
- **Fix:** Read once and parse both fields.

### 22. `.env` Parser Doesn't Handle Values Containing `=`
- **File:** `src/providers/store.rs`
- **Line:** 199
- **Issue:** Edge case with values containing `=`.
- **Fix:** Use `split_once('=')` which already handles this correctly.

### 23. `mask_key` Reveals Too Much for Short Keys
- **File:** `src/providers/store.rs`
- **Lines:** 278–281
- **Issue:** `mask_key("ab")` returns `"ab****"`, revealing the entire key.
- **Fix:** Always mask at least 4 characters.

## P3 — Logic Bugs & Edge Cases

### 24. `needs_fix` Has False-Positive Logic
- **File:** `src/agent/loop.rs`
- **Lines:** 472–479
- **Issue:** Returns `true` if output contains `"error"` (lowercased). Catches legitimate error messages inside passing test output.
- **Fix:** Refine heuristic to only match failure indicators, not general error messages.

### 25. `list_files` Skips Directories
- **File:** `src/tools/fs.rs`
- **Lines:** 187–197
- **Issue:** Only includes files (`is_file()`), not directories.
- **Fix:** Document behavior or add directory listing option.

### 26. `read_file` Step Falls Back to `search_code`
- **File:** `src/agent/executor.rs`
- **Lines:** 221–238
- **Issue:** If a `read_file` step has no `filename`, it silently falls back to `search_code`.
- **Fix:** Return an error or skip the step instead of silently changing operation.

### 27. Retry Logic Can Exceed `max_retries`
- **File:** `src/agent/loop.rs`
- **Lines:** 390–400
- **Issue:** The recursive call can trigger another retry, exceeding `max_retries`.
- **Fix:** Decrement retry count properly or use a loop instead of recursion.

### 28. `allowed_commands` Contains Multi-Word Commands
- **File:** `src/config/settings.rs`
- **Lines:** 36–41
- **Issue:** `"npm test"` is a single allowed command, but `"npm"` alone is rejected. Inconsistent with single-word commands.
- **Fix:** Allow both exact match and prefix match for multi-word commands.

### 29. `extract_path` Heuristic Can Return Invalid Paths
- **File:** `src/agent/executor.rs`
- **Lines:** 144–163
- **Issue:** Can return things like `"a.b.c"` as a path when the step description mentions a version number.
- **Fix:** Add more heuristics to filter out non-path tokens.

### 30. Hardcoded Cloud Model List
- **File:** `src/main.rs`
- **Lines:** 359–369
- **Issue:** `get_cloud_models` returns a hardcoded list.
- **Fix:** Make data-driven from the models.dev catalog.

### 31. `Bench` Command Overwrites Local Results with Cloud
- **File:** `src/main.rs`
- **Lines:** 211–218
- **Issue:** Both local and cloud benchmark results saved to the same file.
- **Fix:** Use separate files for local and cloud results.

## Missing Features (from TODO.md)

| Priority | TODO Item | Status |
|----------|-----------|--------|
| P0 | Timeout + process-group kill for `run_command` | **FIXED** |
| P0 | Parsed executable+args allowlist; reject shell operators | **FIXED** |
| P0 | Per-tool approval policy (ask/allow/deny) | **MISSING** |
| P0 | Workspace diff summary + commit/rollback workflow | **MISSING** |
| P0 | Handle malformed provider-specific arguments gracefully | **PARTIAL** |
| P0 | Configurable max iterations, output caps, timeouts | **PARTIAL** |
| P0 | Tests for path traversal, symlink escape, command injection | **MISSING** |
| P1 | Concurrent independent tool calls | **MISSING** |
| P1 | `tool_choice` + capability filtering per model | **MISSING** |
| P1 | Patch/edit tools (unified diff, targeted replacement) | **MISSING** |
| P1 | Repo discovery tools (list tree, git diff/status) | **MISSING** |
| P1 | Prompts in `prompts/` files, versioned/tested | **MISSING** |
| P1 | Structured planner output with JSON schema | **MISSING** |
| P1 | `FallbackChain` in primary tool loop | **MISSING** (dead code exists) |
| P1 | Provider health checks, retry classification, circuit breaking | **MISSING** |
| P2 | Split local/cloud benchmark result files | **MISSING** |
| P2 | MCP client | **MISSING** |
| P2 | Integration tests with mock provider | **MISSING** |
