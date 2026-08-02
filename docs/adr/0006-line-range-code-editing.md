# ADR 0006 — Line-Range Code Editing (`edit_file`)

**Status:** Accepted  
**Date:** 2026-08-02  
**Author:** Antigravity + Luan  

## Context

The agent harness previously supported only `write_file` (full file overwrite) and `replace_exact` (exact string match anywhere in the file). This caused model editing failures in real-world benchmarks (SWE-bench / LiveCodeBench) due to:
1. Slight whitespace or indentation mismatches causing `replace_exact` to fail.
2. Duplicate strings across a file causing `replace_exact` ambiguity errors (refusing to edit if >1 match).
3. Full file rewrites (`write_file`) wasting output tokens and increasing latency.

All competitive 2026 harnesses (Claude Code `Edit`, Antigravity `replace_file_content`, Cursor Composer) use line-range anchored editing.

## Decision

1. **Implement `edit_file` in `FileTools` (`src/tools/fs.rs`):**
   - Accepts `path`, optional `start_line` and `end_line` (1-based inclusive), optional `old_content`, and `new_content`.
   - When `start_line` and `end_line` are provided, surgically replaces lines `[start_line-1..end_line]` with `new_content`.
   - Validates `old_content` against actual lines at specified range if provided (with helpful error reporting on mismatch).
   - Falls back to `replace_exact` if line numbers are omitted.

2. **Register `edit_file` in Tool Registry & Dispatch (`src/agent/loop.rs`):**
   - Classified as `ToolEffect::Mutation` (invalidates prior verification, subject to `write_tool_policy` approval gate).
   - Registered in `coding_tools()` schema with `start_line`, `end_line`, `old_content`, `new_content`.

## Consequences

- The model can now perform surgical edits without rewriting entire files or failing on string ambiguity.
- Significantly reduces output token consumption and turn latency on multi-line edits.
- Unit tests in `src/tools/fs.rs` verify surgical line-range replacement and bounds checking.
