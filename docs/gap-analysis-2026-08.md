# Gap Analysis — Anamnesic Coder vs. 2026 Coding Agent Harnesses

**Date:** 2026-08-02  
**Scope:** Comparação funcional real entre o Anamnesic Coder e os harnesses líderes de agosto 2026: Claude Code, Codex CLI, Antigravity, Cursor Agent, Aider.

---

## Executive Summary

O Anamnesic Coder tem uma base sólida com roteamento multi-provedor, fallback por tier, transações de workspace, e loop de tool-use competitivo. Porém, está **2-3 gerações atrás** dos líderes em áreas críticas: edição de código, sub-agentes, memória persistente, MCP, e observabilidade. Os gaps mais impactantes para SWE-bench performance são: (1) ausência de edição cirúrgica por linha, (2) nenhum sub-agente, (3) nenhum mecanismo de contexto inteligente.

> [!CAUTION]
> O approval broker está **definido nos tipos mas não está wired** — writes e commands executam sem pedir aprovação. Isso é um gap de segurança crítico.

---

## 1. Tool Inventory — Gap Comparison

| Tool / Capability | Anamnesic | Claude Code | Codex CLI | Antigravity | Cursor | Aider |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Read file** | ✅ | ✅ | ✅ | ✅ (line ranges) | ✅ | ✅ |
| **Write file** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Line-range edit** | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ |
| **Multi-edit (non-contiguous)** | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ |
| **Diff/patch application** | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ |
| **Exact string replace** | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Shell execution** | ✅ | ✅ | ✅ (sandbox) | ✅ | ✅ | ✅ |
| **Grep/ripgrep search** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Glob/file search** | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ |
| **List directory** | ✅ (files only) | ✅ | ✅ | ✅ (files+dirs) | ✅ | ❌ |
| **Symbol search** | ⚠️ (regex) | ✅ (LSP) | ❌ | ❌ | ✅ (LSP) | ✅ (tree-sitter) |
| **Web search** | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **HTTP fetch** | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |
| **Image generation** | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Git operations** | ⚠️ (basic) | ✅ | ✅ | ✅ (via shell) | ✅ | ✅ (native) |
| **Task/TODO tracking** | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Notebook editing** | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ |
| **Background tasks** | ❌ | ❌ | ✅ (cloud) | ✅ | ✅ | ❌ |
| **Timers/cron** | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |

### 🔴 Gap Crítico: Edição de Código

O Anamnesic tem apenas `write_file` (overwrite total) e `replace_exact` (busca exata de string). **Todos os líderes** oferecem pelo menos uma forma de edição cirúrgica:

- **Claude Code**: `Edit` tool com start/end line + replacement content
- **Antigravity**: `replace_file_content` (single block, line-range) + `multi_replace_file_content` (multi-block)
- **Codex CLI**: `apply_diff` (unified diff format)
- **Aider**: Whole-file diff output, parsed and applied

`replace_exact` é frágil porque:
1. Exige match exato (whitespace, indentation) — modelos erram frequentemente
2. Não suporta edições multi-site no mesmo arquivo
3. Não tem line-range anchoring — ambiguidade quando a mesma string aparece múltiplas vezes

**Recomendação**: Implementar `edit_file(path, start_line, end_line, old_content, new_content)` e `multi_edit_file(path, edits[])`.

---

## 2. Agent Architecture

| Capability | Anamnesic | Claude Code | Codex CLI | Antigravity | Cursor | Aider |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Sub-agent spawning** | ❌ | ✅ | ✅ (cloud) | ✅ | ✅ (8x) | ❌ |
| **Parallel agent execution** | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Custom agent types** | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ (arch/edit) |
| **Architect/Editor split** | ⚠️ (planner) | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Tool-use loop** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Planner fallback** | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Max iterations** | 128 | ~200 | ~100 | ~200 | ~100 | Unlimited |
| **Parallel read tools** | ✅ (4) | ✅ | ✅ | ✅ | ✅ | ❌ |

### 🔴 Gap Crítico: Sub-Agentes

Os líderes de SWE-bench (Claude Code ~72%, Codex ~70%) usam sub-agentes para:
- **Pesquisa paralela**: ler múltiplos arquivos simultaneamente via sub-agentes dedicados
- **Delegação de tarefas**: refatorações grandes divididas entre agentes especializados
- **Verificação independente**: um agente edita, outro verifica

O Anamnesic tem apenas um loop sequencial. Para tarefas complexas (multi-arquivo, refatoração), isso resulta em mais iterações, mais tokens, e mais chances de context overflow.

---

## 3. Context Management

| Capability | Anamnesic | Claude Code | Codex CLI | Antigravity | Cursor | Aider |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Max context** | 128K | 1M | 200K | 2M | 200K | Model-dep. |
| **Context compaction** | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Conversation summary** | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Repository map** | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| **File checksums** | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ |
| **Token counting** | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Smart file selection** | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ (repo map) |

### 🔴 Gap Crítico: Context Intelligence

O Anamnesic envia o histórico completo a cada turno (`conversation.clone()`), sem:
- **Compactação**: quando o contexto enche, o loop simplesmente falha
- **Sumarização**: mensagens antigas não são resumidas
- **Repo map**: o agente não sabe quais arquivos são relevantes sem buscar
- **Token counting**: não sabe quanto contexto resta

O compressor (`src/compressor/`) existe mas opera em nível de texto, não de conversa. Não há integração com o agent loop para comprimir o histórico.

---

## 4. Permission & Safety

| Capability | Anamnesic | Claude Code | Codex CLI | Antigravity | Cursor |
|---|:---:|:---:|:---:|:---:|:---:|
| **Per-tool approval** | ⚠️ (types only) | ✅ | ✅ (modes) | ✅ | ✅ |
| **Sandbox isolation** | ❌ | ❌ | ✅ (kernel) | ✅ (cloud) | ❌ |
| **Path-scoped perms** | ❌ | ✅ | N/A | ✅ | ❌ |
| **Network policy** | ❌ | ✅ | ✅ | ✅ | ❌ |
| **Command prefix match** | ✅ | ✅ | N/A | ✅ | ✅ |
| **Persistent settings** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Pre-tool hooks** | ❌ | ✅ | ❌ | ❌ | ❌ |

### 🟡 Gap Moderado: Approval Broker Desconectado

Os tipos `ApprovalRequest`, `ApprovalDecision` e `AgentHooks.on_approval` existem em `src/agent/loop.rs:50-71`, mas o código de dispatch de tools em `execute_tool_call()` **nunca chama `on_approval()`**. Isso significa que mesmo com `write_tool_policy: Ask`, o agente executa sem perguntar.

---

## 5. Memory & Persistence

| Capability | Anamnesic | Claude Code | Codex CLI | Antigravity | Cursor | Aider |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Short-term memory** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Long-term memory** | ❌ | ✅ (CLAUDE.md) | ❌ | ✅ (transcripts) | ✅ (memories) | ❌ |
| **Project context files** | ❌ | ✅ (CLAUDE.md) | ❌ | ✅ (AGENTS.md) | ✅ (.cursorrules) | ❌ |
| **Session persistence** | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ (git) |
| **Vector/semantic search** | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Episodic memory** | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |

### 🟡 Gap: Sem Memória de Projeto

O Anamnesic não lê nenhum arquivo de contexto de projeto (como `CLAUDE.md`, `.cursorrules`, ou `AGENTS.md`). Ironicamente, o projeto tem um `AGENTS.md` na raiz, mas o agente não o lê automaticamente.

---

## 6. Verification & Testing

| Capability | Anamnesic | Claude Code | Codex CLI | Antigravity | Cursor | Aider |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Auto-detect test cmd** | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **Run after mutations** | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **Lint integration** | ❌ | ✅ | ❌ | ✅ (IDE) | ✅ (LSP) | ✅ |
| **Retry on failure** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Rollback on failure** | ✅ | ❌ | ✅ (sandbox) | ❌ | ✅ | ✅ (git) |
| **Adversarial verify** | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |

O Anamnesic é competitivo aqui. Auto-detecção de test command + retry + rollback é forte.

---

## 7. LLM Integration

| Capability | Anamnesic | Claude Code | Codex CLI | Antigravity | Aider |
|---|:---:|:---:|:---:|:---:|:---:|
| **Multi-provider** | ✅ | ❌ (Anthropic) | ❌ (OpenAI) | ❌ (Google) | ✅ (70+) |
| **Same-tier fallback** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Retry with backoff** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Streaming** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Streaming tool deltas** | ❌ | ✅ | ✅ | ✅ | ❌ |
| **Cost tracking** | ❌ | ✅ | ✅ | ❌ | ✅ |
| **Token counting** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Extended thinking** | ❌ | ✅ | ✅ (o3/o4) | ❌ | ❌ |
| **Local GGUF inference** | ✅ | ❌ | ❌ | ❌ | ❌ |

O Anamnesic tem vantagens únicas: multi-provider routing, same-tier fallback, e inferência GGUF local. Nenhum concorrente tem os três.

---

## 8. Extensibility

| Capability | Anamnesic | Claude Code | Codex CLI | Antigravity | Cursor |
|---|:---:|:---:|:---:|:---:|:---:|
| **MCP client** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Plugin/skill system** | ❌ | ✅ (skills) | ❌ | ✅ (skills) | ✅ (ext.) |
| **Custom tool defs** | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Hooks/events** | ✅ (AgentHooks) | ✅ (PreToolUse) | ❌ | ❌ | ❌ |
| **Config files** | ⚠️ (env only) | ✅ (.claude/) | ✅ | ✅ (.gemini/) | ✅ (.cursor/) |

---

## Priority Ranking — What to Fix First

### 🔴 P0 — Blockers para viabilidade real (impactam SWE-bench diretamente)

| # | Gap | Impact | Effort | Recommendation |
|---|-----|--------|--------|----------------|
| G1 | **Sem edição cirúrgica (line-range edit)** | Crítico — `replace_exact` falha quando o modelo erra whitespace ou quando a string aparece múltiplas vezes | Médio | Implementar `edit_file(path, start, end, old, new)` no estilo Antigravity |
| G2 | **Approval broker não wired** | Segurança — writes/commands executam sem gate | Baixo | Wiring o `on_approval` callback no dispatch de `execute_tool_call()` |
| G3 | **Sem context compaction** | Performance — historico cresce até estourar contexto | Alto | Implementar sumarização de mensagens antigas quando `token_count > 0.8 * max_context` |
| G4 | **Sem token counting** | Ops — não sabe quanto contexto resta; relacionado a G3 | Médio | Adicionar `tiktoken`-style counting ou estimate por chars |

### 🟡 P1 — Importantes para competitividade

| # | Gap | Impact | Effort | Recommendation |
|---|-----|--------|--------|----------------|
| G5 | **Sem sub-agentes** | Throughput — tarefas complexas são lentas | Alto | Modelo mínimo: `Task` tool que spawna um segundo loop |
| G6 | **Sem cost tracking** | Ops — não sabe quanto gastou | Baixo | Contar tokens in/out por chamada LLM, acumular por turno |
| G7 | **Sem MCP client** | Extensibilidade — não conecta a servers externos | Alto | Implementar MCP client mínimo (stdio transport) |
| G8 | **Sem project context (AGENTS.md/CLAUDE.md)** | Quality — modelo não tem contexto do projeto | Baixo | Ler e injetar `AGENTS.md` no system prompt automaticamente |
| G9 | **Streaming tool deltas** | UX — tool calls só aparecem quando completos | Médio | Parsear deltas SSE incrementalmente |
| G10 | **`list_files` não lista diretórios** | Quality — modelo não vê a estrutura | Baixo | Adicionar diretórios ao output e/ou tool `list_dir` separado |

### 🟢 P2 — Nice-to-have

| # | Gap | Impact | Effort |
|---|-----|--------|--------|
| G11 | Sem repo map (Aider-style) | Context efficiency | Alto |
| G12 | Sem lint integration | Edit quality | Médio |
| G13 | Sem web search | Research capability | Médio |
| G14 | Sem session persistence | UX | Médio |
| G15 | Sem background tasks | UX | Alto |
| G16 | Sem git branch/stash | Workflow | Baixo |

---

## Competitive Position Matrix

```
                    Edição   Sub-agents   Context    Safety    Memory   MCP    Local LLM
Claude Code         ██████   ██████       ██████     █████░    █████░   █████░   ░░░░░░
Codex CLI           ████░░   █████░       ████░░     ██████    ███░░░   █████░   ░░░░░░
Antigravity         ██████   ██████       ██████     ██████    █████░   ██████   ░░░░░░
Cursor              ██████   ██████       █████░     ████░░    ██████   █████░   ░░░░░░
Aider               ████░░   ██░░░░       ████░░     ████░░   ███░░░   ░░░░░░   ░░░░░░
─────────────────────────────────────────────────────────────────────────────────────────
Anamnesic           ██░░░░   ░░░░░░       ██░░░░     ███░░░   ██░░░░   ░░░░░░   ██████
```

---

## Anamnesic Unique Strengths

Áreas onde o Anamnesic é **melhor ou igual** aos líderes:

1. **Multi-provider routing** — nenhum líder faz roteamento entre providers com fallback por tier
2. **Local GGUF inference** — único harness com engine de inferência local embutido
3. **Same-tier automatic fallback** — `ModelTier` classification é exclusivo
4. **Workspace transactions** — rollback determinístico; Codex usa sandbox, Claude Code não tem rollback
5. **Planner fallback** — funciona com modelos sem tool-calling (útil para modelos locais menores)
6. **Verification gate** — auto-detecção de test command com retry é competitivo

---

## Recommended Roadmap

### Sprint 1 (1-2 dias) — Quick Wins
- [ ] **G2**: Wire approval broker (baixo esforço, alto impacto de segurança)
- [ ] **G8**: Ler `AGENTS.md` no system prompt
- [ ] **G10**: `list_files` incluir diretórios
- [ ] **G6**: Cost tracking básico (tokens in/out por chamada)

### Sprint 2 (3-5 dias) — Edição
- [ ] **G1**: Implementar `edit_file` com line-range anchoring
- [ ] Adicionar `multi_edit_file` para edições não-contíguas
- [ ] Manter `replace_exact` como fallback

### Sprint 3 (1 semana) — Context Intelligence
- [ ] **G4**: Token counting (estimate por chars ou integração tiktoken)
- [ ] **G3**: Context compaction (sumarizar mensagens antigas)
- [ ] **G11**: Repo map básico (tree-sitter ou regex para definições)

### Sprint 4 (1-2 semanas) — Architecture
- [ ] **G5**: Sub-agent mínimo (Task tool)
- [ ] **G9**: Streaming tool deltas
- [ ] **G7**: MCP client (stdio transport)
