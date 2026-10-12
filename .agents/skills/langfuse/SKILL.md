---
name: langfuse
description: Interact with Langfuse and access its documentation. Use when needing to (1) query or modify Langfuse data programmatically via the CLI — traces, prompts, datasets, scores, sessions, and any other API resource, (2) look up Langfuse documentation, concepts, integration guides, or SDK usage, or (3) understand how any Langfuse feature works. This skill covers CLI-based API access (via npx) and multiple documentation retrieval methods.
allowed-tools:
  - WebFetch(domain:langfuse.com)
  - Bash(curl *langfuse.com/*)
---

# Langfuse

This skill helps you use Langfuse effectively across all common workflows: instrumenting applications, migrating prompts, debugging traces, and accessing data programmatically.

## Core Principles

1. **Documentation First**: NEVER implement based on memory. Always fetch current docs before writing code.
2. **Best Practices by Use Case**: Check instrumentation reference before implementing.
3. **Use latest Langfuse versions**: Always use the latest SDK version.
4. **Framework integrations first**: LangChain/LangGraph CallbackHandler captures more context automatically.

## Instrumentation Best Practices (LangChain/LangGraph)

- Use `langfuse.langchain.CallbackHandler` passed via `config["callbacks"]`
- Set `trace_name`, `session_id`, `user_id` on the CallbackHandler per request
- `session_id` should map to the conversation thread ID for continuity
- `trace_name` should be descriptive (e.g., `"rag-chat"`, `"rag-eval"`)
- Flush with `langfuse_client.flush()` on shutdown

Docs: https://langfuse.com/docs/integrations/langchain/tracing
