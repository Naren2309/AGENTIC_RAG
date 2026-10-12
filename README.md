# Agentic RAG System using Ollama Granite Model

## Overview

This project implements an **Agentic Retrieval-Augmented Generation (Agentic RAG)** system using **LangGraph**, **LangChain**, **Qdrant Vector Database**, and **Ollama Granite Model**. Unlike a traditional RAG pipeline, this system behaves like an intelligent agent that can:

- Understand previous conversations
- Rewrite ambiguous user queries
- Search documents intelligently
- Retrieve additional context when needed
- Self-correct if information is insufficient
- Compress context to reduce token usage
- Aggregate responses from multiple agents

The project uses **IBM Granite 4.1 8B** running locally through **Ollama**, making it completely offline without requiring cloud APIs.

---

# Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| LangGraph | Agent Workflow |
| LangChain | LLM Framework |
| Ollama | Local LLM Server |
| Granite 4.1 8B | Chat Model |
| HuggingFace Embeddings | Dense Embeddings |
| BM25 Sparse Embeddings | Keyword Search |
| Qdrant | Vector Database |
| PyMuPDF4LLM | PDF → Markdown Conversion |
| Gradio | Chat Interface |
| RAGAS | Evaluation |

---

# System Architecture

```
                User Question
                      │
                      ▼
         Conversation Memory
                      │
                      ▼
           Query Rewriting Agent
                      │
          ┌───────────┴───────────┐
          │                       │
      Query Clear?           Query Ambiguous?
          │                       │
          ▼                       ▼
 Parallel Retrieval Agents   Ask User Clarification
          │
          ▼
 Search Child Chunks
          │
          ▼
 Retrieve Parent Chunks
          │
          ▼
 Context Compression
          │
          ▼
 Self Correction
          │
          ▼
 Aggregate Answers
          │
          ▼
 Final Response
```

---

# How Ollama is Used

The system uses **Ollama** as the local inference engine.

Instead of sending prompts to OpenAI or Anthropic, every prompt is sent to a locally running Granite model.

```
User
   │
   ▼
LangGraph
   │
   ▼
ChatOllama
   │
   ▼
Ollama Server
   │
   ▼
Granite 4.1 8B
```

Model initialization:

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="granite4.1:8b",
    temperature=0,
    seed=42
)
```

---

# Step-by-Step Working

## Step 1: User Uploads PDFs

The user places PDF documents inside:

```
docs/
```

Example:

```
docs/
    SQL.pdf
    JavaScript.pdf
    Python.pdf
```

---

## Step 2: PDF Conversion

Each PDF is converted into Markdown.

```
SQL.pdf

↓

SQL.md
```

PyMuPDF4LLM extracts

- headings
- paragraphs
- lists
- tables

while preserving document structure.

---

## Step 3: Hierarchical Chunking

Instead of splitting into fixed chunks directly, the system creates two levels.

### Parent Chunks

Large chunks

```
3000–4000 characters
```

Example

```
Chapter 1
Entire Introduction

↓

Parent Chunk
```

---

### Child Chunks

Each Parent Chunk becomes many smaller chunks.

Example

```
Parent Chunk

↓

Child 1
Child 2
Child 3
Child 4
```

Child size:

```
500 characters
```

with

```
100 overlap
```

---

# Why Two-Level Chunking?

Instead of retrieving tiny chunks directly,

the system

1. searches small chunks
2. finds the best parent chunk
3. retrieves the complete parent

This gives

✔ Accurate search

and

✔ Better context.

---

# Step 4: Embedding Generation

Every child chunk is converted into vectors.

Dense Embedding

```
Qwen/Qwen3-Embedding-0.6B
```

Sparse Embedding

```
BM25
```

This enables Hybrid Search.

---

# Step 5: Store in Qdrant

Child chunks are stored inside

```
Qdrant
```

Parent chunks are stored as JSON files.

Example

```
parent_store/

SQL_p1.json

SQL_p2.json

SQL_p3.json
```

Each JSON contains

```
{
   page_content,
   metadata,
   parent_id
}
```

---

# Step 6: User Asks Question

Example

```
How do I install SQL?
```

---

# Step 7: Conversation Memory

The system remembers previous chats.

Example

User

```
Tell me about SQL.
```

Later

```
How do I install it?
```

The system understands

```
it = SQL
```

using conversation summary.

---

# Step 8: Query Rewriting

The LLM rewrites unclear questions.

Example

Input

```
How do I update it?
```

Output

```
How do I update SQL?
```

Another example

Input

```
Explain JavaScript and Python.
```

Output

```
Question 1:
Explain JavaScript.

Question 2:
Explain Python.
```

---

# Step 9: Query Clarification

If the question is vague

```
Tell me about that.
```

The system pauses.

Instead of hallucinating,

it asks

```
What topic are you referring to?
```

---

# Step 10: Multi-Agent Execution

If there are multiple rewritten questions,

LangGraph creates multiple agents.

Example

```
Explain JavaScript

Explain Python
```

becomes

```
Agent 1

↓

JavaScript

Agent 2

↓

Python
```

Both execute simultaneously.

---

# Step 11: Search Child Chunks

Each agent first calls

```
search_child_chunks()
```

This searches the Qdrant vector database.

Returned information

```
Parent ID

Source File

Relevant Chunk
```

---

# Step 12: Retrieve Parent Chunk

If more context is needed,

the agent calls

```
retrieve_parent_chunks(parent_id)
```

Example

```
SQL_p2
```

↓

```
parent_store/SQL_p2.json
```

Entire section is loaded.

---

# Step 13: Self-Correction

If retrieved information is insufficient,

the agent automatically

- changes search query
- searches again
- retries retrieval

until

```
MAX_TOOL_CALLS
```

or

```
MAX_ITERATIONS
```

is reached.

---

# Step 14: Context Compression

Large conversations consume many tokens.

Instead of sending everything repeatedly,

the system summarizes previous research.

Original

```
10 pages
```

↓

Compressed

```
1 page
```

This significantly reduces token usage.

---

# Step 15: Answer Aggregation

If multiple agents answered,

their outputs are merged.

Example

Agent 1

```
JavaScript Answer
```

Agent 2

```
Python Answer
```

↓

Granite combines both into one final response.

---

# Step 16: Final Response

The answer is displayed through

```
Gradio
```

```
Browser

↓

Chat Interface

↓

Granite Response
```

---

# Ollama Workflow

```
User

↓

Gradio

↓

LangGraph

↓

ChatOllama

↓

Ollama Server

↓

Granite 4.1 8B

↓

Response

↓

Gradio
```

---

# Retrieval Workflow

```
Question

↓

Rewrite

↓

Vector Search

↓

Child Chunks

↓

Parent Chunks

↓

LLM

↓

Answer
```

---

# Project Folder Structure

```
project/

│
├── app.py
├── config.py
│
├── core/
│
├── db/
│
├── rag_agent/
│
├── ui/
│
├── docs/
│
├── markdown_docs/
│
├── parent_store/
│
└── qdrant_db/
```

---

# Advantages

- Completely offline
- No OpenAI API cost
- Faster local inference
- Supports conversation memory
- Human-in-the-loop clarification
- Hybrid semantic + keyword search
- Multi-agent reasoning
- Self-correcting retrieval
- Context compression
- Modular architecture
- Easy to replace the LLM provider

---

# Model Used

```
Ollama
```

Model:

```
granite4.1:8b
```

Configuration:

```python
llm = ChatOllama(
    model="granite4.1:8b",
    temperature=0,
    seed=42
)
```

---

# End-to-End Workflow Summary

```
Upload PDFs
        │
        ▼
Convert PDF → Markdown
        │
        ▼
Create Parent Chunks
        │
        ▼
Create Child Chunks
        │
        ▼
Generate Dense + Sparse Embeddings
        │
        ▼
Store Child Chunks in Qdrant
        │
        ▼
Store Parent Chunks as JSON
        │
        ▼
User Question
        │
        ▼
Conversation Memory
        │
        ▼
Query Rewriting
        │
        ▼
Clarification (if required)
        │
        ▼
Parallel Retrieval Agents
        │
        ▼
Search Child Chunks
        │
        ▼
Retrieve Parent Chunks
        │
        ▼
Context Compression
        │
        ▼
Self-Correction
        │
        ▼
Aggregate Answers
        │
        ▼
Granite Generates Final Response
        │
        ▼
Gradio Displays the Answer
```

---

# Conclusion

This project demonstrates a complete **Agentic RAG architecture** powered by **LangGraph** and **Ollama Granite 4.1 8B**. It extends the traditional RAG pipeline with conversation memory, intelligent query rewriting, human-assisted clarification, hierarchical retrieval, multi-agent reasoning, self-correction, and context compression. By running the Granite model locally through Ollama, the system provides a privacy-preserving, cost-effective, and modular solution for building advanced document question-answering applications.
