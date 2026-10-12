# pyrefly: ignore [missing-import]
import os
import sys
import json
import types
import asyncio
import argparse
from pathlib import Path

# --- RAGAS 0.4.3 Import Workaround ---
# Mocks ChatVertexAI to prevent import errors during startup outside Jupyter.
try:
    # pyrefly: ignore [missing-import]
    import langchain_community.chat_models.vertexai  # noqa: F401
except ModuleNotFoundError:
    vertexai_module = types.ModuleType("langchain_community.chat_models.vertexai")
    class ChatVertexAI:
        pass
    vertexai_module.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = vertexai_module

# Ensure 'project' is in python path
project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
load_dotenv(os.path.join(project_dir, ".env"))

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from tqdm import tqdm
# pyrefly: ignore [missing-import]
from openai import AsyncOpenAI
# pyrefly: ignore [missing-import]
from langchain_core.messages import HumanMessage

# RAGAS Imports
# pyrefly: ignore [missing-import]
from ragas.metrics.collections import (
    AnswerAccuracy,
    ContextRelevance,
    ResponseGroundedness,
    ContextPrecision,
    ContextRecall,
)
# pyrefly: ignore [missing-import]
from ragas.llms import llm_factory

import config
from core.rag_system import RAGSystem

# --- Ground Truth Context Extractor ---
def extract_context(doc_texts: dict, markdown_file: str, phrases: list[str], window: int = 1400) -> str:
    """Extract compact reference context from source markdown matching context phrases."""
    text = doc_texts.get(markdown_file)
    if not text:
        return ""
    
    lower_text = text.lower()
    spans = []

    for phrase in phrases:
        position = lower_text.find(phrase.lower())
        if position < 0:
            continue
        start = max(0, position - window // 3)
        end = min(len(text), position + len(phrase) + window)
        spans.append((start, end))

    if not spans:
        return text[:window].strip()

    spans.sort()
    merged_spans = []
    for start, end in spans:
        if merged_spans and start <= merged_spans[-1][1]:
            merged_spans[-1] = (merged_spans[-1][0], max(merged_spans[-1][1], end))
        else:
            merged_spans.append((start, end))

    return "\n\n...\n\n".join(text[start:end].strip() for start, end in merged_spans)

# --- RAG Querying ---
def query_rag(rag_system: RAGSystem, question: str) -> dict:
    """Query the RAG system independently by resetting session thread."""
    rag_system.reset_thread()
    cfg = rag_system.get_config()
    result = rag_system.agent_graph.invoke({"messages": [HumanMessage(content=question)]}, config=cfg)
    
    # Retrieve answer
    answer = result["messages"][-1].content
    
    # Retrieve retrieved contexts
    contexts = []
    for agent_answer in sorted(result.get("agent_answers", []), key=lambda item: item["index"]):
        contexts.extend(agent_answer.get("contexts", []))
    contexts = list(dict.fromkeys(contexts))  # deduplicate
    
    return {"answer": answer, "contexts": contexts}

# --- Asynchronous Scoring Core ---
async def score_answer(row: dict, scorers: dict) -> dict:
    """Score a single QA pair against the active RAGAS scorers."""
    if row.get("run_error"):
        tqdm.write(f"   skipped pipeline failure: {row['run_error']}")
        return {metric_name: np.nan for metric_name in scorers}

    retrieved_contexts = row["retrieved_contexts"]
    if isinstance(retrieved_contexts, str):
        try:
            import ast
            retrieved_contexts = ast.literal_eval(retrieved_contexts)
        except Exception:
            retrieved_contexts = [retrieved_contexts]

    metric_inputs = {
        "answer_accuracy": {
            "user_input": row["user_input"],
            "response": row["response"],
            "reference": row["reference"],
        },
        "context_relevance": {
            "user_input": row["user_input"],
            "retrieved_contexts": retrieved_contexts,
        },
        "response_groundedness": {
            "response": row["response"],
            "retrieved_contexts": retrieved_contexts,
        },
        "context_precision": {
            "user_input": row["user_input"],
            "reference": row["reference"],
            "retrieved_contexts": retrieved_contexts,
        },
        "context_recall": {
            "user_input": row["user_input"],
            "retrieved_contexts": retrieved_contexts,
            "reference": row["reference"],
        },
    }

    scores = {}
    for metric_name, scorer in scorers.items():
        try:
            result = await scorer.ascore(**metric_inputs[metric_name])
            scores[metric_name] = result.value
            tqdm.write(f"   {metric_name}: {result.value:.3f}")
        except Exception as exc:
            scores[metric_name] = np.nan
            tqdm.write(f"   {metric_name}: failed ({exc})")

    return scores

# --- Main Evaluation Flow ---
async def run_evaluation():
    parser = argparse.ArgumentParser(description="Evaluate Agentic RAG system with RAGAS metrics.")
    parser.add_argument(
        "--qa-file", 
        type=str, 
        default=None, 
        help="Path to the curated RAGAS QA JSON file."
    )
    parser.add_argument(
        "--output-file", 
        type=str, 
        default="ragas_evaluation_dataset.csv", 
        help="CSV file to load/save RAG outputs."
    )
    parser.add_argument(
        "--no-cache", 
        action="store_true", 
        help="Query RAG pipeline instead of using existing cache CSV."
    )
    args = parser.parse_args()

    # 1. Locate the QA JSON file
    repo_root = Path(project_dir).parent
    qa_candidates = [
        Path(args.qa_file) if args.qa_file else None,
        repo_root / "notebooks" / "data" / "curated_ragas_qa.json",
        repo_root / "data" / "curated_ragas_qa.json",
        Path("./curated_ragas_qa.json")
    ]
    qa_candidates = [p for p in qa_candidates if p is not None]
    qa_path = next((p for p in qa_candidates if p.exists()), None)
    
    if not qa_path:
        sys.exit(f"❌ Error: Curated QA dataset not found. Checked candidates: {[str(p) for p in qa_candidates]}")

    print(f"📦 Loading curated QA dataset from: {qa_path}")
    curated_records = json.loads(qa_path.read_text(encoding="utf-8"))

    # 2. Extract Reference Contexts from Markdown files
    markdown_dir = Path(config.MARKDOWN_DIR)
    if not markdown_dir.exists():
        sys.exit(f"❌ Error: Markdown source directory {markdown_dir.resolve()} does not exist. Please upload documents first.")

    md_paths = sorted(markdown_dir.glob("*.md"))
    if not md_paths:
        sys.exit(f"❌ Error: No markdown files found in {markdown_dir.resolve()}. Run ingestion or upload files first.")

    doc_texts = {path.name: path.read_text(encoding="utf-8") for path in md_paths}
    for record in curated_records:
        record["reference_contexts"] = [
            extract_context(doc_texts, record["markdown_file"], record["context_phrases"])
        ]

    # 3. Determine if we need to query RAG or use cached CSV
    output_path = Path(args.output_file)
    ragas_df = None
    
    if not args.no_cache and output_path.exists():
        print(f"♻️ Found existing evaluation dataset: {output_path}. Loading cached answers...")
        cached_df = pd.read_csv(output_path)
        # Check if questions match
        if len(cached_df) == len(curated_records) and all(cached_df["user_input"].values == [r["question"] for r in curated_records]):
            ragas_df = cached_df
            print("✅ Cached questions match the dataset. Skipping RAG queries.")
        else:
            print("⚠️ Cached dataset length or questions do not match the current curated JSON. Re-querying RAG...")

    if ragas_df is None:
        print("🔨 Initializing RAG System under test...")
        rag_system = RAGSystem()
        rag_system.initialize()
        print("🚀 Querying RAG system for all QA questions...")
        
        results = []
        for record in tqdm(curated_records, desc="Querying RAG"):
            question = record["question"]
            try:
                rag_out = query_rag(rag_system, question)
                results.append({
                    "source": record["source"],
                    "user_input": question,
                    "retrieved_contexts": rag_out["contexts"],
                    "reference_contexts": record["reference_contexts"],
                    "response": rag_out["answer"],
                    "reference": record["reference"],
                    "run_error": ""
                })
            except Exception as e:
                print(f"\n❌ Error evaluating query '{question[:40]}...': {e}")
                results.append({
                    "source": record["source"],
                    "user_input": question,
                    "retrieved_contexts": [],
                    "reference_contexts": record["reference_contexts"],
                    "response": "",
                    "reference": record["reference"],
                    "run_error": str(e)
                })

        ragas_df = pd.DataFrame(results)
        ragas_df.to_csv(output_path, index=False)
        print(f"💾 Fresh RAG responses saved to {output_path}")
        
        # Flush Langfuse traces if enabled
        if hasattr(rag_system, "observability") and rag_system.observability._enabled:
            print("Flushing Langfuse traces...")
            rag_system.observability.flush()

    # 4. Initialize Ragas Scorer with Judge Model
    print(f"\n🧠 Initializing Ragas Judge (Ollama: {config.JUDGE_MODEL})...")
    ollama_client = AsyncOpenAI(
        base_url="http://localhost:11434/v1/",
        api_key="ollama",
    )
    ragas_llm = llm_factory(
        config.JUDGE_MODEL,
        client=ollama_client,
        temperature=0,
    )

    scorers = {
        "answer_accuracy": AnswerAccuracy(llm=ragas_llm),
        "context_relevance": ContextRelevance(llm=ragas_llm),
        "response_groundedness": ResponseGroundedness(llm=ragas_llm),
        "context_precision": ContextPrecision(llm=ragas_llm),
        "context_recall": ContextRecall(llm=ragas_llm),
    }

    # 5. Execute scoring loop
    print("🔄 Running RAGAS metrics evaluation...")
    score_rows = []
    for idx, row in tqdm(ragas_df.iterrows(), total=len(ragas_df), desc="Evaluating"):
        tqdm.write(f"\n[{idx + 1}/{len(ragas_df)}] {row['user_input'][:90]}")
        scores = await score_answer(row, scorers)
        score_rows.append(scores)

    scores_df = pd.DataFrame(score_rows)
    df_results = pd.concat([ragas_df.reset_index(drop=True), scores_df], axis=1)

    # 6. Aggregate and Display Results
    metric_cols = ["answer_accuracy", "context_relevance", "response_groundedness", "context_precision", "context_recall"]
    agg = df_results.groupby("source")[metric_cols].mean().round(3)
    agg.loc["OVERALL"] = df_results[metric_cols].mean().round(3)

    print("\n" + "=" * 80)
    print("                      RAGAS EVALUATION METRICS REPORT")
    print("=" * 80)
    print(agg.to_string())
    print("=" * 80)
    print(f"Results report exported to csv.")

if __name__ == "__main__":
    try:
        asyncio.run(run_evaluation())
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted.")
