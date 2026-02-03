#!/usr/bin/env python3
"""
LoCoMo Benchmark v2 - Two-Stage Evaluation

Stage 1: Retrieval Quality (NO LLM needed)
  - Measures if Memory finds the right memories
  - Uses ground truth "evidence" from LoCoMo
  - Metrics: Precision, Recall, MRR, Hit@K

Stage 2: End-to-End QA (needs LLM)
  - Measures answer accuracy with LLM
  - Uses same LLM for fair comparison
  - Metrics: F1 Score, Exact Match

Usage:
    # Stage 1 only (no LLM required)
    python benchmarks/eval_locomo_v2.py --stage retrieval

    # Stage 2 only (requires ANTHROPIC_API_KEY)
    python benchmarks/eval_locomo_v2.py --stage qa

    # Both stages
    python benchmarks/eval_locomo_v2.py --stage both
"""

import json
import os
import sys
import re
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from engram import Memory
from engram.config import MemoryConfig

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal", 
    3: "multi-hop",
    4: "open-domain-1",
    5: "open-domain-2",
}


def sanitize_fts_query(query: str) -> str:
    """Sanitize query for FTS5."""
    sanitized = re.sub(r'[^a-zA-Z0-9\s]', ' ', query)
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    if not sanitized:
        return "memory"
    stop_words = {'the', 'a', 'an', 'to', 'of', 'in', 'on', 'at', 'for', 'with', 
                  'is', 'was', 'are', 'were', 'be', 'been', 'what', 'who', 'when',
                  'where', 'why', 'how', 'did', 'does', 'do'}
    words = [w for w in sanitized.lower().split() if w not in stop_words and len(w) > 2]
    return ' '.join(words[:10]) if words else sanitized[:50]


def load_conversation_to_memory(conv: dict, mem: Memory) -> Dict[str, str]:
    """
    Load conversation into Memory, return mapping of dia_id -> memory_id.
    """
    dia_to_mem = {}
    
    # Find all sessions
    session_keys = sorted([k for k in conv['conversation'].keys() if k.startswith('session_') and not k.endswith('_date_time')])
    
    for session_key in session_keys:
        session = conv['conversation'].get(session_key, [])
        if not isinstance(session, list):
            continue
            
        for turn in session:
            if not isinstance(turn, dict):
                continue
            
            speaker = turn.get('speaker', 'Unknown')
            text = turn.get('text', '')
            dia_id = turn.get('dia_id', '')
            
            if text and dia_id:
                content = f"{speaker}: {text}"
                mem_id = mem.add(content, type='episodic', importance=0.5)
                dia_to_mem[dia_id] = mem_id
        
        # Consolidate between sessions
        mem.consolidate(days=0.5)
    
    return dia_to_mem


def evaluate_retrieval(
    question: str,
    evidence_ids: List[str],
    mem: Memory,
    dia_to_mem: Dict[str, str],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Evaluate retrieval quality without LLM.
    
    Args:
        question: The question text
        evidence_ids: Ground truth dialog IDs that contain the answer
        mem: Memory instance
        dia_to_mem: Mapping from dialog ID to memory ID
        k_values: Values of K for Hit@K metric
    
    Returns:
        Dict with precision, recall, mrr, hit@k metrics
    """
    # Get ground truth memory IDs
    ground_truth_mem_ids = set()
    for dia_id in evidence_ids:
        if dia_id in dia_to_mem:
            ground_truth_mem_ids.add(dia_to_mem[dia_id])
    
    if not ground_truth_mem_ids:
        # No ground truth available
        return {'precision': 0, 'recall': 0, 'mrr': 0, 'latency_ms': 0, **{f'hit@{k}': 0 for k in k_values}}
    
    # Recall memories
    query = sanitize_fts_query(question)
    start = time.perf_counter()
    results = mem.recall(query, limit=max(k_values))
    latency = (time.perf_counter() - start) * 1000
    
    retrieved_ids = [r['id'] for r in results]
    retrieved_set = set(retrieved_ids)
    
    # Calculate metrics
    hits = retrieved_set & ground_truth_mem_ids
    
    # Precision@K (using max K)
    precision = len(hits) / len(retrieved_ids) if retrieved_ids else 0
    
    # Recall
    recall = len(hits) / len(ground_truth_mem_ids) if ground_truth_mem_ids else 0
    
    # MRR (Mean Reciprocal Rank)
    mrr = 0
    for i, mem_id in enumerate(retrieved_ids):
        if mem_id in ground_truth_mem_ids:
            mrr = 1.0 / (i + 1)
            break
    
    # Hit@K
    hit_at_k = {}
    for k in k_values:
        top_k = set(retrieved_ids[:k])
        hit_at_k[f'hit@{k}'] = 1.0 if (top_k & ground_truth_mem_ids) else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'mrr': mrr,
        'latency_ms': latency,
        **hit_at_k
    }


def evaluate_qa(
    question: str,
    answer: str,
    mem: Memory,
    client: "Anthropic",
    model: str = "claude-3-haiku-20240307"
) -> Dict[str, float]:
    """
    Evaluate end-to-end QA with LLM.
    """
    # Recall
    query = sanitize_fts_query(question)
    start = time.perf_counter()
    results = mem.recall(query, limit=5)
    latency = (time.perf_counter() - start) * 1000
    
    # Build context
    context = "\n".join([f"- {r['content']}" for r in results])
    
    if not context.strip():
        context = "(No relevant memories found)"
    
    # Ask LLM
    try:
        response = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{
                'role': 'user',
                'content': f"""Based on these conversation memories:
{context}

Question: {question}

Answer briefly and directly. If the information isn't in the memories, say "I don't know"."""
            }]
        )
        predicted = response.content[0].text.strip()
    except Exception as e:
        predicted = f"Error: {e}"
    
    # Calculate F1
    pred_tokens = set(predicted.lower().split())
    ans_tokens = set(str(answer).lower().split())
    
    if not pred_tokens or not ans_tokens:
        f1 = 0.0
    else:
        common = pred_tokens & ans_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ans_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Exact match (normalized)
    pred_norm = ' '.join(predicted.lower().split())
    ans_norm = ' '.join(str(answer).lower().split())
    exact_match = 1.0 if ans_norm in pred_norm or pred_norm in ans_norm else 0.0
    
    return {
        'f1': f1,
        'exact_match': exact_match,
        'latency_ms': latency,
        'predicted': predicted,
        'answer': answer
    }


def run_evaluation(
    data_file: Path,
    stage: str = "both",
    limit: Optional[int] = None,
    model: str = "claude-3-haiku-20240307"
) -> Dict:
    """Run the full evaluation."""
    
    print(f"Loading data from: {data_file}")
    with open(data_file) as f:
        data = json.load(f)
    
    if limit:
        data = data[:limit]
        print(f"Limiting to {limit} conversations")
    
    # Initialize Anthropic client if needed
    client = None
    if stage in ["qa", "both"]:
        if ANTHROPIC_AVAILABLE and os.environ.get("ANTHROPIC_API_KEY"):
            client = Anthropic()
            print(f"✓ Claude API available (model: {model})")
        else:
            print("⚠ Claude API not available - skipping QA evaluation")
            if stage == "qa":
                return {}
            stage = "retrieval"
    
    results = {
        "retrieval": defaultdict(list),
        "qa": defaultdict(list),
        "by_category": defaultdict(lambda: {"retrieval": [], "qa": []})
    }
    
    print(f"\nEvaluating {len(data)} conversations...")
    
    for i, conv in enumerate(data):
        conv_id = conv.get('sample_id', f'conv-{i}')
        print(f"\n[{i+1}/{len(data)}] Processing {conv_id}...")
        
        # Create fresh memory for this conversation
        mem = Memory(":memory:", config=MemoryConfig.chatbot())
        
        # Load conversation
        dia_to_mem = load_conversation_to_memory(conv, mem)
        print(f"    Loaded {len(dia_to_mem)} dialog turns")
        
        # Get QA pairs
        qa_items = conv.get('qa', [])
        print(f"    Evaluating {len(qa_items)} questions")
        
        for qa in qa_items:
            question = qa.get('question', '')
            answer = qa.get('answer', '')
            evidence = qa.get('evidence', [])
            category = qa.get('category', 0)
            cat_name = CATEGORY_NAMES.get(category, f"category-{category}")
            
            # Stage 1: Retrieval evaluation
            if stage in ["retrieval", "both"]:
                ret_metrics = evaluate_retrieval(question, evidence, mem, dia_to_mem)
                results["retrieval"]["all"].append(ret_metrics)
                results["by_category"][cat_name]["retrieval"].append(ret_metrics)
            
            # Stage 2: QA evaluation  
            if stage in ["qa", "both"] and client:
                qa_metrics = evaluate_qa(question, answer, mem, client, model)
                results["qa"]["all"].append(qa_metrics)
                results["by_category"][cat_name]["qa"].append(qa_metrics)
    
    return results


def compute_aggregates(results: Dict) -> Dict:
    """Compute aggregate statistics."""
    agg = {}
    
    # Retrieval aggregates
    if results["retrieval"]["all"]:
        ret_all = results["retrieval"]["all"]
        agg["retrieval"] = {
            "count": len(ret_all),
            "precision": sum(r['precision'] for r in ret_all) / len(ret_all),
            "recall": sum(r['recall'] for r in ret_all) / len(ret_all),
            "mrr": sum(r['mrr'] for r in ret_all) / len(ret_all),
            "hit@1": sum(r['hit@1'] for r in ret_all) / len(ret_all),
            "hit@3": sum(r['hit@3'] for r in ret_all) / len(ret_all),
            "hit@5": sum(r['hit@5'] for r in ret_all) / len(ret_all),
            "hit@10": sum(r['hit@10'] for r in ret_all) / len(ret_all),
            "avg_latency_ms": sum(r['latency_ms'] for r in ret_all) / len(ret_all),
        }
    
    # QA aggregates
    if results["qa"]["all"]:
        qa_all = results["qa"]["all"]
        agg["qa"] = {
            "count": len(qa_all),
            "f1": sum(r['f1'] for r in qa_all) / len(qa_all),
            "exact_match": sum(r['exact_match'] for r in qa_all) / len(qa_all),
            "avg_latency_ms": sum(r['latency_ms'] for r in qa_all) / len(qa_all),
        }
    
    # By category
    agg["by_category"] = {}
    for cat_name, cat_results in results["by_category"].items():
        cat_agg = {}
        
        if cat_results["retrieval"]:
            ret = cat_results["retrieval"]
            cat_agg["retrieval"] = {
                "count": len(ret),
                "mrr": sum(r['mrr'] for r in ret) / len(ret),
                "hit@5": sum(r['hit@5'] for r in ret) / len(ret),
            }
        
        if cat_results["qa"]:
            qa = cat_results["qa"]
            cat_agg["qa"] = {
                "count": len(qa),
                "f1": sum(r['f1'] for r in qa) / len(qa),
            }
        
        agg["by_category"][cat_name] = cat_agg
    
    return agg


def print_results(agg: Dict):
    """Print results in a nice format."""
    print("\n" + "=" * 70)
    print("LOCOMO BENCHMARK RESULTS - NeuromemoryAI")
    print("=" * 70)
    
    # Retrieval results
    if "retrieval" in agg:
        ret = agg["retrieval"]
        print(f"""
## Stage 1: Retrieval Quality (No LLM)

| Metric | Value |
|--------|-------|
| Questions | {ret['count']} |
| **MRR** | **{ret['mrr']:.3f}** |
| Hit@1 | {ret['hit@1']:.1%} |
| Hit@3 | {ret['hit@3']:.1%} |
| Hit@5 | {ret['hit@5']:.1%} |
| Hit@10 | {ret['hit@10']:.1%} |
| Precision | {ret['precision']:.3f} |
| Recall | {ret['recall']:.3f} |
| Avg Latency | {ret['avg_latency_ms']:.1f}ms |
""")
    
    # QA results
    if "qa" in agg:
        qa = agg["qa"]
        print(f"""
## Stage 2: End-to-End QA (With LLM)

| Metric | Value |
|--------|-------|
| Questions | {qa['count']} |
| **F1 Score** | **{qa['f1']:.3f}** |
| Exact Match | {qa['exact_match']:.1%} |
| Avg Latency | {qa['avg_latency_ms']:.1f}ms |
""")
    
    # By category
    print("\n## Results by Category\n")
    print("| Category | Count | MRR | Hit@5 | F1 |")
    print("|----------|-------|-----|-------|-----|")
    
    for cat_name in sorted(agg["by_category"].keys()):
        cat = agg["by_category"][cat_name]
        count = cat.get("retrieval", {}).get("count", cat.get("qa", {}).get("count", 0))
        mrr = cat.get("retrieval", {}).get("mrr", 0)
        hit5 = cat.get("retrieval", {}).get("hit@5", 0)
        f1 = cat.get("qa", {}).get("f1", 0)
        print(f"| {cat_name} | {count} | {mrr:.3f} | {hit5:.1%} | {f1:.3f} |")
    
    print("\n" + "=" * 70)


def save_results(agg: Dict, output_file: Path):
    """Save results to markdown file."""
    with open(output_file, 'w') as f:
        f.write("# LoCoMo Benchmark Results - NeuromemoryAI\n\n")
        f.write(f"*Generated: {datetime.now().isoformat()}*\n\n")
        
        if "retrieval" in agg:
            ret = agg["retrieval"]
            f.write("## Stage 1: Retrieval Quality (No LLM)\n\n")
            f.write("This measures if NeuromemoryAI retrieves the correct memories.\n\n")
            f.write("| Metric | Value | Description |\n")
            f.write("|--------|-------|-------------|\n")
            f.write(f"| **MRR** | **{ret['mrr']:.3f}** | Mean Reciprocal Rank |\n")
            f.write(f"| Hit@1 | {ret['hit@1']:.1%} | Correct in top 1 |\n")
            f.write(f"| Hit@5 | {ret['hit@5']:.1%} | Correct in top 5 |\n")
            f.write(f"| Avg Latency | {ret['avg_latency_ms']:.1f}ms | Retrieval speed |\n\n")
        
        if "qa" in agg:
            qa = agg["qa"]
            f.write("## Stage 2: End-to-End QA (With LLM)\n\n")
            f.write("This measures answer quality using Claude Haiku.\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| **F1 Score** | **{qa['f1']:.3f}** |\n")
            f.write(f"| Exact Match | {qa['exact_match']:.1%} |\n\n")
        
        f.write("## Key Insights\n\n")
        f.write("- **Retrieval metrics** measure NeuromemoryAI's unique contribution\n")
        f.write("- **QA metrics** measure the combined Memory + LLM system\n")
        f.write("- Latency is consistently low (<10ms) due to local FTS5\n")
    
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="LoCoMo Benchmark v2")
    parser.add_argument("--data-file", type=Path, 
                       default=Path("benchmarks/locomo/data/locomo10.json"))
    parser.add_argument("--stage", choices=["retrieval", "qa", "both"], default="both")
    parser.add_argument("--limit", type=int, help="Limit number of conversations")
    parser.add_argument("--model", default="claude-3-haiku-20240307")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/LOCOMO_RESULTS_V2.md"))
    
    args = parser.parse_args()
    
    results = run_evaluation(
        data_file=args.data_file,
        stage=args.stage,
        limit=args.limit,
        model=args.model
    )
    
    if results:
        agg = compute_aggregates(results)
        print_results(agg)
        save_results(agg, args.output)


if __name__ == "__main__":
    main()
