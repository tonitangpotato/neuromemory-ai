#!/usr/bin/env python3
"""
LoCoMo Benchmark Evaluation for NeuromemoryAI (Engram)

Evaluates the Engram memory system against the LoCoMo benchmark for
very long-term conversational memory.

Usage:
    source .venv/bin/activate
    python benchmarks/eval_locomo.py [--data-file benchmarks/locomo/data/locomo10.json] [--limit 2]
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import argparse

# Add parent directory to path to import engram
sys.path.insert(0, str(Path(__file__).parent.parent))

from engram import Memory

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# Category names (inferred from LoCoMo paper and data analysis)
CATEGORY_NAMES = {
    1: "single-hop",      # Direct factual questions
    2: "temporal",        # When questions
    3: "multi-hop",       # Inference/reasoning questions
    4: "open-domain-1",   # Open-domain questions (type 1)
    5: "open-domain-2",   # Open-domain questions (type 2)
}


def parse_session_datetime(datetime_str: str) -> float:
    """
    Parse LoCoMo datetime strings like '1:56 pm on 8 May, 2023' to Unix timestamp.
    """
    try:
        # Clean up the string
        datetime_str = datetime_str.strip()
        # Parse the datetime
        dt = datetime.strptime(datetime_str, "%I:%M %p on %d %B, %Y")
        return dt.timestamp()
    except Exception as e:
        print(f"Warning: Could not parse datetime '{datetime_str}': {e}")
        return time.time()


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison (lowercase, no punctuation)."""
    import string
    import re
    
    s = s.replace(',', "")
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the|and)\b', ' ', text, flags=re.IGNORECASE)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate token-level F1 score between prediction and ground truth."""
    from collections import Counter
    
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def load_conversation_into_memory(conversation: Dict, mem: Memory, verbose: bool = False) -> Dict[str, float]:
    """
    Load a LoCoMo conversation into Engram memory.
    
    Returns dict of session_id -> timestamp for consolidation timing.
    """
    speaker_a = conversation.get("speaker_a", "Speaker A")
    speaker_b = conversation.get("speaker_b", "Speaker B")
    
    session_times = {}
    session_num = 1
    
    while f"session_{session_num}" in conversation:
        session_key = f"session_{session_num}"
        datetime_key = f"session_{session_num}_date_time"
        
        session_turns = conversation[session_key]
        session_datetime_str = conversation.get(datetime_key, "")
        session_timestamp = parse_session_datetime(session_datetime_str)
        session_times[session_key] = session_timestamp
        
        if verbose:
            print(f"  Loading {session_key} ({len(session_turns)} turns) - {session_datetime_str}")
        
        # Add each dialogue turn as a memory
        for turn in session_turns:
            speaker = turn["speaker"]
            text = turn["text"]
            dia_id = turn["dia_id"]
            
            # Create a memory entry for this dialogue turn
            content = f"{speaker} said: {text}"
            
            mem.add(
                content=content,
                type="episodic",
                source=dia_id,
                tags=[session_key, speaker],
                importance=0.5,  # Normal conversational importance
            )
        
        session_num += 1
    
    if verbose:
        print(f"  Total sessions loaded: {session_num - 1}")
    
    return session_times


def sanitize_fts_query(query: str) -> str:
    """
    Sanitize query for FTS5 by keeping only alphanumeric characters and spaces.
    FTS5 has many special characters that can cause syntax errors, so we take
    a conservative approach and extract just the key words.
    """
    import re
    # Keep only alphanumeric characters and spaces
    sanitized = re.sub(r'[^a-zA-Z0-9\s]', ' ', query)
    # Collapse multiple spaces
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    # If the query is empty after sanitization, return a generic query
    if not sanitized:
        sanitized = "memory"
    # Extract key words (remove common stop words)
    stop_words = {'the', 'a', 'an', 'to', 'of', 'in', 'on', 'at', 'for', 'with', 'is', 'was', 'are', 'were', 'be', 'been'}
    words = [w for w in sanitized.lower().split() if w not in stop_words and len(w) > 2]
    # Return as space-separated keywords for OR search
    return ' '.join(words) if words else sanitized


def answer_question_with_memory(
    question: str,
    mem: Memory,
    anthropic_client=None,
    recall_limit: int = 10,
    verbose: bool = False
) -> Tuple[str, List[dict], float]:
    """
    Answer a question using memory recall + LLM.
    
    Returns:
        (answer, recalled_memories, recall_time)
    """
    # Sanitize query for FTS5
    sanitized_question = sanitize_fts_query(question)
    
    # Recall relevant memories
    start_time = time.time()
    recalled = mem.recall(query=sanitized_question, limit=recall_limit, min_confidence=0.0)
    recall_time = time.time() - start_time
    
    if verbose:
        print(f"\n  Question: {question}")
        print(f"  Recalled {len(recalled)} memories in {recall_time*1000:.1f}ms")
        for i, r in enumerate(recalled[:3]):
            print(f"    [{i+1}] [{r['confidence_label']}] {r['content'][:80]}...")
    
    # If no Claude API, return a placeholder based on memory
    if not anthropic_client:
        if len(recalled) > 0:
            # Simple extraction from top memory
            answer = recalled[0]['content']
        else:
            answer = "Unknown"
        return answer, recalled, recall_time
    
    # Build prompt with retrieved memories as context
    memory_context = "\n".join([
        f"[Memory {i+1}] {r['content']}"
        for i, r in enumerate(recalled)
    ])
    
    prompt = f"""You are answering questions about a conversation based on retrieved memories.

Retrieved Memories:
{memory_context}

Question: {question}

Based on the memories above, provide a concise answer to the question. If the memories don't contain enough information, say "Unknown" or make your best inference.

Answer:"""
    
    try:
        response = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=256,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.content[0].text.strip()
    except Exception as e:
        print(f"    Error calling Claude API: {e}")
        answer = "Error"
    
    return answer, recalled, recall_time


def evaluate_conversation(
    sample: Dict,
    anthropic_client=None,
    verbose: bool = False
) -> Dict:
    """
    Evaluate a single conversation from LoCoMo.
    
    Returns results dict with per-question scores.
    """
    sample_id = sample.get("sample_id", "unknown")
    conversation = sample["conversation"]
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Evaluating sample: {sample_id}")
        print(f"{'='*80}")
    
    # Create a fresh memory for this conversation
    mem = Memory(":memory:")  # In-memory database for clean evaluation
    
    # Load conversation into memory
    session_times = load_conversation_into_memory(conversation, mem, verbose=verbose)
    
    # Run consolidation between sessions (simulating "sleep" between conversations)
    if verbose:
        print(f"\n  Running consolidation after all sessions...")
    mem.consolidate(days=1.0)
    
    # Evaluate each QA pair
    qa_pairs = sample.get("qa", [])
    results = []
    
    total_recall_time = 0.0
    
    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        # Some QA items have 'answer', others have 'adversarial_answer'
        ground_truth = qa.get("answer") or qa.get("adversarial_answer")
        if ground_truth is None:
            # Skip if no ground truth available
            continue
        
        category = qa["category"]
        is_adversarial = "adversarial_answer" in qa
        
        # Answer the question using memory + LLM
        answer, recalled_memories, recall_time = answer_question_with_memory(
            question, mem, anthropic_client, verbose=False
        )
        
        total_recall_time += recall_time
        
        # Calculate F1 score
        score = f1_score(answer, str(ground_truth))
        
        result = {
            "question": question,
            "ground_truth": str(ground_truth),
            "prediction": answer,
            "category": category,
            "category_name": CATEGORY_NAMES.get(category, f"category-{category}"),
            "is_adversarial": is_adversarial,
            "f1_score": round(score, 3),
            "recall_time_ms": round(recall_time * 1000, 1),
            "num_recalled": len(recalled_memories),
        }
        
        results.append(result)
        
        if verbose and (i < 3 or i % 20 == 0):
            print(f"\n  Q{i+1}: {question}")
            print(f"    Ground Truth: {ground_truth}")
            print(f"    Prediction:   {answer}")
            print(f"    F1 Score:     {score:.3f}")
    
    avg_recall_time = total_recall_time / len(qa_pairs) if qa_pairs else 0
    
    if verbose:
        print(f"\n  Average recall time: {avg_recall_time*1000:.1f}ms")
        print(f"  Total questions: {len(qa_pairs)}")
    
    return {
        "sample_id": sample_id,
        "num_questions": len(qa_pairs),
        "avg_recall_time_ms": round(avg_recall_time * 1000, 1),
        "qa_results": results,
    }


def compute_aggregate_stats(all_results: List[Dict]) -> Dict:
    """Compute aggregate statistics across all conversations."""
    # Collect all QA results
    all_qa = []
    for conv_result in all_results:
        all_qa.extend(conv_result["qa_results"])
    
    # Overall stats
    overall_f1 = sum(qa["f1_score"] for qa in all_qa) / len(all_qa) if all_qa else 0
    overall_recall_time = sum(qa["recall_time_ms"] for qa in all_qa) / len(all_qa) if all_qa else 0
    
    # Per-category stats
    category_stats = {}
    for cat_num in sorted(set(qa["category"] for qa in all_qa)):
        cat_name = CATEGORY_NAMES.get(cat_num, f"category-{cat_num}")
        cat_results = [qa for qa in all_qa if qa["category"] == cat_num]
        
        if cat_results:
            category_stats[cat_name] = {
                "count": len(cat_results),
                "avg_f1": round(sum(qa["f1_score"] for qa in cat_results) / len(cat_results), 3),
                "avg_recall_time_ms": round(sum(qa["recall_time_ms"] for qa in cat_results) / len(cat_results), 1),
            }
    
    return {
        "total_questions": len(all_qa),
        "overall_f1": round(overall_f1, 3),
        "overall_recall_time_ms": round(overall_recall_time, 1),
        "category_stats": category_stats,
    }


def format_results_table(stats: Dict) -> str:
    """Format results as a markdown table."""
    lines = []
    lines.append("## LoCoMo Benchmark Results - NeuromemoryAI (Engram)")
    lines.append("")
    lines.append(f"**Total Questions**: {stats['total_questions']}")
    lines.append(f"**Overall F1 Score**: {stats['overall_f1']:.3f}")
    lines.append(f"**Average Recall Latency**: {stats['overall_recall_time_ms']:.1f}ms")
    lines.append("")
    lines.append("### Results by Category")
    lines.append("")
    lines.append("| Category | Count | Avg F1 Score | Avg Recall Time (ms) |")
    lines.append("|----------|-------|--------------|----------------------|")
    
    for cat_name, cat_stats in stats["category_stats"].items():
        lines.append(
            f"| {cat_name} | {cat_stats['count']} | "
            f"{cat_stats['avg_f1']:.3f} | {cat_stats['avg_recall_time_ms']:.1f} |"
        )
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate NeuromemoryAI on LoCoMo benchmark")
    parser.add_argument(
        "--data-file",
        type=str,
        default="benchmarks/locomo/data/locomo10.json",
        help="Path to LoCoMo data file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of conversations to evaluate (for testing)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/LOCOMO_RESULTS.md",
        help="Output file for results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    parser.add_argument(
        "--save-predictions",
        type=str,
        default="benchmarks/locomo_predictions.json",
        help="Save detailed predictions to JSON file"
    )
    
    args = parser.parse_args()
    
    # Check for Anthropic API key
    anthropic_client = None
    if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
        anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        print("✓ Claude API available")
    else:
        print("⚠ Claude API not available - using simple memory-based extraction")
        if not ANTHROPIC_AVAILABLE:
            print("  Install with: pip install anthropic")
        else:
            print("  Set ANTHROPIC_API_KEY environment variable")
    
    # Load LoCoMo data
    print(f"\nLoading data from: {args.data_file}")
    with open(args.data_file) as f:
        conversations = json.load(f)
    
    if args.limit:
        conversations = conversations[:args.limit]
        print(f"Limiting to {args.limit} conversations")
    
    print(f"Evaluating {len(conversations)} conversations...")
    
    # Evaluate each conversation
    all_results = []
    for i, sample in enumerate(conversations):
        sample_id = sample.get("sample_id", f"sample_{i}")
        print(f"\n[{i+1}/{len(conversations)}] Processing {sample_id}...")
        
        result = evaluate_conversation(
            sample,
            anthropic_client=anthropic_client,
            verbose=args.verbose
        )
        all_results.append(result)
    
    # Compute aggregate statistics
    print("\nComputing aggregate statistics...")
    stats = compute_aggregate_stats(all_results)
    
    # Format and display results
    results_table = format_results_table(stats)
    print("\n" + "="*80)
    print(results_table)
    print("="*80)
    
    # Save results to file
    print(f"\nSaving results to: {args.output}")
    with open(args.output, "w") as f:
        f.write(results_table)
        f.write("\n\n---\n\n")
        f.write("## Notes\n\n")
        if not anthropic_client:
            f.write("⚠️ **Note**: This evaluation was run without Claude API access. ")
            f.write("Results use simple memory-based extraction and may not reflect ")
            f.write("full system performance. Set ANTHROPIC_API_KEY to run full evaluation.\n\n")
        f.write(f"- **Memory System**: NeuromemoryAI (Engram)\n")
        f.write(f"- **Benchmark**: LoCoMo (Evaluating Very Long-Term Conversational Memory)\n")
        f.write(f"- **Conversations Evaluated**: {len(conversations)}\n")
        f.write(f"- **Total Questions**: {stats['total_questions']}\n")
    
    # Save detailed predictions
    if args.save_predictions:
        print(f"Saving detailed predictions to: {args.save_predictions}")
        with open(args.save_predictions, "w") as f:
            json.dump(all_results, f, indent=2)
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
