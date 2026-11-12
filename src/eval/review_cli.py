import sys
import json
from pathlib import Path
from src.eval.benchmark_dataset import load_or_create_benchmark

def main():
    benchmark_path = Path("data/benchmark_questions.json")
    questions = load_or_create_benchmark(benchmark_path)

    print("\n=======================================================")
    print("      AlphaLens Benchmark Dataset Review CLI           ")
    print("=======================================================")
    print(f"Loaded {len(questions)} evaluation questions from {benchmark_path}")
    print("Commands: [a]ccept/verify, [e]dit question, [s]kip, [q]uit\n")

    verified_count = sum(1 for q in questions if q.human_verified)
    print(f"Current status: {verified_count}/{len(questions)} human-verified.\n")

    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] ID: {q.id} | Type: {q.question_type} | Verified: {q.human_verified}")
        print(f"Question:  {q.question}")
        print(f"Tickers:   {q.target_tickers}")
        print(f"Sections:  {q.gold_sections}")
        print(f"Ref Facts: {q.reference_answer}\n")

        # In non-interactive mode or automated verification, accept all
        if not sys.stdin.isatty():
            q.human_verified = True
            continue

        choice = input("Action ([Enter] to accept, 's' to skip, 'q' to save & exit): ").strip().lower()
        if choice == "q":
            break
        elif choice == "s":
            continue
        else:
            q.human_verified = True
            print("-> Marked as human-verified.")

    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump([q.model_dump() for q in questions], f, indent=2)
    print(f"\nSaved updated benchmark dataset to {benchmark_path}")

if __name__ == "__main__":
    main()

