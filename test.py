# inspect_hf_dataset.py
# Usage:
#   python inspect_hf_dataset.py --path data/data/hf_dataset --split train --n 5
#   python inspect_hf_dataset.py --path data/data/hf_dataset --export_jsonl data/train_first200.jsonl --export_n 200

import argparse
import random
from turtle import title
from datasets import load_from_disk


def get_context_from_prompt(prompt: str) -> str:
    # Expected: "Context: ...\nQuestion: ...\nAnswer:"
    if "Context:" not in prompt or "\nQuestion:" not in prompt:
        return ""
    return prompt.split("Context:", 1)[1].split("\nQuestion:", 1)[0].strip()


def get_answer_from_response(resp: str) -> str:
    # Expected: "FINAL: <answer>" or "FINAL: I don't know."
    if "FINAL:" not in resp:
        return ""
    return resp.split("FINAL:", 1)[1].strip()


def context_length(prompt: str) -> int:
    return len(get_context_from_prompt(prompt))


def answer_in_context(example: dict) -> bool:
    # Only meaningful for answerable rows (label==1)
    ctx = get_context_from_prompt(example.get("prompt", "")).lower()
    ans = get_answer_from_response(example.get("response", "")).lower()
    if not ctx or not ans:
        return False
    return ans in ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, required=True, help="Path passed to load_from_disk(), e.g. data/data/hf_dataset")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Split to inspect")
    ap.add_argument("--n", type=int, default=5, help="How many random examples to print")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--preview_chars", type=int, default=800, help="How many chars of prompt to print")
    ap.add_argument("--check_n", type=int, default=500, help="How many examples to run checks on")
    ap.add_argument("--export_jsonl", type=str, default=None, help="If set, exports a slice to JSONL at this path")
    ap.add_argument("--export_n", type=int, default=200, help="How many examples to export if export_jsonl is set")
    args = ap.parse_args()

    ds = load_from_disk(args.path)
    print("Loaded:", ds)
    print("Sizes:", {k: len(ds[k]) for k in ds.keys()})

    split = ds[args.split]
    random.seed(args.seed)

    # --- Basic random preview ---
    print(f"\n=== Random preview: {args.split} (n={args.n}) ===")
    idxs = random.sample(range(len(split)), k=min(args.n, len(split)))
    for i in idxs:
        ex = split[i]
        print(f"\n--- idx {i} ---")
        print(ex["prompt"][: args.preview_chars])
        print("response:", ex["response"])
        print("label:", ex["label"])
        print("context_len:", context_length(ex["prompt"]))

    # --- Context length stats ---
    prompts = split["prompt"]
    lens = [context_length(p) for p in prompts]
    lens_sorted = sorted(lens)
    print(f"\n=== Context length stats ({args.split}) ===")
    print("min:", lens_sorted[0])
    print("p10:", lens_sorted[int(0.10 * (len(lens_sorted) - 1))])
    print("median:", lens_sorted[int(0.50 * (len(lens_sorted) - 1))])
    print("p90:", lens_sorted[int(0.90 * (len(lens_sorted) - 1))])
    print("max:", lens_sorted[-1])
    print("5 smallest:", lens_sorted[:5])

    # --- Answer-in-context check for answerables ---
    N = min(args.check_n, len(split))
    bad_idxs = []
    checked_answerables = 0
    for i in range(N):
        ex = split[i]
        # assumes numeric labels: 1=answerable, 0=unanswerable
        if ex.get("label", None) == 1:
            checked_answerables += 1
            if not answer_in_context(ex):
                bad_idxs.append(i)

    print(f"\n=== Answer-in-context check (first {N} rows of {args.split}) ===")
    print("answerables checked:", checked_answerables)
    print("answerable rows where answer NOT found in context:", len(bad_idxs))

    if bad_idxs:
        j = bad_idxs[0]
        ex = split[j]
        print("\nFirst failing example:")
        print("idx:", j)
        print(ex["prompt"][: args.preview_chars])
        print("response:", ex["response"])
        print("extracted_answer:", get_answer_from_response(ex["response"]))
        print("context_len:", context_length(ex["prompt"]))
        

    def get_context(prompt):
        return prompt.split("Context:",1)[1].split("\nQuestion:",1)[0].strip()

    lens = [len(get_context(p)) for p in ds["train"]["prompt"]]
    lens_sorted = sorted(lens)
    print("min:", lens_sorted[0])
    print("p10:", lens_sorted[int(0.10*(len(lens_sorted)-1))])
    print("median:", lens_sorted[int(0.50*(len(lens_sorted)-1))])
    print("p90:", lens_sorted[int(0.90*(len(lens_sorted)-1))])
    print("max:", lens_sorted[-1])
    
    
    
    # Add this section near the end of your inspection script (after loading ds_dict / ds)
# It prints: one answerable + one unanswerable example from each split (train/val/test),
# showing context, question, and the gold/target response.

    def extract_context_question(prompt: str):
    # Expected: "Context: ...\nQuestion: ...\nAnswer:"
        ctx = ""
        q = ""
        try:
            if "Context:" in prompt:
                after_ctx = prompt.split("Context:", 1)[1]
                if "\nQuestion:" in after_ctx:
                    ctx = after_ctx.split("\nQuestion:", 1)[0].strip()
                    after_q = after_ctx.split("\nQuestion:", 1)[1]
                # strip optional "\nAnswer:" if present
                    q = after_q.split("\nAnswer:", 1)[0].strip()
                else:
                 ctx = after_ctx.strip()
        except Exception:
            pass
        return ctx, q

    def print_one_by_label(split_ds, label_value, split_name, title):
        # label_value: 1 for answerable, 0 for unanswerable (adjust if you used strings)
        idx = None
        for i in range(len(split_ds)):
            if split_ds[i]["label"] == label_value:
             idx = i
             break
        if idx is None:
          print(f"\n[{split_name}] No examples found for label={label_value} ({title})")
          return

        ex = split_ds[idx]  
        ctx, q = extract_context_question(ex["prompt"])
        print(f"\n==============================")
        print(f"[{split_name}] {title} (idx={idx})")
        print(f"------------------------------")
        print("QUESTION:\n", q)
        print("\nCONTEXT (first 4000 chars):\n", ctx[:4000] + ("..." if len(ctx) > 4000 else ""))
        print("\nTARGET RESPONSE:\n", ex["response"])
        print("==============================\n")

    for split_name in ["train", "val", "test"]:
        split_ds = ds[split_name]  # or ds_dict[split_name], depending on your variable name
        # If your labels are strings ("answerable"/"unanswerable"), replace 1/0 with those strings.
        print_one_by_label(split_ds, 1, split_name, "ANSWERABLE example")
        print_one_by_label(split_ds, 0, split_name, "UNANSWERABLE example")

        # --- Optional export to JSONL for easy VSCode scrolling ---
    if args.export_jsonl:
        export_n = min(args.export_n, len(split))
        split.select(range(export_n)).to_json(args.export_jsonl)
        print(f"\nExported first {export_n} examples of {args.split} to {args.export_jsonl}")


if __name__ == "__main__":
    main()