from datasets import load_dataset, Dataset, DatasetDict
import json
import os

# Load TriviaQA (answerable questions)
print("Loading TriviaQA...")
trivia = load_dataset("mandarjoshi/trivia_qa", "rc", split="train")

# Load SQuAD 2.0 (unanswerable questions)
print("Loading SQuAD 2.0...")
squad = load_dataset("rajpurkar/squad_v2", split="train")

#choose the correct context paragraph for each question/answer. if non match, return None
def pick_context_with_answer(contexts, answer):
    ans = answer.strip().lower()
    for c in contexts:
        if c and ans in c.lower():
            return c
    return None


def shrink_context_around_answer(context: str, answer: str, max_chars: int = 4000, min_window: int = 800):
    """
    Returns a substring of `context` centered around the first occurrence of `answer`,
    with length <= max_chars, attempting to keep sentence boundaries.
    If answer isn't found, returns None.

    max_chars: final cap for context length (characters)
    min_window: if context is already short, keep it (helps avoid over-cropping)
    """
    if not context or not answer:
        return None

    ctx_lower = context.lower()
    ans_lower = answer.strip().lower()
    idx = ctx_lower.find(ans_lower)
    if idx == -1:
        return None

    # If already small enough, keep as-is
    if len(context) <= max_chars:
        return context

    # Center window around answer span
    ans_end = idx + len(answer)
    center = (idx + ans_end) // 2

    half = max_chars // 2
    start = max(0, center - half)
    end = min(len(context), center + half)

    snippet = context[start:end]

    # Try to align to sentence-ish boundaries to look nicer
    # (simple heuristics; good enough for this project)
    if start > 0:
        # move start forward to the next likely boundary
        cut = max(snippet.find(". "), snippet.find("\n"))
        if cut != -1 and cut < 300:  # don't cut too aggressively
            snippet = snippet[cut+2:] if snippet[cut:cut+2] == ". " else snippet[cut+1:]

    if end < len(context):
        # move end backward to last boundary in snippet
        last_period = snippet.rfind(". ")
        last_nl = snippet.rfind("\n")
        cut = max(last_period, last_nl)
        if cut != -1 and (len(snippet) - cut) < 300:
            snippet = snippet[:cut+1]

    # If we somehow over-cropped too hard, ensure at least min_window if possible
    if len(snippet) < min_window and len(context) > min_window:
        # expand a bit around original center
        half2 = min_window // 2
        start2 = max(0, center - half2)
        end2 = min(len(context), center + half2)
        snippet = context[start2:end2]

    return snippet.strip()

def format_trivia(example):
    # Grab first wikipedia context passage
    contexts = example["entity_pages"]["wiki_context"]
    context = pick_context_with_answer(contexts, example["answer"]["value"])
    if not context:
        return None

    question = example["question"]
    answer = example["answer"]["value"]
    
    context = shrink_context_around_answer(context, answer, max_chars=4000)
    if not context:
        return None
    
    return {
        "prompt": f"Context: {context}\nQuestion: {question}\nAnswer:",
        "response": f"FINAL: {answer}",
        "label": 1,
    }
    




def format_squad_unanswerable(example):
    # SQuAD 2.0 has empty answers for unanswerable questions
    if len(example["answers"]["text"]) == 0:
        return {
            "prompt": f"Context: {example['context']}\nQuestion: {example['question']}\nAnswer:",
            "response": "FINAL: I don't know.",
            "label": 0,
        }
    return None


# Format and filter
print("Formatting TriviaQA...")
trivia_formatted = []
for ex in trivia.select(range(50000)):  # try more because of filtering
    r = format_trivia(ex)
    if r:
        trivia_formatted.append(r)
    if len(trivia_formatted) >= 20000:
        break
trivia_formatted = [ex for ex in trivia_formatted if ex is not None]

print("Formatting SQuAD unanswerables...")
squad_unanswerable = []
for ex in squad:
    result = format_squad_unanswerable(ex)
    if result:
        squad_unanswerable.append(result)
    if len(squad_unanswerable) >= 7000:
        break

# Combine
full_dataset = trivia_formatted + squad_unanswerable
print(f"Total examples: {len(full_dataset)}")
print(f"Answerable: {len(trivia_formatted)}, Unanswerable: {len(squad_unanswerable)}")


ds = Dataset.from_list(full_dataset)

# Deterministic shuffle
SEED = 42
ds = ds.shuffle(seed=SEED)

# Train/val/test split (80/10/10)
ds_splits = ds.train_test_split(test_size=0.2, seed=SEED)          # 80/20
tmp = ds_splits["test"].train_test_split(test_size=0.5, seed=SEED) # 10/10
ds_dict = DatasetDict({
    "train": ds_splits["train"],
    "val": tmp["train"],
    "test": tmp["test"],
})

print(ds_dict)
print({k: len(v) for k, v in ds_dict.items()})

# Preview one of each
print("\n--- Train example ---")
print(ds_dict["train"][0])

# ---- Save to disk (recommended) ----
os.makedirs("data/hf_dataset", exist_ok=True)
ds_dict.save_to_disk("data/hf_dataset")
print("Saved HF DatasetDict to data/hf_dataset")