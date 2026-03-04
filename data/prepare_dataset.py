from datasets import load_dataset

# Load TriviaQA (answerable questions)
print("Loading TriviaQA...")
trivia = load_dataset("mandarjoshi/trivia_qa", "rc", split="train")

# Load SQuAD 2.0 (unanswerable questions)
print("Loading SQuAD 2.0...")
squad = load_dataset("rajpurkar/squad_v2", split="train")

def format_trivia(example):
    # Grab first wikipedia context passage
    contexts = example["entity_pages"]["wiki_context"]
    context = contexts[0] if contexts else ""
    if not context:
        return None
    
    question = example["question"]
    answer = example["answer"]["value"]
    
    return {
        "prompt": f"Context: {context}\nQuestion: {question}",
        "response": f"FINAL: {answer}",
        "label": "answerable"
    }

def format_squad_unanswerable(example):
    # SQuAD 2.0 has empty answers for unanswerable questions
    if len(example["answers"]["text"]) == 0:
        return {
            "prompt": f"Context: {example['context']}\nQuestion: {example['question']}",
            "response": "FINAL: I don't know.",
            "label": "unanswerable"
        }
    return None

# Format and filter
print("Formatting TriviaQA...")
trivia_formatted = [format_trivia(ex) for ex in trivia.select(range(20000))]
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

# Preview one of each
print("\n--- Answerable example ---")
print(full_dataset[0])
print("\n--- Unanswerable example ---")
print(full_dataset[-1])
