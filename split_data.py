"""Split LLM-generated data into train/eval by code."""
import json, random

data = json.load(open("generated_training_data/train_data.json"))
codes = sorted(set(d["code"] for d in data))
print(f"Total examples: {len(data)}, Unique codes: {len(codes)}")

rng = random.Random(42)
rng.shuffle(codes)
eval_codes = set(codes[:50])

train = [d for d in data if d["code"] not in eval_codes]
evl = [d for d in data if d["code"] in eval_codes]

train_codes = len(set(d["code"] for d in train))
eval_code_count = len(set(d["code"] for d in evl))
print(f"Train: {len(train)} examples ({train_codes} codes)")
print(f"Eval:  {len(evl)} examples ({eval_code_count} codes)")

json.dump(train, open("generated_training_data/train_split.json", "w"), indent=2, ensure_ascii=False)
json.dump(evl, open("generated_training_data/eval_split.json", "w"), indent=2, ensure_ascii=False)
print("Saved train_split.json and eval_split.json")
