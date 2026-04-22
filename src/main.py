from huggingface_hub import login
import os
login(token=os.environ.get("HF_TOKEN"))

from datasets import load_dataset
import torch
torch.manual_seed(42)
dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")
samples = dataset.select(range(50))
print(f"Dataset loaded! Total samples: {len(samples)}")
print(f"\nExample article:")
print(samples[0]['article'][:300])
print(f"\nReference summary:")
print(samples[0]['highlights'])

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
model_id = "google/gemma-2-2b-it"
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
print("Loading model... (takes 3-5 minutes)")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
print("Gemma model ready!")

import time

print("Running zero-shot prompts...")
zero_shot_summaries = []
zero_shot_times = []

for i, sample in enumerate(samples.select(range(10))):
    article = sample['article'][:800]
    
    messages = [
        {"role": "user", "content": f"Summarize the following article in 3 sentences:\n\n{article}"}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    start = time.time()
    response = gen(
        prompt,
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    end = time.time()
    
    generated = response[0]['generated_text']
    summary = generated[len(prompt):]
    zero_shot_summaries.append(summary.strip())
    zero_shot_times.append(end - start)
    print(f"Sample {i+1}/10 done")

avg_latency = sum(zero_shot_times) / len(zero_shot_times)
print(f"\nZero-shot complete!")
print(f"Average latency: {avg_latency:.2f} seconds per summary")
print(f"\nExample output:\n{zero_shot_summaries[0]}")


print("Running few-shot prompts...")
few_shot_summaries = []

for i, sample in enumerate(samples.select(range(10))):
    article = sample['article'][:800]
    
    messages = [
        {"role": "user", "content": """Summarize the following article in 3 sentences.

Here is an example:
Article: Apple released a new iPhone model with improved camera features and longer battery life. The device will be available in stores next month at a starting price of $999.
Summary: Apple announced a new iPhone with enhanced camera and battery improvements. The phone launches next month starting at $999.

Now summarize this article:
Article: """ + article}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    response = gen(
        prompt,
        max_new_tokens=150,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    generated = response[0]['generated_text']
    summary = generated[len(prompt):]
    few_shot_summaries.append(summary.strip())
    print(f"Sample {i+1}/10 done")

print(f"\nFew-shot complete!")
print(f"\nExample output:\n{few_shot_summaries[0]}")


print("Running chain-of-thought prompts...")
cot_summaries = []

for i, sample in enumerate(samples.select(range(10))):
    article = sample['article'][:800]
    
    messages = [
        {"role": "user", "content": f"""Read the following article and summarize it by thinking step by step.

Step 1: Identify the main topic of the article.
Step 2: Find the most important facts and details.
Step 3: Write a clear 3 sentence summary.

Article: {article}

Now follow the steps and write your summary:"""}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    response = gen(
        prompt,
        max_new_tokens=200,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    generated = response[0]['generated_text']
    summary = generated[len(prompt):]
    cot_summaries.append(summary.strip())
    print(f"Sample {i+1}/10 done")

print(f"\nChain-of-thought complete!")
print(f"\nExample output:\n{cot_summaries[0]}")


from evaluate import load

print("Calculating ROUGE scores...")

rouge = load("rouge")

reference_summaries = [samples[i]['highlights'] for i in range(10)]

# Score zero-shot
zero_shot_scores = rouge.compute(
    predictions=zero_shot_summaries,
    references=reference_summaries
)

# Score few-shot
few_shot_scores = rouge.compute(
    predictions=few_shot_summaries,
    references=reference_summaries
)

# Score chain-of-thought
cot_scores = rouge.compute(
    predictions=cot_summaries,
    references=reference_summaries
)
print("\n========== RESULTS ==========")
print(f"\nZero-Shot:")
print(f"  ROUGE-1: {zero_shot_scores['rouge1']:.4f}")
print(f"  ROUGE-2: {zero_shot_scores['rouge2']:.4f}")
print(f"  ROUGE-L: {zero_shot_scores['rougeL']:.4f}")

print(f"\nFew-Shot:")
print(f"  ROUGE-1: {few_shot_scores['rouge1']:.4f}")
print(f"  ROUGE-2: {few_shot_scores['rouge2']:.4f}")
print(f"  ROUGE-L: {few_shot_scores['rougeL']:.4f}")

print(f"\nChain-of-Thought:")
print(f"  ROUGE-1: {cot_scores['rouge1']:.4f}")
print(f"  ROUGE-2: {cot_scores['rouge2']:.4f}")
print(f"  ROUGE-L: {cot_scores['rougeL']:.4f}")
print("\n==============================")


import json
import os

os.makedirs("results", exist_ok=True)

metrics = {
    "zero_shot": zero_shot_scores,
    "few_shot": few_shot_scores,
    "chain_of_thought": cot_scores,
    "efficiency": {
        "avg_latency_seconds": avg_latency,
        "model": "google/gemma-2-2b-it",
        "samples_evaluated": 10
    }
}

with open("results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

with open("results/generations.txt", "w") as f:
    for i in range(10):
        f.write(f"=== Sample {i+1} ===\n")
        f.write(f"ZERO-SHOT:\n{zero_shot_summaries[i]}\n\n")
        f.write(f"FEW-SHOT:\n{few_shot_summaries[i]}\n\n")
        f.write(f"CHAIN-OF-THOUGHT:\n{cot_summaries[i]}\n\n")
        f.write(f"REFERENCE:\n{reference_summaries[i]}\n\n")
        f.write("="*50 + "\n\n")

print("Results saved!")
print("metrics.json saved")
print("generations.txt saved")