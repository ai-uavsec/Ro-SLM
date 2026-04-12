import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import datetime

model_name = "8B"   # or the -Instruct variant for chat-style SFT
dataset_type = "rm" # "original", "clean", "comments", "no_human","ours"

if model_name == "3B":
    base_model = "meta-llama/Llama-3.2-3B" 
    lora_path = "./llama_finetuned_3b_" + dataset_type
elif model_name == "8B":
    base_model = "meta-llama/Llama-3.1-8B" 
    lora_path = "./llama_finetuned_8b_" + dataset_type
elif model_name == "8B-full":
    base_model = "./llama31_8b_full_ft"
else:
    print("Unknown model name")
    
print(f"**********training {base_model} with dataset: {dataset_type} *****************")
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# # QLoRA config
# bnb = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype="bfloat16",
# )

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
model.config.pad_token_id = tokenizer.pad_token_id

# LoRA config
lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)

# --- Dataset prep ---
ds = load_dataset("json", data_files={"train": "dataset/train_"+ dataset_type +".jsonl", "eval": "dataset/eval_"+ dataset_type +".jsonl"})

def format_row(row):
    parts = []
    for msg in row["messages"]:
        role = msg["role"]        # "system" / "user" / "assistant"
        content = msg["content"]
        parts.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n{content}\n<|eot_id|>"
        )
    text = "".join(parts)
    return {"text": text}

ds = ds.map(format_row)

# keep only the text column so TRL treats this as plain LM data (no chat_template)
cols_to_keep = ["text"]
cols_to_remove = [c for c in ds["train"].column_names if c not in cols_to_keep]
ds = ds.remove_columns(cols_to_remove)

cfg = SFTConfig(
    output_dir= lora_path,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=60, #25-30 for 8b, 30-35 for 3b
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    max_length=2048,
    packing=True,
    save_strategy="epoch",
    dataset_text_field="text",   # <-- goes here, not in SFTTrainer
)

trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=ds["train"],
    eval_dataset=ds.get("eval"),
    processing_class=tokenizer,  # for newer TRL; if this errors, just drop it
    # do NOT pass dataset_text_field or tokenizer here
)

trainer.train()
trainer.save_model()

print(f"training {model_name} complete")
now = datetime.datetime.now()
print("Current date and time:", now.strftime("%Y-%m-%d %H:%M:%S"))
# tokenizer.save_pretrained("./llama32-1b-qlora-sft")
