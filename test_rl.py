import torch
import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from my_utils import *

with open("system_prompts/airsim.txt", "r") as f:
    airsim_sysprompt = f.read()

# -------------------------------------------------------
# Choose base + RL adapter path (GRPO output_dir)
# -------------------------------------------------------
model_name = "8B"
dataset_type = "clean"
reward_type = "llm"

if model_name == "3B":
    # If you GRPO-trained on instruct, keep consistent here too.
    base_model = "meta-llama/Llama-3.2-3B"
    rl_adapter_path = f"./llama_grpo_3b_{dataset_type}_{reward_type}"
elif model_name == "8B":
    # IMPORTANT: match the base model you used during GRPO training.
    # If you trained with "meta-llama/Llama-3.1-8B-Instruct", use it here.
    base_model = "meta-llama/Llama-3.1-8B"
    rl_adapter_path = f"./llama_grpo_8b_{dataset_type}_{reward_type}"
elif model_name == "8B-full":
    base_model = "./llama31_8b_full_ft"
    rl_adapter_path = "./llama_grpo_8b_full_" + dataset_type
else:
    raise ValueError("Unknown model name")

print(f"********** testing RL(GRPO) {model_name} with dataset: {dataset_type} *****************")
print(f"Base: {base_model}")
print(f"Adapter: {rl_adapter_path}")

# -------------------------------------------------------
# Load tokenizer + base model
# -------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

# -------------------------------------------------------
# Load RL fine-tuned LoRA adapter (saved by GRPOTrainer.save_model())
# Standard: load base -> attach adapter
# -------------------------------------------------------
model = PeftModel.from_pretrained(model, rl_adapter_path)
model.eval()

# -------------------------------------------------------
# Prompt building (recommended): use chat template
# -------------------------------------------------------
def build_inputs_from_messages(messages):
    """
    messages: list[{"role": "system"|"user"|"assistant", "content": str}]
    Returns tokenized inputs ready for generate().
    """
    # If tokenizer has a chat template (typical for *-Instruct models),
    # this will format with the correct special tokens and add an assistant prompt.
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,   # adds the assistant header for generation
            return_tensors="pt",
        )
        return {"input_ids": input_ids.to(model.device), "attention_mask": torch.ones_like(input_ids).to(model.device)}

    # Fallback: your manual formatting (use only if no chat template exists)
    parts = []
    for msg in messages:
        parts.append(
            f"<|start_header_id|>{msg['role']}<|end_header_id|>\n{msg['content']}\n<|eot_id|>"
        )
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
    text = "<|begin_of_text|>" + "".join(parts)
    return tokenizer(text, return_tensors="pt").to(model.device)

def single_request(task_prompt):
    messages = [
        {"role": "system", "content": airsim_sysprompt},
        {"role": "user", "content": task_prompt},
    ]
    inputs = build_inputs_from_messages(messages)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=600,
            temperature=0.2,
            top_p=0.3,
            do_sample=True,
            # Optional: if you want generation to stop at EOS more cleanly
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only newly generated tokens
    prompt_len = inputs["input_ids"].shape[1]
    generated = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
    return generated.strip()

# -------------------------------------------------------
# Your existing loop stays the same
# -------------------------------------------------------
repeat = 3
evaluate = 1
generate = 1
task_names = ["advanced_modified"] # "basic_modified", "advanced_modified"

print(f"task: {task_names}, model: {model_name}")

if generate == 1:
    now = datetime.datetime.now()
    overwrite_log("sysprompt")
    log(model_name + dataset_type + "\n" + now.strftime("%Y-%m-%d %H:%M:%S") + "\n" + airsim_sysprompt, "sysprompt")

    for task_name in task_names:
        print(f"********** Generating: {task_name} **********")

        with open("task_prompts/" + task_name + ".txt", "r") as file:
            tasks = file.readlines()

        overwrite_log(task_name + "_code")

        for it in range(repeat):
            for i, task_prompt in enumerate(tasks):
                print(f"Generating code for task {i} on the {it}th repeat.")
                response = single_request(task_prompt)
                code = extract_python_code(response) if "```" in response else response
                log(str(code), task_name + "_code", "\n\n---------\n")

if evaluate:
    from evaluate import *
    for task_name in task_names:
        print(f"Evaluating: {task_name}")
        result = evaluate_task(task_name)
        print(f"*** See log/{task_name}_result.txt ***")
else:
    print(f"Code generation complete. No evaluation for {task_name}!")

now = datetime.datetime.now()
print("Current date and time:", now.strftime("%Y-%m-%d %H:%M:%S"))
print(f"Test on {task_names} complete. Model name: {model_name}. Dataset Type: {dataset_type} Repeat: {repeat}")
