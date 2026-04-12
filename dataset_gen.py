from openai import OpenAI
from my_utils import *
# from evaluate import *
import json
import datetime

with open("system_prompts/airsim.txt", "r") as f:
    airsim_sysprompt = f.read()
with open("system_prompts/task_gen.txt", "r") as f:
    task_gen_sysprompt = f.read()
with open("system_prompts/task_gen_advanced.txt", "r") as f:
    task_gen_advanced_sysprompt = f.read()
with open("system_prompts/task_aug_advanced.txt", "r") as f:
    task_aug_sysprompt = f.read()
with open("system_prompts/feedback.txt", "r") as f:
    feedback_sysprompt = f.read()
with open("system_prompts/prior_feedback_interpret.txt", "r") as f:
    interpret_sysprompt = f.read()
with open("system_prompts/prior_feedback_verify.txt", "r") as f:
    verify_sysprompt = f.read()

code_history = [
    {
        "role": "system",
        "content": airsim_sysprompt
    }
]

task_gen_history = [
    {
        "role": "system",
        "content": task_gen_sysprompt
    }
]

task_gen_advanced_history = [
    {
        "role": "system",
        "content": task_gen_advanced_sysprompt
    }
]

task_aug_history = [
    {
        "role": "system",
        "content": task_aug_sysprompt
    }
]

feedback_history = [
    {
        "role": "system",
        "content": feedback_sysprompt
    }
]

interpret_history = [
    {
        "role": "system",
        "content": interpret_sysprompt
    }
]

verify_history = [
    {
        "role": "system",
        "content": verify_sysprompt
    }
]

print("Initializing GPT...")
client = OpenAI()

def single_request(history, prompt, model_name):
    single_chat = history.copy()
    single_chat.append(
        {
            "role": "user",
            "content": prompt
        }
    )
    completion = client.chat.completions.create(
        model = model_name, # "gpt-4", "gpt-4o-mini", "gpt-4-turbo"
        messages = single_chat
    )
    response = completion.choices[0].message.content

    return response


def request(prompt, model_name):
    chat_history.append(
		{
			"role": "user",
			"content": prompt
		}
	)
    completion = client.chat.completions.create(
		model = model_name, # "gpt-4", "gpt-4o-mini", "gpt-4-turbo"
		messages = chat_history
	)
    response = completion.choices[0].message.content
    chat_history.append(
        {
            "role": "assistant",
            "content": response,
        }
    )
    
    return response


def basic_task_gen(easy, hard):
    # task generation
    overwrite_log("gen_tasks")
    gen_history = task_gen_history.copy()
    tasks = single_request(gen_history ,f"Please generate {easy} tasks like examples.", model_name = "gpt-5.1")
    log(tasks + "\n", "gen_tasks")
    tasks = single_request(gen_history, f"Generate {hard} tasks that fly the drone in XZ or YZ plane like example 4 in the drone's body frame", model_name)
    log(tasks, "gen_tasks")

    clean_gen_tasks()
    print(f'Basic task generation completed! {easy} task, {hard} task')

def advanced_task_gen(num):
    overwrite_log("gen_tasks")
    gen_history = task_gen_advanced_history.copy()
    tasks = single_request(gen_history, f"generate {num} tasks according to the flight path and pattern from the examples I gave you, you should cover all the patterns in the examples", model_name)
    log(tasks, "gen_tasks")
    

def task_aug(times, model_name):
    with open("log/gen_tasks.txt", 'r', encoding="utf-8") as file:
        tasks = file.readlines()
    with open("log/gen_code.txt", "r", encoding="utf-8") as file:
        codes = file.read()
    print(f'Augmenting: total tasks **{len(tasks)}**')
    
    aug_task = ""
    if tasks and not tasks[-1].endswith("\n"):
        aug_task += "\n"

    for aug in range(times):
        for i, task_prompt in enumerate(tasks):
            aug_history = task_aug_history.copy()
            task = single_request(aug_history, "Given task: "+ task_prompt, model_name)
            aug_task += task + '\n'
            print(f'Augmenting task {i+1} in round {aug+1}')

        with open("log/gen_tasks.txt", 'a', encoding="utf-8") as file:
            file.write(aug_task)
        aug_task = ""
        
    if codes and not codes.endswith("\n"):
        codes += "\n"
    codes = codes * (times + 1)
    with open("log/gen_code.txt", "w", encoding="utf-8") as f:
        f.write(codes)
    
    print('*** Task augmentation completed')

def code_gen(model_name):
    with open("log/gen_tasks.txt", 'r', encoding="utf-8") as file:
        tasks = file.readlines()
    print(f'Total tasks {len(tasks)}')
    ## code generation
    overwrite_log("gen_code")
    correction_time = 5
    code_prune = False

    for i, task_prompt in enumerate(tasks):
        global chat_history
        chat_history = code_history.copy()
        chat_interpret = interpret_history.copy()
        chat_verify = verify_history.copy()

        print(f"Generating code for task {i}")
        
        # print(i, task_prompt)
        response =request(task_prompt, model_name)
        if '```' in response:
            code = extract_python_code(response)
        else:
            code = response
            
        if code_prune:
            code = prune_code(response)

        for time_i in range(correction_time):
            interpretation = single_request(chat_interpret, code, model_name)
            interpretation = "Task description: " + task_prompt + "\nDrone actions: " + interpretation
            verification = single_request(chat_verify, interpretation, model_name)
            
            if "YES" in verification:
                break
            elif "NO" in verification:
                if time_i == correction_time and "NO" in verification:
                    print(f"****No further correction for task {i}****")
                    break

                print(f"Correction times on task {i}:", time_i)
                print(task_prompt, "\n", code, "\n", verification)
                
                correction_prompt = "\n The differences between code and task description: " \
                                    + verification.replace("NO", "") + " \n Please reason where you did wrong and revise the code based on the differences. "
                response = request(correction_prompt, model_name)
                
                if '```' in response:
                    response = extract_python_code(response)
                
                if code_prune:
                    code = prune_code(response)
                else:
                    code = response
            else:
                print("Unknown verification result: ", verification)
        
        log(str(code), "gen_code", "\n---------\n") # log generated code of each task
        
    print('*** Code generation completed')
    '''
        observation = NL_observation(code, task_prompt)
        
        for time_i in range(correction_time):
            if not observation:
                print(f"Failed to generate trajectory for task {i}")
                feedback = "error in generated code"
            else:
                prompt = "Task description: " + task_prompt + "\n" + "Action steps: " + observation
                feedback = single_request(feedback_history, prompt, model_name)
                
            print(task_prompt, '\n', code, '\n', observation, '\n',feedback)
            if "YES" in feedback:
                break

            elif "NO" in feedback:
                print(f'Task {i}, correction time {time_i}')
                if time_i == correction_time and "NO" in feedback:
                    print(f"****No further correction for task {i}****")
                    break
                correction_prompt = "\n The differences between actions and task description: " \
                        + feedback.replace("NO", "") + " \n Please reason where you did wrong and refine the code based on the differences. "
                response = request(correction_prompt, model_name)
            
                if '```' in response:
                    code = extract_python_code(response)
                else:
                    code = response

                observation = NL_observation(code, task_prompt)
            else:
                print(f"Unexpected feedback format: {feedback}, regenerate feedback in the next iteration")
            
        log(str(code), "gen_code", "\n---------\n") # log generated code of each task
    '''
def dataset_gen(prune=1):
    ## dataset construction
    with open("log/gen_code.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()

    with open("log/gen_tasks.txt", 'r', encoding="utf-8") as file:
        tasks = file.readlines()

    idx = 0
    code = ""
    dataset = []

    for _, line in enumerate(lines):
        if line == "---------\n":
            if prune:
                code = prune_code(code)
            dataset.append({
                "messages": [
                    {"role": "system", "content": airsim_sysprompt},
                    {"role": "user", "content": tasks[idx]},
                    {"role": "assistant", "content": code},
                ]
            })
            code = ""
            idx += 1
        else:
            code += line

    with open("log/llama3_dataset.jsonl", "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"*** dataset generation completed. code prune = {prune}")

if __name__ == "__main__":
    model_name = "o4-mini" # gpt-5.1
    # basic_task_gen(2,1)
    # advanced_task_gen(10)
    # code_gen(model_name)
    # task_aug(2, model_name)
    dataset_gen(prune = 1)
    
    now = datetime.datetime.now()
    print("Current date and time:", now.strftime("%Y-%m-%d %H:%M:%S"))