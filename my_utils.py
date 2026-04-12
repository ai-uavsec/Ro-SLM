import re
import time
# from ultralytics import YOLO
# import torch

# obj_list = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
#            "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow",
#            "elephant",
#            "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
#            "snowboard", "sports ball", "kite",
#            "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#            "fork", "knife ",
#            "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut",
#            "cake", "chair", "sofa",
#            "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ",
#            "keyboard ", "cell phone", "microwave ",
#            "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ",
#            "hair drier", "toothbrush ")

# model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

# model = YOLO("yolo11l.pt")

code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)
def extract_python_code(content):
    code_blocks = code_block_regex.findall(content)
    if code_blocks:
        full_code = "\n".join(code_blocks)

        if full_code.startswith("python"):
            full_code = full_code[7:]

        return full_code
    else:
        return None

# log response/code 
def log(response, task_name, string = ""):
    file_name = "log/" + task_name + ".txt"
    
    if str(response):
        with open(file_name, "a", encoding="utf-8") as file:
            file.write(str(response) + string)

def overwrite_log(name):
    logfile_name = "log/" + name + ".txt"
    try:
        with open(logfile_name, 'w') as file:
            pass
    except FileNotFoundError:
        open(logfile_name, 'a').close()
        print(f"{logfile_name} not found. An empty file has been created.")

# print evaluations 
def print_evaluation(evaluation):
    lines = evaluation.split('\n')
    last_line = lines[-1]
    if "YES" in last_line:
        print("***", evaluation)
    elif "NO" in last_line:
        print("///", evaluation)
    else:
        print("The evaluator confused!!!")

# convert a number to percentage expression
def number_to_percentage(value):
    return f"{value * 100:.2f}%"

# def detect_objects(img):
#     results = model(img)
#     # rec_result = results.xyxy[0]
#     # rec_result_np = rec_result.cpu().numpy()
#     # obj_locs = rec_result_np[:, 0:4]
#     # obj_list = results.pandas().xyxy[0]['name'].tolist()
#     obj_locs = results[0].boxes.xyxy.cpu().numpy()
#     obj_list = [results[0].names[cls] for cls in results[0].boxes.cls.cpu().numpy()]
    
#     return obj_list, obj_locs

# def sweeping(object_name):
#     if isinstance(object_name, str):
#         print("The variable is a string.")
#         return True
#     else:
#         return False
# def approach(object_name):
#     if isinstance(object_name, str):
#         print("The variable is a string.")
#         return True
#     else:
#         return False

# def orienting(object_name):
#     if isinstance(object_name, str):
#         print("The variable is a string.")
#         return True
#     else:
#         return False

# def following(object_name):
#     if isinstance(object_name, str):
#         print("The variable is a string.")
#         return True
#     else:
#         return False


def eval_state_change(updated_state, current_state, gt):
    state_change = updated_state - current_state
    error = state_change - gt[1:5]
    position_offset = 1
    yaw_offset = 5
    if gt[0] == 1:
        if abs(error[0]) < position_offset and abs(error[1]) < position_offset and abs(error[2]) < position_offset and abs(error[3]) < yaw_offset:
            return True
        else:
            return False
    elif gt[0] == 0:
        return abs(updated_state[2]) < 1

# remove comments, whitespace, empty lines   
def prune_code(code):
    # Remove comments (both single line and multi-line)
    code = re.sub(r'//.*?(\n|$)', '\n', code)  # Remove single-line comments
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # Remove multi-line comments
    code = re.sub(r"\s*#.*", "", code)
    # Remove leading and trailing whitespace from each line
    code = '\n'.join(line.strip() for line in code.split('\n'))

    # Optionally, remove empty lines
    code = '\n'.join(line for line in code.split('\n') if line)

    return code

# remove comments only
def remove_comments(code):
    # Remove single-line comments
    code = re.sub(r'#.*', '', code)
    
    # Remove multi-line comments (triple quotes)
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    
    return code

def extract_code(code_str: str) -> str:
    """
    Extracts code inside ```python ... ``` fences.
    - If fences are at start/end, returns the content inside.
    - If fences appear somewhere else, returns only what's between them.
    - If no fences, returns the original string.
    """
    start_tag = "python```"
    end_tag = "```"

    start_idx = code_str.find(start_tag)
    if start_idx != -1:
        # Move to the end of the starting fence line
        start_idx = code_str.find("\n", start_idx)
        if start_idx == -1:
            return ""  # no content after start fence
        start_idx += 1

        end_idx = code_str.find(end_tag, start_idx)
        if end_idx != -1:
            return code_str[start_idx:end_idx].strip()

    return code_str.strip()

def remove_px_land(code_str: str) -> str:
    lines = code_str.splitlines()
    filtered_lines = [line for line in lines if "px.land()" not in line.strip()]
    return "\n".join(filtered_lines)


def clean_gen_tasks():
    """
    Remove leading 'xx.' numbers from each line of the file,
    modifying the file in place.
    """
    with open("log/gen_tasks.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        # Remove leading "xx." and any surrounding spaces
        line = re.sub(r'^\s*\d+\.\s*', '', line)

        # Skip empty or whitespace-only lines
        if line.strip() == "":
            continue

        cleaned_lines.append(line)


    with open("log/gen_tasks.txt", "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)