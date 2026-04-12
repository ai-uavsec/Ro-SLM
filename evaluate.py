import math
import numpy as np
import datetime
from my_utils import *
from airsim_wrapper import *

aw = AirSimWrapper()

def evaluate_task(task_name):
    print(f"Evaluating {task_name} ...")
    # read task prompt text file with respect to task name
    with open("task_prompts/" + task_name + ".txt", 'r') as file:
        tasks = file.readlines()
    len_tasks = len(tasks)
    
    overwrite_log(task_name + '_result')	# delete history result log

    with open("log/" + task_name + "_code.txt", 'r', encoding="utf-8") as file:
        lines = file.readlines()
    
    if task_name == 'basic' or task_name == 'advanced' or task_name == 'basic_modified' or task_name == 'advanced_modified':
        if task_name == "basic_modified":
            changes = np.loadtxt("state_changes/" + "basic" + ".txt")
        elif task_name == "advanced_modified":
            changes = np.loadtxt("state_changes/" + "advanced" + ".txt")
        else:
            changes = np.loadtxt("state_changes/" + task_name + ".txt")
            
        state_changes = []
        seperate = 0
        for idx, state_change in enumerate(changes):
            if state_change[0] == 2:
                state_changes.append(changes[seperate:idx].tolist())
                seperate = idx + 1
        
        code = ""
        len_state_changes = len(state_changes)
        index = 0
        success = 0
        overall_completeness = 0
        overall_success = 0
        i = 0
        state_change = np.asarray(state_changes[i])
        # init drone position
        aw.reset_airsim()
        # time.sleep(1)
        aw.fly_to([0, 0, -10])
        time.sleep(1)
        
        for idx, line in enumerate(lines):
            
            if line == "---------\n":
                completeness = success/len(state_change)
                overall_completeness += completeness
                print(completeness, i)
                log(str(completeness) + " " +str(i), task_name + '_result', "\n")
                if completeness == 1:
                    overall_success += 1
                
                code = ""
                index = 0
                success = 0
                i += 1
                state_change = np.asarray(state_changes[i%len_state_changes])

                aw.reset_airsim()
                aw.fly_to([0, 0, -12])
                time.sleep(1)

            else:
                
                if line == "python\n":
                    pass
                elif line == "None\n":
                    success = 0
                    print("Error: No code")
                else:
                    
                    initial_state = aw.get_state()
                    # tranform code with line break into one line 
                    try:
                        if code != "":
                            exec(code)
                            code = ""
                        
                        exec(line)
                    except SyntaxError:
                        code += line
                    # except AttributeError as e:
                    #     success = 0
                    #     print(f'AttributeError occurred: {e}')
                    #     continue
                    # except NameError as e:
                    #     success = 0
                    #     print(f'NameError occurred: {e}')
                    #     continue
                    # except TypeError as e:
                    #     success = 0
                    #     print(f'TypeError occurred: {e}')
                    #     continue
                    # except ModuleNotFoundError as e:
                    #     success = 0
                    #     print(f'ModuleNotFoundError occurred: {e}')
                    #     continue
                    
                    except Exception as e:
                        print(f"{type(e).__name__} occurred: {e}")
                        continue
                    # code = execute_line(line, code)
                    # process = Process(target=loop_function)
                    # process.start()
                    # process.join(timeout=5)  # Wait for 5 seconds

                    # if process.is_alive():
                    #     process.terminate()
                    #     print("Timeout reached! Process terminated.")
                    
                    if len(state_change) > index:

                        current_state = aw.get_state()
                        differences = np.asarray(current_state) - np.asarray(initial_state)
                        differences[3] = ((differences[3] + 180) % 360) - 180 # map yaw differences to [-180, 180]
                        
                        if abs(differences[0]) > 2 or abs(differences[1]) > 2 or abs(differences[2]) > 1 or abs(differences[3]) > 30:
                            if round(abs(differences[3])/10) == 18:
                                differences[3] = 180
                            
                            check_equality = differences - state_change[index][1:5]

                            if state_change[index][0] == 1 and abs(check_equality[0]) < 1.5 and abs(check_equality[1]) < 1.5 and abs(check_equality[2]) < 1.5 and abs(check_equality[3]) < 6:
                                success += 1
                            print(state_change[index][1:5], differences, success)
                            if len(state_change) > 1:
                                index += 1
                            else:
                                index = 0

        result = [overall_success/i, overall_completeness/i]
    
    else:
        raise ValueError("Unkown task name!!! Please check your task name")
        result = np.nan
    
    # store result to .txt
    if isinstance(result, list):
        result = ' '.join(map(str, result))	# converts a list of elements into a single string, with each element separated by a space.
        log(result, task_name + '_result', "  " + task_name + "\n")
    else:
        log(result, task_name + '_result', "  " + task_name + "\n")
    print(f"{task_name} result: {result}")
    # store how many times tasks are repeated
    repeat = i/len_tasks
    log("repeat times: ", task_name + '_result', str(repeat) + "\n")
    # store date & time to .txt for reference 
    now = datetime.datetime.now()
    print("Current date and time:", now.strftime("%Y-%m-%d %H:%M:%S"))
    log(now.strftime("%Y-%m-%d %H:%M:%S"), task_name + '_result', "\n")	# transfrom to human-readable format
    
    return result


def evaluate_single_task(response, task_name, task_id):
    
    changes = np.loadtxt("state_changes/" + task_name + ".txt")
    state_changes = []
    seperate = 0
    for idx, state_change in enumerate(changes):
        if state_change[0] == 2:
            state_changes.append(changes[seperate:idx].tolist())
            seperate = idx + 1
    
    code = ""
    index = 0
    success = 0

    i = 0
    # init drone position
    aw.reset_airsim()
    time.sleep(1)
    aw.fly_to([0, 0, -10])
    time.sleep(1)
    
    state_change = np.asarray(state_changes[task_id])

    for idx, line in enumerate(response.splitlines()):
        if line == "python\n":
            pass
        elif line == "None\n":
            success = 0
            print("Error: No code")
        else:
            
            initial_state = aw.get_state()
            # tranform code with line break into one line 
            try:
                if code != "":
                    exec(code)
                    code = ""
                
                exec(line)
                    
            except SyntaxError:
                code += line
            
                
            # code = execute_line(line, code)
            # process = Process(target=loop_function)
            # process.start()
            # process.join(timeout=5)  # Wait for 5 seconds

            # if process.is_alive():
            #     process.terminate()
            #     print("Timeout reached! Process terminated.")
            
            if len(state_change) > index:

                current_state = aw.get_state()
                differences = np.asarray(current_state) - np.asarray(initial_state)
                differences[3] = ((differences[3] + 180) % 360) - 180 # map yaw differences to [-180, 180]
                
                if abs(differences[0]) > 2 or abs(differences[1]) > 2 or abs(differences[2]) > 1 or abs(differences[3]) > 30:
                    if round(abs(differences[3])/10) == 18:
                        differences[3] = 180
                    
                    check_equality = differences - state_change[index][1:5]

                    if state_change[index][0] == 1 and abs(check_equality[0]) < 1.5 and abs(check_equality[1]) < 1.5 and abs(check_equality[2]) < 1.5 and abs(check_equality[3]) < 6:
                        success += 1
                    print(state_change[index][1:5], differences, success)
                    index += 1

        completeness = success/len(state_change)
        if completeness == 1:
            SR = 1
        else:
            SR = 0

    result = [SR, completeness]
    
    return result





def NL_observation(response, task_prompt):
    # overwrite_log("temp")
    # log(code, "temp")
    # with open("log/code_temp.txt", 'r') as file:
    #     lines = file.readlines()
    description = ""
    shape = ""
    code = ""
    action_idx = 1
    reset_height = -10
    aw.reset_airsim()
    aw.fly_to([0, 0, reset_height])
    time.sleep(1)

    for idx, line in enumerate(response.splitlines()):

        if line == "python\n":
            pass
        elif line == "None\n":
            pass
        else:
            last_state = aw.get_state()
            # transform code with line break into one line 
            try:
                if code != "":
                    exec(code)
                    code = ""
                
                exec(line)
                    
            except SyntaxError:
                code += line
            except NameError as e:
                description = "NameError occurred: " + f" {e}"
                return  description
            except AttributeError as e:
                description = f'AttributeError occurred: {e}'
                return  description
            except TypeError as e:
                description = f'TypeError occurred: {e}'
                return  description
            except ModuleNotFoundError as e:
                description = f'ModuleNotFoundError occurred: {e}'
                return  description

                
            current_state = aw.get_state()
            differences = np.asarray(current_state) - np.asarray(last_state)

            if abs(differences[0]) > 2 or abs(differences[1]) > 2 or abs(differences[2]) > 1 or abs(differences[3]) > 20:
                
                x = differences[0]
                y = differences[1]
                z = differences[2]
                yaw = differences[3]
                
                x_round = round(x)
                y_round = round(y)
                z_round = round(z)
                yaw_round = round(yaw/15)*15
                
                yaw_current = current_state[3]
                current_yaw_round = round(yaw_current/15)*15
                
                theta = -current_state[3]
                theta = math.radians(theta)
                x_body = x * math.cos(theta) - y * math.sin(theta)
                y_body = x * math.sin(theta) + y * math.cos(theta)
                z_body = -z
                x_body_round = round(x_body)
                y_body_round = round(y_body)
                z_body_round = -z_round
                
                description += f"Action {action_idx}: "
                action_idx += 1
                ## movement only in z axis
                if x_round == 0 and y_round == 0 and z_round != 0 and yaw_round == 0:
                    
                    if z > 12:
                            description += "Landing."
                    elif z < 0:
                        if -z_round == 2:
                            description += "Take off." 
                        else:
                            description += f"Fly {-z_round} meters up. "
                    elif z > 0 and z <= 15:
                        description += f"Fly {z_round} meters down. "
                    else:
                        pass

                
                ## yaw rotation 
                elif abs(yaw_round) > 20:
                    yaw_180map = ((yaw_round + 180) % 360) - 180 # map yaw_round to [-180, 180]
                    if yaw_180map == -180:
                        yaw_180map = 180
                    if yaw_180map < 0:
                        description += f"Turn {-yaw_180map} degrees counterclockwise. "
                    elif yaw_180map > 0:
                        description += f"Turn {yaw_180map} degrees clockwise. "                        
                    else:
                        pass
                    # # orientation after movement
                    # if yaw_round != 0:
                    #     if current_yaw_round == 90:
                    #         description += "The drone turned to face the local east. "
                    #     elif current_yaw_round == -90:
                    #         description += "The drone turned to face the local west. "
                    #     elif current_yaw_round == -180 or current_yaw_round == 180:
                    #         description += "The drone turned to face the local south. "
                    #     elif current_yaw_round == 0:
                    #         description += "The drone turned to face the local north. "
                    #     else:
                    #         # if abs(yaw_current) > 20:
                    #         #     description += f"The drone turned to face {yaw_current} degrees in the world frame. "
                    #         pass
                
                elif (x_round != 0 or y_round != 0) and yaw_round == 0:
                    if "body frame" not in task_prompt:
                        # orientation in movement
                        if current_yaw_round == 90:
                            orientation = "east"
                        elif current_yaw_round == -90:
                            orientation = "west"
                        elif current_yaw_round == -180 or current_yaw_round == 180:
                            orientation = "south"
                        elif current_yaw_round == 0:
                            orientation = "north"
                        else:
                            orientation = f"{current_yaw_round} degrees"
                        
                        print(f'{current_state[0]:.3f}, {current_state[1]:.3f}, {current_state[2]:.3f}, {current_state[3]:.3f}')

                        ## movement in x axis of world frame
                        if abs(x_round) > 1 and abs(y_round) <= 1:
                            if x_round < 0:
                                movement =  f"Fly {-round(x_round)} meters South in the world axis while facing {orientation}. The drone moves to[{round(current_state[0])}, {round(current_state[1])}]. "
                            else:
                                movement = f"Fly {round(x_round)} meters North in the world axis while facing {orientation}. The drone moves to [{round(current_state[0])}, {round(current_state[1])}]. "
                        elif abs(x_round) <= 1 and abs(y_round) > 1:
                            ## movement in y axis of world frame
                            if y_round < 0:
                                movement = f"Fly {-round(y_round)} meters West in the world axis while facing {orientation}. The drone moves to [{round(current_state[0])}, {round(current_state[1])}]. "
                            else:
                                movement = f"Fly {round(y_round)} meters East in the world axis while facing {orientation}. The drone moves to [{round(current_state[0])}, {round(current_state[1])}]. "
                        else:
                            movement = f"The drone moves to [{round(current_state[0])}, {round(current_state[1])}] while facing {orientation}."
                        description += movement
                        
                    else:
                        ## body frame movement in x-y plane
                        if z_round == 0:
                            if abs(x_body_round) > 1 and abs(y_body_round) <= 1:
                                if x_body_round < 0:
                                    description += f"Fly {-x_body_round} meters backward in drone's body frame. "
                                elif x_body_round > 0:
                                    description += f"Fly {x_body_round} meters forward in drone's body frame. "
                            elif abs(x_body_round) <= 1 and abs(y_body_round) > 1:
                                if y_body_round < 0:
                                    description += f"Fly {-y_body_round} meters left in drone's body frame. "
                                elif y_body_round > 0:
                                    description += f"Fly {y_body_round} meters right in drone's body frame. "
                            else:
                                pass
                                # description += "unknown movement in xy plane" # waiting for comprehensive movement implementation.
                        ## body frame movement in y-z or x-z plane
                        else:
                            move = round((x ** 2 + y ** 2 + z ** 2) ** 0.5)
                            if move > 15:
                                description += "Landing."
                            # body frame movement in y-z plane
                            elif x_body_round == 0:
                                degree = math.degrees(math.atan(abs(z_body)/abs(y_body)))
                                degree = round(degree/15)*15
                                # top right
                                if y_body_round > 0 and z_body_round > 0:
                                    description += f"Fly the drone in the top-right direction at an angle of {degree} degrees from the horizontal axis, in the YZ plane of drone's body frame for a distance of {move} meters. "
                                # bottom right
                                if y_body_round > 0 and z_body_round < 0:
                                    description += f"Fly the drone in the bottom-right direction at an angle of {degree} degrees from the horizontal axis, in the YZ plane of drone's body frame for a distance of {move} meters. "
                                # top left
                                if y_body_round < 0 and z_body_round > 0:
                                    description += f"Fly the drone in the top-left direction at an angle of {degree} degrees from the horizontal axis, in the YZ plane of drone's body frame for a distance of {move} meters. "
                                # bottom left
                                if y_body_round < 0 and z_body_round < 0:
                                    description += f"Fly the drone in the bottom-left direction at an angle of {degree} degrees from the horizontal axis, in the YZ plane of drone's body frame for a distance of {move} meters. "
                            # body frame movement in x-z plane
                            elif y_body_round == 0:
                                degree = math.degrees(math.atan(abs(z_body)/abs(x_body)))
                                degree = round(degree/15)*15
                                # top forward
                                if x_body_round > 0 and z_body_round > 0:
                                    description += f"Fly the drone in the top-forward direction at an angle of {degree} degrees from the horizontal axis, in the XZ plane of drone's body frame for a distance of {move} meters. "
                                # bottom forward
                                if x_body_round > 0 and z_body_round < 0:
                                    description += f"Fly the drone in the bottom-forward direction at an angle of {degree} degrees from the horizontal axis, in the XZ plane of drone's body frame for a distance of {move} meters. "
                                # top backward
                                if x_body_round < 0 and z_body_round > 0:
                                    description += f"Fly the drone in the top-backward direction at an angle of {degree} degrees from the horizontal axis, in the XZ plane of drone's body frame for a distance of {move} meters. "
                                # bottom backward
                                if x_body_round < 0 and z_body_round < 0:
                                    description += f"Fly the drone in the bottom-backward direction at an angle of {degree} degrees from the horizontal axis, in the XZ plane of drone's body frame for a distance of {move} meters. "
            
    return description


def numerical_observation(response):

    observation = ""

    code = ""
    action_idx = 1

    aw.reset_airsim()
    aw.fly_to([0, 0, -10])
    time.sleep(1)

    state = np.zeros(4)

    for idx, line in enumerate(response.splitlines()):

        if line == "python\n":
            pass
        elif line == "None\n":
            pass
        else:
            last_state = aw.get_state()
            # transform code with line break into one line 
            try:
                if code != "":
                    exec(code)
                    code = ""
                
                exec(line)
                    
            except SyntaxError:
                code += line
            except NameError as e:
                description = "NameError occurred: " + f" {e}"
                return  description
            current_state = aw.get_state()
            differences = np.asarray(current_state) - np.asarray(last_state)
            

            if abs(differences[0]) > 2 or abs(differences[1]) > 2 or abs(differences[2]) > 1 or abs(differences[3]) > 20:
            
                # x = current_state[0]
                # y = current_state[1]
                # z = current_state[2]
                # yaw = current_state[3]
                
                # x_round = round(x/5)*5
                # y_round = round(y/5)*5
                # z_round = round(z,2)
                # # if abs(differences[2]) < 2:
                # #     z_round = round(z/5)*5 -1.5
                # # else:
                # #     z_round = round(z/5)*5
                # yaw_round = round(yaw/15)*15
                
                x = differences[0]
                y = differences[1]
                z = differences[2]
                yaw = differences[3]
                
                x_round = round(x)
                y_round = round(y)
                z_round = round(z)
                yaw_round = round(yaw/15)*15

                if abs(x_round) <= 1:
                    x_round = 0
                if abs(y_round) <= 1:
                    y_round = 0

                yaw_round = ((yaw_round + 180) % 360) - 180 # map yaw_round to [-180, 180]
                if yaw_round == -180:
                    yaw_round = 180

                differences = np.asarray([x_round, y_round, z_round, yaw_round])
                state += differences

                if z_round == -2:
                    observation += f"{action_idx}, take off. " 
                else:
                    observation += f"{action_idx}, [{x_round}, {y_round}, {z_round}, {yaw_round}]. "
                    # observation += f"{action_idx}, {state}. "

                action_idx += 1

    return observation



if __name__ == "__main__":
    
    # with open("log/code_temp.txt", "r", encoding="utf-8") as f:
    #     content = f.read()
    # response = reconstruct_code(content, "sss")
    # print(response)
    task_names = ["advanced"] 
    for task in task_names:
        result = evaluate_task(task)
    # result = evaluate_task2()
    print(f"***Evaluate via evaluate.py***    success rate: {result}")
    # result = ' '.join(map(str, result))  # Converts elements to strings and joins them

    # log(result, task_name + '_result', "task2")