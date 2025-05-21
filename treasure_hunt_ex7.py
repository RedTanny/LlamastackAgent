import json
import random
from llama_stack_client import LlamaStackClient, Agent
from datetime import datetime
import pprint

import re
from datetime import datetime

# --- Custom ToolParser Template ---
from llama_stack_client.lib.agents.tool_parser import ToolParser
from llama_stack_client.types.agents.turn import CompletionMessage
from llama_stack_client.types.shared.tool_call import ToolCall
import uuid

class GraniteToolParser(ToolParser):

    def parse_args(self, function_name, arguments)->dict:
            # If arguments is a string, parse it to a dict
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
            # If it can't be parsed, handle as needed
                pass
        if function_name == "think_tool" or function_name == "move_tool":
          if function_name == "think_tool":
            if arguments.get('arguments') is not None:
                out_arguments = {"args":json.dumps(arguments.get('arguments'))}
            else:
                out_arguments = {"args":{"thought":arguments["thought"]}}
          else:
            if arguments.get('arguments') is not None:
                out_arguments = {"args":json.dumps(arguments.get('arguments'))}
            else:
                out_arguments = {"args":{"direction":arguments["direction"]}}            
        return out_arguments    
    
    def create_tool_call(self, function_name, in_arguments) -> ToolCall:
        print(f"[DEBUG] create_tool_call: {function_name} {in_arguments}")

        if function_name == "think_tool" or function_name == "move_tool":         
            arguments = self.parse_args(function_name, in_arguments)
        else:
            arguments = in_arguments    

        return ToolCall(
            call_id=str(uuid.uuid4()),
            tool_name=function_name,
            arguments=arguments,  # If you have arguments, parse them here
        )

    def get_tool_calls(self, output_message: CompletionMessage):
        #print("[PrintToolParser] get_tool_calls called!")
        tool_calls = []        
        try:
            print(output_message.content)
            json_args = json.loads(output_message.content)
            for call in json_args:
                print(f"[DEBUG] call: {call}")
                if "function" in call:
                    if call["function"] == "scan_tool":
                        tool_calls.append(
                            self.create_tool_call("scan_tool", {})
                        )
                    elif call["function"] == "move_tool":
                        # Format: {'json_args': '{"direction":"right"}'}                        
                        tool_calls.append(
                            self.create_tool_call("move_tool", call)
                        )
                    elif call["function"] == "dig_tool":
                        tool_calls.append(self.create_tool_call("dig_tool", {}))
                    elif call["function"] == "think_tool":
                        # Format: {'json_args': '{"thought":"I need to move right"}'}                        
                        tool_calls.append(self.create_tool_call("think_tool", call))
                    elif call["function"] == "grid_map_tool":
                        tool_calls.append(self.create_tool_call("grid_map_tool", {}))
        except Exception as e:
            print(f"[Error:get_tool_calls] {e}/{output_message.content}")
        return tool_calls

                    

def print_grid(width, height, agent_pos, dig_history, treasure_pos, found):
    for y in range(height):
        row = []
        for x in range(width):
            cell = " 0 "
            
            if (x, y) == agent_pos:
                cell = " @ "
            elif (x, y) == treasure_pos:
                cell = " $ "
            elif (x, y) in dig_history:
                cell = " x "
            row.append(cell)
        print("|" + "|".join(row) + "|")
    print()  # Blank line after grid

def grid_to_string(width, height, agent_pos, dig_history, treasure_pos, found):
    lines = []
    for y in range(height):
        row = []
        for x in range(width):
            cell = " 0 "
            if (x, y) == agent_pos:
                cell = " @ "
            elif (x, y) == treasure_pos:
                cell = " $ "
            elif (x, y) in dig_history:
                cell = " x "
            row.append(cell)
        lines.append("|" + "|".join(row) + "|")
    return "\n".join(lines) + "\n"

# --- Treasure Hunt Environment ---
class TreasureHunt:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.agent_pos = (0, 0)
        self.treasure_pos =(random.randint(0, width - 1), random.randint(0, height - 1))
        self.found = False 
        self.dig_history = set()
        self.Memory = []
        self.last_thought =""
        self.last_action  =""
        print(f"[DEBUG] Treasure is hidden at {self.treasure_pos}")

    def move(self, direction):

        x, y = self.agent_pos
        moves = {"up": (x, y - 1), "down": (x, y + 1), "left": (x - 1, y), "right": (x + 1, y)}
        if direction not in moves:
            print(f"[DEBUG] Invalid direction: {direction}")
            return False
        nx, ny = moves[direction]
        if 0 <= nx < self.width and 0 <= ny < self.height:
            print(f"[DEBUG] Moving from {self.agent_pos} to {(nx, ny)}")
            self.agent_pos = (nx, ny)            
            return True
        print(f"[DEBUG] Move blocked: {(nx, ny)} is outside grid bounds.")
        return False

    def scan(self):

        ax, ay = self.agent_pos
        tx, ty = self.treasure_pos
        distance = abs(ax - tx) + abs(ay - ty)
        print(f"[DEBUG] Scanned distance to treasure: {distance} from {self.agent_pos}")        
        return distance

    def dig(self):
        self.dig_history.add(self.agent_pos)
        if self.agent_pos == self.treasure_pos:
            print(f"[DEBUG] Dig success! Treasure found at {self.treasure_pos}")
            self.found = True
            return True
        print(f"[DEBUG] Dig at {self.agent_pos} failed. No treasure.")
        return False

# --- Tool Functions ---
def scan_tool() -> str:
    """
    Perform a scan to estimate the Manhattan distance to the hidden treasure for all valid moves.        
    :return: JSON string with {"current_distance": int, "distances": {"up": int, "down": int, "left": int, "right": int}} indicating distances from each valid move to treasure.
    """
    current_x, current_y = env.agent_pos
    tx, ty = env.treasure_pos
    current_distance = abs(current_x - tx) + abs(current_y - ty)
    distances = {}
    
    # Check each direction
    for direction in ["up", "down", "left", "right"]:
        x, y = current_x, current_y
        if direction == "up":
            y -= 1
        elif direction == "down":
            y += 1
        elif direction == "left":
            x -= 1
        elif direction == "right":
            x += 1
            
        # Only include valid moves (within grid bounds)
        if 0 <= x < env.width and 0 <= y < env.height:
            tx, ty = env.treasure_pos
            distance = abs(x - tx) + abs(y - ty)
            distances[direction] = distance
    
    output = {
        "current_distance": current_distance,
        "distances": distances
    }
    s_action = f'[{datetime.now()}/Action]:scan_tool -> {output}'
    
    env.last_action = s_action
    print(f"[DEBUG] scan_tool -> {output}")
    env.Memory.append(s_action)
    return json.dumps(output)

def think_tool(args: dict)->str:
    """
    Log an internal thought.    
    :param args: {"thought": str}  
    :return: {"logged": true}
    """
    try:
        thought = args.get("thought", "")        
        env.Memory.append(f'[{datetime.now()}/Thought]:{thought}')
        return '{"logged": true}'
    except Exception as e:
        #cleaned = re.sub(r'[^\x20-\x7E]+', '', json_args)  # keep printable ASCII
        #cleaned = cleaned.replace("'", '"')  # convert single quotes to double quotes
        #cleaned = re.sub(r'\\+', '', cleaned)  # remove backslashes        
        #env.Memory.append(f'[{datetime.now()}/Thought]:{cleaned}')
        # Convert dict to JSON string:
        json_str = json.dumps(args)
        print(f"[Fixed] {json_str}" )
        env.Memory.append(f'[{datetime.now()}/Thought]:{json_str}')
        return '{"logged": true}'        
    
def move_tool(args: dict) -> str:
    """
     Move the agent in a specified direction on the grid.
     :param args:  {"direction": str}, where direction is one of "up", "down", "left", "right".
    :return: JSON string with {"moved": bool, "new_position": [x, y]} indicating success and updated position.
    """
    try:
        #args = json.loads(json_args)
        direction = args.get("direction", "")
        success = env.move(direction)    
        output = {"moved": success, "new_position": env.agent_pos}
        s_action =  f"[DEBUG] move_tool({direction}) -> {output}"
        print(s_action)
        env.last_action = s_action
        env.Memory.append(f'[{datetime.now()}/Action]:move_tool({direction}) -> {output}')
        return json.dumps(output)
    except Exception as e:
        print(f"[Error] {e}/{args}")
        directions = ["up", "down", "left", "right"]
        found = None
        for direction in directions:
            # Use a regex to match whole words, e.g. 'up' but not 'cup'
            pattern = rf"\b{re.escape(direction)}\b"
            if re.search(pattern, args, flags=re.IGNORECASE):                
                success = env.move(direction)
                output = {"moved": success, "new_position": env.agent_pos}
                print(f"[Fixed] move_tool {args}")
                env.Memory.append(f'[{datetime.now()}/Action]:move_tool({direction}) -> {output}')
                return json.dumps(output)
        output = {"moved": "Invalid json_args input"}

def dig_tool() -> str:
    """
    Attempt to dig at the current agent location to uncover the treasure.        
    :return: JSON string with {"found": bool, "dig_history": [x, y]} indicating if the treasure was discovered and the dig history.
    """
    found = env.dig()
    # The error "Object of type set is not JSON serializable" " - fixed need to convert the set to list
    output = {"found": found, "dig_history": list(env.dig_history)}
    s_action = f"[DEBUG] dig_tool -> {output}"
    print(s_action)
    env.last_action = s_action
    env.Memory.append(f'[{datetime.now()}/Action]:dig_tool-> {output}')
    return json.dumps(output)

def grid_map_tool() -> str:
    """
    Return a string representation of the current grid map.

    Symbols:
      x = dig location
      $ = treasure location
      @ = agent position
      0 = empty cell    
    :return: String showing the grid with agent, treasure, and dig locations.
    """
    output = grid_to_string(env.width, env.height, env.agent_pos, env.dig_history, env.treasure_pos, env.found)
    s_action = f"[DEBUG] grid_map_tool ->\n {output}"
    env.last_action = s_action
    print(s_action)
    env.Memory.append(f'[{datetime.now()}/Action]:grid_map_tool')
    return output


GRANITE_PROMPT="""
You are a helpful assistant with access to the following function calls. Your task is to produce a list of function calls necessary to generate a response to the user utterance.
Use the following function calls as required.

Available tools (call them in this order):
[
  {
    'type': 'function',
    'function': {
      'name': 'scan_tool',
      'description': 'Scan the environment to get the current Manhattan distance and the up/down/left/right distances to the treasure from the agent position.',
      'parameters': {'type': 'object', 'properties': {}, 'required': []},
      'return': {'type': 'object', 'description': 'A dictionary with {"current_distance": int, "distances": {"up": int, "down": int, "left": int, "right": int}}'}
    }
  },
  {
    'type': 'function',
    'function': {
      'name': 'think_tool',
      'description': 'Log an internal thought for the agent.',
      'parameters': {'type': 'object', 'properties': {'thought': {'type': 'string', 'description': 'The agent\'s internal thought.'}}, 'required': ['thought']},
      'return': {'type': 'object', 'description': '{"logged": true}'}
    }
  },
  {
    'type': 'function',
    'function': {
      'name': 'move_tool',
      'description': 'Move the agent in a specified direction on the grid.',
      'parameters': {'type': 'object', 'properties': {'direction': {'type': 'string', 'description': 'One of "up", "down", "left", "right".'}}, 'required': ['direction']},
      'return': {'type': 'object', 'description': '{"moved": bool, "new_position": [x, y]}'}
    }
  },
  {
    'type': 'function',
    'function': {
      'name': 'dig_tool',
      'description': 'Attempt to dig at the current agent location to uncover the treasure.',
      'parameters': {'type': 'object', 'properties': {}, 'required': []},
      'return': {'type': 'object', 'description': '{"found": bool, "dig_history": [[x, y], ...]}'}
    }
  },
  {
    'type': 'function',
    'function': {
      'name': 'grid_map_tool',
      'description': 'Return a string representation of the current grid map with agent, treasure, and dig locations.',
      'parameters': {'type': 'object', 'properties': {}, 'required': []},
      'return': {'type': 'string', 'description': 'A string showing the grid with agent, treasure, and dig locations.'}
    }
  }
]

Your goal is to find the treasure on grid (x,y) before running out of turns
If current_distance == 0, you are on the treasure cell—immediately dig

For each of the tools, you must emit exactly one JSON function call
- Only respond with a single JSON function call. The function arguments must be a valid JSON object:
    - All strings must use double quotes (e.g., \"right\", not 'right').
    - Do not embed JSON as a string inside another JSON.
    - No other text. No comments. No explanation.
"""

# --- Main Agent Logic ---
if __name__ == "__main__":
    env = TreasureHunt(width=5, height=5)
    client = LlamaStackClient(base_url="http://localhost:8321")

    tools = [scan_tool, move_tool, dig_tool,think_tool,grid_map_tool]
    agent = Agent(
        client=client,
        model="granite3.3:8b", #"meta-llama/Llama-3.2-3B-Instruct",  # or use "phi4" for faster response
        #instructions=SYSTEM_PROMPT,
        instructions=GRANITE_PROMPT,
        tools=tools,
        tool_parser=GraniteToolParser(),  # <--- Use the custom parser here
        sampling_params={"strategy": {"type": "greedy"}, "max_tokens": 200},
        max_infer_iters=100
        
    )

    session_id = agent.create_session("treasure_hunt")
    print(f'[Debug] Agent SessionId {session_id}')
    max_turns = 15
    env.agent_pos = (0,0)
    print_grid(env.width, env.height, env.agent_pos, env.dig_history, env.treasure_pos, env.found)
    #Seeded Memory
    #env.Memory.append(f'[{datetime.now()}/Thought]:I need to estimate distance to treasure using scan_tool')

    for step in range(max_turns):
        print(f"\n===== TURN {step+1} =====")
        print(f"[DEBUG] Agent is at {env.agent_pos}")

        # build the user content with memory
        user_payload = {
            "position": env.agent_pos,
            "grid_size": {"width": env.width, "height": env.height},            
            "turns_left": (max_turns - step )
        }
        
        print(f"[DEBUG] User payload: {user_payload}")
        response = agent.create_turn(
            session_id=session_id,
            messages=[{"role": "user", "content": json.dumps(user_payload)}],
            stream=False
        )

        print("Agent says:\n", response.output_message.content.strip())
        print(grid_to_string(env.width, env.height, env.agent_pos, env.dig_history, env.treasure_pos, env.found))
        print("------------Trace Memory-----------------")
        for trace in env.Memory:
            print(f"{trace}")
        print("-----------------------------")
        session_response = client.agents.session.retrieve(agent_id=agent.agent_id, session_id=session_id)
        # Match 'error' or 'Error'
        if re.search(r'\berror\b|\bError\b', str(session_response)):            
            print(session_response)
               
        #format_session_data(session_response)
        #print(session_response)
        #print_grid(env.width, env.height, env.agent_pos, env.dig_history, env.treasure_pos, env.found)
        if env.found:
            print("🎉 Treasure found at", env.agent_pos)
            break
        

    if not env.found:
        print("❌ Ran out of turns! Treasure was at:", env.treasure_pos)



