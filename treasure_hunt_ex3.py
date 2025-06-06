import json
import random
from llama_stack_client import LlamaStackClient, Agent
from datetime import datetime



def grid_to_string(width, height, agent_pos, dig_history, treasure_pos):
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
        self.Memory = []         #for debuging 
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
        return current_distance ,distances

    def dig(self):
        self.dig_history.add(self.agent_pos)
        if self.agent_pos == self.treasure_pos:
            print(f"[DEBUG] Dig success! Treasure found at {self.treasure_pos}")
            self.found = True
            return True
        print(f"[DEBUG] Dig at {self.agent_pos} failed. No treasure.")
        return False

# --- Tool Functions ---
def scan_tool(json_args: str) -> str:
    """
    Perform a scan to estimate the Manhattan distance to the hidden treasure for all valid moves.
    :param json_args:JSON string , Ignored input (placeholder for tool interface compatibility).    
    :return: JSON string with {"current_distance": int, "distances": {"up": int, "down": int, "left": int, "right": int}} indicating distances from each valid move to treasure.
    """
    
    current_distance , distances = env.scan()
    output = {
        "current_distance": current_distance,
        "distances": distances
    }
    s_action = f'[{datetime.now()}/Action]:scan_tool -> {output}'    
    print(f"[DEBUG] scan_tool -> {output}")
    env.Memory.append(s_action)
    return json.dumps(output)

def think_tool(json_args: str)->str:
    """
    Log an internal thought for debugging and reflection uses.    
    :param json_args: JSON string with {"thought": str}  
    :return: JSON string {"logged": true}
    """
    args = json.loads(json_args)
    thought = args["thought"]
    env.Memory.append(f'[{datetime.now()}/Thought]:{thought}')
    env.last_thought = f'[{datetime.now()}/Thought]:{thought}'
    return json.dumps({"logged": True})
    
def move_tool(json_args: str) -> str:
    """
    Move the agent in a specified direction on the grid.

    :param json_args: JSON string with {"direction": str}, where direction is one of "up", "down", "left", "right".
    :return: JSON string with {"moved": bool, "new_position": [x, y]} indicating success and updated position.
    """
    args = json.loads(json_args)
    direction = args.get("direction", "")
    success = env.move(direction)    
    output = {"moved": success, "new_position": env.agent_pos}
    s_action =  f"[DEBUG] move_tool({direction}) -> {output}"
    print(s_action)    
    env.Memory.append(f'[{datetime.now()}/Action]:move_tool({direction}) -> {output}')
    return json.dumps(output)

def dig_tool(json_args: str) -> str:
    """
    Attempt to dig at the current agent location to uncover the treasure.        
    :param json_args:JSON string , Ignored input (placeholder for tool interface compatibility).
    :return: JSON string with {"found": bool, "dig_history": [x, y]} indicating if the treasure was discovered and the dig history.
    """
    found = env.dig()
    output = {"found": found, "dig_history": env.dig_history}
    s_action = f"[DEBUG] dig_tool -> {output}"
    print(s_action)    
    env.Memory.append(f'[{datetime.now()}/Action]:dig_tool-> {output}')
    return json.dumps(output)

def grid_map_tool(json_args: str) -> str:
    """
    Return a string representation of the current grid map.

    Symbols:
      x = dig location
      $ = treasure location
      @ = agent position
      0 = empty cell
    :param json_args:JSON string , Ignored input (placeholder for tool interface compatibility).
    :return: String showing the grid with agent, treasure, and dig locations.
    """
    output = grid_to_string(env.width, env.height, env.agent_pos, env.dig_history, env.treasure_pos)
    s_action = f"[DEBUG] grid_map_tool ->\n {output}"    
    print(s_action)
    env.Memory.append(f'[{datetime.now()}/Action]:grid_map_tool')
    return output

SYSTEM_PROMPT="""
You are a treasure hunting agent navigating a 2D grid (x, y) .
your goal is to find the treasure before running out of turns.
you must write your thoughts by calling think tool
For each of the tools, you must emit exactly one JSON function call with valid arguments—no other text or explanation.
"""


# --- Main Agent Logic ---
if __name__ == "__main__":
    env = TreasureHunt(width=5, height=5)
    client = LlamaStackClient(base_url="http://localhost:8321")

    tools = [scan_tool, move_tool, dig_tool,think_tool,grid_map_tool]
    agent = Agent(
        client=client,
        model="meta-llama/Llama-3.2-3B-Instruct",  # or use "phi4" for faster response
        #instructions=SYSTEM_PROMPT,
        instructions=SYSTEM_PROMPT,
        tools=tools,
        sampling_params={"strategy": {"type": "greedy"}, "max_tokens": 200}
    )
    """
    Greedy strategy: 
    This sets the decoding strategy to greedy decoding.
    In this approach, at each step of text generation, the model selects the token with the highest probability.
    This method is deterministic, meaning it will produce the same output for the same input every time.
    """
    session_id = agent.create_session("treasure_hunt")
    print(f'[Debug] Agent SessionId {session_id}')
    max_turns = 15
    env.agent_pos = (0,0)
    print(grid_to_string(env.width, env.height, env.agent_pos, env.dig_history, env.treasure_pos))

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
        print(grid_to_string(env.width, env.height, env.agent_pos, env.dig_history, env.treasure_pos))
        print("------------Trace Memory-----------------")
        for trace in env.Memory:
            print(f"{trace}")
        print("-----------------------------")
        
        if env.found:
            print("🎉 Treasure found at", env.agent_pos)
            break
        

    if not env.found:
        print("❌ Ran out of turns! Treasure was at:", env.treasure_pos)



