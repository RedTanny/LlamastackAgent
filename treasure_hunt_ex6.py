import json
import random
from llama_stack_client import LlamaStackClient, Agent
from datetime import datetime
import pprint

import re
from datetime import datetime


def format_session_data(session_response):
    """
    Process and display session data from the API response in a readable format.
    Takes the direct response from client.agents.session.retrieve() and formats it.
    
    Args:
        session_response: The response object from client.agents.session.retrieve()
    
    Returns:
        None (prints formatted output)
    """
    # Convert to string if it's not already one
    if not isinstance(session_response, str):
        session_str = str(session_response)
    else:
        session_str = session_response
    
    # Extract basic session info
    session_id_match = re.search(r"session_id='([^']+)'", session_str)
    session_name_match = re.search(r"session_name='([^']+)'", session_str)
    started_at_match = re.search(r"started_at=datetime\.datetime\(([^)]+)\)", session_str)
    
    session_info = {
        "session_id": session_id_match.group(1) if session_id_match else "Unknown",
        "session_name": session_name_match.group(1) if session_name_match else "Unknown",
        "started_at": started_at_match.group(1).replace(', tzinfo=datetime.timezone.utc', '') if started_at_match else "Unknown"
    }
    
    # Extract and parse tool calls
    tool_calls = re.findall(r"ToolCall\(arguments=([^)]+), call_id='([^']+)', tool_name='([^']+)'", session_str)
    
    # Extract and parse tool responses
    tool_responses = re.findall(r"ToolResponse\(call_id='([^']+)', content='([^']+)', tool_name='([^']+)'", session_str)
    
    # Create a more structured view of the session
    parsed_session = {
        "session_info": session_info,
        "tool_calls": [{"args": tc[0], "call_id": tc[1], "tool_name": tc[2]} for tc in tool_calls],
        "tool_responses": [{"call_id": tr[0], "content": tr[1], "tool_name": tr[2]} for tr in tool_responses]
    }
    
    # Print the formatted session
    _print_formatted_session(parsed_session)
    
    return parsed_session

def _print_formatted_session(parsed_session):
    """
    Helper function to print the parsed session in a readable format
    """
    session_info = parsed_session["session_info"]
    print("=" * 50)
    print(f"SESSION INFORMATION")
    print("=" * 50)
    print(f"Session ID: {session_info['session_id']}")
    print(f"Session Name: {session_info['session_name']}")
    print(f"Started At: {session_info['started_at']}")
    print("\n")
    
    print("=" * 50)
    print(f"TOOL CALLS & RESPONSES SEQUENCE")
    print("=" * 50)
    
    # Create a mapping of call_id to response for easier lookup
    response_map = {r["call_id"]: r for r in parsed_session["tool_responses"]}
    
    # Print each tool call with its corresponding response, if available
    for i, call in enumerate(parsed_session["tool_calls"]):
        print(f"\n--- Step {i+1} ---")
        print(f"Tool: {call['tool_name']}")
        print(f"Call ID: {call['call_id']}")
        
        # Clean up and format the arguments
        args_str = call["args"]
        try:
            # Try to parse as JSON-like structure
            if "json_args" in args_str:
                # Extract just the json_args part
                json_part = re.search(r"'json_args': '?([^']+)'?", args_str)
                if json_part:
                    args_display = json_part.group(1)
                    # Clean up escaping that might make it hard to read
                    args_display = args_display.replace('\\"', '"').replace('\\\'', "'")
                    print(f"Arguments: {args_display}")
                else:
                    print(f"Arguments: {args_str}")
            else:
                print(f"Arguments: {args_str}")
        except Exception:
            print(f"Arguments: {args_str}")
        
        # Show the response if available
        if call["call_id"] in response_map:
            response = response_map[call["call_id"]]
            print("Response:")
            
            # Clean up response content and handle errors
            content = response["content"]
            if content.startswith('"') and content.endswith('"'):
                # Remove outer quotes
                content = content[1:-1]
                # Unescape interior quotes
                content = content.replace('\\"', '"')
            
            if "Error when running tool" in content:
                print(f"  ERROR: {content}")
            else:
                # Handle special formatting for grid maps
                if "|" in content and "\\n" in content:
                    print("  GRID MAP:")
                    grid_lines = content.split("\\n")
                    for line in grid_lines:
                        if line.strip():
                            print(f"  {line}")
                else:
                    print(f"  {content}")
                    

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

SYSTEM_PROMPT="""
You are a treasure-hunting agent navigating a 2D grid (x, y). Your goal is to find the treasure before running out of turns.

Navigation:
- move right → (x + 1, y)
- move left → (x - 1, y)
- move up → (x, y + 1)
- move down → (x, y - 1)

Each turn you must:
1) Scan the environment (no args). This returns the current distance and the right/left/up/down distances to the treasure.

2) Think step by step in detail about your next move:
   - Note values are integers and your current distance to the treasure (CURRENT_DISTANCE)
   - Compare all possible move distances:
     * If I move up, distance would be DISTANCES.up
     * If I move down, distance would be DISTANCES.down
     * If I move left, distance would be DISTANCES.left
     * If I move right, distance would be DISTANCES.right
   - Identify which direction has the SMALLEST distance value
   - This direction will bring you closest to the treasure

3) Take action:
   - If current_distance == 0, you are on the treasure cell—immediately dig
   - Otherwise, move in the direction with the smallest distance value

Always follow this process:
- First scan to get distances
- Compare all distance values and find the minimum
- Move toward the minimum distance (or dig if already at the treasure)

Remember: The direction with the SMALLEST distance value is ALWAYS the best move.

Do not infer or state absolute treasure coordinates. You only know the Manhattan distance from your current cell and its neighbors
For each of the tools, you must emit exactly one JSON function call
- Only respond with a single JSON function call. The function arguments must be a valid JSON object:
    - All strings must use double quotes (e.g., "right", not 'right').
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
        model="meta-llama/Llama-3.2-3B-Instruct",  # or use "phi4" for faster response
        #instructions=SYSTEM_PROMPT,
        instructions=SYSTEM_PROMPT,
        tools=tools,
        sampling_params={"strategy": {"type": "greedy"}, "max_tokens": 200}
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
            format_session_data(session_response)
            print(session_response)
               
        #format_session_data(session_response)
        #print(session_response)
        #print_grid(env.width, env.height, env.agent_pos, env.dig_history, env.treasure_pos, env.found)
        if env.found:
            print("🎉 Treasure found at", env.agent_pos)
            break
        

    if not env.found:
        print("❌ Ran out of turns! Treasure was at:", env.treasure_pos)



