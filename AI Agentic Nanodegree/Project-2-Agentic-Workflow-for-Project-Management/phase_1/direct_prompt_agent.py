# Test script for DirectPromptAgent class
#
# Every test script in phase_1 follows the same 4-part pattern:
#   1. Imports       - bring in the agent class and helper libraries
#   2. Load the key  - read the OpenAI API key from the .env file
#   3. Build & use   - create the agent and send it a prompt
#   4. Show results  - print the response (and any explanation the rubric asks for)

# ---------------------------------------------------------------------------
# Part 1: Imports
# ---------------------------------------------------------------------------
# Go into the workflow_agents folder, open base_agents.py, and bring in the
# DirectPromptAgent class. (The dots are folder/file boundaries.)
from workflow_agents.base_agents import DirectPromptAgent
import os                       # lets Python read environment settings
from dotenv import load_dotenv  # reads the .env file

# ---------------------------------------------------------------------------
# Part 2: Load the API key
# ---------------------------------------------------------------------------
# load_dotenv() copies the values in .env into the environment; os.getenv()
# then pulls out OPENAI_API_KEY. Keeping the key in .env means it is never
# hard-coded here, so it can't be accidentally submitted or shared.
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# If .env is missing (or the name is misspelled), getenv() returns None
# instead of crashing. Without this check, the error would only surface later
# inside the OpenAI call with a confusing message. Fail early and clearly.
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found - check that your .env file exists and contains it.")

# ---------------------------------------------------------------------------
# Part 3: Build the agent and use it
# ---------------------------------------------------------------------------
prompt = "What is the Capital of France?"

direct_agent = DirectPromptAgent(openai_api_key)      # runs __init__ -> stores the key
direct_agent_response = direct_agent.respond(prompt)  # runs respond() -> API call, returns text

# ---------------------------------------------------------------------------
# Part 4: Show the results
# ---------------------------------------------------------------------------
print(direct_agent_response)

# Explain where the agent's knowledge came from (required by the rubric)
print("\nKnowledge source: this answer came only from the general knowledge the "
      "gpt-3.5-turbo model learned during training. The agent added no persona, "
      "no system prompt, and no extra knowledge -- the prompt was passed straight through.")
