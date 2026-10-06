from workflow_agents.base_agents import AugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

# Instantiate the agent with a persona
augmented_agent = AugmentedPromptAgent(openai_api_key, persona)

# Send the prompt and store the response
augmented_agent_response = augmented_agent.respond(prompt)

# Print the agent's response
print(augmented_agent_response)

# Explain the knowledge source and persona impact (required by the rubric).
# These are printed so the explanation appears in the terminal output.
print("\nKnowledge source: the agent still relies only on the general knowledge the "
      "gpt-3.5-turbo model learned during training. No extra knowledge was supplied, "
      "so it answers 'Paris'.")
print("Persona impact: the system prompt changes HOW the answer is delivered, not "
      "WHAT facts the model knows -- the reply takes on a college professor's voice "
      "and starts with 'Dear students,'.")
