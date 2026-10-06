from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"

persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"

knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)
knowledge_agent_response = knowledge_agent.respond(prompt)

print(knowledge_agent_response)
print("\nNote: the answer says London, which comes from the knowledge we supplied "
      "to the agent, not from the model's own training (which would say Paris).")
