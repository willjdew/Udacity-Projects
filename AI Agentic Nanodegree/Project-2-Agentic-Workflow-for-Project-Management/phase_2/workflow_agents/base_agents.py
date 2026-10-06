from openai import OpenAI

# Udacity/Vocareum keys (starting with "voc-") only work through Vocareum's gateway.
# If you switch to a regular OpenAI key, set this to None to go straight to OpenAI.
OPENAI_BASE_URL = "https://openai.vocareum.com/v1"
import numpy as np
import pandas as pd
import re
import csv
import uuid
from datetime import datetime

# DirectPromptAgent class definition
class DirectPromptAgent:

    def __init__(self, openai_api_key):
        # Initialize the agent
        # Store the API key so respond() can build a client with it.
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        # Generate a response using the OpenAI API
        client = OpenAI(base_url=OPENAI_BASE_URL, api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                # Only a user message -- no system prompt, so the model
                # answers purely from its own general (training) knowledge.
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        # The API returns a full response object; we only want the text
        # of the first (and only) choice.
        return response.choices[0].message.content
        
# AugmentedPromptAgent class definition
class AugmentedPromptAgent:
    def __init__(self, openai_api_key, persona):
        """Initialize the agent with given attributes."""
        self.persona = persona
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Generate a response using OpenAI API."""
        client = OpenAI(base_url=OPENAI_BASE_URL, api_key=self.openai_api_key)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                # The system message sets WHO the model is before it sees the question.
                {"role": "system",
                 "content": f"{self.persona}. Forget all previous context."},
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )

        return response.choices[0].message.content

# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent:
    def __init__(self, openai_api_key, persona, knowledge):
        """Initialize the agent with provided attributes."""
        self.persona = persona
        self.knowledge = knowledge
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Generate a response using the OpenAI API."""
        client = OpenAI(base_url=OPENAI_BASE_URL, api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                # Persona + knowledge + "use only this knowledge" all live in the system message.
                {"role": "system",
                 "content": (
                     f"You are {self.persona} knowledge-based assistant. Forget all previous context. "
                     f"Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge} "
                     f"Answer the prompt based on this knowledge, not your own."
                 )},
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )
        return response.choices[0].message.content

# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = OpenAI(base_url=OPENAI_BASE_URL, api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], 0, 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end
            })

            # FIX: stop once we've reached the end of the text. Without this, the
            # last chunk keeps resetting start to (len - overlap) forever -> infinite
            # loop that eats all memory and crashes the workspace.
            if end >= len(text):
                break

            start = end - self.chunk_overlap
            chunk_id += 1

        with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['text'].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding='utf-8', index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']

        client = OpenAI(base_url=OPENAI_BASE_URL, api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context."},
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"}
            ],
            temperature=0
        )

        return response.choices[0].message.content

class EvaluationAgent:

    def __init__(self, openai_api_key, persona, evaluation_criteria, worker_agent, max_interactions):
        # Initialize the EvaluationAgent with given attributes.
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent          # the agent whose answers we judge
        self.max_interactions = max_interactions  # safety cap so the loop can't run forever

    def evaluate(self, initial_prompt):
        # This method manages interactions between agents to achieve a solution.
        client = OpenAI(base_url=OPENAI_BASE_URL, api_key=self.openai_api_key)
        prompt_to_evaluate = initial_prompt

        for i in range(self.max_interactions):
            print(f"\n--- Interaction {i+1} ---")

            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            response_from_worker = self.worker_agent.respond(prompt_to_evaluate)
            print(f"Worker Agent Response:\n{response_from_worker}")

            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}\n"
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"{self.persona}. Forget all previous context."},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0
            )
            evaluation = response.choices[0].message.content.strip()
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            print(" Step 3: Check if evaluation is positive")
            if evaluation.lower().startswith("yes"):
                print("✅ Final solution accepted.")
                break
            else:
                print(" Step 4: Generate instructions to correct the response")
                instruction_prompt = (
                    f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                )
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": f"{self.persona}. Forget all previous context."},
                        {"role": "user", "content": instruction_prompt}
                    ],
                    temperature=0
                )
                instructions = response.choices[0].message.content.strip()
                print(f"Instructions to fix:\n{instructions}")

                print(" Step 5: Send feedback to worker agent for refinement")
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}"
                )
        # WARNING: if the loop hits max_interactions without a "Yes", the response
        # below is the LAST FAILED attempt, not an approved answer. "iterations"
        # alone can't tell you this (passing on the final try looks the same), so
        # callers should check the "accepted" flag before trusting final_response.
        accepted = evaluation.lower().startswith("yes")
        if not accepted:
            print(f"⚠️ WARNING: response NOT accepted after {self.max_interactions} interactions.")
        return {
            "final_response": response_from_worker,
            "evaluation": evaluation,
            "iterations": i + 1,
            "accepted": accepted
        }

class RoutingAgent():

    def __init__(self, openai_api_key, agents):
        # Initialize the agent with given attributes
        self.openai_api_key = openai_api_key
        # Each agent is a dict: {"name": ..., "description": ..., "func": callable}
        self.agents = agents

    def get_embedding(self, text):
        client = OpenAI(base_url=OPENAI_BASE_URL, api_key=self.openai_api_key)
        # Turn the text into a vector of numbers that captures its meaning
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        # Extract and return the embedding vector from the response
        embedding = response.data[0].embedding
        return embedding

    def route(self, user_input):
        """Send user_input to the agent whose description is most similar in meaning."""
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            agent_emb = self.get_embedding(agent["description"])
            if agent_emb is None:
                continue

            # Cosine similarity: 1.0 = same direction (same meaning), ~0 = unrelated
            similarity = np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            print(similarity)

            # Keep the highest-scoring agent seen so far
            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)

class ActionPlanningAgent:

    def __init__(self, openai_api_key, knowledge):
        # Initialize the agent attributes
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt):

        client = OpenAI(base_url=OPENAI_BASE_URL, api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": (
                     "You are an action planning agent. Using your knowledge, you extract from the user prompt "
                     "the steps requested to complete the action the user is asking for. "
                     "Only use steps that are described in your knowledge; do not add steps from general knowledge. "
                     "Return ONLY the steps as a numbered list, one step per line, in the order they must be done. "
                     "Do not include any introduction, heading, or conclusion. Forget any previous context. "
                     f"This is your knowledge: {self.knowledge}"
                 )},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        response_text = response.choices[0].message.content

        # Clean up: one step per line. Drop blank lines and headings (lines ending in ":"),
        # and strip list markers like "1." / "1)" / "-" / "*" so each step is plain text.
        steps = []
        for line in response_text.split("\n"):
            line = line.strip()
            if not line or line.endswith(":"):
                continue
            line = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line)
            if line:
                steps.append(line)

        return steps