# UdaPlay: AI Game Research Agent

An AI research agent that answers questions about video games. It searches a local ChromaDB vector database of game data first, judges whether the results are good enough, and falls back to a Tavily web search when they are not, saving new facts back to the database for next time.

## Folder layout

```
Udaplay_01_solution_project.ipynb   # Part 1: build the vector database (offline RAG)
Udaplay_02_solution_project.ipynb   # Part 2: the agent and its tools
lib/                                # agent, LLM, memory, RAG, tooling and state machine modules
games/                              # game data, one JSON file per game
docs/project_instructions.md        # Udacity's assignment
```

## Agent tools

| Tool | Purpose |
|---|---|
| `retrieve_game` | Semantic search over the local ChromaDB collection |
| `evaluate_retrieval` | Uses the LLM to judge whether the retrieved documents answer the question |
| `game_web_search` | Tavily web search, used only when local knowledge is not enough |
| `store_game_fact` | Saves a fact found on the web back into ChromaDB |

## Running it

1. Install Python 3.11+ and the packages: `chromadb`, `openai`, `tavily-python`, `python-dotenv`.
2. Create a `.env` file in this folder with `OPENAI_API_KEY`, `CHROMA_OPENAI_API_KEY` and `TAVILY_API_KEY`.
3. Run the Part 1 notebook to build the database, then the Part 2 notebook.
