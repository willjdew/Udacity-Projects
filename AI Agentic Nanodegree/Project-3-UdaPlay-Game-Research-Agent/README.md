# UdaPlay: AI Game Research Agent

An AI research agent that answers questions about video games. It searches a local ChromaDB vector database of game data first, judges whether the results are good enough, and falls back to a Tavily web search when they are not, saving new facts back to the database.

- `starter/Udaplay_01_solution_project.ipynb`: Part 1, building the vector database (offline RAG)
- `starter/Udaplay_02_solution_project.ipynb`: Part 2, the agent and its tools
- `starter/lib/`: agent, memory, RAG, tooling and state machine modules
- `starter/games/`: game data in JSON

See [starter/README.md](starter/README.md) for setup.
