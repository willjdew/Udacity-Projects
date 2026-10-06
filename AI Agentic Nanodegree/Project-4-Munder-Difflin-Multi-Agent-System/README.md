# Munder Difflin Multi-Agent System

A multi-agent system for a fictional paper company. It reads customer requests in plain text and handles inventory checks, quotes and order fulfillment against a SQLite database, then writes a reply to the customer.

## Agents

Built with [smolagents](https://github.com/huggingface/smolagents) and `gpt-4o-mini`. See the [workflow diagram](docs/workflow_diagram.png).

| Agent | Job |
|---|---|
| Orchestrator | Reads each request, calls the workers in order and writes the customer reply |
| Inventory | Checks stock and delivery dates for the requested items |
| Quoting | Prices the order using past quotes and bulk discounts |
| Sales | Records the sale, and restocks when supplies run low |

## Results

On the 20 sample requests, the system recorded 7 orders and declined 13, giving a reason each time. See `test_results.csv`.

## Folder layout

```
project_starter.py          # the full system (the agents start below "YOUR MULTI AGENT STARTS HERE")
test_*.py                   # tests for each agent and for the whole system
quotes.csv, quote_requests.csv, quote_requests_sample.csv   # input data
test_results.csv            # output of the 20-request run
requirements.txt
docs/
├── workflow_diagram.png    # diagram of the agents and tools (.mmd is the editable source)
└── project_instructions.md, project_brief.txt   # Udacity's assignment
```

The code and tests sit together in the top folder because the script loads the CSV files from its own folder and the tests import it from there.

## Running it

1. `pip install -r requirements.txt smolagents`
2. Create a `.env` file in this folder with `UDACITY_OPENAI_API_KEY=your-key`.
3. `python project_starter.py`
