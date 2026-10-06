# AI-Powered Agentic Workflow for Project Management

A library of reusable AI agents, and a project management workflow built from them. Given a product spec, the workflow produces user stories, product features and engineering tasks. The pilot input is the spec for an "Email Router" product.

## How it works

- **Phase 1** builds seven agent classes in `workflow_agents/base_agents.py`: Direct Prompt, Augmented Prompt, Knowledge Augmented Prompt, RAG Knowledge Prompt, Evaluation, Routing and Action Planning. Each one has its own test script.
- **Phase 2** combines them. An Action Planning agent breaks the request into steps, and a Routing agent sends each step to a Product Manager, Program Manager or Development Engineer role. Each role is a Knowledge Augmented agent paired with an Evaluation agent that checks its answer and sends it back for correction until it passes.

## Folder layout

```
phase_1/
├── workflow_agents/base_agents.py   # the agent library
├── *_agent.py                       # one test script per agent
└── outputs/                         # output of the test scripts
phase_2/
├── agentic_workflow.py              # the project management workflow
├── route_check.py                   # quick check of the routing step on its own
├── Product-Spec-Email-Router.txt    # pilot product spec
├── workflow_agents/base_agents.py   # copy of the agent library
└── outputs/                         # final output and earlier test runs
docs/
├── reflection.md                    # strengths, limitations and improvements
├── STUDY_GUIDE.md                   # my study notes
└── project_overview.md, phase_*_instructions.md   # Udacity's assignment
```

## Running it

1. `pip install -r requirements.txt`
2. Create a `.env` file in this folder containing `OPENAI_API_KEY=your-key`.
3. Run an agent test: `cd phase_1` then `python direct_prompt_agent.py`
4. Run the workflow: `cd phase_2` then `python -X utf8 agentic_workflow.py`
