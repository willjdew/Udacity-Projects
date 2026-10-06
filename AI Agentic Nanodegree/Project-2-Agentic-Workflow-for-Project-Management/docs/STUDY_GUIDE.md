# Study Guide: AI-Powered Agentic Workflows

*Udacity Project 2, InnovateNext "Email Router" pilot. Covers Phase 1 (the agent library) and Phase 2 (the complete workflow, its four test runs, and the design lessons from debugging them).*

---

## 1. The Big Picture

An **agentic workflow** is a system where several small, single-purpose AI agents work together to complete a larger task. Instead of one giant prompt that tries to do everything, each agent has **one job**, and the workflow coordinates them.

Think of it like a well-run team at work:

| Team role | Agent equivalent |
|---|---|
| Project lead who breaks down the work | **Action Planning Agent** |
| Dispatcher who assigns work to the right group | **Routing Agent** |
| Specialists who do the work | **Knowledge Augmented Prompt Agents** |
| Reviewers / QA who check the work | **Evaluation Agents** |

**Key idea:** each agent is a reusable building block. Phase 1 builds the blocks. Phase 2 assembles them into a workflow.

---

## 2. LLM API Fundamentals

Every agent in this project comes down to one API call:

```python
client = OpenAI(base_url=OPENAI_BASE_URL, api_key=self.openai_api_key)
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "...rules / persona / knowledge..."},
        {"role": "user",   "content": "...the actual question..."}
    ],
    temperature=0
)
return response.choices[0].message.content
```

### Message roles
- **`system`**: the rules of the game: who the model is and what it must follow. Models give it high priority, so user text can't easily override it.
- **`user`**: the request or question.
- Keeping them separate lets you **reuse** the same persona or rules with any question.

### `temperature`
- Controls randomness. **0 = consistent, repeatable output**, which is what you want when other agents rely on the result.
- Watch out: consistent does **not** mean correct. At temperature 0 a model will give the *same wrong answer* every time if the prompt steers it there (see the London example).

### `response.choices[0].message.content`
- The API returns a large object. We only want the text.
- `choices` is a list because you *can* request several completions of the same prompt with the `n` parameter (e.g. `n=3`, then pick the best). We don't set `n`, so the default is 1 and `[0]` is the only item.
- It's an API feature, not the model "reasoning" several options.

---

## 3. Setup and Configuration

### API keys and `.env`
```python
from dotenv import load_dotenv
import os
load_dotenv()                                  # reads .env into the environment
openai_api_key = os.getenv("OPENAI_API_KEY")   # pulls out the key
```
- Keeping the key in `.env` means it's **never hard-coded**, so it can't be accidentally submitted or shared.
- `.env` holds one line: `OPENAI_API_KEY=your-key` (no quotes, no spaces around `=`).
- **Where it goes:** the top-level `PROJECT_2` folder. `load_dotenv()` starts in the running script's folder and **walks up** parent folders until it finds a `.env`, so one file covers both phases.
- **Never submit `.env`.** On Windows, make sure it isn't secretly saved as `.env.txt`.
- If the key is missing, `os.getenv` returns **`None`** (no error). The failure only shows up later, when the API call runs.

### The Vocareum gateway (`base_url`)
- Udacity keys start with **`voc-`** and only work through Vocareum's gateway.
- Sending one straight to OpenAI gives **401: Incorrect API key**. Think of calling an internal extension without going through the switchboard.
- Fix: one constant at the top of `base_agents.py`, used by every client:
```python
OPENAI_BASE_URL = "https://openai.vocareum.com/v1"
client = OpenAI(base_url=OPENAI_BASE_URL, api_key=self.openai_api_key)
```
- **Why a constant?** If the address changes, or you switch to a regular OpenAI key (set it to `None`), you edit **one line instead of eight**.

### Python packages and imports
```python
from workflow_agents.base_agents import DirectPromptAgent
```
- Read it as: folder `workflow_agents` → file `base_agents.py` → class `DirectPromptAgent`.
- The starter code said `WorkflowAgents`, which doesn't exist, so the import would have failed. Names must match the folders exactly.
- `__init__.py` (even when empty) marks a folder as an importable Python package.
- Code wrapped in `'''` triple quotes is treated as a **string/comment**, not code. The starter classes were "switched off" this way until uncommented.

---

## 4. The Seven Agents: Building Up in Layers

Each agent adds **one ingredient** to the one before it, like a controlled experiment where you change one thing at a time.

| # | Agent | Persona | Knowledge | Output | Test result |
|---|---|---|---|---|---|
| 1 | Direct Prompt | – | – | text | "The capital of France is Paris." |
| 2 | Augmented Prompt | ✔ | – | text | "Dear students, the capital of France is Paris." |
| 3 | Knowledge Augmented | ✔ | ✔ (supplied) | text | "Dear students, … London, not Paris." |
| 4 | RAG Knowledge | ✔ | ✔ (retrieved) | text | Clara's podcast "Crosscurrents" |
| 5 | Evaluation | manages another agent | criteria | dict | "London" after 2 iterations |
| 6 | Routing | picks an agent | agent descriptions | another agent's reply | Texas / Europe / Math routed correctly |
| 7 | Action Planning | ✔ | ✔ | **list** of steps | 8 scrambled-egg steps |

### 4.1 Direct Prompt Agent
- Passes the prompt straight to the model: **user message only, no system prompt**.
- Knowledge source: **only the model's training data**.
- It's the baseline the other agents are compared against.

### 4.2 Augmented Prompt Agent
- Adds a **system message with a persona**, plus "Forget all previous context."
- **The persona changes HOW the answer is delivered, not WHAT facts it knows.** It still says Paris, just in a professor's voice.

### 4.3 Knowledge Augmented Prompt Agent
- System message = persona + supplied knowledge + "use only this knowledge, not your own."
- The test deliberately uses **false knowledge** ("The capital of France is London"). A wrong answer is the only way to **prove** the agent used the supplied knowledge. If it said Paris, you couldn't tell where the answer came from.
- **Supplied knowledge beats training**, usually. It isn't guaranteed, which is one reason evaluators exist.
- **Prompt-template gotcha:** the template is `"You are {persona} knowledge-based assistant..."`, but the persona string already begins "You are a college professor...". The model receives a sentence inside a sentence, and it showed up in real output: *"Dear students, knowledge-based assistant."* It's harmless in a small agent but can matter in complex ones. **Write persona strings that fit the template.**
- This is the **worker agent** that Phase 2 runs on.

### 4.4 RAG Knowledge Prompt Agent (provided by Udacity)
**RAG = Retrieval-Augmented Generation.** Instead of pasting all the knowledge into the prompt, you:
1. **Chunk** a large text into pieces (here 500 characters, with 200 characters of **overlap** so ideas aren't cut in half at the edges).
2. **Embed** each chunk (turn it into a vector of numbers).
3. At question time, embed the question and **retrieve** the most similar chunk.
4. Answer using only that chunk.

Use it when the knowledge is too big to paste into every prompt.

### 4.5 Evaluation Agent
The first agent that **manages another agent** instead of answering by itself.

```
prompt ──► Worker answers
              │
              ▼
        Evaluator (temp 0): "Does this meet the criteria? Yes/No + why"
              │
     ┌── Yes ─┴─ No ──┐
     ▼                ▼
   done        LLM writes "how to fix it" instructions
                      │
        new prompt = original + bad answer + instructions
                      └──► back to the worker (≤ max_interactions)
```

**Design principles:**
- **Separation of duties.** The evaluator never rewrites the answer. It only sends feedback, and the worker revises, like a code reviewer and an author. This keeps the worker's knowledge and persona in charge, so the evaluator can't slip in its own facts (e.g. change London back to Paris).
- **`max_interactions`** is a safety cap so the loop can't run forever (or run up cost).
- **Returns a dictionary:** `final_response`, `evaluation`, `iterations`, plus our addition, **`accepted`**.

**Why we added `accepted`:**
- If all 10 tries fail, `final_response` holds the **last failed** answer.
- `iterations == 10` can't tell you whether it passed or gave up, because passing on the 10th try looks the same.
- `accepted = evaluation.lower().startswith("yes")` makes it explicit, and a ⚠️ warning prints when it's `False`.
- Real-world parallel: an unapproved change should never ship looking approved.

**Observed:** round 1 gave "Dear students, … London, not Paris." and got "No" (a sentence, not a city name). Round 2 gave "London" and got "Yes." Result: `iterations: 2, accepted: True`.
- Note the **conflicting instructions** in the test: the persona says "always start with Dear students," and the criteria say "only a city name." The evaluation loop resolved the conflict.

### 4.6 Routing Agent
A dispatcher: it **chooses which agent answers** and doesn't answer anything itself.

**Embeddings:** a model (`text-embedding-3-large`) turns text into a long vector of numbers. **Texts with similar meanings point in similar directions.**

**Cosine similarity:** measures how closely two vectors point the same way.
```
similarity = dot(a, b) / (|a| × |b|)      # 1.0 = same meaning, ~0 = unrelated
```

**Algorithm:** embed the prompt → embed each agent's description → keep the highest score → call that agent's `func`.

**Routing by meaning, not keywords:** "Rome, Italy" went to the *Europe* agent even though "Europe" never appears in the prompt. The embedding model learned that Italy and Europe belong together.

| Prompt | Texas | Europe | Math | Winner |
|---|---|---|---|---|
| Rome, Texas | **0.386** | 0.165 | 0.003 | Texas |
| Rome, Italy | 0.144 | **0.288** | 0.032 | Europe |
| 2 days × 20 stories | 0.059 | 0.083 | **0.130** | Math (low confidence, thin margin) |

**Lesson:** low scores and thin margins mean the router is guessing. **The descriptions you write decide how well routing works.**

**Lambdas: storing a function instead of a result**
```python
"func": lambda x: texas_agent.respond(x)
```
- Like a phone directory entry: it stores the **number to call**, not the answer. You can't have the answer before you know the question.
- If you wrote `texas_agent.respond(...)` directly, Python would run it **immediately** while building the list, and call *every* agent.
- The router picks first, then makes **one** call: `best_agent["func"](user_input)`.

**Cost consideration (caching):**
- Per prompt: 1 prompt embedding + 3 description embeddings + 1 chat call = **5 API calls, 3 of them repeated work**.
- Descriptions never change, so their embeddings could be **computed once and cached**.
- We left it as is (the rubric describes embedding inside the loop, and the cost is tiny with 3 agents). In a real system it's the first optimization to make.

### 4.7 Action Planning Agent
- A knowledge agent whose output is a **Python list** of steps instead of a paragraph. That's what lets a workflow **loop over steps**.
- It only returns steps found in its knowledge. Asking for scrambled eggs gave the 8 scrambled-egg steps and none of the boiled-egg steps.
- Cleanup line:
```python
steps = [line.strip() for line in response_text.split("\n") if line.strip()]
```
  In plain English: split the text into lines, trim whitespace, drop blank lines.
  **Caveat:** chatbot filler ("Sure! Here are the steps:") would still get through as a "step."
  **This happened in Phase 2:** a heading became step 1 and the numbering was doubled (`1. 1.`). The cleanup now also drops headings and strips list markers, and the system prompt is stricter. See **7.1**.

---

## 5. Debugging Lessons Learned

| Symptom | Cause | Fix / lesson |
|---|---|---|
| `ImportError` (would occur) | Starter said `WorkflowAgents`; folder is `workflow_agents` | Import paths must match folder names exactly |
| Class "not found" | Classes wrapped in `'''` | Remove the triple quotes around each class |
| **401 Incorrect API key** | `voc-` key sent to OpenAI directly | Add Vocareum `base_url` |
| Workspace freezes → "Cannot reconnect" | **Infinite loop** in provided RAG `chunk_text()` ate all memory | Add a `break` once `end >= len(text)` |
| Ran the wrong test | Up-arrow picked an old command | Read the command before hitting Enter |
| Old errors in screenshots | Terminal scrollback | Type `clear` before each run |

### The RAG infinite loop, step by step
```python
while start < len(text):
    end = min(start + 500, len(text))
    chunks.append(...)
    start = end - 200        # step back for overlap
```
On the final chunk, `end == len(text)`, so `start = len(text) - 200`, which is **still < len(text)**. The loop repeats the same chunk forever, and the list grows until memory runs out.
**Fix:**
```python
    if end >= len(text):
        break
```
**General lesson:** any loop that *steps back* (overlap, retries) needs an explicit exit condition. Always ask: "What happens on the last iteration?"

### Reading a traceback
- Read from the **bottom up**. The last line is the actual error (`AuthenticationError: 401`).
- The lines above show the path the error took. Look for the first line in **your** files (e.g. `base_agents.py, line 20, in respond`).

---

## 6. Phase 2: Building the Workflow

### 6.1 Final architecture

```
TPM prompt: "What would the development tasks for this product be?"
        │
        ▼
 Action Planning Agent ──► [1. define user stories, 2. group into features, 3. define tasks]
        │
        ▼  for each step: routing_agent.route(step)   (routes on the step text only)
 Routing Agent ──► picks one team, calls its support function
        ├── Product Manager:  Knowledge Agent (+ product spec) ⇄ Evaluator  → user stories
        ├── Program Manager:  Knowledge Agent ⇄ Evaluator                   → features
        └── Dev Engineer:     Knowledge Agent ⇄ Evaluator                   → tasks
        │
        │  support function = evaluate(step + results of earlier steps)
        ▼
 completed_steps ──► printed as ONE consolidated plan: Part 1 stories / Part 2 features / Part 3 tasks
```

Three decisions are made at run time, by three different agents:
- **WHICH** steps are needed → the planner
- **WHO** handles each step → the router
- **WHETHER** an answer is good enough → each evaluator

### 6.2 TODO by TODO: what we wrote and why

| TODO | What | Key point |
|---|---|---|
| 1 | Import the 4 agent classes | Only works when run from `phase_2/`, next to the `workflow_agents` package |
| 2 | Load the API key | Added a **fail-fast** check: `raise ValueError` if the key is missing, instead of a confusing API error later |
| 3 | Load the spec | Path built from `__file__` (works from any folder); **`encoding="utf-8"`**, because Windows defaults to a different encoding and the spec has special characters |
| 4 | Action Planning Agent | Its knowledge is the recipe: stories → features → tasks |
| 5 | Append the spec to the PM's knowledge | Needs **`+ product_spec`**: Python auto-joins only *literal* strings inside parentheses, never variables |
| 6 | PM knowledge agent | Arguments in the same order as `__init__(openai_api_key, persona, knowledge)` |
| 7 | PM evaluation agent | Persona and criteria copied **word for word**; worker passed **by position** (see below); `max_interactions=MAX_EVAL_INTERACTIONS` |
| 8, 9 | Program Mgr / Dev Eng agents + evaluators | Same pattern; criteria written with parentheses instead of `\` line continuations (same string, PEP 8 style, no trailing-space trap) |
| 10 | Routing agent | Route descriptions + lambdas; list assigned to `routing_agent.agents` |
| 11 | Support functions | Call `evaluate(step)`, return `result["final_response"]` |
| 12 | Workflow loop | plan → for each step: print, route, append, print → print a consolidated plan |

**`worker_agent` vs `agent_to_evaluate`:** the README calls the parameter `agent_to_evaluate`, but Udacity's own starter `base_agents.py` names it `worker_agent`. We passed it **by position**, so the name never appears.
- **Lesson:** when docs and code disagree, follow the code. In a library, the class signatures are the contract every workflow depends on, so you don't rename them to match one README.

**`MAX_EVAL_INTERACTIONS = 5`:** one constant shared by all three evaluators.
- **Cost of a failed round:** 3 API calls (the worker's answer, the verdict, the fix instructions).
- **The Product Manager is worse:** every worker call resends the ~12 KB spec.
- **Why 5:** at 10, one stubborn step could cost 30 calls, so 5 works as a cost cap.

**Why the lambdas avoid a `NameError`:** the routes (TODO 10) are defined *before* the support functions (TODO 11).
- **What happens:** `lambda x: product_manager_support_function(x)` looks the name up only **when it is called**, and by then the function exists.
- **What would fail:** `"func": product_manager_support_function` (no lambda) looks it up immediately, so it would crash.
- **The same trick later:** `with_context()` reads `completed_steps`, which is defined further down in the loop.

**Support functions: README vs rubric vs code.**
- **The README said:** `response = knowledge_agent.respond(step)`, then `evaluate(response)`.
- **The problem:** `evaluate(initial_prompt)` expects the **prompt** and calls its worker's `respond()` itself. Following the README wastes a worker call and makes the worker answer its own output.
- **What we did:** we followed the code: `evaluate(step)`.
- **The rubric agreed:** *"passing the step itself… the support function does not call respond() first."* We read the implementation before trusting the instructions, and the grading criteria backed that up.

**Final output.**
- **The README:** says to print the last item of `completed_steps`.
- **The rubric:** requires *"a single consolidated project plan… in clearly labelled sections. Printing only the last item… does not satisfy this requirement."*
- **What we did:** we print all three parts (stories, features, tasks).
- **Lesson:** check the rubric, not just the README.

### 6.3 Answers to the earlier open design questions
1. **Who knows what?** Only the Product Manager gets the spec, because it's the first stage of the pipeline. Later agents are *supposed* to get their input from earlier steps.
   - **The catch:** the README's loop never passes earlier results forward, so the Program Manager and Dev Engineer knew nothing about the Email Router and invented generic features and tasks (runs 1–2).
   - **The fix:** `with_context()` (see 7.3).
2. **Cost:** a failed evaluation round costs 3 calls, and the Product Manager's calls each carry the 12 KB spec. So `max_interactions` became a named cost cap of 5. In the final run every step passed on the first try, so the whole workflow cost only 3 evaluation rounds.

---

## 7. Phase 2 Test Runs: Debugging an Agentic Workflow

The code "worked" from the first run, with no crashes and every step accepted, but the *output* was wrong. Debugging an agentic workflow means **reading the output critically**, not just checking for errors.

| Run | What happened | Root cause | Fix |
|---|---|---|---|
| 1 | 10 generic software-lifecycle steps (including a heading as a "step"); Product Manager **never** chosen; final output: one invented "Deployment" feature | Planner answered from general knowledge; route descriptions too similar | Stricter planner prompt + cleanup (7.1); route descriptions rewritten (7.2) |
| 2 | Clean plan, correct routing, but only 2 steps (tasks + "create a development plan"); tasks about user login and dashboards, nothing about the Email Router | Planner took the prompt literally; no step order in its knowledge; no context passed forward | Step order added to the planner's knowledge; `with_context()` (7.3) |
| 3 | 3 correct steps, correct routing, real Email Router stories and features; tasks covered only **1 of 6** features | Correction loop lost content (7.4) | Required format put into the workers' knowledge (7.4) |
| 4 | Every step passed its evaluator on the **first try**; exact formats; tasks quote their parent story | (submitted) | One story still missing tasks; a known limitation |

**Method that worked:** fix the **most upstream** problem first. Every bad step fed the next one, so fixing routing or context before fixing the planner would have been wasted effort.

### 7.1 The planner ignored its knowledge
- **Symptom:** gpt-3.5 returned a generic 10-step plan despite *"Only return the steps in your knowledge."*
- **Fix 1, a stronger system prompt:** *"do not add steps from general knowledge… Return ONLY the steps as a numbered list, one step per line… no introduction, heading, or conclusion."*
- **Fix 2, stricter cleanup:** drop headings (lines ending in `:`) and strip list markers like `1.`, `2)`, `-`, `*`:
```python
for line in response_text.split("\n"):
    line = line.strip()
    if not line or line.endswith(":"):
        continue
    line = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line)
```
- **Where to fix it:** in the **library**, because the Phase 1 README already made the agent responsible for *"removing any empty or irrelevant lines"* (a Phase 1 gap showing up in Phase 2). The fix stays generic, with no mention of stories or tasks, so the agent remains reusable. Anything project-specific goes in the knowledge string instead.
- **Cost of changing the library:**
  - update **both copies** of `base_agents.py` (Phase 1 and Phase 2) so they stay identical
  - **re-run the Phase 1 test**; the only change was that the egg steps lost their numbers
- **Why not switch models?** The project requires `gpt-3.5-turbo`, and a stronger model would only have fixed 1 of our 3 problems. Routing uses the *embedding* model, and missing context is a design issue no model can fix.

### 7.2 Embeddings don't understand "not"
- **The README-style description:** *"Responsible for defining product personas and user stories only. **Does not define features or tasks.**"*
- **The problem:** to an embedding model, "does not define features or tasks" mostly **adds** the meaning of *features* and *tasks*, pulling the Product Manager's description toward the other two roles. It won **zero** steps in run 1.
- **The fix:** describe only what each role **does**, using the **same words the planner uses** (from `knowledge_action_planning`), and echo each role's evaluation criteria (e.g. *"As a…"*, *"feature name"*, *"task ID"*).
- **Measured with `route_check.py`:** it scores sample steps against each description (about a dozen embedding calls, no chat calls). Old vs new:

| Step | Old margin | New margin |
|---|---|---|
| Define user stories… | 0.052 | **0.280** |
| Define features… | 0.180 | **0.234** |
| Define tasks… | 0.044 | **0.295** |

  **Margin** = the winner's score minus the runner-up's. A large margin means a confident route; one under about 0.03 is a coin flip. The result went from 4/5 to **5/5** steps routed correctly.
- **Test the cheap part cheaply:** routing can be checked in seconds without running the whole workflow.

### 7.3 Passing context forward
- **The problem:** each step was routed with only its own text, so later agents never saw the stories.
- **The fix:** keep `route(step)` unchanged, so routing is based only on the step (as tested), and add context *after* the route is chosen:
```python
def with_context(query):
    if not completed_steps:
        return query
    return query + "\n\nUse these results from the previous steps:\n\n" + "\n\n".join(completed_steps)

def program_manager_support_function(query):
    result = program_manager_evaluation_agent.evaluate(with_context(query))
    return result["final_response"]
```
- **The design principle:** separate *what decides the route* from *what the worker needs to do the job*. Passing context through `route()` would have changed the embeddings, and so the routing.
- **The trade-off:** prompts grow with every step, which means more tokens.

The planner also needed the **order** stated in its knowledge:
> *"To define development tasks, first define the user stories from the product spec, then group the stories into features, then define the tasks for each story. The development plan is the result of these three steps, not a separate step."*

Without it, the stricter planner returned only what the prompt literally asked for (tasks). This is project-specific knowledge, so it goes in the workflow file, not the library.

### 7.4 Workers never see their evaluator's criteria
- **What happened in run 3:**
  - The Dev Engineer's first answer covered **all 6 features**, in a loose format.
  - The evaluator rejected it for the format, which was correct.
  - The rewrite fixed the format but kept only **1 feature**: asked to rewrite a long answer, gpt-3.5 took a shortcut.
- **Root cause:** the worker only learns the required format by being **rejected**, and correction loops can **lose content**.
- **The fix:** put the required format into each worker's knowledge, plus *"one to three tasks for EVERY user story"*, *"Related User Story: quote the parent story"*, and *"Only output the tasks. No introduction, conclusion or comments."*
- **Result in run 4:** every step passed on the **first try**, going from 5 evaluation rounds to 3.
- **General rule:** **tell the worker what the evaluator checks.** The evaluator should be a safety net, not the way the worker discovers the rules.

### 7.5 Evaluators check format, not content
- **Run 1:** an evaluator approved a "corrected" answer that just repeated the fix instructions (*"To fix the answer, the worker agent should…"*), because it had the right *shape*.
- **Run 4:** one user story got no tasks, and no evaluator noticed.
- **Lesson:** LLM evaluators are good at checking **structure**. **Completeness** (did every story get tasks?) is better checked with plain Python: deterministic, cheap, and it can't be sweet-talked. This is the improvement proposed in `reflection.md`.

### 7.6 Environment gotchas
| Symptom | Cause | Fix |
|---|---|---|
| "I don't see your changes" | Editor still showing the old version | Close and reopen, or *File: Revert File* in VS Code |
| `route_check.py` "OLD" and "CURRENT" tables identical to 3 decimals | Ran against a **stale copy** of `agentic_workflow.py` in the Udacity workspace | Sync files between OneDrive and the workspace; identical numbers mean identical inputs |
| `UnicodeEncodeError` risk when saving output | Library prints ✅ ⚠️; Windows' default encoding can't write them to a file | `python -X utf8 script.py > output.txt` |
| File "looked" truncated | Last line had no trailing newline | Check the raw bytes before assuming data is missing |

---

## 8. Design Lessons (the takeaways)

1. **Follow the code and the rubric, not just the README.** Three times the README disagreed with the code or the rubric (`agent_to_evaluate`, `respond()` then `evaluate()`, printing only the last step).
2. **A run without errors isn't a correct run.** Run 1 had no crashes and every step was ✅ accepted, and the output was still useless. Read the output.
3. **Fix upstream first.** In a pipeline, every bad step feeds the next one.
4. **Prompts are specs.** A weak model does what you *literally* say: "only steps in your knowledge" wasn't enough until the prompt spelled out *how* to answer and the knowledge spelled out the *order*.
5. **Embeddings work by shared vocabulary, not logic.** Write route descriptions positively, in the same words as the inputs they should match, and **measure** them (score and margin).
6. **Separate routing input from worker input.** Route on the step; give the worker the step plus context.
7. **Tell workers what the evaluator checks.** First-try passes are cheaper and avoid lossy rewrites.
8. **Know what your evaluator can't see.** LLM evaluators check form; use code to check completeness.
9. **Build cheap test tools.** Stub agents (no API calls) for wiring and `route_check.py` for routing let us test pieces in seconds.
10. **Change one thing at a time, and keep backups.** Every change had a backup (`originals_backup/`), and each run was saved to its own output file, so every result could be compared with the one before.

---

## 9. Self-Check Questions

1. What's the difference between a `system` and a `user` message, and why keep them separate?
2. Why is `temperature=0` used for agents, and what does it *not* guarantee?
3. Why does `choices` exist as a list?
4. Why does the knowledge-agent test use deliberately false knowledge?
5. What does a persona change, and what does it not change?
6. Why doesn't the evaluator fix answers itself?
7. After hitting `max_interactions` without passing, what's in the result dictionary? How do you tell "passed" from "gave up"?
8. What are embeddings and cosine similarity, in one sentence each?
9. Why does the router store `lambda x: agent.respond(x)` instead of calling `agent.respond(x)`?
10. How many API calls does one routed prompt make, and which ones could be cached?
11. Why does the action planner return a list?
12. What caused the RAG infinite loop, and what general lesson does it teach?
13. What does a 401 error tell you, and why did a valid `voc-` key get one?

**Phase 2**

14. Why does `"func": lambda x: product_manager_support_function(x)` work even though that function is defined later in the file?
15. Why must TODO 5 use `+ product_spec` instead of just placing the variable after the string literals?
16. Why do the support functions call `evaluate(step)` instead of `respond()` then `evaluate()`?
17. Why is `max_interactions` a cost decision? How many API calls does one failed evaluation round make?
18. Why did "Does not define features or tasks" make routing *worse*?
19. What is a routing "margin," and why does it matter more than the pass/fail result?
20. Why does `with_context()` add the earlier results inside the support functions instead of passing them to `route()`?
21. In run 3, the tasks covered only 1 of 6 features. What happened, and what fixed it?
22. What can an LLM evaluator check well, and what should be checked with plain code?
23. Run 1 had no errors and every step was accepted. Why was it still a failed run?
24. Why was the planner fix made in `base_agents.py`, but the step order added in `agentic_workflow.py`?

<details>
<summary>Answers</summary>

1. System = rules/persona (high priority); user = the request. Separation lets you reuse personas and protects the rules from being overridden.
2. It gives consistent, repeatable output. It doesn't guarantee correctness; the model can be consistently wrong.
3. You can request several completions with `n`; by default there's one, at `[0]`.
4. A correct answer (Paris) can't prove where it came from; a wrong one (London) proves the supplied knowledge was used.
5. Style and voice change; facts don't.
6. Separation of duties: it keeps the worker's knowledge and persona in charge and avoids the evaluator slipping in its own facts.
7. The last failed answer, a "No…" evaluation, and `iterations=10`. Check the evaluation text, or better, the `accepted` flag.
8. Embedding: text turned into a vector that captures meaning. Cosine similarity: how closely two vectors point the same way (1 = same meaning).
9. Storing the function defers the call until the router has chosen, so exactly one agent gets called, with the real question.
10. Five (1 prompt embedding + 3 description embeddings + 1 chat call). The 3 description embeddings could be cached.
11. So the workflow can loop over steps one at a time.
12. The overlap step-back never let `start` reach the end. Loops that step back need an explicit exit; check the last iteration.
13. The server received the key but rejected it. `voc-` keys only work through Vocareum's gateway, not directly at OpenAI.
14. A lambda looks the name up only when it's called, and by the time a step is routed the support function exists. Without the lambda, the name is looked up immediately and raises a `NameError`.
15. Python auto-joins adjacent *literal* strings only. A variable next to a string is a `SyntaxError`; you need `+`.
16. `evaluate(initial_prompt)` calls its own worker's `respond()`. Passing it an answer wastes a call and makes the worker respond to its own output. The rubric requires passing the step.
17. Each failed round costs 3 calls (worker answer, verdict, fix instructions), and Product Manager calls carry the 12 KB spec, so the cap limits the worst-case cost.
18. Embeddings don't understand negation. Mentioning features and tasks pulled the Product Manager's description toward those roles, so it won no steps.
19. The winner's score minus the runner-up's. A pass with a tiny margin is a coin flip that could flip with slightly different wording.
20. Routing should depend only on the step. Adding context before routing would change the embedding and could change the route.
21. The first answer covered everything but in the wrong format. The rewrite fixed the format and dropped content. Putting the format in the worker's knowledge made it pass first try, with no rewrite.
22. LLM: structure and format. Code: completeness and coverage (e.g. every story has a task), which is deterministic and can't be fooled by a well-shaped non-answer.
23. The output was wrong: generic steps, no user stories, and an invented feature. Checking for crashes isn't testing; read the output.
24. The planner fix is generic (follow your knowledge, clean output), which belongs in the reusable library. The stories→features→tasks order is specific to this project, so it belongs in the workflow's knowledge.
</details>

---

## 10. Glossary

- **Agent**: a class that wraps an LLM call with a specific role, prompt structure, or behavior.
- **Agentic workflow**: multiple agents coordinated to complete a multi-step task.
- **Persona**: a role description in the system prompt that shapes tone and behavior.
- **Knowledge augmentation**: supplying facts in the prompt that the model must use instead of its training.
- **RAG**: Retrieval-Augmented Generation; find the relevant chunks of a large text, then answer from them.
- **Chunking / overlap**: splitting text into pieces, with shared edges so context isn't lost.
- **Embedding**: a numeric vector representing the meaning of text.
- **Cosine similarity**: a measure (−1 to 1) of how similar two embeddings are in direction.
- **Worker agent**: the agent that produces answers.
- **Evaluation agent**: the agent that judges answers against criteria and sends back feedback.
- **Routing**: choosing which agent should handle a given input.
- **Lambda**: a small anonymous function, used here to store "call this agent" for later.
- **Caching**: storing the result of an expensive computation so it isn't repeated.
- **`.env` / dotenv**: a file and library for keeping secrets out of your code.
- **`base_url`**: the server address the API client sends requests to.
- **Traceback**: Python's error report; read it from the bottom up.
- **Support function**: the function a route calls; runs a role's worker ⇄ evaluator loop and returns the final answer.
- **Context passing**: giving a later agent the results of earlier steps (`with_context()`).
- **Routing margin**: the winning route's similarity score minus the runner-up's; a measure of routing confidence.
- **Fail fast**: stop immediately with a clear error when something required (like an API key) is missing.
- **Stub / fake agent**: a stand-in class that returns canned answers, used to test the workflow's wiring without API calls.
- **Consolidated plan**: the final output combining every step's result (stories, features, tasks) in labelled sections.
- **Rubric**: the grading criteria; when it disagrees with the README, the rubric decides.
