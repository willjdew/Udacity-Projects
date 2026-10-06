# agentic_workflow.py

# Notes: The four agent classes from the Phase 1 library. This import only works when the
# script is run from the phase_2 folder, next to the workflow_agents package.
from workflow_agents.base_agents import (
    ActionPlanningAgent,             # breaks the TPM's prompt into steps
    KnowledgeAugmentedPromptAgent,   # the "workers": PM, Program Mgr, Dev Engineer
    EvaluationAgent,                 # checks each worker's output against criteria
    RoutingAgent,                    # sends each step to the right worker
)

import os
import re
from dotenv import load_dotenv

# Notes: Load the OpenAI API key from .env. Fail fast with a clear message if it is missing,
# instead of a confusing authentication error from the first API call later on.
load_dotenv()  # copies the values in .env into the environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found. Check your .env file.")

# Notes: Load the product spec. The path is built from this script's own folder so it works
# from any working directory; utf-8 is explicit because Windows defaults to a different encoding.
spec_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Product-Spec-Email-Router.txt")
with open(spec_path, "r", encoding="utf-8") as f:
    product_spec = f.read()

# Instantiate all the agents

# Notes: Cap on worker -> evaluator -> fix rounds, shared by all three evaluation agents.
# Each failed round costs 3 API calls, so this bounds the cost if an evaluator never says "Yes".
MAX_EVAL_INTERACTIONS = 5

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
    # Notes: Added to the provided knowledge. Without it the planner only returned the steps the
    # prompt literally asked for (tasks), skipped stories and features, and turned the line above
    # into an extra "create a development plan" step that no persona owns.
    ". \n"
    "To define development tasks, first define the user stories from the product spec, "
    "then group the stories into features, then define the tasks for each story. "
    "The development plan is the result of these three steps, not a separate step."
)
# Notes: The planner turns the workflow prompt into ordered steps (stories -> features -> tasks),
# using only the kinds of steps described in its knowledge.
action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge_action_planning)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    # Notes: Added after the second review. One story began "As a User Interface": the model used a product
    # feature as the persona. Personas must be the people in section 2.2 of the spec (User Classes).
    "The persona in every story must be one of the user classes in section 2.2 of the spec: "
    "Customer Support Representative, Subject Matter Expert, or IT Administrator. "
    "Never use a product feature or system component (such as 'User Interface' or 'Routing Logic') as the persona; "
    "for example, the dashboard and configuration panel story belongs to an IT Administrator. "
    # Notes: Added after reviewer feedback. The first submission's stories skipped the Response
    # Generation Engine (RAG drafts for routine inquiries + human approval before sending), and
    # since features and tasks are built only from the stories, that capability vanished from the
    # whole plan. Asking for coverage of every product feature in section 2.1 fixes it at the source.
    "Cover the whole product: write at least one story for EVERY product feature listed in section 2.1 "
    "of the spec (Email Ingestion System, Message Classification Module, Knowledge Base Integration, "
    "Response Generation Engine, Routing Logic, User Interface). "
    "For the Response Generation Engine, include a story about automatically answering routine inquiries "
    "with draft responses generated from retrieved knowledge, which a human reviews and approves before "
    "they are sent. "
    "For the Knowledge Base Integration, include a story about the knowledge base being updated automatically "
    "with new information from resolved inquiries. "
    "Every story must use the full template, including the words 'so that' followed by the benefit. "
    "Only output the stories. Do not add an introduction, a conclusion, or comments.\n\n"
    # Notes: '+' is required here. Python only auto-joins literal strings, not variables.
    # Only the Product Manager gets the spec: it is the first stage of the pipeline.
    + product_spec
)
# Notes: Worker agent that writes user stories from the spec.
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key, persona_product_manager, knowledge_product_manager
)

# Product Manager - Evaluation Agent
# Notes: Checks the Product Manager's stories against the "As a..., I want..., so that..." template.
# The worker agent is passed by position: base_agents.py names that parameter 'worker_agent'
# (the README calls it 'agent_to_evaluate'), so positional avoids the naming mismatch.
persona_product_manager_eval = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria_product_manager = (
    "The answer should be stories that follow the following structure: "
    "As a [type of user], I want [an action or feature] so that [benefit/value]. "
    # Notes: In the resubmission run, 4 of 6 stories ended with "to ensure..." instead of "so that..."
    # and were still accepted, so the evaluator now checks each story for all three parts.
    "Every story must contain 'As a', 'I want', and 'so that'; reject the answer if any story is missing one. "
    # Notes: Added after the second review ("As a User Interface" was accepted).
    "The [type of user] in every story must be a person: a Customer Support Representative, Subject Matter "
    "Expert, or IT Administrator. Reject the answer if any story uses a product feature or system component "
    "(for example 'As a User Interface') as the user. "
    # Notes: Added after reviewer feedback, so the worker -> evaluator loop also enforces coverage.
    # The evaluator never sees the spec, so the six product features are named here.
    "Together, the stories must cover all six product features: email ingestion, message classification, "
    "knowledge base retrieval, response generation (automated draft responses to routine inquiries using "
    "retrieved knowledge, with human review and approval before dispatch), routing to subject matter "
    "experts, and the user interface/dashboard."
)
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_product_manager_eval,
    evaluation_criteria_product_manager,
    product_manager_knowledge_agent,  # worker_agent: the agent whose answers get checked
    max_interactions=MAX_EVAL_INTERACTIONS,
)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = (
    "Features of a product are defined by organizing similar user stories into cohesive groups."
    # Notes: Added to the provided knowledge. Workers never see their evaluator's criteria, so we
    # give them the required format up front; otherwise they only learn it by being rejected.
    " Write each feature in exactly this format:\n"
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user\n"
    "Only output the features. Do not add an introduction, a conclusion, or comments."
)
# Notes: Worker agent that groups user stories into features. No spec here: its input is
# the stories produced by the Product Manager step (passed in by with_context).
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key, persona_program_manager, knowledge_program_manager
)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."

# Notes: Checks that each feature has all four labeled fields. Structural criteria like these
# give more consistent Yes/No verdicts than quality-based ones.
evaluation_criteria_program_manager = (
    "The answer should be product features that follow the following structure: "
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user"
)
program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_program_manager_eval,
    evaluation_criteria_program_manager,
    program_manager_knowledge_agent,  # worker_agent: the agent whose answers get checked
    max_interactions=MAX_EVAL_INTERACTIONS,
)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = (
    "Development tasks are defined by identifying what needs to be built to implement each user story."
    # Notes: Added to the provided knowledge. In run 3 this agent covered all 6 features in a loose
    # format, was rejected for the format, and the rewrite kept only 1 feature. Giving it the
    # format up front (and asking for every story) avoids that lossy rewrite.
    " Write one to three tasks for EVERY user story, not just the first one."
    # Notes: Added after reviewer feedback. A story such as automated responses involves several
    # separate pieces of work (retrieve knowledge, generate the draft, human review before sending),
    # and each needs its own task with its own acceptance criteria.
    " When a story involves several distinct pieces of engineering work, write a separate task for each."
    " For the automated response story, write separate tasks for knowledge base retrieval, draft response"
    " generation, and the human review and approval step before a response is sent."
    " (RAG means retrieval-augmented generation: relevant knowledge is retrieved first, then used to"
    " generate the draft, so retrieval and generation are two separate tasks.)"
    # Notes: Added after the second review. The Knowledge Base Integration story promises updates from
    # resolved inquiries, but no task planned that work; the retrieval task (for draft responses) is a
    # different part of the feature.
    " For the Knowledge Base Integration story, write a separate task for the continuous learning mechanism"
    " that updates the knowledge base with new information from resolved inquiries (for example: when an"
    " inquiry is marked resolved, extract the question and the approved answer, embed them, and store them in"
    " the vector database). This is different from the task that retrieves knowledge for draft responses."
    " Every user story must have at least one task whose Related User Story quotes it."
    # Notes: The worker must never shrink the plan when it is asked to correct formatting.
    " When asked to correct your answer, keep every task from your previous answer; fix the format only"
    " and never remove tasks."
    # Notes: Added after the fourth run, where some tasks had Assignee/Status/Deadline instead of Dependencies.
    " Every task must have all seven fields below and no other fields (no Assignee, Status or Deadline);"
    " always include Dependencies, and write 'None' when a task has no dependencies."
    " Write each task in exactly this format:\n"
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Quote the parent user story (the 'As a ...' sentence), not a feature name\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first\n"
    "Only output the tasks. Do not add an introduction, a conclusion, or comments."
)
# Notes: Worker agent that turns user stories into development tasks.
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key, persona_dev_engineer, knowledge_dev_engineer
)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
# Notes: Checks that each task has all seven labeled fields. This is the strictest evaluator,
# so it is the most likely to hit MAX_EVAL_INTERACTIONS.
# Notes: In the resubmission run the worker wrote all 10 tasks correctly (including knowledge retrieval,
# draft generation and human review), but the evaluator rejected the answer for "including multiple
# tasks", and the fix instructions made the worker cut it down to a single task. The criteria now say
# plainly that a list of many tasks is expected, and that tasks must not be removed.
evaluation_criteria_dev_engineer = (
    # Notes: Second resubmission run: the evaluator rejected a correct 10-task answer twice with a vague
    # reason, and the third try dropped the SMTP, Routing and UI tasks. The evaluator can't see the stories,
    # so it is now asked only to check the labels on each task, which it can actually verify.
    "The answer should be a list of development tasks. A long list of many tasks is correct and expected. "
    "Answer Yes if every task in the list contains all seven labeled fields below; do not judge the number "
    "of tasks or their content, and never ask for tasks to be removed. "
    "Each task must have these fields: "
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first"
)
development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_dev_engineer_eval,
    evaluation_criteria_dev_engineer,
    development_engineer_knowledge_agent,  # worker_agent: the agent whose answers get checked
    max_interactions=MAX_EVAL_INTERACTIONS,
)


# Routing Agent
# The router embeds each step and each route "description", then sends the step to the
# route with the highest cosine similarity. So the descriptions ARE the routing logic.
# They only say what each role DOES: embeddings don't understand "not", so a phrase like
# "does not define features" pulls a description TOWARD features. Each one reuses the planner's
# wording (from knowledge_action_planning) and echoes its role's evaluation criteria.
# Check changes cheaply with route_check.py before running the full workflow.
# The lambdas delay the name lookup until a step is routed, so the support functions can be
# defined further down the file without a NameError.
routes = [
    {
        "name": "Product Manager",
        "description": "Defines user stories from the product specification by identifying each persona "
                       "(type of user), the action they want, and the desired outcome. "
                       "Writes stories as: As a [user], I want [action] so that [benefit].",
        "func": lambda x: product_manager_support_function(x),
    },
    {
        "name": "Program Manager",
        "description": "Defines product features by grouping related user stories into cohesive groups. "
                       "Each feature has a feature name, description, key functionality, and user benefit.",
        "func": lambda x: program_manager_support_function(x),
    },
    {
        "name": "Development Engineer",
        "description": "Defines development tasks: the engineering work required to build the product "
                       "for each story, with task ID, acceptance criteria, estimated effort, and dependencies.",
        "func": lambda x: development_engineer_support_function(x),
    },
]
routing_agent = RoutingAgent(openai_api_key, [])
routing_agent.agents = routes  # assign the list of routes to the routing agent's 'agents' attribute

# Job function persona support functions
# Notes: Each support function takes one step from the action plan, passes that step to its
# EvaluationAgent's evaluate(), and returns the "final_response" of the result. It does NOT call
# the knowledge agent's respond() first: evaluate() in base_agents.py calls its own worker agent
# internally (worker -> evaluator -> fix loop), as the rubric describes. (The phase_2 README's
# "respond() first, then evaluate() the answer" would waste a worker call and make the worker
# respond to its own output.)
#
# Notes: with_context() passes earlier results forward. The step itself is still what each support
# function hands to evaluate(); when earlier steps are done, their results are appended after it
# as reference material. The router still routes on the step text only (route(step) in the loop
# below), so routing stays as tested with route_check.py. Why: without this, the Program Manager
# and Development Engineer never see the user stories. In test runs 1-2 they invented generic
# features and tasks unrelated to the Email Router; with it (runs 3-4) every feature and task is
# built from the real stories, and each task's "Related User Story" quotes one of them.
# completed_steps is defined further down, in the workflow loop; like the lambdas, the name is
# only looked up when the function runs.
def with_context(query):
    if not completed_steps:
        return query
    return query + "\n\nUse these results from the previous steps:\n\n" + "\n\n".join(completed_steps)


# Notes: Code-level format checks. In the resubmission runs the gpt-3.5 evaluators were not reliable:
# they accepted stories without "so that", and accepted task lists with no separate knowledge-retrieval
# task. These checks are plain Python, so they give the same verdict every time. If a check fails, the
# step is run again (up to MAX_FORMAT_RETRIES times) with a note saying exactly what to fix.
MAX_FORMAT_RETRIES = 3


def story_format_problems(text):
    """Return the stories that don't use the 'As a ..., I want ... so that ...' template."""
    stories = [line.strip() for line in text.splitlines() if line.strip().startswith("As a")]
    if not stories:
        return ["(no stories found)"]
    return [s for s in stories if " I want " not in s or " so that " not in s]


# Notes: Added after the second review. The personas must be the user classes from section 2.2 of the spec.
ALLOWED_PERSONAS = ["customer support representative", "subject matter expert", "sme", "it administrator"]


def story_persona_problems(text):
    """Return the stories whose persona is not one of the spec's user classes (e.g. 'As a User Interface')."""
    problems = []
    for line in text.splitlines():
        line = line.strip()
        match = re.match(r"As an? (.+?),? I want", line)
        if match and not any(p in match.group(1).lower() for p in ALLOWED_PERSONAS):
            problems.append(line)
    return problems


# Notes: The tasks each kind of story needs, with the instruction given to the worker for each one.
TASK_HINTS = {
    "knowledge base retrieval": "retrieving the relevant knowledge from the knowledge base (vector database)",
    "draft response generation": "generating the draft response from the retrieved knowledge",
    "human review and approval": "the human review and approval step before a response is sent",
    "knowledge base update from resolved inquiries": "updating the knowledge base with new information from "
        "resolved inquiries (extract the question and approved answer, embed them, store them in the vector database)",
}


def required_tasks_for_story(story):
    """Return which of the TASK_HINTS tasks a story needs, based on what the story asks for."""
    s = story.lower()
    needed = []
    if "draft" in s:  # not "generat": "Response Generation Engine" appears in stories that aren't about drafts
        needed += ["knowledge base retrieval", "draft response generation"]
    if "approv" in s or "review" in s:
        needed.append("human review and approval")
    if "resolved" in s:
        needed.append("knowledge base update from resolved inquiries")
    return needed


def missing_response_tasks(text, required=None):
    """Return which required tasks have no Task Title of their own (all of them if required is None)."""
    titles = [line.lower() for line in text.splitlines() if line.strip().startswith("Task Title:")]
    retrieval = {t for t in titles if "retriev" in t and "knowledge" in t}
    drafting = {t for t in titles if "draft" in t or "generat" in t}
    found = {
        "knowledge base retrieval": bool(retrieval),
        # Notes: Fixed after the third run: "Generate draft response using retrieved knowledge" was wrongly
        # rejected because it mentions retrieval. Now it passes if there is a drafting title and retrieval and
        # drafting are covered by at least two different tasks (not one task doing both).
        "draft response generation": bool(drafting) and len(retrieval | drafting) >= 2,
        "human review and approval": any("review" in t or "approv" in t for t in titles),
        "knowledge base update from resolved inquiries": any(
            "resolved" in t or ("knowledge" in t and ("updat" in t or "learning" in t)) for t in titles),
    }
    required = list(found) if required is None else required
    return [name for name in required if not found[name]]


# Notes: Added after the fourth run. The evaluator accepted tasks that were missing "Dependencies" (or had
# Assignee/Status/Deadline instead), so the seven required fields are now checked in code for every task.
TASK_FIELDS = ["Task ID", "Task Title", "Related User Story", "Description",
               "Acceptance Criteria", "Estimated Effort", "Dependencies"]


def task_field_problems(text):
    """Return a list like 'S1-T1: missing Dependencies' for tasks without exactly the seven required fields."""
    problems = []
    for block in re.split(r"(?m)^\s*(?=Task ID:)", text):
        if not block.strip().startswith("Task ID:"):
            continue
        task_id = block.splitlines()[0].split(":", 1)[1].strip()
        labels = re.findall(r"(?m)^\s*([A-Z][A-Za-z ]{1,30}):", block)
        missing = [f for f in TASK_FIELDS if f not in labels]
        extra = [l for l in labels if l not in TASK_FIELDS]
        if missing:
            problems.append(f"{task_id}: missing {', '.join(missing)}")
        if extra:
            problems.append(f"{task_id}: extra field(s) {', '.join(extra)}")
    return problems


def product_manager_support_function(query):
    note = ""
    for attempt in range(1, MAX_FORMAT_RETRIES + 1):
        result = product_manager_evaluation_agent.evaluate(with_context(query) + note)
        problems = story_format_problems(result["final_response"])
        bad_personas = story_persona_problems(result["final_response"])
        if not problems and not bad_personas:
            break
        print(f"\n[Format check] {len(problems)} story(ies) missing 'I want' or 'so that', "
              f"{len(bad_personas)} story(ies) with a persona that is not a user type "
              f"(attempt {attempt}/{MAX_FORMAT_RETRIES}); running this step again.")
        note = ("\n\nIMPORTANT: every story must use exactly this template, with the words 'I want' and "
                "'so that': As a [type of user], I want [an action or feature] so that [benefit/value]. "
                "The type of user must be a Customer Support Representative, Subject Matter Expert, or "
                "IT Administrator, never a product feature such as 'User Interface'.")
    return result["final_response"]


def program_manager_support_function(query):
    result = program_manager_evaluation_agent.evaluate(with_context(query))
    return result["final_response"]


# Notes: Rewritten after the third run. Asking for every story's tasks in ONE answer was not reliable: in that
# run the worker wrote 8 good tasks, the evaluator rejected them with just "No.", and the "corrected" answer
# kept a single task, so 5 of 8 stories ended up with no task. Now the tasks are planned one story at a time:
# each story still goes through the worker -> evaluator loop (evaluate()), but each answer is short, so the
# evaluator judges it reliably and a correction can't wipe out the rest of the plan. Every story is
# guaranteed its own tasks, and the tasks each story needs (TASK_HINTS) are checked in code.
def development_engineer_support_function(query):
    stories = [line.strip() for step in completed_steps for line in step.splitlines()
               if line.strip().startswith("As a")]
    if not stories:  # no stories to work from: fall back to a single request
        return development_engineer_evaluation_agent.evaluate(with_context(query))["final_response"]

    all_tasks = []
    earlier_tasks = []  # "S1-T1: title" lines, so later stories can list them as dependencies
    for n, story in enumerate(stories, start=1):
        print(f"\n[Development Engineer] Planning tasks for story {n}/{len(stories)}")
        needed = required_tasks_for_story(story)
        prompt = (f"{query}\n\nDefine the development tasks for this user story only:\n{story}\n\n"
                  f"Use the Task IDs S{n}-T1, S{n}-T2, and so on.")
        if needed:
            prompt += " Write a separate task for each of: " + "; ".join(TASK_HINTS[x] for x in needed) + "."
        if earlier_tasks:
            prompt += ("\nTasks already planned for earlier stories (list their Task IDs under Dependencies "
                       "when a task needs one of them first):\n" + "\n".join(earlier_tasks))
        note = ""
        for attempt in range(1, MAX_FORMAT_RETRIES + 1):
            tasks = development_engineer_evaluation_agent.evaluate(prompt + note)["final_response"]
            missing = missing_response_tasks(tasks, needed)
            field_problems = task_field_problems(tasks)
            if "Task ID:" in tasks and not missing and not field_problems:
                break
            print(f"\n[Coverage check] Story {n}: missing tasks: {', '.join(missing) or '-'}; "
                  f"field problems: {'; '.join(field_problems) or '-'} "
                  f"(attempt {attempt}/{MAX_FORMAT_RETRIES}); running this story again.")
            note = ("\n\nIMPORTANT: every task must have exactly these seven fields and no others: "
                    + ", ".join(TASK_FIELDS) + ". Always include Dependencies (write 'None' if there are none).")
            if missing or needed:
                note += (" Write a separate task for each of: "
                         + "; ".join(TASK_HINTS[x] for x in missing or needed) + ".")
        # Notes: Each call covers exactly one story, so its tasks' "Related User Story" is set to that story's
        # exact text. This keeps the link between every task and its story exact.
        tasks = re.sub(r"(?m)^(\s*Related User Story:).*$", lambda m: m.group(1) + " " + story, tasks)
        all_tasks.append(tasks.strip())
        ids = re.findall(r"(?m)^\s*Task ID:\s*(.+)$", tasks)
        titles = re.findall(r"(?m)^\s*Task Title:\s*(.+)$", tasks)
        earlier_tasks += [f"{i.strip()}: {t.strip()}" for i, t in zip(ids, titles)]
    return "\n\n".join(all_tasks)


# Run the workflow

print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
workflow_prompt = "What would the development tasks for this product be?"
# ****
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")
# Notes: The agentic workflow. The planner decides WHICH steps are needed for this prompt,
# the router decides WHO handles each step, and each support function runs that persona's
# worker -> evaluator loop. Each step is routed on its own text only (as the README specifies);
# the support functions add the earlier results in completed_steps as context (see with_context).
workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
print(f"\nWorkflow steps ({len(workflow_steps)}):")
for n, step in enumerate(workflow_steps, start=1):
    print(f"  {n}. {step}")

completed_steps = []
for n, step in enumerate(workflow_steps, start=1):
    print(f"\n{'=' * 70}\nExecuting step {n}/{len(workflow_steps)}: {step}\n{'=' * 70}")
    result = routing_agent.route(step)
    completed_steps.append(result)
    print(f"\nResult of step {n}:\n{result}")

print("\n*** Workflow execution completed ***\n")
# Notes: The README asks for the last completed step as the final output. The project overview
# asks for "a final, structured output representing the comprehensively planned project", so we
# print every step's result (stories, features, tasks) as one plan; the last section is that
# last completed step.
if completed_steps:
    print("Final output of the workflow: development plan\n")
    for n, (step, result) in enumerate(zip(workflow_steps, completed_steps), start=1):
        print(f"{'#' * 70}\n# Part {n}: {step}\n{'#' * 70}\n")
        print(f"{result}\n")
else:
    print("The action planning agent returned no steps, so there is no output.")
