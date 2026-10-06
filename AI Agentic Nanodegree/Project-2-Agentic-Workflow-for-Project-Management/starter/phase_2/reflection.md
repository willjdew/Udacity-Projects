# Reflection: Agentic Workflow for Project Management (Email Router pilot)

## How the workflow works

A TPM prompt ("What would the development tasks for this product be?") goes to an
**Action Planning Agent**, which breaks it into steps. A **Routing Agent** sends each step to
the closest-matching role by embedding similarity. Each role is a **Knowledge Augmented Prompt
Agent** (the worker) paired with an **Evaluation Agent** that checks the worker's answer against
fixed structural criteria and sends it back with correction instructions until it passes.
Results from earlier steps are passed forward as context, and the workflow prints one
consolidated plan: user stories, then product features, then engineering tasks.

## How it got there: test runs and review fixes

| Run | What happened | What we changed |
|---|---|---|
| 1 | Planner returned 10 generic software-lifecycle steps (plus a heading line as a "step"). The router never picked the Product Manager, so the product spec was never used. Final output: one invented "Deployment" feature. | Stricter planner system prompt, and cleanup that drops headings and list numbers. Rewrote the route descriptions (see below). |
| 2 | Clean plan, correct routing, but only 2 steps: the planner returned just what the prompt literally asked for (tasks). Tasks were generic (user login, dashboards), not about the Email Router. | Told the planner the order (stories, then features, then tasks). Passed earlier results forward as context. |
| 3 | Correct 3-step plan and routing; Email Router stories and features. But the tasks covered only 1 of 6 features. | Put the required output format into each worker's knowledge. |
| 4 | Every step passed its evaluator on the first try; features and tasks in the exact required format; tasks quote their parent story. One of 5 stories got no tasks. | Small code cleanup (explicit `routing_agent.agents` assignment); no behavior change. |
| Submission 1 | 7 stories, 7 features, 8 tasks. Reviewer: the automated draft-response workflow (retrieval, drafting, human approval) had no tasks. | Asked the Product Manager to cover all six spec features and the Development Engineer to write separate retrieval, drafting and review tasks. |
| Submission 2 | Response workflow covered. Reviewer: one story used "As a User Interface" as its persona, and no task planned the knowledge base update from resolved inquiries. | Personas limited to the spec's user classes (section 2.2); added plain-Python checks (see below). |
| Resubmission runs | First run: the Development Engineer wrote 8 good tasks, the evaluator rejected them with just "No.", and the "corrected" answer kept one task, so 5 of 8 stories had none. Next run: every story had tasks, but four tasks lacked Dependencies (one had Assignee/Status/Deadline instead) and the evaluator accepted them. | Tasks are now planned one story at a time, and code checks the seven task fields. |
| Final | `phase_2_final_output.txt`: 6 stories (all real user types), 6 features, 20 tasks. Every story has tasks, every task has exactly the seven fields, and three tasks (S3-T1 to S3-T3) update the knowledge base from resolved inquiries. | |

### Code checks added after review

The LLM evaluators proved unreliable in both directions: they accepted stories without "so that"
and tasks missing Dependencies, and rejected correct answers with no reason given. So the support
functions now run plain-Python checks after `evaluate()`, and re-run the step (up to 3 times) with
a note naming exactly what to fix:

- **Stories:** every story has "As a", "I want" and "so that", and its persona is a Customer
  Support Representative, Subject Matter Expert or IT Administrator.
- **Tasks:** planned one story at a time (IDs like S3-T1), so every story gets tasks and one bad
  correction can't wipe out the rest of the plan. Each task must have exactly the seven required
  fields, and the stories that need them get separate tasks for knowledge retrieval, draft
  generation, human approval, and knowledge base updates from resolved inquiries.

**Note on the two warnings in the final log.** For story 2 (classification) and story 6
(dashboard), the log shows "response NOT accepted after 5 interactions". The gpt-3.5 evaluator
kept answering "No, the answer does not meet the criteria" without giving a reason, so its
correction instructions had nothing to act on. The code field check passed those tasks, and the
final output confirms all seven fields are present on each one. The warnings reflect the
evaluator's unreliability, not a defect in the submitted plan.

## Strengths

- **Reusable building blocks.** The same agent classes serve every role; a role is just a persona,
  a knowledge string and evaluation criteria. A new product only needs a new spec, and a new
  kind of plan mostly needs new planner knowledge.
- **Built-in quality control.** Every worker's output is checked against explicit structural
  criteria, so the plan always comes out in a predictable, reviewable format.
- **Routing by meaning.** The router matches steps to roles by semantic similarity, so the
  planner's step wording does not have to match exact keywords.
- **Traceability.** The consolidated plan links each task back to the user story it implements,
  and tasks list their dependencies.

## Limitations

- **The LLM evaluators are unreliable.** In run 1 an evaluator accepted a "fixed" answer that
  just repeated the correction instructions back, because it had the right shape. Later it
  accepted tasks missing a required field, and rejected correct answers without a reason. Coverage
  and format are now enforced by code checks instead, but those live outside the Evaluation Agent.
- **Correction loops can lose content.** In run 3 the Development Engineer's first answer covered
  all 6 features but in the wrong format; the rewrite fixed the format and dropped 5 of the 6
  features. Giving workers the required format up front (run 4) avoided the rewrite, but did not
  make the loop itself safer.
- **Embedding routing is sensitive to wording.** Embeddings do not understand negation: route
  descriptions like "Does not define features" pulled the Product Manager's description *toward*
  features. The descriptions only worked once they said what each role does, using the same
  words as the planner's steps. A cheap check script (route_check.py) was needed to tune them.
- **Behavior is model-sensitive.** With gpt-3.5-turbo the planner needed very explicit
  instructions and knowledge to stop answering from general knowledge.
- **Cost grows with context.** The spec (about 12 KB) goes to the Product Manager, and every
  later step carries all earlier results, so prompts get longer at each step.

## One specific improvement: build the code checks into the Evaluation Agent

The coverage check proposed in the first version of this reflection is now implemented (see
"Code checks added after review"), but it runs *after* `evaluate()` returns. When the LLM
evaluator rejects a correct answer, as it did for stories 2 and 6 in the final run, the loop
still spends five interactions on vague corrections before the code check gets a say.

The improvement: let `EvaluationAgent` take an optional `check` function alongside its criteria.
Inside the loop, run the check first. If it reports problems, use them directly as the correction
instructions ("S1-T1: missing Dependencies"), skipping the LLM's vaguer ones. If it passes, the
LLM evaluator only judges what code can't, such as whether a task is really about its story.
This keeps the worker -> evaluator -> fix loop the rubric describes, cuts wasted API calls, and
means an accepted answer always passes the deterministic checks.
