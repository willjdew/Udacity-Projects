# route_check.py
# Cheap check of the routing descriptions, without running the full workflow.
# It embeds some sample steps and every route description, then prints the cosine
# similarity table the RoutingAgent would use, and which route would win each step.
# Cost: about a dozen embedding calls. No worker or evaluation agents are run.
#
# Usage:  python -X utf8 route_check.py
#         python -X utf8 route_check.py "Your own step text" "Another step"

import ast
import os
import sys

import numpy as np
from dotenv import load_dotenv

from workflow_agents.base_agents import RoutingAgent

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found. Check your .env file.")

HERE = os.path.dirname(os.path.abspath(__file__))


def load_current_routes():
    """Read the route names/descriptions straight out of agentic_workflow.py.

    We parse the file instead of importing it, because importing would run the whole workflow.
    This keeps a single copy of the descriptions: whatever is in the workflow is what gets tested.
    """
    with open(os.path.join(HERE, "agentic_workflow.py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "routes" for t in node.targets):
            routes = []
            for d in node.value.elts:
                fields = {k.value: v for k, v in zip(d.keys, d.values)}
                routes.append((ast.literal_eval(fields["name"]), ast.literal_eval(fields["description"])))
            return routes
    raise RuntimeError("Could not find 'routes = [...]' in agentic_workflow.py")


# The previous descriptions (README style, with "Does not ..." phrases), kept for comparison.
OLD_ROUTES = [
    ("Product Manager", "Responsible for defining product personas and user stories only. "
                        "Does not define features or tasks. Does not group stories"),
    ("Program Manager", "Responsible for defining product features by grouping related user stories. "
                        "Does not define user stories or development tasks"),
    ("Development Engineer", "Responsible for defining the development tasks needed to implement each user story. "
                             "Does not define user stories or features"),
]

# (step text, route we expect to win). The first two came from the first workflow run;
# the last three are the kind of steps the fixed planner should now produce.
SAMPLE_STEPS = [
    ("Analyzing the product spec to identify user stories.", "Product Manager"),
    ("Breaking down user stories into specific tasks.", "Development Engineer"),
    ("Define user stories from the product spec by identifying a persona, an action, and a desired outcome", "Product Manager"),
    ("Define features by grouping related user stories", "Program Manager"),
    ("Define tasks for each story representing the engineering work required to develop the product", "Development Engineer"),
]

router = RoutingAgent(openai_api_key, [])
_cache = {}


def embed(text):
    if text not in _cache:
        _cache[text] = np.array(router.get_embedding(text))
    return _cache[text]


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def report(title, routes, steps):
    short = {"Product Manager": "ProdMgr", "Program Manager": "ProgMgr", "Development Engineer": "DevEng"}
    print(f"\n=== {title} ===")
    print(f"{'ProdMgr':>8} {'ProgMgr':>8} {'DevEng':>8}  {'margin':>6}  result  step")
    passed = 0
    for step, expected in steps:
        scores = [cosine(embed(step), embed(desc)) for _, desc in routes]
        order = np.argsort(scores)[::-1]
        winner = routes[order[0]][0]
        margin = scores[order[0]] - scores[order[1]]  # how clearly the winner won
        if expected is None:
            result = short[winner]
        elif winner == expected:
            result, passed = "PASS  ", passed + 1
        else:
            result = f"FAIL->{short[winner]}"
        cells = " ".join(f"{s:8.3f}" for s in scores)
        print(f"{cells}  {margin:6.3f}  {result}  {step[:70]}")
    graded = sum(1 for _, e in steps if e is not None)
    if graded:
        print(f"{passed}/{graded} steps routed as expected")


if len(sys.argv) > 1:
    steps = [(s, None) for s in sys.argv[1:]]  # your own steps: no expected route, just show the winner
else:
    steps = SAMPLE_STEPS

report("OLD descriptions (README style)", OLD_ROUTES, steps)
report("CURRENT descriptions (from agentic_workflow.py)", load_current_routes(), steps)
print("\nmargin = winner's score minus runner-up's. Small margins (< ~0.03) are fragile.")
