"""
Stand-alone check of the Quoting agent, for the Udacity workspace.

Run it from the folder that holds project_starter.py and the CSV files:

    python test_quoting_agent.py

Part 1 checks the two quoting tools directly (no API calls).
Part 2 gives the Quoting agent eight orders (this uses the API key in your
.env file). The items are the ones the Inventory agent found it could supply
for those sample requests. For each order the script checks that the agent
passed the items to calculate_quote unchanged, returned the tool's quote
unchanged, looked at the quote history, and copied the history note that
the history tool wrote.

Nothing here is part of the graded submission. It resets the practice
database (munder_difflin.db) each time it runs.
"""

import sys

import smolagents
from smolagents import LogLevel

MINIMUM_SMOLAGENTS_VERSION = (1, 20)

QUOTE_TOOL_NAME = "calculate_quote"
HISTORY_TOOL_NAME = "find_similar_quotes"

# Orders to quote: (sample request number, customer's event, request date,
# items that can be supplied as (catalog item_name, quantity in units)).
QUOTE_TEST_CASES = [
    (1, "ceremony", "2025-04-01", [("Glossy paper", 200), ("Cardstock", 100), ("Colored paper", 100)]),
    (4, "reception", "2025-04-05", [("Cardstock", 500), ("A4 paper", 250)]),
    (5, "party", "2025-04-05", [("Colored paper", 500), ("Cardstock", 300), ("Decorative adhesive tape (washi tape)", 200)]),
    (6, "assembly", "2025-04-06", [("Construction paper", 500), ("Standard copy paper", 300), ("Cardstock", 200)]),
    (10, "show", "2025-04-08", [("Glossy paper", 500), ("Cardstock", 300)]),
    (12, "party", "2025-04-08", [("Cardstock", 200), ("Standard copy paper", 500), ("Paper napkins", 100)]),
    # Partial orders: only the items that could be supplied are quoted.
    (2, "parade", "2025-04-03", [("Poster paper", 500), ("Party streamers", 300)]),
    (3, "conference", "2025-04-04", [("A4 paper", 10000), ("Standard copy paper", 250000)]),
]


def print_heading(title: str) -> None:
    """Print a section heading that is easy to find in the console output."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_environment() -> None:
    """Stop early with a clear message if smolagents is too old."""
    version_parts = tuple(int(part) for part in smolagents.__version__.split(".")[:2])
    print(f"smolagents version: {smolagents.__version__}")
    if version_parts < MINIMUM_SMOLAGENTS_VERSION:
        sys.exit(
            "This project needs smolagents 1.20 or newer. "
            "Run: pip install -U smolagents"
        )


def to_order_lines(items: list) -> list:
    """Turn (item_name, quantity) pairs into the order lines the quote tool expects."""
    return [{"item_name": item_name, "quantity": quantity} for item_name, quantity in items]


def run_tool_checks(project) -> bool:
    """
    Call each quoting tool directly and compare with known answers.

    Args:
        project: The imported project_starter module.

    Returns:
        bool: True when every check passed.
    """
    quote = project.calculate_quote

    def cardstock_quote(units: int) -> str:
        """Quote a single line of Cardstock, which costs $0.15 a sheet."""
        return quote(to_order_lines([("Cardstock", units)]))

    three_items = quote(to_order_lines([("Glossy paper", 200), ("Cardstock", 100), ("Colored paper", 100)]))
    history = project.find_similar_quotes("ceremony, cardstock")

    checks = [
        ("499 units: no discount, total $74.85", "BULK DISCOUNT: 0%" in cardstock_quote(499) and "TOTAL: $74.85" in cardstock_quote(499)),
        ("500 units: 5% off, total $71.25", "BULK DISCOUNT: 5%" in cardstock_quote(500) and "TOTAL: $71.25" in cardstock_quote(500)),
        ("1,000 units: 10% off, total $135.00", "BULK DISCOUNT: 10%" in cardstock_quote(1000) and "TOTAL: $135.00" in cardstock_quote(1000)),
        ("5,000 units: 15% off, total $637.50", "BULK DISCOUNT: 15%" in cardstock_quote(5000) and "TOTAL: $637.50" in cardstock_quote(5000)),
        ("three items, 400 units: subtotal and total $65.00", "SUBTOTAL: $65.00" in three_items and "TOTAL: $65.00" in three_items),
        ("below the top tier, the quote says how many units reach the next tier", "NEXT DISCOUNT TIER: 100 more units" in three_items),
        ("an unknown item is refused with a plain message", "cannot be priced" in quote(to_order_lines([("balloons", 5)]))),
        ("find_similar_quotes returns past quotes for 'ceremony, cardstock'", history["status"] == "ok" and history["match_count"] > 0),
        ("find_similar_quotes returns an empty list for 'balloons'", project.find_similar_quotes("balloons")["match_count"] == 0),
        (
            "the history note states the counts the tool found",
            history["history_note"]
            == f"HISTORY NOTE: {history['match_count']} similar past quote(s) found; "
               f"{history['discount_mention_count']} of them mention a bulk discount.",
        ),
        (
            "with no matches the history note says so",
            project.find_similar_quotes("balloons")["history_note"] == "HISTORY NOTE: No similar past quotes were found.",
        ),
    ]

    all_passed = True
    for description, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {description}")
        all_passed = all_passed and passed
    return all_passed


def build_task(event: str, request_date: str, items: list) -> str:
    """
    Write the task the way the orchestrator will hand it to the Quoting agent.

    Args:
        event (str): The event the customer mentioned.
        request_date (str): The request date as YYYY-MM-DD.
        items (list): (item_name, quantity) pairs that can be supplied.

    Returns:
        str: The task text.
    """
    item_lines = "\n".join(f"- {item_name}: {quantity} units" for item_name, quantity in items)
    return (
        "Prepare a quote for this order.\n"
        f"Items that can be supplied:\n{item_lines}\n"
        f"Customer's event: {event}\n"
        f"Request date: {request_date}"
    )


def collect_tool_calls(agent) -> dict:
    """
    Gather the calls the agent made to the two quoting tools.

    Args:
        agent: The agent that has just finished a run.

    Returns:
        dict: 'quote_calls' (list of order_lines lists) and 'history_calls'
              (list of search-term strings).
    """
    calls = {"quote_calls": [], "history_calls": []}
    for step in agent.memory.steps:
        for tool_call in getattr(step, "tool_calls", None) or []:
            arguments = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
            if tool_call.name == QUOTE_TOOL_NAME:
                calls["quote_calls"].append(arguments.get("order_lines") or [])
            elif tool_call.name == HISTORY_TOOL_NAME:
                calls["history_calls"].append(str(arguments.get("search_terms")))
    return calls


def same_items(sent_lines: list, expected_items: list) -> bool:
    """Check that the agent sent exactly the items it was given, in any order."""
    sent = sorted(
        (str(line.get("item_name")), int(line.get("quantity", 0)))
        for line in sent_lines if isinstance(line, dict)
    )
    return sent == sorted(expected_items)


def squeeze(text: str) -> str:
    """Collapse all spacing so texts can be compared regardless of line breaks."""
    return " ".join(str(text).split())


def run_agent_checks(project) -> list:
    """
    Give every test order to the Quoting agent and print what it did.

    Args:
        project: The imported project_starter module.

    Returns:
        list: One scoreboard row per order.
    """
    agent = project.quoting_agent
    # Keep the console readable: this script prints its own summary instead.
    agent.logger.level = LogLevel.ERROR

    scoreboard = []
    for number, event, request_date, items in QUOTE_TEST_CASES:
        expected_quote = project.calculate_quote(to_order_lines(items))
        expected_total_line = expected_quote.splitlines()[-1]
        task = build_task(event, request_date, items)

        print_heading(f"Order from sample request {number}")
        print(task)

        row = {"number": number, "items": False, "quote": False, "history": False, "note": False}
        try:
            final_answer = str(agent.run(task))
        except Exception as error:  # keep going so one failure does not hide the rest
            print(f"\n--- The agent run failed: {error}")
            scoreboard.append(row)
            continue

        calls = collect_tool_calls(agent)
        row["items"] = len(calls["quote_calls"]) == 1 and same_items(calls["quote_calls"][0], items)
        row["quote"] = squeeze(expected_quote) in squeeze(final_answer)
        row["history"] = len(calls["history_calls"]) >= 1
        # The note must be the one the tool wrote for the terms the agent searched.
        expected_note = (
            project.find_similar_quotes(calls["history_calls"][-1]).get("history_note", "")
            if calls["history_calls"] else ""
        )
        row["note"] = bool(expected_note) and squeeze(expected_note) in squeeze(final_answer)

        print(f"\n--- Tool calls: {len(calls['quote_calls'])} to {QUOTE_TOOL_NAME}, "
              f"{len(calls['history_calls'])} to {HISTORY_TOOL_NAME} {calls['history_calls']} ---")
        print("\n--- Final answer ---")
        print(final_answer)
        print(f"\n--- Expected: {expected_total_line} ---")
        if not row["items"]:
            print(f"NOTE: the items sent to {QUOTE_TOOL_NAME} were not the items given: {calls['quote_calls']}")
        if not row["quote"]:
            print("NOTE: the final answer does not contain the tool's quote unchanged, which was:")
            print(expected_quote)
        if not row["note"]:
            print(f"NOTE: the final answer does not end with the tool's history note, which was: {expected_note!r}")
        scoreboard.append(row)
    return scoreboard


def print_scoreboard(scoreboard: list) -> None:
    """
    Print one line per order and the totals.

    Args:
        scoreboard (list): Rows produced by run_agent_checks.
    """
    labels = [
        ("items", "items passed unchanged"),
        ("quote", "quote returned unchanged"),
        ("history", "history searched"),
        ("note", "history note copied unchanged"),
    ]
    for row in scoreboard:
        all_good = all(row[key] for key, _ in labels)
        problems = ", ".join(f"NOT {label}" for key, label in labels if not row[key])
        print(f"[{'OK' if all_good else 'XX'}] order from request {row['number']:>2}{': ' + problems if problems else ''}")
    print()
    for key, label in labels:
        print(f"{label}: {sum(row[key] for row in scoreboard)} of {len(scoreboard)}")


def main() -> None:
    """Run the environment check, the tool checks, then the agent checks."""
    check_environment()

    try:
        import project_starter as project
    except RuntimeError as error:  # raised when the API key is missing
        sys.exit(f"Could not load project_starter.py: {error}")

    print(f"Model: {project.MODEL_ID}")
    project.init_database(project.db_engine)

    print_heading("PART 1 - quoting tools (no API calls)")
    if not run_tool_checks(project):
        sys.exit("A tool check failed, so the agent test was skipped.")

    print_heading(f"PART 2 - Quoting agent on {len(QUOTE_TEST_CASES)} orders (uses the API)")
    scoreboard = run_agent_checks(project)

    print_heading("SCOREBOARD")
    print_scoreboard(scoreboard)
    print("\nSend back the whole output for review.")


if __name__ == "__main__":
    main()
