"""
Stand-alone check of the Inventory agent, for the Udacity workspace.

Run it from the folder that holds project_starter.py and the CSV files:

    python test_inventory_agent.py

Part 1 checks the three inventory tools directly (no API calls).
Part 2 sends all 20 requests from quote_requests_sample.csv to the Inventory
agent (this uses the API key in your .env file). For each one it prints the
order lines the agent filled in and the report the tool produced, and it
checks that the agent passed that report on unchanged. A scoreboard at the
end compares each 'ORDER COMPLETE' verdict with the expected one.

Nothing here is part of the graded submission. It resets the practice
database (munder_difflin.db) each time it runs. Every request is checked
against opening stock, because the Inventory agent never changes the database.
"""

import sys

import pandas as pd
import smolagents
from smolagents import LogLevel

MINIMUM_SMOLAGENTS_VERSION = (1, 20)

# The tool whose input (the order lines) and output (the report) are printed.
ORDER_TOOL_NAME = "check_order_availability"

# Sample requests (1-based) that can be supplied in full at opening stock.
# Every other request includes an item we do not sell, a size we do not
# carry, or a restock that would arrive after the deadline.
EXPECTED_COMPLETE_REQUESTS = {1, 4, 5, 6, 10, 12}


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


def make_order_line(requested: str, item_name: str, quantity: int, unit: str = "sheets", sheet_size: str = "") -> dict:
    """Build one order line in the shape check_order_availability expects."""
    return {
        "requested": requested,
        "item_name": item_name,
        "quantity": quantity,
        "unit": unit,
        "sheet_size": sheet_size,
    }


def run_tool_checks(project) -> bool:
    """
    Call each inventory tool directly and compare with known answers.

    Args:
        project: The imported project_starter module.

    Returns:
        bool: True when every check passed.
    """
    opening_date = "2025-04-01"
    deadline = "2025-04-15"
    check_order = project.check_order_availability

    catalog = project.list_catalog_stock(opening_date)
    in_stock = check_order([make_order_line("cardstock", "Cardstock", 100)], opening_date, deadline)
    restock_in_time = check_order([make_order_line("A4 paper", "A4 paper", 500)], opening_date, deadline)
    restock_too_late = check_order([make_order_line("A4 paper", "A4 paper", 500)], "2025-04-08", "2025-04-10")
    reams = check_order([make_order_line("copy paper", "Standard copy paper", 2, "reams")], opening_date, deadline)
    wrong_size = check_order([make_order_line("A3 glossy paper", "Glossy paper", 100, "sheets", "A3")], opening_date, deadline)
    packets = check_order([make_order_line("envelopes", "Envelopes", 50, "packets")], opening_date, deadline)
    not_sold = check_order([make_order_line("balloons", "", 200, "each")], opening_date, deadline)
    literal_name = check_order([make_order_line("colorful construction paper", "Colored paper", 500)], opening_date, deadline)
    printer_wrong_match = check_order([make_order_line("white printer paper", "A4 paper", 300)], opening_date, deadline)
    printing_no_match = check_order([make_order_line("standard printing paper", "", 1000)], opening_date, deadline)
    printer_a4 = check_order(
        [make_order_line("A4 size printer paper", "Standard copy paper", 250, "sheets", "A4")], opening_date, deadline
    )
    printer_a3 = check_order(
        [make_order_line("A3 printer paper", "A4 paper", 100, "sheets", "A3")], opening_date, deadline
    )
    mixed_order = check_order(
        [make_order_line("cardstock", "Cardstock", 100), make_order_line("balloons", "", 200, "each")],
        opening_date,
        deadline,
    )

    checks = [
        (
            "list_catalog_stock lists all 46 catalog items",
            catalog["status"] == "ok" and catalog["item_count"] == 46,
        ),
        (
            "stock covers the order: complete, available on the request date",
            "available_from: 2025-04-01" in in_stock and "ORDER COMPLETE: yes" in in_stock,
        ),
        (
            "a shortfall restocked by 04-05: complete, available from that date",
            "available_from: 2025-04-05" in restock_in_time and "ORDER COMPLETE: yes" in restock_in_time,
        ),
        (
            "a restock arriving 04-12 misses a 04-10 deadline: not complete",
            "available_from: 2025-04-12" in restock_too_late
            and "after the deadline 2025-04-10" in restock_too_late
            and "ORDER COMPLETE: no" in restock_too_late,
        ),
        (
            "the report gives no stock counts and no supplier quantities",
            all(
                hidden not in report
                for report in (in_stock, restock_in_time, restock_too_late)
                for hidden in ("units_on_hand", "shortfall", "restock", "228", "272", "595")
            ),
        ),
        (
            "2 reams are converted to 1000 sheets",
            "quantity_units: 1000" in reams,
        ),
        (
            "a size we do not carry (A3) is reported as NOT SOLD",
            "catalog_item: NOT SOLD" in wrong_size and "ORDER COMPLETE: no" in wrong_size,
        ),
        (
            "a quantity in packets is reported as QUANTITY UNCLEAR",
            "QUANTITY UNCLEAR" in packets and "ORDER COMPLETE: no" in packets,
        ),
        (
            "an item with no catalog match is reported as NOT SOLD",
            "catalog_item: NOT SOLD" in not_sold and "ORDER COMPLETE: no" in not_sold,
        ),
        (
            "a catalog name in the customer's words wins over a wrong match",
            "catalog_item: Construction paper" in literal_name,
        ),
        (
            "'white printer paper' is Standard copy paper, even when the agent proposes A4 paper",
            "catalog_item: Standard copy paper" in printer_wrong_match,
        ),
        (
            "'standard printing paper' is Standard copy paper, even when the agent proposes nothing",
            "catalog_item: Standard copy paper" in printing_no_match,
        ),
        (
            "'A4 size printer paper' is A4 paper, even when the agent proposes Standard copy paper",
            "catalog_item: A4 paper" in printer_a4,
        ),
        (
            "'A3 printer paper' is still refused for its size",
            "catalog_item: NOT SOLD" in printer_a3 and "A3 size" in printer_a3,
        ),
        (
            "one unsuppliable item makes the whole order not complete",
            "ITEMS THAT CAN BE SUPPLIED: 1 of 2" in mixed_order and "ORDER COMPLETE: no" in mixed_order,
        ),
        (
            "check_min_stock: nothing is low at opening stock",
            project.check_min_stock(opening_date)["low_stock_count"] == 0,
        ),
    ]

    all_passed = True
    for description, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {description}")
        all_passed = all_passed and passed
    return all_passed


def load_test_requests() -> list:
    """
    Build every request text exactly as run_test_scenarios() will send it.

    Returns:
        list: One dict per sample request with 'number', 'request_text' (what
              the agent receives) and 'display_text' (the same on one line).
    """
    sample = pd.read_csv("quote_requests_sample.csv")
    sample["request_date"] = pd.to_datetime(sample["request_date"], format="%m/%d/%y")

    test_requests = []
    for row_position, row in enumerate(sample.to_dict("records")):
        request_date = row["request_date"].strftime("%Y-%m-%d")
        request_text = f"{row['request']} (Date of request: {request_date})"
        test_requests.append({
            "number": row_position + 1,
            "request_text": request_text,
            "display_text": " ".join(request_text.split()),
        })
    return test_requests


def collect_order_tool_calls(agent) -> list:
    """
    Gather what the agent sent to, and got back from, the order tool.

    Args:
        agent: The agent that has just finished a run.

    Returns:
        list: One dict per call with 'order_lines' (list) and 'report' (str).
    """
    calls = []
    for step in agent.memory.steps:
        for tool_call in getattr(step, "tool_calls", None) or []:
            if tool_call.name == ORDER_TOOL_NAME:
                arguments = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
                calls.append({
                    "order_lines": arguments.get("order_lines") or [],
                    "report": str(getattr(step, "observations", "") or ""),
                })
    return calls


def same_text(first: str, second: str) -> bool:
    """Compare two texts while ignoring differences in spacing and line breaks."""
    return " ".join(str(first).split()) == " ".join(str(second).split())


def run_agent_checks(project) -> list:
    """
    Send every sample request to the Inventory agent and print what it did.

    Args:
        project: The imported project_starter module.

    Returns:
        list: One scoreboard row per request.
    """
    agent = project.inventory_agent
    # Keep the console readable: this script prints its own summary instead.
    agent.logger.level = LogLevel.ERROR

    scoreboard = []
    for test_request in load_test_requests():
        number = test_request["number"]
        expected_complete = number in EXPECTED_COMPLETE_REQUESTS
        print_heading(f"Sample request {number}")
        print(test_request["display_text"])

        try:
            final_answer = str(agent.run(
                "Prepare an availability report for this customer request:\n\n"
                + test_request["request_text"]
            ))
        except Exception as error:  # keep going so one failure does not hide the rest
            print(f"\n--- The agent run failed: {error}")
            scoreboard.append({"number": number, "expected": expected_complete, "actual": None, "copied": False})
            continue

        tool_calls = collect_order_tool_calls(agent)
        print(f"\n--- Order lines the agent sent ({len(tool_calls)} call(s) to {ORDER_TOOL_NAME}) ---")
        for tool_call in tool_calls:
            for line in tool_call["order_lines"]:
                if isinstance(line, dict):
                    print(
                        f"  {line.get('requested')!r} -> item_name={line.get('item_name')!r}, "
                        f"quantity={line.get('quantity')!r}, unit={line.get('unit')!r}, "
                        f"sheet_size={line.get('sheet_size')!r}"
                    )
                else:
                    print(f"  (not a proper line) {line!r}")

        tool_report = tool_calls[-1]["report"] if tool_calls else ""
        copied_unchanged = bool(tool_report) and same_text(tool_report, final_answer)
        print("\n--- Final answer ---")
        print(final_answer)
        if not copied_unchanged:
            print("\n--- NOTE: the final answer differs from the tool's report, which was ---")
            print(tool_report or "(the tool was not called)")

        actual_complete = "ORDER COMPLETE: yes" in final_answer
        scoreboard.append({
            "number": number,
            "expected": expected_complete,
            "actual": actual_complete,
            "copied": copied_unchanged,
        })
    return scoreboard


def print_scoreboard(scoreboard: list) -> None:
    """
    Print one line per request and the totals.

    Args:
        scoreboard (list): Rows produced by run_agent_checks.
    """
    def verdict(value) -> str:
        """Turn True/False/None into a short label."""
        return {True: "complete", False: "not complete", None: "run failed"}[value]

    correct = 0
    copied = 0
    for row in scoreboard:
        matches = row["actual"] == row["expected"]
        correct += matches
        copied += row["copied"]
        print(
            f"[{'OK' if matches else 'XX'}] request {row['number']:>2}: "
            f"expected {verdict(row['expected'])}, got {verdict(row['actual'])}"
            f"{'' if row['copied'] else '  (report was changed or missing)'}"
        )
    print(f"\nORDER COMPLETE verdict correct: {correct} of {len(scoreboard)}")
    print(f"Tool report passed on unchanged: {copied} of {len(scoreboard)}")


def main() -> None:
    """Run the environment check, the tool checks, then the agent checks."""
    check_environment()

    try:
        import project_starter as project
    except RuntimeError as error:  # raised when the API key is missing
        sys.exit(f"Could not load project_starter.py: {error}")

    print(f"Model: {project.MODEL_ID}")
    project.init_database(project.db_engine)

    print_heading("PART 1 - inventory tools (no API calls)")
    if not run_tool_checks(project):
        sys.exit("A tool check failed, so the agent test was skipped.")

    print_heading("PART 2 - Inventory agent on all 20 sample requests (uses the API)")
    scoreboard = run_agent_checks(project)

    print_heading("SCOREBOARD")
    print_scoreboard(scoreboard)
    print("\nSend back the whole output for review.")


if __name__ == "__main__":
    main()
