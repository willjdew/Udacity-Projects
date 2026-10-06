"""
Stand-alone check of the Sales agent, for the Udacity workspace.

Run it from the folder that holds project_starter.py and the CSV files:

    python test_sales_agent.py

Part 1 checks the three sales tools directly (no API calls).
Part 2 gives the Sales agent eleven tasks (this uses the API key in your
.env file): seven orders to record, three top-up tasks and one financial
summary. Every task starts from a freshly reset practice database. Sales
only records an order that has been through the Inventory availability
check and found complete, so before each order task the script runs that
check itself, as the Inventory agent would have. The script first runs the
tool itself to learn the correct outcome, sets everything up again, and
then lets the agent do the same task. For each task it checks that the agent
called the right tool once, passed the items and dates unchanged, returned
the tool's report unchanged, and left the database exactly as the tool alone
would have.

Supplier costs and cash figures are never in a report. The tools print them
to the console as 'INTERNAL LOG' lines, which appear above each task's
results in Part 2.

Nothing here is part of the graded submission. It resets the practice
database (munder_difflin.db) many times while it runs.
"""

import contextlib
import io
import sys

import pandas as pd
import smolagents
from smolagents import LogLevel

MINIMUM_SMOLAGENTS_VERSION = (1, 20)

ORDER_TOOL_NAME = "fulfill_order"
RESTOCK_TOOL_NAME = "buy_restock"
SNAPSHOT_TOOL_NAME = "financial_snapshot"
SALES_TOOL_NAMES = (ORDER_TOOL_NAME, RESTOCK_TOOL_NAME, SNAPSHOT_TOOL_NAME)

# Orders to record: (label, request date, deadline, items as (catalog
# item_name, quantity in units), items the availability check covered).
# None in the last place means the check covered the same items. The first
# three are complete orders from the sample requests; the last four must be
# refused.
ORDER_TEST_CASES = [
    ("sample request 1: everything is in stock", "2025-04-01", "2025-04-15",
     [("Glossy paper", 200), ("Cardstock", 100), ("Colored paper", 100)], None),
    ("sample request 5: one item must be bought in", "2025-04-05", "2025-04-15",
     [("Colored paper", 500), ("Cardstock", 300), ("Decorative adhesive tape (washi tape)", 200)], None),
    ("sample request 12: two items must be bought in", "2025-04-08", "2025-04-15",
     [("Cardstock", 200), ("Standard copy paper", 500), ("Paper napkins", 100)], None),
    ("refusal: the restock would arrive after the deadline", "2025-04-01", "2025-04-03",
     [("Cardstock", 5000)], None),
    ("refusal: one item is not in the catalog", "2025-04-01", "2025-04-15",
     [("Glossy paper", 100), ("Balloons", 50)], None),
    ("refusal: the restock costs more than the cash", "2025-04-01", "2025-04-30",
     [("Notepads", 30000)], None),
    ("refusal: only the in-stock part of a declined order is sent", "2025-04-01", "2025-04-15",
     [("Glossy paper", 100)], [("Glossy paper", 100), ("Balloons", 50)]),
]

# Top-up tasks: (label, items as (catalog item_name, quantity in units)).
# Before each one, the items in SOLD_OUT_ITEMS are sold out, so those two
# (and only those two) are below their minimum stock level.
RESTOCK_DATE = "2025-04-01"
SOLD_OUT_ITEMS = ("Glossy paper", "Cardstock")
RESTOCK_TEST_CASES = [
    ("top-ups: the two low items", [("Cardstock", 500), ("Glossy paper", 500)]),
    ("top-ups: a quantity that is too high and an item that is not low", [("Cardstock", 5000), ("Photo paper", 500)]),
    ("top-ups: the task says nothing is low", []),
]

# A deadline far enough away that set-up sales are never refused for time.
FAR_DEADLINE = "2025-12-31"


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


# ---------------------------------------------------------------------------
# Small helpers shared by both parts
# ---------------------------------------------------------------------------

def to_lines(items: list) -> list:
    """Turn (item_name, quantity) pairs into the lines the sales tools expect."""
    return [{"item_name": item_name, "quantity": quantity} for item_name, quantity in items]


def quietly(function, *args, **kwargs):
    """Call a function while hiding the starter code's debug prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def to_cents(amount: float) -> int:
    """Convert dollars to whole cents so amounts can be compared exactly."""
    return int(round(float(amount) * 100))


def units_on_hand(project, item_name: str, as_of_date: str) -> int:
    """Read the stock of one item straight from the starter helper."""
    return int(project.get_stock_level(item_name, as_of_date)["current_stock"].iloc[0])


def count_rows(project) -> int:
    """Count the rows in the transactions table."""
    return int(pd.read_sql("SELECT COUNT(*) AS row_count FROM transactions", project.db_engine)["row_count"].iloc[0])


def rows_after(project, first_new_row: int) -> list:
    """
    List the transactions written after a starting point, in a comparable form.

    Args:
        project: The imported project_starter module.
        first_new_row (int): The row count before the writes being checked.

    Returns:
        list: Sorted (item_name, type, units, price in cents, date) tuples.
    """
    transactions = pd.read_sql("SELECT * FROM transactions", project.db_engine).iloc[first_new_row:]
    return sorted(
        (str(row["item_name"]), str(row["transaction_type"]), int(row["units"]),
         to_cents(row["price"]), str(row["transaction_date"]))
        for _, row in transactions.iterrows()
    )


def describe_rows(rows: list) -> str:
    """Summarize written rows as 'N purchase(s), M sale(s)'."""
    purchases = sum(1 for row in rows if row[1] == "stock_orders")
    sales = sum(1 for row in rows if row[1] == "sales")
    return f"{purchases} purchase(s), {sales} sale(s)"


def run_availability_check(project, items: list, request_date: str, deadline: str) -> str:
    """
    Run the Inventory availability check on an order, without the agent.

    Sales only records an order that this check has found complete, so the
    script runs it wherever the Inventory agent would have gone first.

    Args:
        project: The imported project_starter module.
        items (list): (catalog item_name, quantity in units) pairs.
        request_date (str): The request date as YYYY-MM-DD.
        deadline (str): The delivery deadline as YYYY-MM-DD.

    Returns:
        str: The availability report.
    """
    checked_lines = [
        {"requested": item_name, "item_name": item_name, "quantity": quantity, "unit": "each", "sheet_size": ""}
        for item_name, quantity in items
    ]
    return quietly(
        project.check_order_availability,
        order_lines=checked_lines, request_date=request_date, deadline=deadline,
    )


def reset_database(project, sell_out_first: bool = False) -> None:
    """
    Put the practice database back to opening stock, with no order checked.

    Args:
        project: The imported project_starter module.
        sell_out_first (bool): When True, also sell every unit of the items
            in SOLD_OUT_ITEMS, so that they fall below their minimum.
    """
    quietly(project.init_database, project.db_engine)
    project.start_new_request()
    if sell_out_first:
        sold_out_items = [(item_name, units_on_hand(project, item_name, RESTOCK_DATE)) for item_name in SOLD_OUT_ITEMS]
        run_availability_check(project, sold_out_items, RESTOCK_DATE, FAR_DEADLINE)
        quietly(
            project.fulfill_order,
            order_lines=to_lines(sold_out_items), request_date=RESTOCK_DATE, deadline=FAR_DEADLINE,
        )


def squeeze(text: str) -> str:
    """Collapse all spacing so texts can be compared regardless of line breaks."""
    return " ".join(str(text).split())


# ---------------------------------------------------------------------------
# Part 1: the tools on their own
# ---------------------------------------------------------------------------

def run_tool_checks(project) -> bool:
    """
    Call each sales tool directly and compare with known answers.

    Args:
        project: The imported project_starter module.

    Returns:
        bool: True when every check passed.
    """
    day = "2025-04-01"
    checks = []

    def order(items: list, deadline: str = FAR_DEADLINE, checked_items: list = None) -> str:
        """
        Check availability, then record an order for the test day.

        The availability check covers the same items unless checked_items
        says otherwise; pass an empty list to skip the check altogether.
        """
        if checked_items is None:
            checked_items = items
        if checked_items:
            run_availability_check(project, checked_items, day, deadline)
        return quietly(project.fulfill_order, order_lines=to_lines(items), request_date=day, deadline=deadline)

    def top_up(items: list) -> str:
        """Buy top-ups on the test day and return the restock report."""
        return quietly(project.buy_restock, items=to_lines(items), request_date=day)

    def cash_cents() -> int:
        """Read the cash balance on the test day, in cents."""
        return to_cents(project.get_cash_balance(day))

    # An order covered by stock: three sales, cash up by the quoted $65.00.
    reset_database(project)
    cash_before, rows_before = cash_cents(), count_rows(project)
    report = order([("Glossy paper", 200), ("Cardstock", 100), ("Colored paper", 100)])
    checks.append((
        "order from stock: recorded, three sales written, cash up by exactly $65.00",
        "ORDER RECORDED: yes" in report and "ORDER TOTAL: $65.00" in report
        and describe_rows(rows_after(project, rows_before)) == "0 purchase(s), 3 sale(s)"
        and cash_cents() - cash_before == 6500,
    ))

    # An order 300 units above stock: the exact shortfall is bought first.
    reset_database(project)
    cardstock_on_hand = units_on_hand(project, "Cardstock", day)
    rows_before = count_rows(project)
    report = order([("Cardstock", cardstock_on_hand + 300)])
    new_rows = rows_after(project, rows_before)
    checks.append((
        "order above stock: buys exactly the 300-unit shortfall for $45.00, then sells; stock ends at 0",
        "ORDER RECORDED: yes" in report and describe_rows(new_rows) == "1 purchase(s), 1 sale(s)"
        and ("Cardstock", "stock_orders", 300, 4500, day) in new_rows
        and units_on_hand(project, "Cardstock", day) == 0 and "DELIVERY DATE: 2025-04-05" in report,
    ))
    checks.append((
        "the report of that order says nothing about the restock or the cash",
        "restock" not in report.lower() and "cash" not in report.lower() and report.endswith("ORDER RECORDED: yes"),
    ))

    # Refusals must leave the database untouched.
    refusal_cases = [
        ("a restock that would arrive after the deadline", [("Cardstock", 5000)], "2025-04-03", "after the deadline 2025-04-03"),
        ("one late item among items in stock", [("Glossy paper", 100), ("Cardstock", 5000)], "2025-04-03", "Cardstock (5000 units)"),
        ("an item that is not in the catalog", [("Glossy paper", 100), ("Balloons", 50)], FAR_DEADLINE, "cannot be priced"),
        ("a restock that costs more than the cash", [("Notepads", 30000)], FAR_DEADLINE, "cannot buy in the stock"),
    ]
    for description, items, deadline, expected_reason in refusal_cases:
        reset_database(project)
        rows_before = count_rows(project)
        report = order(items, deadline)
        checks.append((
            f"refused, nothing written: {description}",
            "ORDER RECORDED: no" in report and expected_reason in report and count_rows(project) == rows_before
            and "$" not in report,
        ))

    # Only the complete order that the Inventory check approved is recorded.
    guard_cases = [
        ("an order that never went through the availability check",
         [("Glossy paper", 100)], [], "has not been through an availability check"),
        ("the in-stock part of an order that was declined",
         [("Glossy paper", 100)], [("Glossy paper", 100), ("Balloons", 50)], "no part of it can be recorded"),
        ("a quantity that differs from the one that was checked",
         [("Glossy paper", 150)], [("Glossy paper", 100)], "not the order that was checked"),
    ]
    for description, items, checked_items, expected_reason in guard_cases:
        reset_database(project)
        rows_before = count_rows(project)
        report = order(items, checked_items=checked_items)
        checks.append((
            f"refused, nothing written: {description}",
            "ORDER RECORDED: no" in report and expected_reason in report and count_rows(project) == rows_before,
        ))

    reset_database(project)
    rows_before = count_rows(project)
    first_report = order([("Glossy paper", 100)])
    second_report = order([("Glossy paper", 100)], checked_items=[])
    checks.append((
        "an order is recorded once: a second call for the same order is refused",
        "ORDER RECORDED: yes" in first_report and "already been recorded" in second_report
        and describe_rows(rows_after(project, rows_before)) == "0 purchase(s), 1 sale(s)",
    ))

    # Top-ups: only items that are really low, never more than recommended.
    reset_database(project)
    rows_before = count_rows(project)
    report = top_up([("Cardstock", 500)])
    checks.append((
        "top-up refused for an item that is not low",
        "ITEMS BOUGHT: 0 of 1" in report and count_rows(project) == rows_before,
    ))

    reset_database(project, sell_out_first=True)
    cash_before, rows_before = cash_cents(), count_rows(project)
    report = top_up([("Glossy paper", 500), ("Cardstock", 5000)])
    checks.append((
        "top-ups for the two low items: both bought for $175.00, 5,000 cut back to 500, cheapest first",
        "ITEMS BOUGHT: 2 of 2" in report and cash_before - cash_cents() == 17500
        and "Quantity reduced from 5000 to the recommended top-up of 500." in report
        and report.index("Cardstock") < report.index("Glossy paper")
        and rows_after(project, rows_before) == [
            ("Cardstock", "stock_orders", 500, 7500, day),
            ("Glossy paper", "stock_orders", 500, 10000, day),
        ],
    ))

    # With $80 in the bank, only the cheaper of the two top-ups can be bought.
    reset_database(project, sell_out_first=True)
    project.create_transaction("A4 paper", "stock_orders", 0, project.get_cash_balance(day) - 80, day)
    report = top_up([("Glossy paper", 500), ("Cardstock", 500)])
    checks.append((
        "top-ups with $80 of cash: the $75 one is bought, the $100 one is skipped",
        "ITEMS BOUGHT: 1 of 2" in report and "Not enough cash for this top-up." in report and cash_cents() == 500,
    ))
    checks.append((
        "the restock report has no costs or cash figures in it",
        "$" not in report,
    ))

    # The snapshot matches the starter's own report and hides the opening cash row.
    reset_database(project)
    snapshot = project.financial_snapshot(day)
    full_report = project.generate_financial_report(day)
    checks.append((
        "financial snapshot matches the starter report and lists no sales before any sale",
        snapshot["status"] == "ok" and to_cents(snapshot["cash_balance"]) == to_cents(full_report["cash_balance"])
        and to_cents(snapshot["inventory_value"]) == to_cents(full_report["inventory_value"])
        and snapshot["top_selling_products"] == [],
    ))

    all_passed = True
    for description, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {description}")
        all_passed = all_passed and passed
    return all_passed


# ---------------------------------------------------------------------------
# Part 2: the agent
# ---------------------------------------------------------------------------

def item_list_text(items: list) -> str:
    """Write items as the bullet list used in every task."""
    return "\n".join(f"- {item_name}: {quantity} units" for item_name, quantity in items)


def build_test_cases() -> list:
    """
    Write every task the way the orchestrator will hand it to the Sales agent.

    Returns:
        list: One dict per task with 'label', 'task' (the text the agent
              receives), 'tool' (the tool it should call, or None), 'arguments'
              (what that tool should receive), 'sell_out_first' (whether
              the low-stock set-up is needed) and 'checked_order' (the order
              the availability check covers during set-up, or None).
    """
    test_cases = []
    for label, request_date, deadline, items, checked_items in ORDER_TEST_CASES:
        test_cases.append({
            "label": label,
            "task": (
                "Record this complete order.\n"
                f"Items:\n{item_list_text(items)}\n"
                f"Request date: {request_date}\n"
                f"Delivery deadline: {deadline}"
            ),
            "tool": ORDER_TOOL_NAME,
            "arguments": {"order_lines": to_lines(items), "request_date": request_date, "deadline": deadline},
            "sell_out_first": False,
            "checked_order": (checked_items or items, request_date, deadline),
        })

    for label, items in RESTOCK_TEST_CASES:
        if items:
            task = (
                "Buy these low-stock top-ups.\n"
                f"Items:\n{item_list_text(items)}\n"
                f"Request date: {RESTOCK_DATE}"
            )
        else:
            task = (
                "Buy the low-stock top-ups.\n"
                "Items: none. The Inventory agent found no items at or below their minimum stock level.\n"
                f"Request date: {RESTOCK_DATE}"
            )
        test_cases.append({
            "label": label,
            "task": task,
            "tool": RESTOCK_TOOL_NAME if items else None,
            "arguments": {"items": to_lines(items), "request_date": RESTOCK_DATE},
            "sell_out_first": bool(items),
            "checked_order": None,
        })

    test_cases.append({
        "label": "internal financial summary",
        "task": f"Give the internal financial summary.\nRequest date: {RESTOCK_DATE}",
        "tool": SNAPSHOT_TOOL_NAME,
        "arguments": {"as_of_date": RESTOCK_DATE},
        "sell_out_first": False,
        "checked_order": None,
    })
    return test_cases


def set_up_task(project, test_case: dict) -> None:
    """
    Put the database and the availability check in the state a task starts from.

    Args:
        project: The imported project_starter module.
        test_case (dict): One task from build_test_cases.
    """
    reset_database(project, test_case["sell_out_first"])
    if test_case["checked_order"] is not None:
        run_availability_check(project, *test_case["checked_order"])


def run_tool_directly(project, test_case: dict):
    """
    Run the task's tool without the agent, to learn the correct outcome.

    Args:
        project: The imported project_starter module.
        test_case (dict): One task from build_test_cases.

    Returns:
        The tool's result: a report (str), a snapshot (dict), or None when
        the task calls for no tool.
    """
    if test_case["tool"] is None:
        return None
    tool = getattr(project, test_case["tool"])
    return quietly(tool, **test_case["arguments"])


def collect_tool_calls(agent) -> list:
    """
    Gather the calls the agent made to the three sales tools.

    Args:
        agent: The agent that has just finished a run.

    Returns:
        list: One dict per call with 'name' and 'arguments'.
    """
    calls = []
    for step in agent.memory.steps:
        for tool_call in getattr(step, "tool_calls", None) or []:
            if tool_call.name in SALES_TOOL_NAMES:
                arguments = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
                calls.append({"name": tool_call.name, "arguments": arguments})
    return calls


def same_lines(sent_lines, expected_lines: list) -> bool:
    """Check that the agent sent exactly the items it was given, in any order."""
    def as_pairs(lines) -> list:
        """Reduce lines to sorted (item_name, quantity) pairs."""
        pairs = []
        for line in lines if isinstance(lines, list) else []:
            if not isinstance(line, dict):
                return [("not a proper line", -1)]
            try:
                quantity = int(float(str(line.get("quantity", "")).replace(",", "")))
            except ValueError:
                quantity = -1
            pairs.append((str(line.get("item_name")), quantity))
        return sorted(pairs)

    return as_pairs(sent_lines) == as_pairs(expected_lines)


def same_arguments(sent: dict, expected: dict) -> bool:
    """Check that items and dates reached the tool exactly as they were given."""
    for name, expected_value in expected.items():
        if isinstance(expected_value, list):
            if not same_lines(sent.get(name), expected_value):
                return False
        elif str(sent.get(name, "")).strip()[:10] != expected_value:
            return False
    return True


def snapshot_in_answer(snapshot: dict, final_answer: str) -> bool:
    """Check that the summary states the three main figures the tool returned."""
    answer = final_answer.replace(",", "")
    figures = [snapshot["cash_balance"], snapshot["inventory_value"], snapshot["total_assets"]]
    # 45059.7 matches '45059.7', '45059.70' and '$45,059.70' alike.
    return all(f"{figure:.2f}".rstrip("0").rstrip(".") in answer for figure in figures)


def run_agent_checks(project) -> list:
    """
    Give every task to the Sales agent and print what it did.

    Args:
        project: The imported project_starter module.

    Returns:
        list: One scoreboard row per task.
    """
    agent = project.sales_agent
    # Keep the console readable: this script prints its own summary instead.
    agent.logger.level = LogLevel.ERROR

    scoreboard = []
    for number, test_case in enumerate(build_test_cases(), start=1):
        print_heading(f"Task {number} - {test_case['label']}")
        print(test_case["task"])

        # Learn the correct outcome by running the tool itself.
        set_up_task(project, test_case)
        rows_before = count_rows(project)
        expected_result = run_tool_directly(project, test_case)
        expected_rows = rows_after(project, rows_before)

        # Start again from the same state and let the agent do the task.
        set_up_task(project, test_case)
        row = {"number": number, "label": test_case["label"], "calls": False, "inputs": False, "report": False, "database": False}
        try:
            final_answer = str(agent.run(test_case["task"]))
        except Exception as error:  # keep going so one failure does not hide the rest
            print(f"\n--- The agent run failed: {error}")
            row["database"] = rows_after(project, rows_before) == expected_rows
            scoreboard.append(row)
            continue

        calls = collect_tool_calls(agent)
        actual_rows = rows_after(project, rows_before)
        expected_tool = test_case["tool"]

        if expected_tool is None:
            # No tool should run, so there is nothing to pass on or copy.
            row["calls"] = not calls
            row["inputs"] = row["calls"]
            row["report"] = row["calls"]
        else:
            row["calls"] = [call["name"] for call in calls] == [expected_tool]
            matching_calls = [call for call in calls if call["name"] == expected_tool]
            row["inputs"] = bool(matching_calls) and same_arguments(matching_calls[0]["arguments"], test_case["arguments"])
            if isinstance(expected_result, dict):
                row["report"] = snapshot_in_answer(expected_result, final_answer)
            else:
                row["report"] = squeeze(expected_result) in squeeze(final_answer)
        row["database"] = actual_rows == expected_rows

        print(f"\n--- Tool calls ({len(calls)}) ---")
        for call in calls:
            print(f"  {call['name']}: {call['arguments']}")
        print("\n--- Final answer ---")
        print(final_answer)
        expected_call = f"{expected_tool} called once" if expected_tool else "no tool call"
        print(f"\n--- Expected: {expected_call}; rows written: {describe_rows(expected_rows)} ---")
        if not row["calls"]:
            print("NOTE: the agent did not make exactly the expected tool call.")
        if expected_tool is not None and not row["inputs"]:
            print(f"NOTE: the tool should have received: {test_case['arguments']}")
        if expected_tool is not None and not row["report"]:
            print("NOTE: the final answer does not contain the tool's result unchanged, which was:")
            print(expected_result)
        if not row["database"]:
            print(f"NOTE: the database differs. Expected rows: {expected_rows}")
            print(f"      Rows the agent's run wrote: {actual_rows}")
        scoreboard.append(row)
    return scoreboard


def print_scoreboard(scoreboard: list) -> None:
    """
    Print one line per task and the totals.

    Args:
        scoreboard (list): Rows produced by run_agent_checks.
    """
    labels = [
        ("calls", "right tool called exactly once"),
        ("inputs", "items and dates passed unchanged"),
        ("report", "tool result returned unchanged"),
        ("database", "database left as expected"),
    ]
    for row in scoreboard:
        all_good = all(row[key] for key, _ in labels)
        problems = ", ".join(f"NOT {label}" for key, label in labels if not row[key])
        print(f"[{'OK' if all_good else 'XX'}] task {row['number']:>2} ({row['label']}){': ' + problems if problems else ''}")
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

    print_heading("PART 1 - sales tools (no API calls)")
    if not run_tool_checks(project):
        sys.exit("A tool check failed, so the agent test was skipped.")

    test_case_count = len(ORDER_TEST_CASES) + len(RESTOCK_TEST_CASES) + 1
    print_heading(f"PART 2 - Sales agent on {test_case_count} tasks (uses the API)")
    scoreboard = run_agent_checks(project)

    print_heading("SCOREBOARD")
    print_scoreboard(scoreboard)
    print("\nSend back the whole output for review.")


if __name__ == "__main__":
    main()
