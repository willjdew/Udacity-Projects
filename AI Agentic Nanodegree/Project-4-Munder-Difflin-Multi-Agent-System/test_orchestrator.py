"""
Stand-alone check of the Orchestrator, for the Udacity workspace.

Run it from the folder that holds project_starter.py and the CSV files:

    python test_orchestrator.py

Part 1 checks how the Orchestrator is wired up and that its completion check
names the right missing step (no API calls).
Part 2 sends ten of the sample requests through the whole system (this uses
the API key in your .env file, and takes several minutes): the six that can
be filled in full at opening stock, and four that must be declined for
different reasons. Every request starts from a freshly reset practice
database, so the right outcome is known in advance. The script works that
outcome out with the tools alone, resets the database, and then lets the
Orchestrator handle the request. For each request it prints every task the
Orchestrator gave to a worker, the worker's answer, each time the completion
check sent a final answer back, the reply to the customer, and who wrote it.
The Orchestrator writes the confirmation of a recorded order. The reply to an
order that was not placed is written in code, from the tools' record.

The scoreboard has six checks: the Inventory agent received the customer's
request word for word; the workers were called in the workflow's order; the
database ended up exactly as expected; the reply gives the right outcome;
the reply states the details it must (total and delivery date for a recorded
order, the partial-order total for a declined one); and no internal wording
is in the reply.

Read the replies as well as the scoreboard: the script can check that a
reply contains the right total and date, but not whether it reads well.

Nothing here is part of the graded submission. It resets the practice
database (munder_difflin.db) many times while it runs.
"""

import contextlib
import io
import json
import sys

import pandas as pd
import smolagents
from smolagents import LogLevel

MINIMUM_SMOLAGENTS_VERSION = (1, 20)

INVENTORY = "inventory_agent"
QUOTING = "quoting_agent"
SALES = "sales_agent"
WORKER_NAMES = (INVENTORY, QUOTING, SALES)

# Sample requests (1-based) that can be filled in full at opening stock, with
# the catalog items and unit quantities that should be recorded for each.
# These are the items the Inventory agent found when it was tested.
EXPECTED_ORDERS = {
    1: [("Glossy paper", 200), ("Cardstock", 100), ("Colored paper", 100)],
    4: [("Cardstock", 500), ("A4 paper", 250)],
    5: [("Colored paper", 500), ("Cardstock", 300), ("Decorative adhesive tape (washi tape)", 200)],
    6: [("Construction paper", 500), ("Standard copy paper", 300), ("Cardstock", 200)],
    10: [("Glossy paper", 500), ("Cardstock", 300)],
    12: [("Cardstock", 200), ("Standard copy paper", 500), ("Paper napkins", 100)],
}

# Every complete sample request asks for delivery by this date.
EXPECTED_ORDER_DEADLINE = "2025-04-15"

# Sample requests that must be declined, and why.
DECLINED_REQUESTS = {
    2: "balloons are not in the catalog",
    3: "A3 paper is a size we do not carry",
    9: "A3 size not carried, and envelopes asked for in packets",
    13: "the A4 paper restock would arrive after the deadline",
}

# Words that show the customer was told the order cannot be filled.
DECLINE_WORDS = ("unable", "cannot", "can't", "can not", "not able", "unfortunately", "regret", "not possible")

# Text that must never appear in a reply to a customer: report field names,
# agent names, and internal stock or money matters.
INTERNAL_MARKERS = (
    "units_on_hand", "units on hand", "on hand", "shortfall", "catalog_item", "quantity_units",
    "available_from", "can_supply", "history note", "order recorded:", "order complete:",
    "internal", "top-up", "top up", "top_up", "min_stock", "minimum stock", "cash",
    "restock", "supplier", "inventory_agent", "quoting_agent", "sales_agent", "orchestrator",
)


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
# Small helpers
# ---------------------------------------------------------------------------

def to_lines(items: list) -> list:
    """Turn (item_name, quantity) pairs into the lines the sales tools expect."""
    return [{"item_name": item_name, "quantity": quantity} for item_name, quantity in items]


def quietly(function, *args, **kwargs):
    """Call a function while hiding the console prints it makes."""
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def to_cents(amount: float) -> int:
    """Convert dollars to whole cents so amounts can be compared exactly."""
    return int(round(float(amount) * 100))


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


def reset_database(project) -> None:
    """Put the practice database back to opening stock, with no order checked."""
    quietly(project.init_database, project.db_engine)
    project.start_new_request()


def load_test_requests() -> list:
    """
    Build the chosen request texts exactly as run_test_scenarios() will send them.

    Returns:
        list: One dict per chosen sample request with 'number', 'request_date'
              and 'request_text' (what the Orchestrator receives).
    """
    sample = pd.read_csv("quote_requests_sample.csv")
    sample["request_date"] = pd.to_datetime(sample["request_date"], format="%m/%d/%y")

    chosen_numbers = sorted(set(EXPECTED_ORDERS) | set(DECLINED_REQUESTS))
    test_requests = []
    for row_position, row in enumerate(sample.to_dict("records")):
        number = row_position + 1
        if number in chosen_numbers:
            request_date = row["request_date"].strftime("%Y-%m-%d")
            test_requests.append({
                "number": number,
                "request_date": request_date,
                "request_text": f"{row['request']} (Date of request: {request_date})",
            })
    return test_requests


def work_out_expected_outcome(project, test_request: dict) -> dict:
    """
    Use the tools alone, without any agent, to produce the correct outcome.

    For a complete order this checks availability, records the order, and
    then buys the top-up for every item the sale left at or below its minimum. For a request that
    must be declined, nothing is written.

    Args:
        project: The imported project_starter module.
        test_request (dict): One request from load_test_requests.

    Returns:
        dict: 'complete' (bool), 'rows' (the transactions that should be
              written), 'total' (the order total as text such as '65.00', or
              None), 'delivery_date' (YYYY-MM-DD, or None) and
              'low_item_count' (how many top-ups are expected).
    """
    reset_database(project)
    items = EXPECTED_ORDERS.get(test_request["number"])
    if items is None:
        return {"complete": False, "rows": [], "total": None, "delivery_date": None, "low_item_count": 0}

    rows_before = count_rows(project)
    request_date = test_request["request_date"]
    # Sales only records an order that the availability check found complete.
    checked_lines = [
        {"requested": item_name, "item_name": item_name, "quantity": quantity, "unit": "each", "sheet_size": ""}
        for item_name, quantity in items
    ]
    quietly(
        project.check_order_availability,
        order_lines=checked_lines, request_date=request_date, deadline=EXPECTED_ORDER_DEADLINE,
    )
    sales_report = quietly(
        project.fulfill_order,
        order_lines=to_lines(items), request_date=request_date, deadline=EXPECTED_ORDER_DEADLINE,
    )
    low_items = project.find_low_stock_items(request_date)
    if low_items:
        top_ups = [{"item_name": item["item_name"], "quantity": item["top_up_quantity"]} for item in low_items]
        quietly(project.buy_restock, items=top_ups, request_date=request_date)

    total_lines = [line for line in sales_report.splitlines() if line.startswith("ORDER TOTAL: $")]
    date_lines = [line for line in sales_report.splitlines() if line.startswith("DELIVERY DATE: ")]
    return {
        "complete": "ORDER RECORDED: yes" in sales_report,
        "rows": rows_after(project, rows_before),
        "total": total_lines[0].replace("ORDER TOTAL: $", "").replace(",", "") if total_lines else None,
        "delivery_date": date_lines[0].replace("DELIVERY DATE: ", "") if date_lines else None,
        "low_item_count": len(low_items),
    }


# ---------------------------------------------------------------------------
# Part 1: the wiring
# ---------------------------------------------------------------------------

def sales_refuses_unchecked_order(project) -> bool:
    """Check that the Sales tool will not record an order Inventory never checked."""
    reset_database(project)
    rows_before = count_rows(project)
    report = quietly(
        project.fulfill_order,
        order_lines=to_lines([("Glossy paper", 100)]), request_date="2025-04-01", deadline=EXPECTED_ORDER_DEADLINE,
    )
    return "ORDER RECORDED: no" in report and count_rows(project) == rows_before


def run_completion_check_cases(project) -> list:
    """
    Walk two orders through the tools, asking the completion check at each stage.

    No agent is involved: the script calls the tools itself, in workflow
    order, and after each one asks find_reply_problem what is still missing.

    Args:
        project: The imported project_starter module.

    Returns:
        list: (description, passed) pairs.
    """
    day, deadline = "2025-04-08", "2025-04-15"

    def problem(reply: str = "Thank you.") -> str:
        """Ask the completion check about a reply; '' means the reply is ready."""
        return project.find_reply_problem(reply) or ""

    def check_availability(items: list) -> None:
        """Run the availability check on (item_name, quantity) pairs."""
        lines = [
            {"requested": item_name, "item_name": item_name, "quantity": quantity, "unit": "each", "sheet_size": ""}
            for item_name, quantity in items
        ]
        quietly(project.check_order_availability, order_lines=lines, request_date=day, deadline=deadline)

    cases = []

    # A complete order that leaves Glossy paper below its minimum (sample request 10).
    order = [("Glossy paper", 500), ("Cardstock", 300)]
    reset_database(project)
    cases.append(("nothing done yet: step 1 is named", "STEP 1" in problem()))
    check_availability(order)
    cases.append(("after the availability check: step 2 (quote) is named, with the exact item lines to send",
                  "STEP 2" in problem() and "- Glossy paper: 500 units\n- Cardstock: 300 units" in problem()))
    quietly(project.calculate_quote, order_lines=to_lines([("Glossy paper", 500)]))
    cases.append(("a quote that leaves out an item is not accepted", "not for exactly the items" in problem()))
    quietly(project.calculate_quote, order_lines=to_lines([("Glossy paper", 500), ("Cardstock", 3)]))
    cases.append(("a quote with a wrong quantity is not accepted, and the right lines are given again",
                  "not for exactly the items" in problem() and "- Cardstock: 300 units" in problem()))
    quietly(project.calculate_quote, order_lines=to_lines(order))
    cases.append(("complete order quoted: step 3 (record the order) is named, with the items and both dates",
                  "STEP 3" in problem() and "- Cardstock: 300 units" in problem()
                  and f"Delivery deadline: {deadline}" in problem()))
    quietly(project.check_min_stock, day)
    quietly(project.fulfill_order, order_lines=to_lines(order), request_date=day, deadline=deadline)
    cases.append(("order recorded: step 4 is named, and a low-stock check made before the sale does not count",
                  "STEP 4" in problem()))
    quietly(project.check_min_stock, day)
    cases.append(("the low-stock check found an item: step 5 (top-ups) is named, with the top-up to buy",
                  "STEP 5" in problem() and "- Glossy paper: 500 units" in problem()))
    quietly(project.buy_restock, items=to_lines([("Glossy paper", 500)]), request_date=day)
    cases.append(("all steps done: a reply that does not say the order is confirmed is sent back",
                  "must say that the order is confirmed" in problem("Total $137.75, delivery on April 8, 2025.")))
    cases.append(("a confirmation that also says the order was not placed is sent back",
                  "must not say that it was not placed" in problem(
                      "Your order is confirmed but has not been placed yet. Total $137.75 on April 8, 2025.")))
    cases.append(("a confirmation without the total is sent back",
                  "order total, $137.75" in problem("Your order is confirmed for delivery on April 8, 2025.")))
    cases.append(("a confirmation that gives the deadline in place of the delivery date is sent back",
                  "April 8, 2025" in problem("Your order is confirmed. Total $137.75, delivered by April 15, 2025.")))
    cases.append(("a confirmation with a report field name in it is sent back",
                  "catalog_item" in problem("Confirmed. catalog_item: Glossy paper. Total $137.75 on April 8, 2025.")))
    cases.append(("a confirmation with the total and the delivery date is accepted, however the date is written",
                  all(problem(f"Your order is confirmed. The total is $137.75, delivered on {written}.") == ""
                      for written in ("April 8, 2025", "2025-04-08", "8 April 2025", "April 8th, 2025"))))
    code_reply = project.write_reply_in_code()
    cases.append(("the fallback reply for a recorded order confirms it, with prices, the total and the date",
                  problem(code_reply) == "" and "confirmed" in code_reply and "Total: $137.75" in code_reply
                  and "April 8, 2025" in code_reply and "Glossy paper: 500 units at $0.20 each" in code_reply))

    # A declined order with one item that can be supplied.
    reset_database(project)
    check_availability([("Glossy paper", 100), ("Balloons", 50)])
    cases.append(("declined order with a suppliable item: step 2 (quote) is still required, and Sales is ruled out",
                  "STEP 2" in problem() and "- Glossy paper: 100 units" in problem()
                  and "do not give it to the sales_agent" in problem()))
    quietly(project.calculate_quote, order_lines=to_lines([("Glossy paper", 100)]))
    cases.append(("declined and quoted: the Orchestrator may finish, whatever it wrote",
                  problem("ORDER NOT PLACED") == "" and problem("We can supply the order as requested.") == ""))
    code_reply = project.write_reply_in_code()
    cases.append(("the reply written in code for a declined order says nothing was ordered",
                  "not able to fill this order" in code_reply and "nothing has been ordered or charged" in code_reply))
    cases.append(("it gives the reason for the item that cannot be supplied",
                  "- 50 Balloons: This item is not in our catalog." in code_reply))
    cases.append(("it offers the rest as a priced partial order that has not been placed",
                  "Glossy paper: 100 units at $0.20 each, $20.00" in code_reply and "Total: $20.00" in code_reply
                  and "has not been placed" in code_reply))

    # A declined order with nothing that can be supplied needs no quote.
    reset_database(project)
    check_availability([("Balloons", 50)])
    cases.append(("declined order with nothing to supply: no quote is required",
                  problem("ORDER NOT PLACED") == ""))
    code_reply = project.write_reply_in_code()
    cases.append(("with nothing to supply, the reply written in code gives the reason and makes no offer",
                  "- 50 Balloons: This item is not in our catalog." in code_reply and "partial order" not in code_reply))

    # The Inventory agent is given the customer's own words for the availability task.
    request_text = "I need 100 sheets of glossy paper by April 15, 2025. Thank you. (Date of request: 2025-04-08)"
    reset_database(project)
    project.CURRENT_REQUEST["text"] = request_text
    received = project.inventory_task_for("Prepare an availability report: 100 glossy.")
    cases.append(("before an order is recorded, Inventory is given the customer's request word for word",
                  received.endswith(request_text) and "100 glossy." not in received))
    project.REQUEST_PROGRESS["order_recorded"] = True
    low_stock_task = "Low-stock check after a recorded sale.\nRequest date: 2025-04-08"
    cases.append(("after an order is recorded, the low-stock task is passed through unchanged",
                  project.inventory_task_for(low_stock_task) == low_stock_task))
    project.CURRENT_REQUEST["text"] = None
    reset_database(project)
    return cases


def run_wiring_checks(project) -> bool:
    """
    Check how the Orchestrator is set up, without calling the model.

    Args:
        project: The imported project_starter module.

    Returns:
        bool: True when every check passed.
    """
    orchestrator = project.orchestrator
    workers = [getattr(project, name) for name in WORKER_NAMES]

    expected_outcomes = [
        work_out_expected_outcome(project, test_request) for test_request in load_test_requests()
    ]
    recorded = [outcome for outcome in expected_outcomes if outcome["complete"]]

    checks = [
        ("the Orchestrator has the three workers as team members",
         sorted(orchestrator.managed_agents) == sorted(WORKER_NAMES)),
        ("the Orchestrator has no tools of its own (only final_answer)",
         list(orchestrator.tools) == ["final_answer"]),
        ("the system has four agents, within the limit of five",
         len(workers) + 1 <= 5),
        ("each worker receives its task as plain text, with no three-part answer demanded",
         all(worker.prompt_templates["managed_agent"]["task"] == "{{task}}" for worker in workers)),
        ("each worker still has its own instructions and tools",
         all(len(worker.tools) >= 3 and "Munder Difflin" in worker.system_prompt for worker in workers)),
        ("with the tools alone, the six complete sample orders are recorded and the four others write nothing",
         len(recorded) == len(EXPECTED_ORDERS)
         and all(not outcome["rows"] for outcome in expected_outcomes if not outcome["complete"])),
        ("handle_customer_request is available for the test harness",
         callable(getattr(project, "handle_customer_request", None))),
        ("Sales refuses an order that has not been through the availability check",
         sales_refuses_unchecked_order(project)),
        ("the Orchestrator's replies go through the completion check",
         [check.__name__ for check in orchestrator.final_answer_checks] == ["reply_is_ready"]),
    ]
    checks += run_completion_check_cases(project)

    all_passed = True
    for description, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {description}")
        all_passed = all_passed and passed
    return all_passed


# ---------------------------------------------------------------------------
# Part 2: the whole system
# ---------------------------------------------------------------------------

def read_tool_call(tool_call) -> tuple:
    """
    Read the name and arguments of a tool call, in either form smolagents keeps.

    Args:
        tool_call: A call from a step's tool_calls, or from the model's raw
            message when the step was not completed.

    Returns:
        tuple: (name, arguments as a dict).
    """
    function = getattr(tool_call, "function", None)
    name = getattr(tool_call, "name", None) or getattr(function, "name", "")
    arguments = getattr(tool_call, "arguments", None)
    if arguments is None:
        arguments = getattr(function, "arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            arguments = {"task": arguments}
    return name, arguments if isinstance(arguments, dict) else {}


def collect_delegations(orchestrator) -> list:
    """
    Gather every task the Orchestrator gave to a worker, with the answer.

    Args:
        orchestrator: The Orchestrator after it has finished a run.

    Returns:
        list: One dict per step that called a worker, with 'workers' (the
              names called in that step), 'tasks' (the task texts) and
              'answer' (what came back).
    """
    delegations = []
    for step in orchestrator.memory.steps:
        tool_calls = getattr(step, "tool_calls", None)
        answer = str(getattr(step, "observations", "") or "")
        if not tool_calls:
            # A task sent in the same step as a final answer still runs, but
            # smolagents rejects the step and keeps only the model's message.
            message = getattr(step, "model_output_message", None)
            tool_calls = getattr(message, "tool_calls", None) or []
            answer = "(not kept: this task was sent in the same step as a final answer, which smolagents rejects)"

        worker_calls = [read_tool_call(tool_call) for tool_call in tool_calls]
        worker_calls = [(name, arguments) for name, arguments in worker_calls if name in WORKER_NAMES]
        if not worker_calls:
            continue
        delegations.append({
            "workers": [name for name, _ in worker_calls],
            "tasks": [str(arguments.get("task", arguments)) for _, arguments in worker_calls],
            "answer": answer,
        })
    return delegations


def collect_refusals(orchestrator) -> list:
    """
    Gather the messages of the completion check each time it sent a reply back.

    Args:
        orchestrator: The Orchestrator after it has finished a run.

    Returns:
        list: One message per reply that was not accepted.
    """
    refusals = []
    for step in orchestrator.memory.steps:
        error_text = str(getattr(step, "error", None) or "")
        if "reply_is_ready" in error_text:
            refusals.append(error_text.split("failed with error: ", 1)[-1])
    return refusals


def squeeze(text: str) -> str:
    """Collapse all spacing so texts can be compared regardless of line breaks."""
    return " ".join(str(text).split())


def steps_in_order(called: list, expected: dict) -> bool:
    """
    Check that the workers were called in the order the workflow sets.

    Args:
        called (list): Worker names in the order they were called.
        expected (dict): The expected outcome from work_out_expected_outcome.

    Returns:
        bool: True when the order is one the workflow allows.
    """
    if expected["complete"]:
        # The top-ups may be sent as one task or as one task per item.
        allowed = [
            [INVENTORY, QUOTING, SALES, INVENTORY] + [SALES] * top_up_tasks
            for top_up_tasks in range(1, expected["low_item_count"] + 1)
        ]
        if expected["low_item_count"] == 0:
            # With nothing low, there is no top-up step.
            allowed.append([INVENTORY, QUOTING, SALES, INVENTORY])
    else:
        # A declined request is checked, then quoted for the partial order,
        # and never reaches Sales. (Each of the four declined test requests
        # has at least one item that can be supplied.)
        allowed = [[INVENTORY, QUOTING]]
    return called in allowed


def reply_gives_outcome(reply: str, expected: dict) -> bool:
    """
    Check that the reply tells the customer the right outcome.

    Args:
        reply (str): The reply to the customer.
        expected (dict): The expected outcome from work_out_expected_outcome.

    Returns:
        bool: For a recorded order, True when the reply states the order
              total and does not read as a refusal. For a declined request,
              True when the reply says the order cannot be filled.
    """
    text = reply.lower().replace(",", "")
    sounds_declined = any(word in text for word in DECLINE_WORDS)
    if expected["complete"]:
        return expected["total"] in text and "confirm" in text and not sounds_declined
    return sounds_declined


def reply_has_details(project, reply: str, expected: dict) -> bool:
    """
    Check that the reply states the facts the customer needs.

    Args:
        project: The imported project_starter module.
        reply (str): The reply to the customer.
        expected (dict): The expected outcome from work_out_expected_outcome.

    Returns:
        bool: For a recorded order, True when the reply has the expected
              total and delivery date. For a declined request, True when
              the reply has the right total for the partial order (or when
              nothing could be supplied, so there is none to offer).
    """
    text = squeeze(reply.lower().replace(",", ""))
    if expected["complete"]:
        has_date = any(spelling in text for spelling in project.date_spellings(expected["delivery_date"]))
        return expected["total"] in text and has_date
    # The right partial-order total is worked out here, from the items the
    # availability check found could be supplied, not taken from the run.
    suppliable_units = project.LAST_AVAILABILITY_CHECK.get("suppliable_units_by_item")
    if not suppliable_units:
        return True
    partial_order = project.price_order(
        [{"item_name": item_name, "quantity": units} for item_name, units in suppliable_units.items()]
    )
    return f"{partial_order['total']:.2f}" in text


def internal_markers_in(reply: str) -> list:
    """List the internal words or field names found in a reply."""
    text = reply.lower()
    return [marker for marker in INTERNAL_MARKERS if marker in text]


def run_system_checks(project) -> list:
    """
    Send every chosen request through the Orchestrator and print what happened.

    Args:
        project: The imported project_starter module.

    Returns:
        list: One scoreboard row per request.
    """
    # Keep the console readable: this script prints its own summary instead.
    for agent in [project.orchestrator] + [getattr(project, name) for name in WORKER_NAMES]:
        agent.logger.level = LogLevel.ERROR

    scoreboard = []
    for test_request in load_test_requests():
        number = test_request["number"]
        expected = work_out_expected_outcome(project, test_request)
        if expected["complete"]:
            expected_text = (
                f"order recorded, total ${expected['total']}; rows written: {describe_rows(expected['rows'])}"
                f" ({expected['low_item_count']} top-up(s))"
            )
        else:
            expected_text = f"declined ({DECLINED_REQUESTS[number]}); nothing written"

        print_heading(f"Sample request {number} - expected: {expected_text}")
        print(" ".join(test_request["request_text"].split()))

        # Start from opening stock again and let the whole system handle it.
        reset_database(project)
        rows_before = count_rows(project)
        print("\n--- Console log while the system ran ---")
        reply = project.handle_customer_request(test_request["request_text"])
        actual_rows = rows_after(project, rows_before)
        delegations = collect_delegations(project.orchestrator)
        refusals = collect_refusals(project.orchestrator)
        called = [worker for delegation in delegations for worker in delegation["workers"]]
        markers = internal_markers_in(reply)
        reply_source = project.CURRENT_REQUEST["reply_source"]
        fallback_used = reply_source == "code, as a fallback"
        inventory_tasks = project.CURRENT_REQUEST["inventory_tasks"]
        first_task = inventory_tasks[0] if inventory_tasks else ""

        row = {
            "number": number,
            "request": squeeze(test_request["request_text"]) in squeeze(first_task),
            "order": steps_in_order(called, expected),
            "database": actual_rows == expected["rows"],
            "outcome": reply_gives_outcome(reply, expected),
            "details": reply_has_details(project, reply, expected),
            "clean": not markers,
            "refusals": len(refusals),
            "fallback": fallback_used,
        }

        for step_number, delegation in enumerate(delegations, start=1):
            print(f"\n--- Step {step_number}: task given to {' and '.join(delegation['workers'])} ---")
            for task in delegation["tasks"]:
                print(task)
            if step_number == 1 and INVENTORY in delegation["workers"]:
                print("(The Inventory agent is given the customer's request above, word for word, with this task.)")
            print(f"\n--- Step {step_number}: answer ---")
            print(delegation["answer"])
        print(f"\n--- Completion check sent the final answer back {len(refusals)} time(s) ---")
        for refusal in refusals:
            print(f"  {refusal}")
        print(f"\n--- Reply to the customer (written by: {reply_source}) ---")
        print(reply)

        print(f"\n--- Workers called: {', '.join(called) or 'none'} ---")
        print(f"--- Rows written: {describe_rows(actual_rows)} ---")
        if fallback_used:
            print("NOTE: the Orchestrator failed or ran out of steps, so the reply written in code was sent as a fallback.")
        if not row["request"]:
            print("NOTE: the Inventory agent was not given the customer's request word for word. It was given:")
            print(first_task or "(nothing)")
        if not row["details"]:
            print("NOTE: the reply does not state the required details (total and delivery date, or the partial-order total).")
        if not row["order"]:
            print("NOTE: the workers were not called in the order the workflow sets.")
        if not row["database"]:
            print(f"NOTE: the database differs. Expected rows: {expected['rows']}")
            print(f"      Rows this run wrote: {actual_rows}")
        if not row["outcome"]:
            print(f"NOTE: the reply does not give the expected outcome ({expected_text}).")
        if markers:
            print(f"NOTE: the reply contains internal wording: {markers}")
        scoreboard.append(row)
    return scoreboard


def print_scoreboard(scoreboard: list) -> None:
    """
    Print one line per request and the totals.

    Args:
        scoreboard (list): Rows produced by run_system_checks.
    """
    labels = [
        ("request", "Inventory given the request word for word"),
        ("order", "workers called in the right order"),
        ("database", "database left as expected"),
        ("outcome", "reply gives the right outcome"),
        ("details", "reply states the required details"),
        ("clean", "no internal wording in the reply"),
    ]
    for row in scoreboard:
        all_good = all(row[key] for key, _ in labels)
        problems = ", ".join(f"NOT {label}" for key, label in labels if not row[key])
        kind = "complete" if row["number"] in EXPECTED_ORDERS else "declined"
        print(f"[{'OK' if all_good else 'XX'}] request {row['number']:>2} ({kind}){': ' + problems if problems else ''}")
    print()
    for key, label in labels:
        print(f"{label}: {sum(row[key] for row in scoreboard)} of {len(scoreboard)}")
    print()
    print(f"final answers the completion check sent back: {sum(row['refusals'] for row in scoreboard)}"
          f" (across {sum(1 for row in scoreboard if row['refusals'])} request(s))")
    print(f"fallback replies used: {sum(row['fallback'] for row in scoreboard)}")


def main() -> None:
    """Run the environment check, the wiring checks, then the system checks."""
    check_environment()

    try:
        import project_starter as project
    except RuntimeError as error:  # raised when the API key is missing
        sys.exit(f"Could not load project_starter.py: {error}")

    print(f"Model: {project.MODEL_ID}")

    print_heading("PART 1 - the wiring and the completion check (no API calls)")
    if not run_wiring_checks(project):
        sys.exit("A wiring check failed, so the system test was skipped.")

    request_count = len(EXPECTED_ORDERS) + len(DECLINED_REQUESTS)
    print_heading(f"PART 2 - the whole system on {request_count} sample requests (uses the API)")
    scoreboard = run_system_checks(project)

    print_heading("SCOREBOARD")
    print_scoreboard(scoreboard)
    print("\nSend back the whole output for review.")


if __name__ == "__main__":
    main()
