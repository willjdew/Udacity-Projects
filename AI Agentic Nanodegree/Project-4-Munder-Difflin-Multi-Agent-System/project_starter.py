import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine
from smolagents import OpenAIServerModel, Tool, ToolCallingAgent, tool

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.

# Used to turn down the agents' step-by-step console output for the full run.
from smolagents import LogLevel

# Name of the environment variable that holds the API key (read from .env).
API_KEY_ENV_NAME = "UDACITY_OPENAI_API_KEY"

# OpenAI-compatible proxy provided for this course.
API_BASE_URL = "https://openai.vocareum.com/v1"

# Model used by every agent. Change it here to switch all agents at once.
MODEL_ID = "gpt-4o-mini"

# Read the key from a .env file kept next to this script (never hard-code it).
dotenv.load_dotenv()
api_key = os.getenv(API_KEY_ENV_NAME)
if not api_key:
    raise RuntimeError(
        f"{API_KEY_ENV_NAME} is not set. Add it to a .env file in the same "
        "folder as this script."
    )

# One shared model connection, reused by every agent.
model = OpenAIServerModel(
    model_id=MODEL_ID,
    api_base=API_BASE_URL,
    api_key=api_key,
)


"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


# Tools for inventory agent

# ---------------------------------------------------------------------------
# Shared lookups and helpers used by the agent tools
# ---------------------------------------------------------------------------

# Catalog keyed by lower-cased item name. Tools use it to confirm an item is
# sold at all and to recover the exact spelling stored in the database.
CATALOG_BY_LOWER_NAME = {item["item_name"].lower(): item for item in paper_supplies}

# Low-stock top-ups are bought from the supplier in whole lots of this size
# (one ream of paper is 500 sheets; other products use the same lot size).
RESTOCK_LOT_SIZE = 500

# Customers may order paper by the ream; the catalog counts single sheets.
SHEETS_PER_REAM = 500

# Pack words with no fixed size. A quantity given in one of these cannot be
# turned into a number of single units.
UNSIZED_PACK_UNITS = ("packet", "pack", "box", "case", "carton", "bundle")

# Sheet sizes that every standard paper item is supplied in.
STANDARD_SHEET_SIZES = {"a4", "letter", "8.5x11"}

# Sizes known by name. Size codes (A3) and dimensions (24x36) are
# recognized by the digits they contain.
NAMED_SHEET_SIZES = {"letter", "legal", "tabloid", "ledger"}

# Everyday names customers use for plain office paper. None of them is a
# catalog name, so the agent had to translate them, and it did so differently
# from one run to the next. Code now settles it: these phrases mean
# EVERYDAY_PAPER_ITEM, or EVERYDAY_PAPER_ITEM_A4 when the customer says A4.
EVERYDAY_PAPER_NAMES = ("printer paper", "printing paper", "copy paper")
EVERYDAY_PAPER_ITEM = "Standard copy paper"
EVERYDAY_PAPER_ITEM_A4 = "A4 paper"


def normalize_request_date(date_text: str) -> str:
    """
    Convert a date string to the YYYY-MM-DD form used for every database lookup.

    The starter helpers compare dates as plain strings, so a date in any other
    format silently gives wrong results. Every tool passes its date through here.

    Args:
        date_text (str): A date such as '2025-04-01' or '2025-04-01T00:00:00'.

    Returns:
        str: The same date as 'YYYY-MM-DD'.

    Raises:
        ValueError: If the text is not an ISO-formatted date.
    """
    return datetime.fromisoformat(str(date_text).strip()).strftime("%Y-%m-%d")


def invalid_date_result(date_text: str) -> Dict:
    """
    Build the result every tool returns when it is given a malformed date.

    Args:
        date_text (str): The date text that could not be read.

    Returns:
        Dict: A result with status 'invalid_date' and a plain-language message.
    """
    return {
        "status": "invalid_date",
        "message": f"'{date_text}' is not a valid date. Use YYYY-MM-DD.",
    }


# Stock counts, supplier costs and cash figures must stay inside the company.
# The tools never put them in the text they return, because that text is
# read by the agent that writes to the customer. They go to the console log.


def log_internal(source: str, details: List[str]) -> None:
    """
    Print company-only details to the console log.

    Nothing printed here is returned to an agent, so it cannot reach a
    customer reply.

    Args:
        source (str): Which tool is logging, and for what date.
        details (List[str]): One line per detail.
    """
    print(f"INTERNAL LOG ({source}):")
    for detail in details:
        print(f"  - {detail}")


def find_catalog_item(item_name: str) -> Union[Dict, None]:
    """
    Look up an item in the product catalog, ignoring case and outer spaces.

    Args:
        item_name (str): The item name to look up.

    Returns:
        Dict or None: The catalog entry (item_name, category, unit_price),
                      or None when the company does not sell that item.
    """
    return CATALOG_BY_LOWER_NAME.get(str(item_name).strip().lower())


def convert_to_single_units(quantity: int, unit: str) -> Union[int, None]:
    """
    Convert a customer's quantity into single catalog units (sheets, cups, rolls).

    Args:
        quantity (int): The number the customer wrote.
        unit (str): The unit the customer wrote, such as 'sheets' or 'reams'.

    Returns:
        int or None: The number of single units, or None when the unit is a
                     pack with no fixed size, so the quantity cannot be known.
    """
    unit_text = str(unit or "").strip().lower()
    if unit_text.startswith("ream"):
        return quantity * SHEETS_PER_REAM
    if unit_text.startswith(UNSIZED_PACK_UNITS):
        return None
    return quantity


def normalize_size_text(size_text: str) -> str:
    """
    Reduce a size description to a bare key that can be compared.

    Args:
        size_text (str): A size such as '24" x 36 inches', 'A3' or 'Letter-sized'.

    Returns:
        str: The size without spaces, quote marks or filler words, in lower
             case, such as '24x36', 'a3' or 'letter'.
    """
    size_key = str(size_text or "").lower().replace("×", "x")
    for filler in ("inches", "inch", "-sized", "-size", "sized", "size", '"', "'", " "):
        size_key = size_key.replace(filler, "")
    return size_key


def is_sheet_size(size_key: str) -> bool:
    """
    Decide whether a normalized key really names a size.

    This guards against a colour or quality word arriving in the size field.

    Args:
        size_key (str): A key produced by normalize_size_text.

    Returns:
        bool: True for named sizes ('letter'), size codes ('a3') and
              dimensions ('24x36'); False for anything else, including ''.
    """
    return size_key in NAMED_SHEET_SIZES or any(character.isdigit() for character in size_key)


def size_is_carried(catalog_name: str, sheet_size: str) -> bool:
    """
    Decide whether a catalog item is supplied in the size the customer asked for.

    Standard sheets cover A4 and letter. Any other size is carried only when the
    catalog name itself includes it, as in 'Large poster paper (24x36 inches)'.

    Args:
        catalog_name (str): Exact catalog name of the matched item.
        sheet_size (str): The size the customer stated, or '' when none.

    Returns:
        bool: True when no size was stated or the size is one we supply.
    """
    size_key = normalize_size_text(sheet_size)
    if not is_sheet_size(size_key) or size_key in STANDARD_SHEET_SIZES:
        return True
    return size_key in normalize_size_text(catalog_name)


def is_iso_size_code(word: str) -> bool:
    """
    Recognize a paper size code such as 'A4' or 'B5' inside a sentence.

    Args:
        word (str): A single lower-case word.

    Returns:
        bool: True when the word is a letter a, b or c followed by digits.
    """
    code = word.strip("(),.;:")
    return len(code) in (2, 3) and code[0] in "abc" and code[1:].isdigit()


def resolve_catalog_item(requested_wording: str, proposed_name: str, sheet_size: str = "") -> Union[Dict, None]:
    """
    Settle which catalog item a requested item refers to.

    The agent proposes a catalog name, and two rules in code can overrule it:
    1. When the customer's own words contain exactly one catalog name, that
       literal match wins, so 'colorful construction paper' is always
       'Construction paper'.
    2. Otherwise, when the words contain an everyday name for plain office
       paper ('printer paper', 'printing paper', 'copy paper'), the item is
       'Standard copy paper', or 'A4 paper' when the customer asked for A4.

    Args:
        requested_wording (str): The customer's words for the item.
        proposed_name (str): The catalog name proposed by the agent, or ''.
        sheet_size (str, optional): The size the customer stated, or ''.

    Returns:
        Dict or None: The catalog entry, or None when we do not sell the item.
    """
    proposed_item = find_catalog_item(proposed_name)

    # Drop size codes first so 'glossy A4 paper' reads as 'glossy paper'.
    wording = " ".join(
        word for word in str(requested_wording or "").lower().split()
        if not is_iso_size_code(word)
    )
    named_items = [item for item in paper_supplies if item["item_name"].lower() in wording]

    # 'Poster paper' also sits inside '220 gsm poster paper': keep the longest.
    most_specific_items = [
        item for item in named_items
        if not any(
            item is not other and item["item_name"].lower() in other["item_name"].lower()
            for other in named_items
        )
    ]

    # Rule 1: a catalog name in the customer's words.
    if proposed_item is not None and proposed_item in most_specific_items:
        return proposed_item
    if len(most_specific_items) == 1:
        return most_specific_items[0]

    # Rule 2: an everyday name for plain office paper.
    if any(everyday_name in wording for everyday_name in EVERYDAY_PAPER_NAMES):
        asked_for_a4 = normalize_size_text(sheet_size) == "a4"
        return find_catalog_item(EVERYDAY_PAPER_ITEM_A4 if asked_for_a4 else EVERYDAY_PAPER_ITEM)

    return proposed_item


def assess_item_supply(
    catalog_name: str,
    quantity_units: int,
    order_date: str,
    deadline_date: str,
    units_already_allocated: int = 0,
) -> Dict:
    """
    Work out whether stock, or stock plus a supplier restock, covers one item.

    Args:
        catalog_name (str): Exact catalog name of the item.
        quantity_units (int): Number of single units the customer wants.
        order_date (str): Request date as YYYY-MM-DD.
        deadline_date (str): Delivery deadline as YYYY-MM-DD.
        units_already_allocated (int, optional): Units of this item promised to
            earlier lines of the same order. Default is 0.

    Returns:
        Dict: 'units_on_hand', 'shortfall', 'restock_arrival_date' (None when
              no restock is needed), 'ready_date', 'can_supply' and 'reason'.
              The reason is safe to pass on to a customer: it gives dates
              but no stock counts or supplier quantities.
    """
    # Starter helper: net units = stock orders minus sales up to the date.
    stock_frame = get_stock_level(catalog_name, order_date)
    units_on_hand = int(stock_frame["current_stock"].iloc[0])
    units_free = max(0, units_on_hand - units_already_allocated)
    shortfall = max(0, quantity_units - units_free)

    if shortfall == 0:
        # Stock covers the order, so it is ready on the day it is requested.
        restock_arrival_date = None
        ready_date = order_date
    else:
        # Starter helper: lead time grows with the size of the supplier order,
        # so only the shortfall is ordered, never the full quantity.
        restock_arrival_date = get_supplier_delivery_date(order_date, shortfall)
        ready_date = restock_arrival_date

    # Dates are all YYYY-MM-DD here, so comparing the strings compares the dates.
    can_supply = ready_date <= deadline_date
    if can_supply and shortfall == 0:
        reason = "Covered by stock on hand."
    elif can_supply:
        reason = f"Available from {restock_arrival_date}, by the deadline."
    elif shortfall == 0:
        reason = f"The deadline {deadline_date} is before the request date {order_date}."
    else:
        reason = (
            f"The full quantity would not be available until "
            f"{restock_arrival_date}, after the deadline {deadline_date}."
        )

    return {
        "units_on_hand": units_on_hand,
        "shortfall": shortfall,
        "restock_arrival_date": restock_arrival_date,
        "ready_date": ready_date,
        "can_supply": can_supply,
        "reason": reason,
    }


def evaluate_order_line(
    order_line: Dict,
    order_date: str,
    deadline_date: str,
    allocated_units: Dict[str, int],
) -> Dict:
    """
    Apply every availability rule to one line of a customer order.

    The rules run in a fixed order: is the item sold, is the size carried, is
    the quantity clear, and finally can stock or a restock cover it in time.

    Args:
        order_line (Dict): 'requested', 'item_name', 'quantity', 'unit' and
            'sheet_size' as filled in by the agent.
        order_date (str): Request date as YYYY-MM-DD.
        deadline_date (str): Delivery deadline as YYYY-MM-DD.
        allocated_units (Dict[str, int]): Units already promised per item by
            earlier lines of this order. Updated in place.

    Returns:
        Dict: The line's result: 'requested', 'catalog_item' (None when not
              sold), 'quantity_units' (None when unclear), 'units_on_hand',
              'shortfall', 'restock_arrival_date', 'ready_date', 'can_supply'
              and 'reason'.
    """
    if not isinstance(order_line, dict):
        order_line = {"requested": str(order_line)}

    quantity_text = str(order_line.get("quantity", "")).replace(",", "").strip()
    unit = str(order_line.get("unit", "") or "").strip()
    sheet_size = str(order_line.get("sheet_size", "") or "").strip()
    wording = str(order_line.get("requested", "") or order_line.get("item_name", "")).strip()

    # If the size field was left empty, pick up a size code written in the wording.
    if not sheet_size:
        size_codes = [word.strip("(),.;:") for word in wording.lower().split() if is_iso_size_code(word)]
        sheet_size = size_codes[0].upper() if size_codes else ""

    # Show the request the way the customer phrased it, for the report. When
    # the wording already starts with the quantity, it is used as it stands.
    if not quantity_text or wording.replace(",", "").startswith(quantity_text):
        requested_text = wording
    elif unit and unit.lower() != "each":
        requested_text = f"{quantity_text} {unit} of {wording}"
    else:
        requested_text = f"{quantity_text} {wording}"

    result = {
        "requested": requested_text,
        "catalog_item": None,
        "quantity_units": None,
        "units_on_hand": None,
        "shortfall": None,
        "restock_arrival_date": None,
        "ready_date": None,
        "can_supply": False,
        "reason": "",
    }

    # Rule 1: the item must be in the catalog.
    catalog_item = resolve_catalog_item(wording, order_line.get("item_name", ""), sheet_size)
    if catalog_item is None:
        result["reason"] = "This item is not in our catalog."
        return result

    # Rule 2: the size must be one we carry for that item.
    if not size_is_carried(catalog_item["item_name"], sheet_size):
        result["reason"] = f"We do not carry this item in {sheet_size} size."
        return result
    result["catalog_item"] = catalog_item["item_name"]

    # Rule 3: the quantity must be a clear number of single units.
    try:
        quantity = int(float(quantity_text))
    except ValueError:
        quantity = 0
    if quantity <= 0:
        result["reason"] = "The request does not give a clear quantity for this item."
        return result
    quantity_units = convert_to_single_units(quantity, unit)
    if quantity_units is None:
        result["reason"] = (
            f"The quantity is given in {unit}, which have no fixed size. "
            "Please give the number of individual items."
        )
        return result
    result["quantity_units"] = quantity_units

    # Rule 4: stock, or stock plus a restock, must cover it by the deadline.
    supply = assess_item_supply(
        catalog_item["item_name"],
        quantity_units,
        order_date,
        deadline_date,
        allocated_units.get(catalog_item["item_name"], 0),
    )
    allocated_units[catalog_item["item_name"]] = (
        allocated_units.get(catalog_item["item_name"], 0) + quantity_units
    )
    result.update({
        "units_on_hand": supply["units_on_hand"],
        "shortfall": supply["shortfall"],
        "restock_arrival_date": supply["restock_arrival_date"],
        "ready_date": supply["ready_date"],
        "can_supply": supply["can_supply"],
        "reason": supply["reason"],
    })
    return result


def is_order_complete(line_results: List[Dict]) -> bool:
    """
    Decide whether a checked order can be supplied in full.

    Args:
        line_results (List[Dict]): One result per line from evaluate_order_line.

    Returns:
        bool: True only when there is at least one line and every line can
              be supplied by the deadline.
    """
    return bool(line_results) and all(line["can_supply"] for line in line_results)


# The most recent availability check. The Sales tool reads it to confirm
# that the order it is asked to record is the same order that was checked
# and found complete. It is kept in memory only, which is enough because
# requests are handled one at a time.
LAST_AVAILABILITY_CHECK = {}

# What has been done so far for the order that was last checked. The tools
# fill it in as they run, and the Orchestrator's completion check reads it,
# so "which steps have been done" is known from what the tools actually did,
# not from what a model says it did. A new availability check starts it again.
REQUEST_PROGRESS = {}

# The customer request being handled, exactly as it arrived, and how its
# reply was produced. Set by handle_customer_request. The Inventory agent is
# given 'text' for its availability task, so the customer's words reach it
# unchanged whatever the Orchestrator typed.
CURRENT_REQUEST = {"text": None, "reply_source": None, "inventory_tasks": []}


def start_new_request() -> None:
    """Forget the last availability check and its progress, ready for a new request."""
    LAST_AVAILABILITY_CHECK.clear()
    REQUEST_PROGRESS.clear()
    REQUEST_PROGRESS.update({
        "quoted_units_by_item": None,   # items of the latest quote, or None
        "quote_total_cents": None,      # total of the latest quote
        "sale_attempted": False,        # fulfill_order was called
        "order_recorded": False,        # fulfill_order recorded the order
        "order_total_cents": None,      # total of the recorded order
        "delivery_date": None,          # delivery date of the recorded order
        "sale_refusal_reasons": [],     # why fulfill_order refused, if it did
        "low_stock_items": None,        # None until the low-stock check runs after a sale
        "top_ups_attempted": False,     # buy_restock was called after the sale
    })


start_new_request()


def total_units_by_item(item_quantities: List[tuple]) -> Dict[str, int]:
    """
    Add up the units asked for per catalog item.

    Args:
        item_quantities (List[tuple]): (catalog item name, units) pairs. The
            same item may appear more than once.

    Returns:
        Dict[str, int]: Total units keyed by item name.
    """
    units_by_item = {}
    for item_name, units in item_quantities:
        units_by_item[item_name] = units_by_item.get(item_name, 0) + int(units)
    return units_by_item


def remember_availability_check(order_date: str, deadline_date: str, line_results: List[Dict]) -> None:
    """
    Keep the outcome of an availability check for the Sales tool to compare with.

    Args:
        order_date (str): Request date as YYYY-MM-DD.
        deadline_date (str): Delivery deadline as YYYY-MM-DD.
        line_results (List[Dict]): One result per line from evaluate_order_line.
    """
    order_complete = is_order_complete(line_results)
    suppliable_units = total_units_by_item(
        [(line["catalog_item"], line["quantity_units"]) for line in line_results if line["can_supply"]]
    )

    # One customer request records at most one order. If this request already
    # has a recorded order, a later check changes nothing, so the order
    # cannot be checked again and then recorded a second time.
    if REQUEST_PROGRESS["order_recorded"]:
        return

    # Otherwise a new check replaces the last one, and earlier progress
    # (a quote, a refused sale) no longer applies.
    start_new_request()
    LAST_AVAILABILITY_CHECK.update({
        "order_date": order_date,
        "deadline_date": deadline_date,
        "order_complete": order_complete,
        # The items a quote should cover, complete order or not.
        "suppliable_units_by_item": suppliable_units,
        # Only a complete order can be recorded, so only then are items kept.
        "units_by_item": suppliable_units if order_complete else {},
        "recorded": False,
        # What the customer asked for and the outcome of each line, for a
        # reply written in code. The reasons are safe to give to a customer.
        "lines": [
            {"requested": line["requested"], "can_supply": line["can_supply"], "reason": line["reason"]}
            for line in line_results
        ],
    })


def format_availability_report(order_date: str, deadline_date: str, line_results: List[Dict]) -> str:
    """
    Write the availability report that the Inventory agent returns.

    The report holds only what a customer may be told: what can be supplied,
    from which date, and why not when it cannot. Stock counts and supplier
    orders are left out (see describe_availability_internals).

    The 'ORDER COMPLETE' verdict is decided in code, by is_order_complete.

    Args:
        order_date (str): Request date as YYYY-MM-DD.
        deadline_date (str): Delivery deadline as YYYY-MM-DD.
        line_results (List[Dict]): One result per line from evaluate_order_line.

    Returns:
        str: The full report as plain text.
    """
    report_lines = [
        "AVAILABILITY REPORT",
        f"Request date: {order_date}",
        f"Deadline: {deadline_date}",
    ]
    for line_number, line in enumerate(line_results, start=1):
        if line["catalog_item"] is None:
            quantity_shown = "n/a"
        else:
            quantity_shown = "QUANTITY UNCLEAR" if line["quantity_units"] is None else line["quantity_units"]
        report_lines += [
            f"Item {line_number}",
            f"- requested: {line['requested']}",
            f"- catalog_item: {line['catalog_item'] or 'NOT SOLD'}",
            f"- quantity_units: {quantity_shown}",
            f"- available_from: {line['ready_date'] or 'n/a'}",
            f"- can_supply_by_deadline: {'yes' if line['can_supply'] else 'no'}",
            f"- reason: {line['reason']}",
        ]

    suppliable_count = sum(1 for line in line_results if line["can_supply"])
    order_complete = is_order_complete(line_results)
    report_lines.append(f"ITEMS THAT CAN BE SUPPLIED: {suppliable_count} of {len(line_results)}")
    if order_complete:
        # The whole order is ready when its slowest item is ready.
        report_lines.append(f"ORDER READY DATE: {max(line['ready_date'] for line in line_results)}")
    report_lines.append(f"ORDER COMPLETE: {'yes' if order_complete else 'no'}")
    return "\n".join(report_lines)


def describe_availability_internals(line_results: List[Dict]) -> List[str]:
    """
    List the stock position behind an availability report, for the console log.

    Args:
        line_results (List[Dict]): One result per line from evaluate_order_line.

    Returns:
        List[str]: For each item that reached the stock check, the units on
                   hand, the shortfall and the supplier arrival date.
    """
    details = []
    for line in line_results:
        if line["units_on_hand"] is None:
            continue  # refused before the stock check: not sold, wrong size or unclear quantity
        details.append(
            f"{line['catalog_item']}: {line['quantity_units']} wanted, {line['units_on_hand']} on hand, "
            f"shortfall {line['shortfall']}, supplier arrival {line['restock_arrival_date'] or 'not needed'}"
        )
    return details


# ---------------------------------------------------------------------------
# Inventory agent tools
# ---------------------------------------------------------------------------

@tool
def list_catalog_stock(as_of_date: str) -> dict:
    """
    List every item the company sells, with the units on hand as of a date.

    Call this first for a new request, to match the customer's wording to exact
    catalog names. The list is the full catalog: any product that does not
    appear in it is not sold, and an item showing 0 units is sold but would
    need a supplier restock.

    Args:
        as_of_date: The date of the customer's request, in YYYY-MM-DD format.

    Returns:
        dict: Always contains 'status', which is one of:
            - 'ok': also contains 'as_of_date', 'item_count' and 'items', a
              list of {'item_name', 'category', 'units_on_hand'} in catalog order.
            - 'invalid_date': also contains 'message'.
    """
    # Reject a malformed date up front rather than return misleading counts.
    try:
        lookup_date = normalize_request_date(as_of_date)
    except ValueError:
        return invalid_date_result(as_of_date)

    # Starter helper: returns only the items that currently have stock.
    stocked_units = get_all_inventory(lookup_date)

    # Walk the whole catalog so unstocked items are listed with 0 units
    # instead of being left out and mistaken for products we do not sell.
    items = [
        {
            "item_name": catalog_item["item_name"],
            "category": catalog_item["category"],
            "units_on_hand": int(stocked_units.get(catalog_item["item_name"], 0)),
        }
        for catalog_item in paper_supplies
    ]

    return {
        "status": "ok",
        "as_of_date": lookup_date,
        "item_count": len(items),
        "items": items,
    }


class CheckOrderAvailabilityTool(Tool):
    """
    Inventory tool that checks a whole customer order in one call.

    It is written as a Tool subclass, rather than with the @tool decorator,
    so that the fields of each order line can be described to the model.
    """

    name = "check_order_availability"
    description = (
        "Checks every item of one customer order against stock and supplier "
        "lead times, and returns the finished availability report as text. "
        "Call it once per order with one line for each item the customer asked "
        "for. It converts reams to sheets, rejects sizes and items we do not "
        "sell, works out shortfalls and restock dates, and decides whether the "
        "order is complete. Copy quantities, units and sizes as the customer "
        "wrote them: do not convert or judge them yourself."
    )
    inputs = {
        "order_lines": {
            "type": "array",
            "description": "One entry for each item the customer asked for.",
            "items": {
                "type": "object",
                "properties": {
                    "requested": {
                        "type": "string",
                        "description": "The customer's own words for the item, without the quantity, for example 'A4 glossy paper'.",
                    },
                    "item_name": {
                        "type": "string",
                        "description": "Exact catalog item_name for the same kind of product, copied from list_catalog_stock. Use '' when no catalog item matches.",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "The number the customer wrote, for example 500.",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit the customer wrote, for example 'sheets', 'reams', 'rolls' or 'packets'. Use 'each' for countable products.",
                    },
                    "sheet_size": {
                        "type": "string",
                        "description": "The size the customer stated, for example 'A4', 'A3', '8.5x11' or '24x36'. Use '' when no size was stated.",
                    },
                },
                "required": ["requested", "item_name", "quantity", "unit", "sheet_size"],
            },
        },
        "request_date": {
            "type": "string",
            "description": "The date of the customer's request, in YYYY-MM-DD format.",
        },
        "deadline": {
            "type": "string",
            "description": "The date the customer needs delivery by, in YYYY-MM-DD format.",
        },
    }
    output_type = "string"

    def forward(self, order_lines: list, request_date: str, deadline: str) -> str:
        """
        Check each order line and return the availability report.

        Helper functions used: get_stock_level and get_supplier_delivery_date
        (both through assess_item_supply).

        Args:
            order_lines (list): One dict per requested item, as described in `inputs`.
            request_date (str): The date of the customer's request (YYYY-MM-DD).
            deadline (str): The delivery deadline (YYYY-MM-DD).

        Returns:
            str: The availability report, or a one-line message when the
                 dates or the order lines cannot be read.
        """
        # Reject bad inputs up front rather than return a misleading report.
        try:
            order_date = normalize_request_date(request_date)
        except ValueError:
            return invalid_date_result(request_date)["message"]
        try:
            deadline_date = normalize_request_date(deadline)
        except ValueError:
            return invalid_date_result(deadline)["message"]
        if not isinstance(order_lines, list) or not order_lines:
            return "No order lines were given. Pass one line for each item the customer asked for."

        # Units promised to earlier lines, so two lines for the same item
        # cannot both count the same stock.
        allocated_units = {}
        line_results = [
            evaluate_order_line(order_line, order_date, deadline_date, allocated_units)
            for order_line in order_lines
        ]
        # Kept so that Sales can only record this order if it is complete.
        remember_availability_check(order_date, deadline_date, line_results)

        # Stock counts go to the console log, never into the returned report.
        internal_details = describe_availability_internals(line_results)
        if internal_details:
            log_internal(f"check_order_availability, {order_date}", internal_details)
        return format_availability_report(order_date, deadline_date, line_results)


# The tool instance handed to the Inventory agent.
check_order_availability = CheckOrderAvailabilityTool()


def find_low_stock_items(as_of_date: str) -> List[Dict]:
    """
    Find the stocked items that are at or below their minimum stock level.

    This is the single place where the minimum-stock rule and the top-up
    quantity are worked out. The Inventory tool reports the result, and the
    Sales tool checks against it again before it buys anything.

    Args:
        as_of_date (str): The date to check stock on, as YYYY-MM-DD.

    Returns:
        List[Dict]: One dict per low item with 'item_name', 'units_on_hand',
                    'min_stock_level' and 'top_up_quantity'. Empty when
                    nothing needs a top-up.
    """
    # Minimum levels exist only for the items in the inventory reference table.
    minimum_levels = pd.read_sql(
        "SELECT item_name, min_stock_level FROM inventory", db_engine
    )

    # Starter helper: leaves out items with no stock, so those default to 0.
    stocked_units = get_all_inventory(as_of_date)

    low_stock_items = []
    for _, inventory_row in minimum_levels.iterrows():
        min_stock_level = int(inventory_row["min_stock_level"])
        units_on_hand = int(stocked_units.get(inventory_row["item_name"], 0))

        if units_on_hand <= min_stock_level:
            # Fewest whole lots that lift the stock back above its minimum.
            lots_needed = (min_stock_level - units_on_hand) // RESTOCK_LOT_SIZE + 1
            low_stock_items.append({
                "item_name": inventory_row["item_name"],
                "units_on_hand": units_on_hand,
                "min_stock_level": min_stock_level,
                "top_up_quantity": lots_needed * RESTOCK_LOT_SIZE,
            })
    return low_stock_items


@tool
def check_min_stock(as_of_date: str) -> dict:
    """
    List the stocked items that are at or below their minimum stock level.

    Call this after a sale has been recorded, to find items that need a top-up.
    Each flagged item comes with a recommended top-up quantity in whole supplier
    lots. This tool only reports; buying the top-up is the Sales agent's job.
    Stock counts and minimum levels are not returned: they go to the console log.

    Args:
        as_of_date: The date of the customer's request, in YYYY-MM-DD format.

    Returns:
        dict: Always contains 'status', which is one of:
            - 'ok': also contains 'as_of_date', 'low_stock_count' and
              'low_stock_items', a list of {'item_name', 'top_up_quantity'}.
              The list is empty when nothing needs a top-up.
            - 'invalid_date': also contains 'message'.
    """
    try:
        lookup_date = normalize_request_date(as_of_date)
    except ValueError:
        return invalid_date_result(as_of_date)

    # Starter helper used inside: get_all_inventory.
    low_stock_items = find_low_stock_items(lookup_date)

    # Noted for the Orchestrator's completion check: the low-stock step ran.
    REQUEST_PROGRESS["low_stock_items"] = [
        {"item_name": item["item_name"], "top_up_quantity": item["top_up_quantity"]}
        for item in low_stock_items
    ]

    # Stock counts go to the console log, never into the returned result.
    if low_stock_items:
        log_internal(f"check_min_stock, {lookup_date}", [
            f"{item['item_name']}: {item['units_on_hand']} on hand, minimum {item['min_stock_level']}"
            for item in low_stock_items
        ])

    return {
        "status": "ok",
        "as_of_date": lookup_date,
        "low_stock_count": len(low_stock_items),
        "low_stock_items": [
            {"item_name": item["item_name"], "top_up_quantity": item["top_up_quantity"]}
            for item in low_stock_items
        ],
    }



# Tools for quoting agent

# Bulk discount by the total number of units in an order, lowest tier first.
# Each entry is (minimum units, percent off). This is the whole pricing
# policy: change it here and nowhere else.
BULK_DISCOUNT_TIERS = [(0, 0), (500, 5), (1000, 10), (5000, 15)]


def to_cents(dollar_amount: float) -> int:
    """
    Convert a dollar amount to whole cents.

    All quote arithmetic is done in cents so that totals never pick up the
    small rounding errors that come from adding decimal fractions.

    Args:
        dollar_amount (float): An amount in dollars, such as 0.05.

    Returns:
        int: The same amount in cents, such as 5.
    """
    return int(round(dollar_amount * 100))


def format_money(cents: int) -> str:
    """
    Show an amount held in cents as dollars.

    Args:
        cents (int): An amount in cents, such as 6175.

    Returns:
        str: The amount as text, such as '$61.75'.
    """
    return f"${cents / 100:,.2f}"


def find_discount_tier(total_units: int) -> int:
    """
    Find which bulk-discount tier an order falls into.

    Args:
        total_units (int): Total units across every line of the order.

    Returns:
        int: Position of the tier in BULK_DISCOUNT_TIERS (0 is the lowest).
    """
    tier_index = 0
    for index, (minimum_units, _) in enumerate(BULK_DISCOUNT_TIERS):
        if total_units >= minimum_units:
            tier_index = index
    return tier_index


def describe_discount_tier(tier_index: int) -> str:
    """
    Explain a discount tier in one sentence a customer can read.

    Args:
        tier_index (int): Position of the tier in BULK_DISCOUNT_TIERS.

    Returns:
        str: For example 'Orders of 500 to 999 units receive 5% off.'
    """
    minimum_units, percent = BULK_DISCOUNT_TIERS[tier_index]
    is_top_tier = tier_index == len(BULK_DISCOUNT_TIERS) - 1

    if is_top_tier:
        return f"Orders of {minimum_units:,} units or more receive {percent}% off."
    next_minimum_units = BULK_DISCOUNT_TIERS[tier_index + 1][0]
    if percent == 0:
        return f"Orders under {next_minimum_units:,} units are charged at list price."
    return f"Orders of {minimum_units:,} to {next_minimum_units - 1:,} units receive {percent}% off."


def price_order(order_lines: List[Dict]) -> Dict:
    """
    Price an order from the catalog and apply the bulk-discount tier.

    This is the single place where prices are worked out. The quote tool uses
    it, and the sales tool can call it again so that the amount recorded for a
    sale always equals the amount that was quoted.

    The discount is shared out across the lines so that the discounted line
    totals add up exactly to the order total.

    Args:
        order_lines (List[Dict]): One dict per item with 'item_name' (exact
            catalog name) and 'quantity' (single units).

    Returns:
        Dict: Always contains 'status'.
            - 'ok': also contains 'lines' (each with 'item_name', 'quantity',
              'unit_price', 'line_total' and 'discounted_line_total'),
              'total_units', 'subtotal', 'discount_percent',
              'discount_amount', 'total', 'discount_reason',
              'units_to_next_tier' and 'next_tier_percent' (both None at the
              top tier). Money values are in dollars.
            - 'invalid_order': also contains 'message'.
    """
    if not isinstance(order_lines, list) or not order_lines:
        return {
            "status": "invalid_order",
            "message": "No order lines were given. Pass one line for each item to be quoted.",
        }

    # Price every line at the catalog list price, in cents.
    priced_lines = []
    for order_line in order_lines:
        if not isinstance(order_line, dict):
            return {"status": "invalid_order", "message": f"'{order_line}' is not an order line."}

        catalog_item = find_catalog_item(order_line.get("item_name", ""))
        if catalog_item is None:
            return {
                "status": "invalid_order",
                "message": f"'{order_line.get('item_name')}' is not an exact catalog item_name, so it cannot be priced.",
            }
        try:
            quantity = int(float(str(order_line.get("quantity", "")).replace(",", "")))
        except ValueError:
            quantity = 0
        if quantity <= 0:
            return {
                "status": "invalid_order",
                "message": f"The quantity for '{catalog_item['item_name']}' must be a whole number above 0.",
            }

        unit_price_cents = to_cents(catalog_item["unit_price"])
        priced_lines.append({
            "item_name": catalog_item["item_name"],
            "quantity": quantity,
            "unit_price_cents": unit_price_cents,
            "line_cents": unit_price_cents * quantity,
        })

    # The tier depends on the total number of units, not on the dollar value.
    total_units = sum(line["quantity"] for line in priced_lines)
    subtotal_cents = sum(line["line_cents"] for line in priced_lines)
    tier_index = find_discount_tier(total_units)
    discount_percent = BULK_DISCOUNT_TIERS[tier_index][1]
    discount_cents = (subtotal_cents * discount_percent + 50) // 100  # nearest cent
    total_cents = subtotal_cents - discount_cents

    # Share the discount across the lines; the last line takes any odd cents
    # left over, so the discounted lines add up exactly to the total.
    line_discounts = [line["line_cents"] * discount_percent // 100 for line in priced_lines]
    line_discounts[-1] += discount_cents - sum(line_discounts)

    # Work out how far the order is from the next tier, if there is one.
    if tier_index + 1 < len(BULK_DISCOUNT_TIERS):
        next_minimum_units, next_tier_percent = BULK_DISCOUNT_TIERS[tier_index + 1]
        units_to_next_tier = next_minimum_units - total_units
    else:
        units_to_next_tier, next_tier_percent = None, None

    return {
        "status": "ok",
        "lines": [
            {
                "item_name": line["item_name"],
                "quantity": line["quantity"],
                "unit_price": line["unit_price_cents"] / 100,
                "line_total": line["line_cents"] / 100,
                "discounted_line_total": (line["line_cents"] - line_discount) / 100,
            }
            for line, line_discount in zip(priced_lines, line_discounts)
        ],
        "total_units": total_units,
        "subtotal": subtotal_cents / 100,
        "discount_percent": discount_percent,
        "discount_amount": discount_cents / 100,
        "total": total_cents / 100,
        "discount_reason": describe_discount_tier(tier_index),
        "units_to_next_tier": units_to_next_tier,
        "next_tier_percent": next_tier_percent,
    }


def format_quote(pricing: Dict) -> str:
    """
    Write a priced order as the plain-text quote the Quoting agent returns.

    Args:
        pricing (Dict): A result from price_order with status 'ok'.

    Returns:
        str: The itemized quote, with the discount and the reason for it.
    """
    quote_lines = ["QUOTE"]
    for line_number, line in enumerate(pricing["lines"], start=1):
        quote_lines += [
            f"Item {line_number}",
            f"- catalog_item: {line['item_name']}",
            f"- quantity_units: {line['quantity']}",
            f"- unit_price: {format_money(to_cents(line['unit_price']))}",
            f"- line_total: {format_money(to_cents(line['line_total']))}",
        ]

    quote_lines += [
        f"TOTAL UNITS: {pricing['total_units']}",
        f"SUBTOTAL: {format_money(to_cents(pricing['subtotal']))}",
        f"BULK DISCOUNT: {pricing['discount_percent']}% (-{format_money(to_cents(pricing['discount_amount']))})",
        f"DISCOUNT REASON: {pricing['discount_reason']}",
    ]
    if pricing["units_to_next_tier"] is not None:
        quote_lines.append(
            f"NEXT DISCOUNT TIER: {pricing['units_to_next_tier']} more units "
            f"would bring the discount to {pricing['next_tier_percent']}%."
        )
    quote_lines.append(f"TOTAL: {format_money(to_cents(pricing['total']))}")
    return "\n".join(quote_lines)


class CalculateQuoteTool(Tool):
    """
    Quoting tool that prices an order and applies the bulk-discount tier.

    It uses none of the starter's database helpers: it is a pure calculation
    on the catalog price list (paper_supplies) and BULK_DISCOUNT_TIERS. Like
    the order-availability tool it is a Tool subclass, so that the fields of
    each order line can be described to the model.
    """

    name = "calculate_quote"
    description = (
        "Prices the items of an order at catalog list prices, applies the bulk "
        "discount for the total number of units, and returns the finished "
        "quote as text. Pass only the items that can be supplied, using the "
        "exact catalog item_name and the quantity in single units from the "
        "availability report. All of the arithmetic is done here: never work "
        "out a price, a discount or a total yourself."
    )
    inputs = {
        "order_lines": {
            "type": "array",
            "description": "One entry for each item to be quoted.",
            "items": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "Exact catalog item_name, for example 'Glossy paper'.",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Number of single units, for example 200.",
                    },
                },
                "required": ["item_name", "quantity"],
            },
        },
    }
    output_type = "string"

    def forward(self, order_lines: list) -> str:
        """
        Price the order and return the quote.

        Args:
            order_lines (list): One dict per item with 'item_name' and 'quantity'.

        Returns:
            str: The itemized quote, or a one-line message when a line
                 cannot be priced.
        """
        pricing = price_order(order_lines)
        if pricing["status"] != "ok":
            return pricing["message"]

        # Noted for the Orchestrator's completion check: what was quoted.
        REQUEST_PROGRESS["quoted_units_by_item"] = total_units_by_item(
            [(line["item_name"], line["quantity"]) for line in pricing["lines"]]
        )
        REQUEST_PROGRESS["quote_total_cents"] = to_cents(pricing["total"])
        return format_quote(pricing)


# The tool instance handed to the Quoting agent.
calculate_quote = CalculateQuoteTool()


# Most past quotes returned by one history search, and how much of each
# long text field is kept, so the result stays short enough to read quickly.
MAX_SIMILAR_QUOTES = 3
HISTORY_EXCERPT_LENGTH = 240


def shorten_text(text: str, max_length: int = HISTORY_EXCERPT_LENGTH) -> str:
    """
    Cut a long text down to an excerpt on one line.

    Args:
        text (str): The full text.
        max_length (int, optional): Most characters to keep.

    Returns:
        str: The text on one line, ending in '...' when it was cut.
    """
    single_line = " ".join(str(text or "").split())
    if len(single_line) <= max_length:
        return single_line
    return single_line[:max_length].rstrip() + "..."


@tool
def find_similar_quotes(search_terms: str) -> dict:
    """
    Look up past quotes for similar jobs in the quote history.

    Use this to see how comparable orders were quoted before, so that the way
    a discount is explained stays consistent. Past quotes are internal
    background only: never repeat another customer's wording or amounts.

    Args:
        search_terms: One to three single words separated by commas, such as
            the customer's event and a main item: 'ceremony, cardstock'.

    Returns:
        dict: Always contains 'status', which is one of:
            - 'ok': also contains 'search_terms' (list), 'match_count',
              'discount_mention_count', 'history_note' (a finished sentence
              to copy into the answer) and 'quotes', a list of
              {'total_amount', 'job_type', 'event_type', 'order_size',
              'request_excerpt', 'explanation_excerpt'}. The list is empty
              when nothing similar was found.
            - 'invalid_terms': also contains 'message'.
    """
    terms = [term.strip() for term in str(search_terms or "").split(",") if term.strip()]
    if not terms:
        return {
            "status": "invalid_terms",
            "message": "Give at least one search word, for example 'ceremony'.",
        }

    # Starter helper: it only returns quotes that match EVERY term it is given,
    # so each term is searched on its own.
    results_per_term = [search_quote_history([term], limit=MAX_SIMILAR_QUOTES) for term in terms]

    # Take results in turn from each term so no single term crowds out the
    # others, skipping repeats and history rows with no usable total.
    matches = []
    seen_requests = set()
    for position in range(MAX_SIMILAR_QUOTES):
        for term_results in results_per_term:
            if position >= len(term_results):
                continue
            past_quote = term_results[position]
            is_repeat = past_quote["original_request"] in seen_requests
            has_total = isinstance(past_quote["total_amount"], (int, float)) and past_quote["total_amount"] > 0
            if is_repeat or not has_total:
                continue
            seen_requests.add(past_quote["original_request"])
            matches.append({
                "total_amount": float(past_quote["total_amount"]),
                "job_type": past_quote["job_type"],
                "event_type": past_quote["event_type"],
                "order_size": past_quote["order_size"],
                "request_excerpt": shorten_text(past_quote["original_request"]),
                "explanation_excerpt": shorten_text(past_quote["quote_explanation"]),
                # Checked on the full explanation, before it is shortened.
                "mentions_discount": "discount" in str(past_quote["quote_explanation"]).lower(),
            })

    matches = matches[:MAX_SIMILAR_QUOTES]
    discount_mention_count = sum(1 for match in matches if match.pop("mentions_discount"))

    # The note is written here so that it states only what was actually found.
    if not matches:
        history_note = "HISTORY NOTE: No similar past quotes were found."
    else:
        history_note = (
            f"HISTORY NOTE: {len(matches)} similar past quote(s) found; "
            f"{discount_mention_count} of them mention a bulk discount."
        )

    return {
        "status": "ok",
        "search_terms": terms,
        "match_count": len(matches),
        "discount_mention_count": discount_mention_count,
        "history_note": history_note,
        "quotes": matches,
    }


# Tools for ordering agent

# ---------------------------------------------------------------------------
# Sales agent tools
# ---------------------------------------------------------------------------

# These are the only tools that write to the database. Each one finishes
# every check before its first write, so a refusal never leaves part of an
# order behind.

def find_order_mismatch(priced_lines: List[Dict], order_date: str, deadline_date: str) -> Union[str, None]:
    """
    Compare an order Sales is asked to record with the last availability check.

    Orders are all-or-nothing. This stops part of a declined order from being
    recorded, and stops one order from being recorded twice.

    Args:
        priced_lines (List[Dict]): The order's lines from price_order, each
            with 'item_name' and 'quantity'.
        order_date (str): Request date as YYYY-MM-DD.
        deadline_date (str): Delivery deadline as YYYY-MM-DD.

    Returns:
        str or None: None when the order is the one that was checked and
                     found complete, and it has not been recorded yet.
                     Otherwise a sentence saying why it cannot be recorded.
    """
    last_check = LAST_AVAILABILITY_CHECK
    if not last_check:
        return "This order has not been through an availability check, so it cannot be recorded."
    if last_check["recorded"]:
        return "An order has already been recorded for this request. It cannot be recorded again."
    if not last_check["order_complete"]:
        return (
            "The availability check found that this order cannot be supplied in "
            "full, so no part of it can be recorded."
        )

    units_by_item = total_units_by_item([(line["item_name"], line["quantity"]) for line in priced_lines])
    same_order = (
        units_by_item == last_check["units_by_item"]
        and order_date == last_check["order_date"]
        and deadline_date == last_check["deadline_date"]
    )
    if not same_order:
        return (
            "This is not the order that was checked for availability. The items, "
            "quantities, request date and deadline must all be the same."
        )
    return None


def plan_order(order_lines: List[Dict], order_date: str, deadline_date: str) -> Dict:
    """
    Run every check on a customer order and work out what would be recorded.

    This function writes nothing. It prices the order, repeats the Inventory
    agent's stock and deadline check on every line, confirms that this is the
    complete order the Inventory agent checked, and confirms that cash covers
    any supplier restock. The order is 'ready' only when all of that passes.

    Args:
        order_lines (List[Dict]): One dict per item with 'item_name' (exact
            catalog name) and 'quantity' (single units).
        order_date (str): Request date as YYYY-MM-DD.
        deadline_date (str): Delivery deadline as YYYY-MM-DD.

    Returns:
        Dict: Always contains 'status'.
            - 'ready': also contains 'pricing' (the price_order result),
              'supply' (one assess_item_supply result per priced line, in the
              same order), 'restock_cost_cents' and 'ready_date'.
            - 'refused': also contains 'reasons' (sentences the customer may
              be told) and 'internal_notes' (details for the company only).
    """
    # Price first: this also rejects unknown items and unusable quantities.
    pricing = price_order(order_lines)
    if pricing["status"] != "ok":
        return {"status": "refused", "reasons": [pricing["message"]], "internal_notes": []}

    # Repeat the Inventory check on every line. Units promised to earlier
    # lines are tracked so two lines cannot both count the same stock.
    allocated_units = {}
    supply_results = []
    for line in pricing["lines"]:
        supply = assess_item_supply(
            line["item_name"],
            line["quantity"],
            order_date,
            deadline_date,
            allocated_units.get(line["item_name"], 0),
        )
        allocated_units[line["item_name"]] = allocated_units.get(line["item_name"], 0) + line["quantity"]
        supply_results.append(supply)

    # All-or-nothing: one item that cannot arrive in time refuses the order.
    late_item_reasons = [
        f"{line['item_name']} ({line['quantity']} units): {supply['reason']}"
        for line, supply in zip(pricing["lines"], supply_results)
        if not supply["can_supply"]
    ]
    if late_item_reasons:
        return {"status": "refused", "reasons": late_item_reasons, "internal_notes": []}

    # It must also be the whole order that the availability check approved,
    # not a part of a declined order and not an order already recorded.
    mismatch_reason = find_order_mismatch(pricing["lines"], order_date, deadline_date)
    if mismatch_reason is not None:
        return {"status": "refused", "reasons": [mismatch_reason], "internal_notes": []}

    # Shortfalls are bought from the supplier at the catalog unit price.
    restock_cost_cents = sum(
        to_cents(line["unit_price"]) * supply["shortfall"]
        for line, supply in zip(pricing["lines"], supply_results)
    )

    # Starter helper: cash is checked once, against the whole restock cost.
    cash_cents = to_cents(get_cash_balance(order_date))
    if restock_cost_cents > cash_cents:
        return {
            "status": "refused",
            "reasons": ["We cannot buy in the stock this order needs at present."],
            "internal_notes": [
                f"Restock cost {format_money(restock_cost_cents)} is more than "
                f"the cash balance {format_money(cash_cents)}."
            ],
        }

    return {
        "status": "ready",
        "pricing": pricing,
        "supply": supply_results,
        "restock_cost_cents": restock_cost_cents,
        # The whole order is ready when its slowest item is ready.
        "ready_date": max(supply["ready_date"] for supply in supply_results),
    }


def format_sales_refusal(order_date: str, deadline_date: str, reasons: List[str]) -> str:
    """
    Write the Sales report for an order that was not recorded.

    Args:
        order_date (str): Request date as YYYY-MM-DD.
        deadline_date (str): Delivery deadline as YYYY-MM-DD.
        reasons (List[str]): Why the order was refused, one sentence each.

    Returns:
        str: The full report as plain text.
    """
    report_lines = [
        "SALES REPORT",
        f"Request date: {order_date}",
        f"Deadline: {deadline_date}",
        "ORDER RECORDED: no",
        "REASON:",
    ]
    report_lines += [f"- {reason}" for reason in reasons]
    report_lines.append("Nothing was recorded: no sale and no stock purchase.")
    return "\n".join(report_lines)


def format_sales_confirmation(order_date: str, deadline_date: str, plan: Dict) -> str:
    """
    Write the Sales report for an order that was recorded in full.

    The report holds only what the customer may be told: the items, the
    amounts charged, the total and the delivery date.

    Args:
        order_date (str): Request date as YYYY-MM-DD.
        deadline_date (str): Delivery deadline as YYYY-MM-DD.
        plan (Dict): A 'ready' result from plan_order.

    Returns:
        str: The full report as plain text.
    """
    pricing = plan["pricing"]
    report_lines = [
        "SALES REPORT",
        f"Request date: {order_date}",
        f"Deadline: {deadline_date}",
    ]
    for line_number, line in enumerate(pricing["lines"], start=1):
        report_lines += [
            f"Item {line_number}",
            f"- catalog_item: {line['item_name']}",
            f"- quantity_units: {line['quantity']}",
            f"- amount_recorded: {format_money(to_cents(line['discounted_line_total']))}",
        ]
    report_lines += [
        f"ORDER TOTAL: {format_money(to_cents(pricing['total']))}",
        f"DELIVERY DATE: {plan['ready_date']}",
        "ORDER RECORDED: yes",
    ]
    return "\n".join(report_lines)


def describe_order_internals(plan: Dict, cash_after: float) -> List[str]:
    """
    List the company-only details of a recorded order, for the console log.

    Args:
        plan (Dict): A 'ready' result from plan_order.
        cash_after (float): Cash balance in dollars after the order was written.

    Returns:
        List[str]: The supplier purchases, their total cost and the cash balance.
    """
    restocked_lines = [
        (line, supply)
        for line, supply in zip(plan["pricing"]["lines"], plan["supply"])
        if supply["shortfall"] > 0
    ]
    details = []
    if not restocked_lines:
        details.append("restock purchases: none, the order was covered by stock on hand")
    for line, supply in restocked_lines:
        restock_cost = format_money(to_cents(line["unit_price"]) * supply["shortfall"])
        details.append(
            f"restock purchase: {supply['shortfall']} units of {line['item_name']} "
            f"for {restock_cost}, arriving {supply['restock_arrival_date']}"
        )
    details += [
        f"restock cost: {format_money(plan['restock_cost_cents'])}",
        f"cash balance after this order: {format_money(to_cents(cash_after))}",
    ]
    return details


class FulfillOrderTool(Tool):
    """
    Sales tool that records one complete customer order, or nothing at all.

    Like the other whole-order tools it is a Tool subclass, so that the
    fields of each order line can be described to the model.
    """

    name = "fulfill_order"
    description = (
        "Records one complete customer order in the database and returns the "
        "Sales report as text. Call it once per order, with every item of the "
        "order, using the exact catalog item_name and the quantity in single "
        "units from the quote. It only records an order that the Inventory "
        "agent's availability report found complete, with the same items, "
        "quantities and dates, and it records it once. It checks stock, "
        "supplier delivery dates against the deadline, and cash; it buys any "
        "shortfall from the supplier and records each sale at the quoted "
        "price. If any check fails it records nothing and the report says "
        "why. Never work out a "
        "price or decide yourself whether the order can be filled."
    )
    inputs = {
        "order_lines": {
            "type": "array",
            "description": "One entry for each item of the order.",
            "items": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "Exact catalog item_name, for example 'Glossy paper'.",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Number of single units, for example 200.",
                    },
                },
                "required": ["item_name", "quantity"],
            },
        },
        "request_date": {
            "type": "string",
            "description": "The date of the customer's request, in YYYY-MM-DD format.",
        },
        "deadline": {
            "type": "string",
            "description": "The date the customer needs delivery by, in YYYY-MM-DD format.",
        },
    }
    output_type = "string"

    def forward(self, order_lines: list, request_date: str, deadline: str) -> str:
        """
        Check the order, then record all of it or none of it.

        Helper functions used: get_stock_level and get_supplier_delivery_date
        (both through assess_item_supply), get_cash_balance and
        create_transaction.

        Args:
            order_lines (list): One dict per item with 'item_name' and 'quantity'.
            request_date (str): The date of the customer's request (YYYY-MM-DD).
            deadline (str): The delivery deadline (YYYY-MM-DD).

        Returns:
            str: The Sales report, ending with 'ORDER RECORDED: yes', or
                 containing 'ORDER RECORDED: no' and the reason.
        """
        # Reject bad dates up front: every transaction is dated the request date.
        try:
            order_date = normalize_request_date(request_date)
        except ValueError:
            return "ORDER RECORDED: no. " + invalid_date_result(request_date)["message"]
        try:
            deadline_date = normalize_request_date(deadline)
        except ValueError:
            return "ORDER RECORDED: no. " + invalid_date_result(deadline)["message"]

        # Noted for the Orchestrator's completion check: Sales was asked.
        REQUEST_PROGRESS["sale_attempted"] = True

        # Every check runs here, before anything is written.
        plan = plan_order(order_lines, order_date, deadline_date)
        if plan["status"] != "ready":
            REQUEST_PROGRESS["sale_refusal_reasons"] = plan["reasons"]
            if plan["internal_notes"]:
                log_internal(f"fulfill_order, {order_date}, order refused", plan["internal_notes"])
            return format_sales_refusal(order_date, deadline_date, plan["reasons"])

        # Starter helper: buy each shortfall first, so stock never goes negative.
        for line, supply in zip(plan["pricing"]["lines"], plan["supply"]):
            if supply["shortfall"] > 0:
                create_transaction(
                    line["item_name"],
                    "stock_orders",
                    supply["shortfall"],
                    to_cents(line["unit_price"]) * supply["shortfall"] / 100,
                    order_date,
                )

        # Starter helper: record each sale at its share of the quoted total,
        # so the cash received equals the quote to the cent.
        for line in plan["pricing"]["lines"]:
            create_transaction(
                line["item_name"],
                "sales",
                line["quantity"],
                line["discounted_line_total"],
                order_date,
            )

        # The checked order is now used up: a second call cannot record it again.
        LAST_AVAILABILITY_CHECK["recorded"] = True

        # Noted for the completion check. The low-stock steps start afresh
        # here, so only a check made after this sale counts.
        REQUEST_PROGRESS.update({
            "order_recorded": True,
            "order_total_cents": to_cents(plan["pricing"]["total"]),
            "delivery_date": plan["ready_date"],
            "low_stock_items": None,
            "top_ups_attempted": False,
        })

        # Costs and cash go to the console log, never into the returned report.
        log_internal(
            f"fulfill_order, {order_date}, order recorded",
            describe_order_internals(plan, get_cash_balance(order_date)),
        )
        return format_sales_confirmation(order_date, deadline_date, plan)


# The tool instance handed to the Sales agent.
fulfill_order = FulfillOrderTool()


def read_top_up_line(top_up_line: Dict, low_stock_by_name: Dict[str, Dict]) -> Dict:
    """
    Read one requested top-up, confirm it is needed, and price it.

    The item must be at or below its minimum stock level, and the quantity is
    never more than the top-up that the minimum-stock rule recommends.

    Args:
        top_up_line (Dict): 'item_name' (exact catalog name) and 'quantity'
            (single units to buy).
        low_stock_by_name (Dict[str, Dict]): The items that are low right
            now, from find_low_stock_items, keyed by item name.

    Returns:
        Dict: 'item_name', 'quantity', 'cost_cents', 'note' and 'problem'.
              'problem' is None for a usable line; otherwise it says what is
              wrong and the cost is None. 'note' is set when the quantity
              was reduced, and is None otherwise.
    """
    if not isinstance(top_up_line, dict):
        top_up_line = {"item_name": str(top_up_line)}

    result = {
        "item_name": str(top_up_line.get("item_name", "")).strip(),
        "quantity": None,
        "cost_cents": None,
        "note": None,
        "problem": None,
    }

    catalog_item = find_catalog_item(result["item_name"])
    if catalog_item is None:
        result["problem"] = "This is not an exact catalog item_name."
        return result
    result["item_name"] = catalog_item["item_name"]

    # Repeat the Inventory check: only items that are really low are bought.
    low_stock_item = low_stock_by_name.get(catalog_item["item_name"])
    if low_stock_item is None:
        result["problem"] = "This item is not at or below its minimum stock level."
        return result

    try:
        quantity = int(float(str(top_up_line.get("quantity", "")).replace(",", "")))
    except ValueError:
        quantity = 0
    if quantity <= 0:
        result["problem"] = "The quantity must be a whole number above 0."
        return result

    # Never buy more than the recommended top-up.
    if quantity > low_stock_item["top_up_quantity"]:
        result["note"] = (
            f"Quantity reduced from {quantity} to the recommended top-up "
            f"of {low_stock_item['top_up_quantity']}."
        )
        quantity = low_stock_item["top_up_quantity"]

    result["quantity"] = quantity
    result["cost_cents"] = to_cents(catalog_item["unit_price"]) * quantity
    return result


class BuyRestockTool(Tool):
    """
    Sales tool that buys low-stock top-ups from the supplier.

    Top-ups do not depend on each other, so this tool is not all-or-nothing:
    it buys the cheapest first and stops when the cash runs out. It repeats
    the Inventory agent's minimum-stock check, so it never buys an item that
    is not low or more than the recommended top-up.
    """

    name = "buy_restock"
    description = (
        "Buys low-stock top-ups from the supplier and returns the restock "
        "report as text. Call it once with every item from the Inventory "
        "agent's low-stock list, using each exact catalog item_name and its "
        "top_up_quantity. It confirms that each item is at or below its "
        "minimum stock level, never buys more than the recommended top-up, "
        "prices each top-up, buys the cheapest first, and skips any that the "
        "remaining cash cannot cover. Never work out a cost or decide "
        "yourself which top-ups to buy."
    )
    inputs = {
        "items": {
            "type": "array",
            "description": "One entry for each item to top up.",
            "items": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "Exact catalog item_name, for example 'Cardstock'.",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Number of single units to buy, for example 500.",
                    },
                },
                "required": ["item_name", "quantity"],
            },
        },
        "request_date": {
            "type": "string",
            "description": "The date of the customer's request, in YYYY-MM-DD format.",
        },
    }
    output_type = "string"

    def forward(self, items: list, request_date: str) -> str:
        """
        Buy the top-ups that cash allows, cheapest first.

        Helper functions used: get_all_inventory (through
        find_low_stock_items), get_cash_balance, create_transaction and
        get_supplier_delivery_date.

        Args:
            items (list): One dict per item with 'item_name' and 'quantity'.
            request_date (str): The date of the customer's request (YYYY-MM-DD).

        Returns:
            str: The restock report, or a one-line message when the date or
                 the items cannot be read.
        """
        try:
            purchase_date = normalize_request_date(request_date)
        except ValueError:
            return "NOTHING BOUGHT. " + invalid_date_result(request_date)["message"]
        if not isinstance(items, list) or not items:
            return "NOTHING BOUGHT. No items were given. Pass one entry for each item to top up."

        # Noted for the Orchestrator's completion check: the top-up step ran.
        REQUEST_PROGRESS["top_ups_attempted"] = True

        # Starter helper inside: what is low right now, checked in code
        # rather than taken on trust from the list that was passed in.
        low_stock_by_name = {
            low_item["item_name"]: low_item for low_item in find_low_stock_items(purchase_date)
        }
        top_ups = [read_top_up_line(top_up_line, low_stock_by_name) for top_up_line in items]

        # Cheapest first, so limited cash covers as many items as possible.
        # Lines that cannot be read go last and are never bought.
        top_ups.sort(key=lambda top_up: (
            top_up["problem"] is not None,
            top_up["cost_cents"] or 0,
            top_up["item_name"],
        ))

        # Starter helper: read the cash once, then count it down in cents.
        cash_before_cents = to_cents(get_cash_balance(purchase_date))
        cash_left_cents = cash_before_cents

        # The report says what was bought. Costs and cash go to the log.
        report_lines = ["RESTOCK REPORT", f"Request date: {purchase_date}"]
        log_details = [f"cash before: {format_money(cash_before_cents)}"]
        bought_items = set()
        for line_number, top_up in enumerate(top_ups, start=1):
            report_lines += [
                f"Item {line_number}",
                f"- catalog_item: {top_up['item_name'] or 'n/a'}",
                f"- quantity_units: {'n/a' if top_up['quantity'] is None else top_up['quantity']}",
            ]

            if top_up["note"] is not None:
                report_lines.append(f"- note: {top_up['note']}")

            if top_up["problem"] is not None:
                report_lines += ["- bought: no", f"- reason: {top_up['problem']}"]
            elif top_up["item_name"] in bought_items:
                # The same item listed twice is only topped up once.
                report_lines += ["- bought: no", "- reason: This item was already topped up in this call."]
            elif top_up["cost_cents"] > cash_left_cents:
                report_lines += ["- bought: no", "- reason: Not enough cash for this top-up."]
                log_details.append(
                    f"not bought: {top_up['quantity']} units of {top_up['item_name']} would cost "
                    f"{format_money(top_up['cost_cents'])}, only {format_money(cash_left_cents)} is left"
                )
            else:
                # Starter helpers: record the purchase, dated the request date,
                # and look up when the supplier would deliver it.
                create_transaction(
                    top_up["item_name"],
                    "stock_orders",
                    top_up["quantity"],
                    top_up["cost_cents"] / 100,
                    purchase_date,
                )
                cash_left_cents -= top_up["cost_cents"]
                bought_items.add(top_up["item_name"])
                arrival_date = get_supplier_delivery_date(purchase_date, top_up["quantity"])
                report_lines += ["- bought: yes", f"- arrival: {arrival_date}"]
                log_details.append(
                    f"bought: {top_up['quantity']} units of {top_up['item_name']} "
                    f"for {format_money(top_up['cost_cents'])}"
                )

        report_lines.append(f"ITEMS BOUGHT: {len(bought_items)} of {len(top_ups)}")
        log_details += [
            f"total spent: {format_money(cash_before_cents - cash_left_cents)}",
            f"cash after: {format_money(cash_left_cents)}",
        ]
        log_internal(f"buy_restock, {purchase_date}", log_details)
        return "\n".join(report_lines)


# The tool instance handed to the Sales agent.
buy_restock = BuyRestockTool()


@tool
def financial_snapshot(as_of_date: str) -> dict:
    """
    Summarize the company's finances as of a date. For internal use only.

    Use this to report the cash and inventory position after orders have been
    recorded. Nothing in the result may be shown to a customer.

    Args:
        as_of_date: The date of the customer's request, in YYYY-MM-DD format.

    Returns:
        dict: Always contains 'status', which is one of:
            - 'ok': also contains 'as_of_date', 'cash_balance',
              'inventory_value', 'total_assets' (all in dollars),
              'items_in_stock' (how many stocked items have units on hand)
              and 'top_selling_products', a list of {'item_name',
              'total_units', 'total_revenue'} with the highest revenue first.
            - 'invalid_date': also contains 'message'.
    """
    try:
        report_date = normalize_request_date(as_of_date)
    except ValueError:
        return invalid_date_result(as_of_date)

    # Starter helper: cash, inventory value and the best sellers in one call.
    report = generate_financial_report(report_date)

    # The opening cash is stored as a sale with no item name, so the helper
    # lists it among the best sellers. Only real products are kept here.
    top_selling_products = [
        {
            "item_name": product["item_name"],
            "total_units": int(product["total_units"]),
            "total_revenue": round(float(product["total_revenue"]), 2),
        }
        for product in report["top_selling_products"]
        if isinstance(product["item_name"], str)
    ]

    # The item-by-item inventory list is left out to keep the result short.
    return {
        "status": "ok",
        "as_of_date": report_date,
        "cash_balance": round(float(report["cash_balance"]), 2),
        "inventory_value": round(float(report["inventory_value"]), 2),
        "total_assets": round(float(report["total_assets"]), 2),
        "items_in_stock": sum(1 for item in report["inventory_summary"] if item["stock"] > 0),
        "top_selling_products": top_selling_products,
    }


# Set up your agents and create an orchestration agent that will manage them.

# ---------------------------------------------------------------------------
# Inventory agent
# ---------------------------------------------------------------------------

# Shown to the orchestrator so it knows when and how to call this agent.
INVENTORY_AGENT_DESCRIPTION = (
    "Checks what the company can supply and by when. Give it the customer's "
    "requested items and quantities, the delivery deadline and the date of the "
    "request; it returns an availability report for every item. It can also "
    "list the items that have fallen to or below their minimum stock level."
)

# Standing instructions added to the agent's system prompt.
INVENTORY_AGENT_INSTRUCTIONS = """
You are the Inventory agent for Munder Difflin, a paper supply company.
You report what the company can supply and by when. You never quote prices,
never spend money, and never write a reply to the customer.

Every date you pass to a tool must be in YYYY-MM-DD format. Convert dates such
as 'April 15, 2025' before calling a tool.

TASK A - availability report (a customer request with items and a deadline)
1. Call list_catalog_stock with the request date. Its list is the full catalog.
2. Write one order line for each item the customer asked for. Use exactly one
   line per requested item: never split one item into two lines, never add an
   item, and never leave one out, even when we do not sell it.
   - requested: the customer's own words for the item, without the quantity.
   - item_name: the exact catalog item_name for the same kind of product,
     copied character for character from the catalog list. Use '' when no
     catalog item is the same kind of product.
   - quantity and unit: the number and the unit exactly as the customer wrote
     them (500 and 'reams' stays 500 and 'reams').
   - sheet_size: the size the customer stated ('A4', 'A3', '8.5x11',
     '24x36'), or '' when none was stated.
   How to choose item_name:
   - If the customer's words contain a catalog name, use that item
     ('colorful construction paper' is 'Construction paper').
   - Ignore colour, finish and quality words that the catalog name does not
     include (for example 'white', 'assorted colors', 'high-quality', 'heavy').
   - A size is not part of the name: 'glossy A4 paper' is ONE line with
     item_name 'Glossy paper' and sheet_size 'A4'.
   - When more than one catalog item fits, choose the plainest name. Do not
     pick an item whose name adds a weight, size or finish the customer did
     not ask for (for 'poster paper' choose 'Poster paper', not
     '220 gsm poster paper').
   - Customers use everyday names. Match them by meaning, not by spelling:
     'printer paper', 'printing paper' and 'copy paper' mean 'Standard copy
     paper' (or 'A4 paper' when the customer says A4); 'poster board' means
     'Poster paper'; 'napkins' means 'Paper napkins'.
   - Never substitute a different product and never invent a catalog name.
3. Call check_order_availability ONCE with all the order lines, the request
   date and the deadline. The tool applies every rule: it converts reams,
   rejects sizes and items we do not sell, checks stock and restock dates, and
   decides whether the order is complete. Do not do any of that yourself.
4. Give the tool's report as your final answer, copied exactly: every line, in
   the same order, with nothing added, removed or reworded.

TASK B - low-stock check (asked for after a sale has been recorded)
Call check_min_stock with the request date. List each low item with its
top_up_quantity, or state that no items are low.
"""

AVAILABILITY_TASK_HEADING = "Prepare an availability report for this customer request:"


def inventory_task_for(task: str) -> str:
    """
    Settle the task the Inventory agent works on when the Orchestrator calls it.

    Before an order is recorded, the only job the Orchestrator has for
    Inventory is the availability report. For that job the customer's
    original request is used in place of whatever the Orchestrator typed,
    so no item, quantity or date can be lost or reworded on the way. After a
    recorded order the task is the low-stock check, which is passed through.

    Args:
        task (str): The task text written by the Orchestrator.

    Returns:
        str: The task the Inventory agent should work on.
    """
    request_text = CURRENT_REQUEST["text"]
    if request_text and not REQUEST_PROGRESS["order_recorded"]:
        return f"{AVAILABILITY_TASK_HEADING}\n{request_text}"
    return task


class InventoryWorker(ToolCallingAgent):
    """
    The Inventory agent, with one addition for when it works for the Orchestrator.

    smolagents calls __call__ when a manager agent gives this agent a task.
    The task is passed through inventory_task_for first. Running the agent
    directly with run() is unchanged.
    """

    def __call__(self, task: str, **kwargs):
        """Work on the Orchestrator's task, reading the customer's own words."""
        task_to_do = inventory_task_for(task)
        # Kept so that a test can show what this agent was actually given.
        CURRENT_REQUEST["inventory_tasks"].append(task_to_do)
        return super().__call__(task_to_do, **kwargs)


# Worker agent that owns the three inventory tools. It only reads the database.
# Note: the `instructions` argument needs smolagents 1.20 or newer.
inventory_agent = InventoryWorker(
    tools=[list_catalog_stock, check_order_availability, check_min_stock],
    model=model,
    name="inventory_agent",
    description=INVENTORY_AGENT_DESCRIPTION,
    instructions=INVENTORY_AGENT_INSTRUCTIONS,
    max_steps=10,
)


# ---------------------------------------------------------------------------
# Quoting agent
# ---------------------------------------------------------------------------

# Shown to the orchestrator so it knows when and how to call this agent.
QUOTING_AGENT_DESCRIPTION = (
    "Prices an order. Give it the items that can be supplied, each with its "
    "exact catalog item_name and its quantity in single units as shown in the "
    "availability report, plus the event the customer mentioned. It returns "
    "an itemized quote with the bulk discount and the reason for it."
)

# Standing instructions added to the agent's system prompt.
QUOTING_AGENT_INSTRUCTIONS = """
You are the Quoting agent for Munder Difflin, a paper supply company.
You price the items you are given. You never check stock, never spend money,
and never write a reply to the customer.

Follow these steps in order:
1. Call calculate_quote ONCE with one line for each item you were given. Copy
   every item_name and quantity exactly as given. Never add, drop, rename or
   change an item. The tool does all the arithmetic: never work out a price,
   a discount or a total yourself.
2. Call find_similar_quotes ONCE with one or two single words taken from the
   task, such as the customer's event ('ceremony') and a main item
   ('cardstock').
3. Give your final answer in two parts:
   - First, the quote from calculate_quote, copied exactly: every line, in
     the same order, with nothing added, removed or reworded.
   - Then, as the last line, the 'history_note' text returned by
     find_similar_quotes, copied exactly. Do not write this line yourself.
   Past quotes are internal background. Never include another customer's
   wording, job, or amounts in your answer.

If calculate_quote answers with a one-line message instead of a quote, give
that message as your final answer.
"""

# Worker agent that owns the two quoting tools. It only reads the database.
quoting_agent = ToolCallingAgent(
    tools=[calculate_quote, find_similar_quotes],
    model=model,
    name="quoting_agent",
    description=QUOTING_AGENT_DESCRIPTION,
    instructions=QUOTING_AGENT_INSTRUCTIONS,
    max_steps=8,
)


# ---------------------------------------------------------------------------
# Sales agent
# ---------------------------------------------------------------------------

# Shown to the orchestrator so it knows when and how to call this agent.
SALES_AGENT_DESCRIPTION = (
    "Records orders and buys stock; it is the only agent that changes the "
    "database. To record an order, give it every item of a complete order, "
    "each with its exact catalog item_name and its quantity in single units, "
    "plus the request date and the delivery deadline; it records the whole "
    "order or refuses it and gives the reason. To restock, give it the "
    "Inventory agent's low-stock items with their top_up_quantity and the "
    "request date; it buys the top-ups that cash allows. It can also give an "
    "internal financial summary for a date."
)

# Standing instructions added to the agent's system prompt.
SALES_AGENT_INSTRUCTIONS = """
You are the Sales agent for Munder Difflin, a paper supply company.
You record customer orders and buy stock from the supplier. You never quote
prices, never decide whether an order can be filled, and never write a reply
to the customer. The tools make every decision; you pass the task to the
right tool and hand back its report.

Every date you pass to a tool must be in YYYY-MM-DD format. Convert dates such
as 'April 15, 2025' before calling a tool.

Do only the one task you were given.

TASK A - record an order (items with quantities, a request date and a deadline)
1. Call fulfill_order ONCE with one line for each item you were given, the
   request date and the deadline. Copy every item_name and quantity exactly
   as given. Never add, drop, rename or change an item or a quantity.
2. Give the tool's report as your final answer, copied exactly: every line, in
   the same order, with nothing added, removed or reworded.
The report is final. If it says 'ORDER RECORDED: no', the order was refused:
do not call fulfill_order again and do not change the order to make it pass.
If it says 'ORDER RECORDED: yes', the order is already in the database:
calling fulfill_order again would record it twice.
Never call buy_restock during this task.

TASK B - buy low-stock top-ups (a list of low items and a request date)
1. Call buy_restock ONCE with one entry for each item you were given and the
   request date. Copy every item_name and quantity exactly as given, even if
   one looks wrong: the tool checks them.
2. Give the tool's report as your final answer, copied exactly.
If the task says that no items are low, do not call any tool. Your final
answer is: 'No top-ups were needed.'

TASK C - internal financial summary (only when the task asks for one)
Call financial_snapshot with the request date. In your final answer give the
cash_balance, inventory_value, total_assets and top_selling_products exactly
as the tool returned them.
"""

# Worker agent that owns the three sales tools. It is the only agent whose
# tools write to the database.
sales_agent = ToolCallingAgent(
    tools=[fulfill_order, buy_restock, financial_snapshot],
    model=model,
    name="sales_agent",
    description=SALES_AGENT_DESCRIPTION,
    instructions=SALES_AGENT_INSTRUCTIONS,
    max_steps=6,
)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

# By default smolagents wraps every task given to a managed agent in a
# template that demands a three-part answer ('short version', 'extremely
# detailed version', 'additional context'). The workers must instead hand
# back their tool's report unchanged, so each one receives its task as
# plain text.
PLAIN_TASK_TEMPLATE = "{{task}}"
for worker_agent in (inventory_agent, quoting_agent, sales_agent):
    worker_agent.prompt_templates["managed_agent"]["task"] = PLAIN_TASK_TEMPLATE

# Shown if this agent is ever listed as a team member itself.
ORCHESTRATOR_DESCRIPTION = (
    "Handles one customer request from start to finish: it asks the "
    "Inventory, Quoting and Sales agents for what it needs, in order, and "
    "writes the reply to the customer."
)

# Standing instructions added to the agent's system prompt.
ORCHESTRATOR_INSTRUCTIONS = """
You are the Orchestrator for Munder Difflin, a paper supply company.
A customer request arrives as text. The date of the request is at the end of
it, like '(Date of request: 2025-04-01)'. You handle the request by giving
tasks to three team members, and then you write the reply to the customer
when the order is confirmed.

You have no tools of your own. Never work out stock, dates, prices, discounts
or totals yourself, and never invent them: every fact in your reply must come
from a team member's answer.

Give ONE task per step and wait for its answer before the next step. Write
each task exactly in the form shown. Follow the steps in this order.

STEP 1 - inventory_agent (always)
Task:
  Prepare an availability report for the customer's request.
The customer's request is attached to this task for you, exactly as it
arrived, so do not copy it out.
The answer is an AVAILABILITY REPORT. Its last line is 'ORDER COMPLETE: yes'
or 'ORDER COMPLETE: no'.

STEP 2 - quoting_agent (skip only when the report says that 0 items can be supplied)
Task:
  Prepare a quote for this order.
  Items that can be supplied:
  - <catalog_item>: <quantity_units> units
  Customer's event: <the event the customer mentioned, or 'not stated'>
  Request date: <Request date from the report>
Write one '- item: quantity units' line for each item whose
can_supply_by_deadline is yes, and no line for any other item. Copy
catalog_item and quantity_units exactly from the report. quantity_units is
the number to use, even when the customer wrote a different one: '500 reams'
in the request is 'quantity_units: 250000' in the report, so write 250000.
The answer is a QUOTE.

If the report says 'ORDER COMPLETE: no', go to STEP 6 now. Never give the
sales_agent an order that is not complete.

STEP 3 - sales_agent (only when the report says 'ORDER COMPLETE: yes')
Task:
  Record this complete order.
  Items:
  - <catalog_item>: <quantity_units> units
  Request date: <Request date from the report>
  Delivery deadline: <Deadline from the report>
Write one line for every item of the report.
The answer is a SALES REPORT with 'ORDER RECORDED: yes' or 'ORDER RECORDED: no'.
If it says no, the order was refused: do not try again, and go to STEP 6.

STEP 4 - inventory_agent (always after 'ORDER RECORDED: yes'; the order is
not finished until this is done)
Task:
  Low-stock check after a recorded sale.
  Request date: <Request date from the report>
The answer lists the items that are low, each with a top_up_quantity, or says
that no items are low.

STEP 5 - sales_agent (only when step 4 listed at least one low item)
Task:
  Buy these low-stock top-ups.
  Items:
  - <item_name>: <top_up_quantity> units
  Request date: <Request date from the report>

STEP 6 - give your final answer, as set out under THE REPLY below.
Your final answer is checked before it is accepted. If a step that applies
has not been done, or a confirmation leaves out a required detail, you will
be told what is missing: do that, then give your final answer again.

THE REPLY
When the Sales report says 'ORDER RECORDED: yes', write a short, polite
message to the customer in plain sentences:
- Start by saying that the order is confirmed.
- List each item with its quantity, unit price and line total from the quote.
- Give the subtotal, the bulk discount with the reason for it, and the total.
- Give the delivery date exactly as the DELIVERY DATE in the Sales report,
  written like 'April 8, 2025'. It can be as early as the day of the
  request. Never give the customer's deadline in its place.

In every other case the order was NOT placed. Do not write a reply to the
customer: the company's standard reply, with the reason for each item and
the partial-order offer, is written from the reports and sent for you. Give
exactly this as your final answer:
  ORDER NOT PLACED

Never put any of these in a reply: stock levels, supplier orders or
top-ups, the history note, other customers, the names of team members or
tools, or field names from the reports.
"""

# ---------------------------------------------------------------------------
# Completion check
# ---------------------------------------------------------------------------

# Report field names and agent names. None of them belongs in a reply to a
# customer; finding one means part of an internal report was copied across.
INTERNAL_WORDING = (
    "catalog_item", "quantity_units", "available_from", "can_supply_by_deadline",
    "amount_recorded", "top_up_quantity", "history note", "order recorded:",
    "order complete:", "inventory_agent", "quoting_agent", "sales_agent",
)


def format_long_date(iso_date: str) -> str:
    """
    Write a date the way a customer reply should give it.

    Args:
        iso_date (str): A date as YYYY-MM-DD.

    Returns:
        str: The date in words, such as 'April 8, 2025'.
    """
    date_value = datetime.fromisoformat(iso_date)
    return f"{date_value.strftime('%B')} {date_value.day}, {date_value.year}"


def date_spellings(iso_date: str) -> List[str]:
    """
    List the ways a reply may write one date, in lower case without commas.

    Args:
        iso_date (str): A date as YYYY-MM-DD.

    Returns:
        List[str]: For example '2025-04-08', 'april 8 2025', 'april 8th 2025',
                   '8 april 2025' and '4/8/2025'.
    """
    date_value = datetime.fromisoformat(iso_date)
    day, year = date_value.day, date_value.year
    month = date_value.strftime("%B").lower()
    short_month = date_value.strftime("%b").lower()
    ordinal = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return [
        iso_date,
        f"{month} {day} {year}",
        f"{month} {day:02d} {year}",
        f"{month} {day}{ordinal} {year}",
        f"{short_month} {day} {year}",
        f"{day} {month} {year}",
        f"{day}{ordinal} {month} {year}",
        f"{date_value.month}/{day}/{year}",
        f"{date_value.month:02d}/{day:02d}/{year}",
    ]


def format_item_lines(units_by_item: Dict[str, int]) -> str:
    """
    Write items the way every task to a worker lists them.

    Args:
        units_by_item (Dict[str, int]): Units keyed by catalog item name.

    Returns:
        str: One '- item: quantity units' line per item.
    """
    return "\n".join(f"- {item_name}: {units} units" for item_name, units in units_by_item.items())


def find_missing_step() -> Union[str, None]:
    """
    Name the first workflow step that should have been done but was not.

    This reads what the tools recorded as they ran (LAST_AVAILABILITY_CHECK
    and REQUEST_PROGRESS), so it does not depend on what any agent reports.
    Each message gives the exact task to send, with the item names and
    quantities filled in from that record, so the Orchestrator has nothing
    to retype from memory.

    Returns:
        str or None: An instruction for the Orchestrator, or None when every
                     step that applies has been done.
    """
    last_check = LAST_AVAILABILITY_CHECK
    progress = REQUEST_PROGRESS

    if not last_check:
        return "STEP 1 has not been done. Give the inventory_agent the availability task first."
    request_date = last_check["order_date"]

    # Step 2: a quote for exactly the items that can be supplied.
    suppliable_units = last_check["suppliable_units_by_item"]
    if suppliable_units and progress["quoted_units_by_item"] != suppliable_units:
        if progress["quoted_units_by_item"] is None:
            opening = "STEP 2 has not been done."
        else:
            opening = "The quote was not for exactly the items and quantities that can be supplied."
        if last_check["order_complete"]:
            closing = ""
        else:
            closing = (
                "\nAfter the quote, give ORDER NOT PLACED as your final answer. This "
                "order is not complete, so do not give it to the sales_agent."
            )
        return (
            f"{opening} Give the quoting_agent this task now, with exactly these item lines:\n"
            "Prepare a quote for this order.\n"
            "Items that can be supplied:\n"
            f"{format_item_lines(suppliable_units)}\n"
            "Customer's event: <the event the customer mentioned, or 'not stated'>\n"
            f"Request date: {request_date}"
            f"{closing}"
        )

    # Step 3: a complete order must be offered to Sales.
    if last_check["order_complete"] and not progress["sale_attempted"]:
        return (
            "STEP 3 has not been done. The report says ORDER COMPLETE: yes, so "
            "give the sales_agent this task now:\n"
            "Record this complete order.\n"
            "Items:\n"
            f"{format_item_lines(last_check['units_by_item'])}\n"
            f"Request date: {request_date}\n"
            f"Delivery deadline: {last_check['deadline_date']}"
        )

    # Steps 4 and 5: after a recorded sale, check stock and buy any top-ups.
    if progress["order_recorded"] and progress["low_stock_items"] is None:
        return (
            "STEP 4 has not been done. The order was recorded, so give the "
            "inventory_agent this task now:\n"
            "Low-stock check after a recorded sale.\n"
            f"Request date: {request_date}"
        )
    if progress["order_recorded"] and progress["low_stock_items"] and not progress["top_ups_attempted"]:
        top_up_units = {item["item_name"]: item["top_up_quantity"] for item in progress["low_stock_items"]}
        return (
            "STEP 5 has not been done. The low-stock check listed items, so give "
            "the sales_agent this task now:\n"
            "Buy these low-stock top-ups.\n"
            "Items:\n"
            f"{format_item_lines(top_up_units)}\n"
            f"Request date: {request_date}"
        )
    return None


# Wording that says an order was not placed. It must not appear in the
# confirmation of an order that was recorded.
NOT_PLACED_WORDING = (
    "cannot fill", "can not fill", "unable to fill", "not able to fill",
    "nothing has been ordered", "has not been placed", "order not placed",
)


def find_reply_problem(reply: str) -> Union[str, None]:
    """
    Decide whether the Orchestrator may finish with this reply.

    Every workflow step that applies must have been done. When an order was
    recorded, the reply is the confirmation the customer will receive, so it
    must also say that the order is confirmed, state the order total and the
    delivery date that the tools recorded, and contain no wording that
    belongs to a declined order or to an internal report.

    When no order was recorded, the wording is not checked, because the
    reply to a declined order is written in code (write_reply_in_code).

    Args:
        reply (str): The reply the Orchestrator wants to give.

    Returns:
        str or None: What is missing, written as an instruction for the
                     Orchestrator, or None when it may finish.
    """
    missing_step = find_missing_step()
    if missing_step is not None:
        return missing_step

    progress = REQUEST_PROGRESS
    if not progress["order_recorded"]:
        return None

    reply_text = " ".join(str(reply).lower().replace(",", "").split())

    if "confirm" not in reply_text:
        return "The order was recorded, so the reply must say that the order is confirmed."
    contradictions = [wording for wording in NOT_PLACED_WORDING if wording in reply_text]
    if contradictions:
        return (
            "The order was recorded, so the reply must not say that it was not "
            f"placed. Take this out: {', '.join(contradictions)}."
        )
    order_total = format_money(progress["order_total_cents"])
    if order_total.replace(",", "") not in reply_text:
        return f"The reply must state the order total, {order_total}."
    if not any(spelling in reply_text for spelling in date_spellings(progress["delivery_date"])):
        return (
            "The reply must give the delivery date from the Sales report, written "
            f"as {format_long_date(progress['delivery_date'])}. Do not replace it "
            "with the customer's deadline."
        )
    internal_words = [wording for wording in INTERNAL_WORDING if wording in reply_text]
    if internal_words:
        return f"Take this internal wording out of the reply: {', '.join(internal_words)}."
    return None


def reply_is_ready(final_answer, memory, agent=None) -> bool:
    """
    Final-answer check for the Orchestrator, in the form smolagents expects.

    smolagents calls this when the Orchestrator gives its final answer. If it
    raises, the answer is not accepted: the Orchestrator is shown the
    message and carries on, so it can do the missing step and reply again.

    Args:
        final_answer: The reply the Orchestrator wants to send.
        memory: The Orchestrator's memory (not needed: the tools' own record
            of what was done is used instead).
        agent: The Orchestrator itself (not needed).

    Returns:
        bool: True when the reply is ready.

    Raises:
        ValueError: With the instruction from find_reply_problem.
    """
    problem = find_reply_problem(str(final_answer))
    if problem is not None:
        raise ValueError(problem)
    return True


# The manager agent. It has no tools: its only actions are giving a task to
# a worker and giving the final answer. max_tool_threads=1 makes it run the
# workers one after another, in the order asked, even if the model requests
# two at once. final_answer_checks holds the completion check; max_steps
# leaves room to do a missed step and reply again.
orchestrator = ToolCallingAgent(
    tools=[],
    model=model,
    managed_agents=[inventory_agent, quoting_agent, sales_agent],
    name="orchestrator",
    description=ORCHESTRATOR_DESCRIPTION,
    instructions=ORCHESTRATOR_INSTRUCTIONS,
    max_steps=12,
    max_tool_threads=1,
    final_answer_checks=[reply_is_ready],
)


def describe_priced_items(units_by_item: Dict[str, int]) -> List[str]:
    """
    List items with their prices, discount and total, for a reply written in code.

    Args:
        units_by_item (Dict[str, int]): Units keyed by catalog item name.

    Returns:
        List[str]: One line per item, then the subtotal, the bulk discount
                   with its reason, and the total. All figures come from
                   price_order, the same pricing the quote and the sale use.
    """
    pricing = price_order(
        [{"item_name": item_name, "quantity": units} for item_name, units in units_by_item.items()]
    )
    priced_lines = [
        f"- {line['item_name']}: {line['quantity']:,} units at "
        f"{format_money(to_cents(line['unit_price']))} each, {format_money(to_cents(line['line_total']))}"
        for line in pricing["lines"]
    ]
    return priced_lines + [
        f"Subtotal: {format_money(to_cents(pricing['subtotal']))}",
        f"Bulk discount: {pricing['discount_percent']}% "
        f"(-{format_money(to_cents(pricing['discount_amount']))}). {pricing['discount_reason']}",
        f"Total: {format_money(to_cents(pricing['total']))}",
    ]


def write_reply_in_code() -> str:
    """
    Write the reply to the customer in code, from what the tools recorded.

    This is the reply for every order that was not recorded: it says that
    nothing was ordered, gives the reason for each item that cannot be
    supplied, and offers the rest as a priced partial order. It is also the
    fallback when the Orchestrator's confirmation cannot be used, in which
    case it confirms the order that is in the database.

    Every statement comes from the tools' record, so the reply is true
    whatever any agent wrote.

    Returns:
        str: The reply to send to the customer.
    """
    last_check = LAST_AVAILABILITY_CHECK
    progress = REQUEST_PROGRESS

    if progress["order_recorded"]:
        return "\n".join(
            ["Thank you for your order. It is confirmed.", ""]
            + describe_priced_items(last_check["units_by_item"])
            + ["", f"The delivery date is {format_long_date(progress['delivery_date'])}."]
        )

    reply_lines = [
        "Thank you for your request. We are not able to fill this order as "
        "requested, and nothing has been ordered or charged."
    ]
    try_again = "Please send your request again or contact us for help."
    if not last_check:
        # Nothing is known about the order, so no reason can be given.
        return f"{reply_lines[0]} {try_again}"

    # Why: the items that cannot be supplied, or the reason Sales gave.
    reasons = [
        f"- {line['requested']}: {line['reason']}" for line in last_check["lines"] if not line["can_supply"]
    ]
    if last_check["order_complete"]:
        # Every item could be supplied, so the refusal came from Sales.
        reasons += [f"- {reason}" for reason in progress["sale_refusal_reasons"]]
    if reasons:
        reply_lines += ["", "The reason:"] + reasons

    # What can still be offered when only part of the order was the problem.
    suppliable_units = last_check["suppliable_units_by_item"]
    offers_partial_order = bool(suppliable_units) and not last_check["order_complete"]
    if offers_partial_order:
        reply_lines += (
            ["", "We can supply these items as a partial order, which has not been placed:"]
            + describe_priced_items(suppliable_units)
            + ["", "If you would like them, please send a new request for these items."]
        )
    if not reasons and not offers_partial_order:
        reply_lines += ["", try_again]
    return "\n".join(reply_lines)


def handle_customer_request(request_text: str) -> str:
    """
    Run one customer request through the multi-agent system.

    Always send requests through this function. It clears the record of the
    previous request, gives the Inventory agent the customer's exact words,
    and decides which reply goes out:
    - an order was recorded and the Orchestrator's confirmation passes the
      completion check: the Orchestrator's confirmation;
    - no order was recorded: the reply written in code, with the reasons
      and the partial-order offer;
    - the Orchestrator failed, or ran out of steps with a reply that does
      not pass the check: the reply written in code, as a fallback.

    Args:
        request_text (str): The customer's request, ending with
            '(Date of request: YYYY-MM-DD)'.

    Returns:
        str: The reply to send to the customer.
    """
    start_new_request()
    CURRENT_REQUEST.update({"text": request_text, "reply_source": None, "inventory_tasks": []})
    try:
        reply = str(orchestrator.run(request_text))
        problem = find_reply_problem(reply)
    except Exception as error:
        reply, problem = "", f"the system raised an error: {error}"

    if problem is not None:
        # The reason goes to the console log; the customer gets a true reply.
        print(f"FALLBACK REPLY USED (handle_customer_request): {problem}")
        CURRENT_REQUEST["reply_source"] = "code, as a fallback"
        return write_reply_in_code()
    if not REQUEST_PROGRESS["order_recorded"]:
        CURRENT_REQUEST["reply_source"] = "code, for an order that was not placed"
        return write_reply_in_code()
    CURRENT_REQUEST["reply_source"] = "Orchestrator"
    return reply


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############

    # The four agents are created when this file is loaded (see above), so
    # there is nothing to build here. Their step-by-step console output is
    # turned down so that this run's own progress lines stay readable. The
    # tools' INTERNAL LOG lines and any errors are still printed.
    for agent in (orchestrator, inventory_agent, quoting_agent, sales_agent):
        agent.logger.level = LogLevel.ERROR

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        # One call handles the whole request: availability, quote, sale,
        # low-stock top-ups and the reply to the customer.
        response = handle_customer_request(request_with_date)

        # Read from the tools' own record, so the results file shows plainly
        # which requests ended in a recorded order.
        order_recorded = "yes" if REQUEST_PROGRESS["order_recorded"] else "no"

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Order recorded: {order_recorded}")
        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "order_recorded": order_recorded,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
