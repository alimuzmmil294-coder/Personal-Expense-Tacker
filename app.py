"""
Personal Expense Tracker
Built with Streamlit, SQLite (embedded local persistence), and Google Gemini (google-genai).
"""

from __future__ import annotations

import datetime
from decimal import Decimal, InvalidOperation
import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ExpenseTracker")

# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================

DB_FILE_PATH: str = os.getenv("EXPENSE_DB_PATH", "expenses.db")
DEFAULT_CURRENCY: str = (
    st.secrets.get("APP_DEFAULT_CURRENCY")
    if hasattr(st, "secrets") and "APP_DEFAULT_CURRENCY" in st.secrets
    else os.getenv("APP_DEFAULT_CURRENCY", "PKR")
)

ENABLE_AI_FEATURES: bool = str(
    st.secrets.get("ENABLE_AI_FEATURES")
    if hasattr(st, "secrets") and "ENABLE_AI_FEATURES" in st.secrets
    else os.getenv("ENABLE_AI_FEATURES", "true")
).lower() == "true"

GEMINI_API_KEY: Optional[str] = (
    st.secrets.get("GEMINI_API_KEY")
    if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets
    else os.getenv("GEMINI_API_KEY")
)

SUPPORTED_CATEGORIES: List[str] = [
    "Food",
    "Transport",
    "Education",
    "Shopping",
    "Bills",
    "Healthcare",
    "Other",
]

PAYMENT_METHODS: List[str] = ["Cash", "Card", "Bank Transfer"]

# ==============================================================================
# 2. SQLITE DATA PERSISTENCE LAYER
# ==============================================================================


class SQLiteDatabaseService:
    """Encapsulates thread-safe, local, zero-credential SQLite database operations."""

    @staticmethod
    def get_connection() -> sqlite3.Connection:
        conn = sqlite3.connect(DB_FILE_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    @classmethod
    def initialize_db(cls) -> None:
        """Creates the expenses table and indexes if they do not exist."""
        with cls.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS expenses (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    amount REAL NOT NULL,
                    currency TEXT NOT NULL,
                    category TEXT NOT NULL,
                    transaction_date TEXT NOT NULL,
                    payment_method TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_date ON expenses (user_id, transaction_date DESC)"
            )
            conn.commit()
            logger.info("SQLite database schema verified successfully.")

    @classmethod
    def get_all(cls, user_id: str) -> List[Dict[str, Any]]:
        with cls.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM expenses
                WHERE user_id = ?
                ORDER BY transaction_date DESC, created_at DESC
                """,
                (user_id,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    @classmethod
    def get_by_id(cls, user_id: str, expense_id: str) -> Optional[Dict[str, Any]]:
        with cls.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM expenses WHERE id = ? AND user_id = ?",
                (expense_id, user_id),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    @classmethod
    def create(cls, user_id: str, data: Dict[str, Any]) -> str:
        import uuid

        doc_id = str(uuid.uuid4())
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with cls.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO expenses (
                    id, user_id, title, amount, currency, category,
                    transaction_date, payment_method, notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    user_id,
                    data["title"],
                    float(data["amount"]),
                    data["currency"],
                    data["category"],
                    data["transaction_date"],
                    data["payment_method"],
                    data.get("notes", ""),
                    now,
                    now,
                ),
            )
            conn.commit()
            return doc_id

    @classmethod
    def update(cls, user_id: str, expense_id: str, data: Dict[str, Any]) -> None:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with cls.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE expenses
                SET title = ?, amount = ?, currency = ?, category = ?,
                    transaction_date = ?, payment_method = ?, notes = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    data["title"],
                    float(data["amount"]),
                    data["currency"],
                    data["category"],
                    data["transaction_date"],
                    data["payment_method"],
                    data.get("notes", ""),
                    now,
                    expense_id,
                    user_id,
                ),
            )
            if cursor.rowcount == 0:
                raise FileNotFoundError(f"Expense {expense_id} not found for user {user_id}")
            conn.commit()

    @classmethod
    def delete(cls, user_id: str, expense_id: str) -> None:
        with cls.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM expenses WHERE id = ? AND user_id = ?",
                (expense_id, user_id),
            )
            if cursor.rowcount == 0:
                raise FileNotFoundError(f"Expense {expense_id} not found for user {user_id}")
            conn.commit()


# ==============================================================================
# 3. AI INTEGRATION LAYER (GOOGLE GENAI SDK)
# ==============================================================================


class GeminiAnalyticsService:
    """Generates spending insights via Google Gen AI SDK without sending PII."""

    @staticmethod
    def generate_spending_insights(
        currency: str,
        total_spent: float,
        category_breakdown: Dict[str, float],
        expense_count: int,
    ) -> str:
        if not ENABLE_AI_FEATURES:
            return "AI Insights are disabled via configuration."

        if not GEMINI_API_KEY:
            return "Gemini API key is not configured. Set GEMINI_API_KEY in your environment or secrets."

        if expense_count == 0:
            return "No transactions found. Add some expenses first to analyze your spending."

        try:
            from google import genai

            client = genai.Client(api_key=GEMINI_API_KEY)

            # Strict privacy preservation: transmit only aggregated sums
            prompt = f"""
You are an expert personal finance budgeting analyst.
Review the following aggregated spending summary and generate 3 to 4 concise, actionable insights or recommendations.
Keep the tone encouraging, professional, and clear. Note that this is automated guidance, not formal financial advice.

Summary Data:
- Total Spending: {currency} {total_spent:,.2f}
- Transaction Count: {expense_count}
- Category Breakdown:
{json.dumps(category_breakdown, indent=2)}

Format with markdown bullet points. Highlight anomalies or dominant expense categories.
"""
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return response.text or "No response received from the AI model."
        except Exception as err:
            logger.error(f"Gemini API request failed: {err}")
            return f"Unable to generate spending insights at this time: {str(err)}"


# ==============================================================================
# 4. CONTROLLER & STATE INITIALIZATION
# ==============================================================================


def init_session_state() -> None:
    """Initializes Streamlit session state keys."""
    if "user_id" not in st.session_state:
        st.session_state.user_id = "default_user"
    if "editing_expense" not in st.session_state:
        st.session_state.editing_expense = None
    if "pending_delete_id" not in st.session_state:
        st.session_state.pending_delete_id = None
    if "ai_insights" not in st.session_state:
        st.session_state.ai_insights = ""


def reset_edit_state() -> None:
    st.session_state.editing_expense = None


# ==============================================================================
# 5. USER INTERFACE COMPONENTS
# ==============================================================================


def main() -> None:
    st.set_page_config(
        page_title="Personal Expense Tracker",
        page_icon="💰",
        layout="wide",
    )

    # Initialize SQLite Database
    SQLiteDatabaseService.initialize_db()
    init_session_state()

    st.title("💰 Personal Expense Tracker")
    st.caption("Self-contained local budgeting engine powered by SQLite & Google Gemini AI")

    # Fetch User Records
    try:
        expenses = SQLiteDatabaseService.get_all(st.session_state.user_id)
    except Exception as e:
        st.error(f"Database read failure: {e}")
        expenses = []

    # KPI Metrics
    total_spent = sum(float(e.get("amount", 0.0)) for e in expenses)
    total_tx = len(expenses)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Spending", f"{DEFAULT_CURRENCY} {total_spent:,.2f}")
    col2.metric("Total Transactions", total_tx)
    col3.metric("Currency", DEFAULT_CURRENCY)

    st.divider()

    # --------------------------------------------------------------------------
    # Expense Form (Create & Update)
    # --------------------------------------------------------------------------
    is_editing = st.session_state.editing_expense is not None
    form_heading = "✏️ Edit Expense" if is_editing else "➕ Add New Expense"
    edit_data = st.session_state.editing_expense or {}

    with st.expander(form_heading, expanded=is_editing or total_tx == 0):
        with st.form("expense_form", clear_on_submit=not is_editing):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                title_input = st.text_input(
                    "Title / Description *",
                    value=edit_data.get("title", ""),
                    placeholder="e.g. Groceries or Fuel",
                )
                amount_input = st.text_input(
                    f"Amount ({DEFAULT_CURRENCY}) *",
                    value=str(edit_data.get("amount", "")),
                    placeholder="e.g. 1500.50",
                )
            with f_col2:
                default_cat_idx = (
                    SUPPORTED_CATEGORIES.index(edit_data["category"])
                    if edit_data.get("category") in SUPPORTED_CATEGORIES
                    else 0
                )
                category_input = st.selectbox(
                    "Category *",
                    SUPPORTED_CATEGORIES,
                    index=default_cat_idx,
                )
                default_date = (
                    datetime.date.fromisoformat(edit_data["transaction_date"])
                    if "transaction_date" in edit_data
                    else datetime.date.today()
                )
                date_input = st.date_input("Transaction Date *", value=default_date)
            with f_col3:
                default_pm_idx = (
                    PAYMENT_METHODS.index(edit_data["payment_method"])
                    if edit_data.get("payment_method") in PAYMENT_METHODS
                    else 0
                )
                pm_input = st.selectbox(
                    "Payment Method *",
                    PAYMENT_METHODS,
                    index=default_pm_idx,
                )
                notes_input = st.text_input(
                    "Optional Notes",
                    value=edit_data.get("notes", ""),
                    placeholder="e.g. Monthly utility bill",
                )

            submit_label = "Update Expense" if is_editing else "Save Expense"
            submitted = st.form_submit_button(submit_label, type="primary")

            if submitted:
                # Validation: Title
                clean_title = title_input.strip()
                if not clean_title:
                    st.error("Expense title cannot be empty.")
                    st.stop()

                # Validation: Exact Numeric & Positive Check
                try:
                    dec_amt = Decimal(amount_input.strip())
                    if dec_amt <= Decimal("0"):
                        st.error("Expense amount must be strictly greater than zero.")
                        st.stop()
                except (InvalidOperation, ValueError):
                    st.error("Please provide a valid numeric amount.")
                    st.stop()

                payload = {
                    "title": clean_title,
                    "amount": float(dec_amt),
                    "currency": DEFAULT_CURRENCY,
                    "category": category_input,
                    "transaction_date": date_input.isoformat(),
                    "payment_method": pm_input,
                    "notes": notes_input.strip(),
                }

                try:
                    if is_editing:
                        SQLiteDatabaseService.update(
                            st.session_state.user_id,
                            edit_data["id"],
                            payload,
                        )
                        st.success("Expense successfully updated.")
                        reset_edit_state()
                    else:
                        SQLiteDatabaseService.create(st.session_state.user_id, payload)
                        st.success("Expense saved.")
                    st.rerun()
                except Exception as err:
                    st.error(f"Persistence error: {err}")

        if is_editing:
            if st.button("Cancel Edit"):
                reset_edit_state()
                st.rerun()

    # --------------------------------------------------------------------------
    # Delete Confirmation Dialog
    # --------------------------------------------------------------------------
    if st.session_state.pending_delete_id:
        st.warning("⚠️ Are you sure you want to permanently delete this expense?")
        d_col1, d_col2, _ = st.columns([1, 1, 4])
        with d_col1:
            if st.button("Yes, Delete", type="primary"):
                try:
                    SQLiteDatabaseService.delete(
                        st.session_state.user_id,
                        st.session_state.pending_delete_id,
                    )
                    st.session_state.pending_delete_id = None
                    st.success("Expense deleted.")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Deletion failed: {ex}")
        with d_col2:
            if st.button("Cancel"):
                st.session_state.pending_delete_id = None
                st.rerun()

    # --------------------------------------------------------------------------
    # Search, Filters, and Expenses Display
    # --------------------------------------------------------------------------
    st.subheader("📑 Expense Records")

    s1, s2, s3 = st.columns([2, 1, 1])
    with s1:
        query_text = st.text_input("🔍 Search Title or Notes", placeholder="Search...")
    with s2:
        selected_category = st.selectbox("Category Filter", ["All"] + SUPPORTED_CATEGORIES)
    with s3:
        sort_rule = st.selectbox(
            "Sort By",
            ["Date (Newest)", "Date (Oldest)", "Amount (Highest)", "Amount (Lowest)"],
        )

    # Filter Pipeline
    filtered = list(expenses)
    if selected_category != "All":
        filtered = [e for e in filtered if e.get("category") == selected_category]

    if query_text.strip():
        kw = query_text.strip().lower()
        filtered = [
            e
            for e in filtered
            if kw in str(e.get("title", "")).lower()
            or kw in str(e.get("notes", "")).lower()
        ]

    if sort_rule == "Date (Newest)":
        filtered.sort(key=lambda x: str(x.get("transaction_date", "")), reverse=True)
    elif sort_rule == "Date (Oldest)":
        filtered.sort(key=lambda x: str(x.get("transaction_date", "")))
    elif sort_rule == "Amount (Highest)":
        filtered.sort(key=lambda x: float(x.get("amount", 0.0)), reverse=True)
    elif sort_rule == "Amount (Lowest)":
        filtered.sort(key=lambda x: float(x.get("amount", 0.0)))

    if not filtered:
        st.info("No expense records match the specified filters.")
    else:
        for item in filtered:
            r1, r2, r3, r4, r5, r6 = st.columns([1.5, 3, 1.5, 2, 1, 1])
            r1.write(f"📅 `{item.get('transaction_date')}`")
            r2.write(f"**{item.get('title')}**\n*{item.get('notes') or '-'}*")
            r3.caption(f"🏷️ {item.get('category')} ({item.get('payment_method')})")
            r4.write(f"**{item.get('currency')} {float(item.get('amount', 0.0)):,.2f}**")

            with r5:
                if st.button("Edit", key=f"edit_{item['id']}"):
                    st.session_state.editing_expense = item
                    st.rerun()

            with r6:
                if st.button("Delete", key=f"del_{item['id']}"):
                    st.session_state.pending_delete_id = item["id"]
                    st.rerun()
            st.divider()

    # --------------------------------------------------------------------------
    # AI Spending Analysis (Gemini)
    # --------------------------------------------------------------------------
    st.subheader("🤖 AI Spending Insights")
    st.caption("Automated budgeting and categorization review powered by Google Gemini.")

    if st.button("Generate AI Insights", type="secondary"):
        if not expenses:
            st.warning("Please record some expenses before requesting an AI review.")
        else:
            with st.spinner("Analyzing spending breakdown with Gemini..."):
                cat_summary: Dict[str, float] = {}
                for exp in expenses:
                    c = exp.get("category", "Other")
                    val = float(exp.get("amount", 0.0))
                    cat_summary[c] = cat_summary.get(c, 0.0) + val

                insights = GeminiAnalyticsService.generate_spending_insights(
                    currency=DEFAULT_CURRENCY,
                    total_spent=total_spent,
                    category_breakdown=cat_summary,
                    expense_count=len(expenses),
                )
                st.session_state.ai_insights = insights

    if st.session_state.ai_insights:
        st.markdown(
            f"""
            <div style="background-color: #f8fafc; padding: 16px; border-radius: 8px; border-left: 4px solid #3b82f6; color: #1e293b;">
            {st.session_state.ai_insights}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Note: Insights are generated by AI using aggregated category summaries. Always review personal financial decisions independently."
        )


if __name__ == "__main__":
    main()
