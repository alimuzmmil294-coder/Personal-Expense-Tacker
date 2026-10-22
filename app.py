"""
Personal Expense Tracker
Built with Streamlit, Cloud Firestore (Firebase Admin SDK), and Google Gemini (google-genai).
"""

from __future__ import annotations

import datetime
from decimal import Decimal, InvalidOperation
import json
import logging
import os
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

# Streamlit secrets or OS environment variable fallback
DEFAULT_CURRENCY: str = (
    st.secrets.get("APP_DEFAULT_CURRENCY")
    or os.getenv("APP_DEFAULT_CURRENCY", "PKR")
)

ENABLE_AI_FEATURES: bool = str(
    st.secrets.get("ENABLE_AI_FEATURES")
    or os.getenv("ENABLE_AI_FEATURES", "true")
).lower() == "true"

GEMINI_API_KEY: Optional[str] = (
    st.secrets.get("GEMINI_API_KEY")
    or os.getenv("GEMINI_API_KEY")
)

FIREBASE_CREDENTIALS_PATH: Optional[str] = (
    st.secrets.get("FIREBASE_CREDENTIALS_PATH")
    or os.getenv("FIREBASE_CREDENTIALS_PATH")
)

# ==============================================================================
# 2. FIREBASE FIRESTORE DATA PERSISTENCE LAYER
# ==============================================================================


@st.cache_resource
def get_firestore_client():
    """Initializes and returns the Firestore client singleton via Firebase Admin SDK."""
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            cred = None

            # 1. Check Streamlit secrets for parsed dictionary credentials
            if "firebase_service_account" in st.secrets:
                cred_dict = dict(st.secrets["firebase_service_account"])
                cred = credentials.Certificate(cred_dict)
            # 2. Check JSON string in environment variable
            elif os.getenv("FIREBASE_CREDENTIALS_JSON"):
                cred_dict = json.loads(os.getenv("FIREBASE_CREDENTIALS_JSON", "{}"))
                cred = credentials.Certificate(cred_dict)
            # 3. Check JSON file path
            elif FIREBASE_CREDENTIALS_PATH and os.path.exists(FIREBASE_CREDENTIALS_PATH):
                cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
            else:
                logger.warning("No explicit Firebase credentials found; using Application Default.")
                cred = credentials.ApplicationDefault()

            firebase_admin.initialize_app(cred)

        db = firestore.client()
        logger.info("Firestore client initialized successfully.")
        return db
    except Exception as e:
        logger.error(f"Failed to initialize Firebase Admin SDK: {e}")
        return None


class FirestoreService:
    """Encapsulates all database operations for isolated user expense collections."""

    @staticmethod
    def _collection(user_id: str):
        db = get_firestore_client()
        if db is None:
            raise RuntimeError(
                "Database connection unavailable. Verify your Firebase service account configuration."
            )
        return db.collection("users").document(user_id).collection("expenses")

    @classmethod
    def get_all(cls, user_id: str) -> List[Dict[str, Any]]:
        collection = cls._collection(user_id)
        docs = collection.order_by("transaction_date", direction="DESCENDING").stream()
        records = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            records.append(data)
        return records

    @classmethod
    def create(cls, user_id: str, data: Dict[str, Any]) -> str:
        collection = cls._collection(user_id)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        data["created_at"] = now
        data["updated_at"] = now
        doc_ref = collection.document()
        doc_ref.set(data)
        return doc_ref.id

    @classmethod
    def update(cls, user_id: str, doc_id: str, data: Dict[str, Any]) -> None:
        collection = cls._collection(user_id)
        doc_ref = collection.document(doc_id)
        if not doc_ref.get().exists:
            raise FileNotFoundError(f"Expense {doc_id} does not exist.")
        data["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        doc_ref.update(data)

    @classmethod
    def delete(cls, user_id: str, doc_id: str) -> None:
        collection = cls._collection(user_id)
        doc_ref = collection.document(doc_id)
        if not doc_ref.get().exists:
            raise FileNotFoundError(f"Expense {doc_id} does not exist.")
        doc_ref.delete()


# ==============================================================================
# 3. AI INTEGRATION LAYER (GOOGLE GENAI SDK)
# ==============================================================================


class GeminiAnalyticsService:
    """Handles financial analysis generation using the official Google Gen AI SDK."""

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
            return "Gemini API key is not configured. Set GEMINI_API_KEY to enable insights."

        if expense_count == 0:
            return "No transactions found to analyze. Please add some expenses first."

        try:
            from google import genai

            client = genai.Client(api_key=GEMINI_API_KEY)

            # Sanitized aggregation prompt: no personal descriptions or notes are transmitted
            prompt = f"""
You are an expert personal finance budgeting analyst.
Review the following aggregated spending summary and generate 3 to 4 concise, actionable insights or recommendations.
Keep the tone professional, encouraging, and clear. Note that this is automated guidance, not formal financial advice.

Summary Data:
- Total Spending: {currency} {total_spent:,.2f}
- Transaction Count: {expense_count}
- Category Breakdown:
{json.dumps(category_breakdown, indent=2)}

Format with markdown bullet points. Highlight key spending categories or areas of potential savings.
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
# 4. SESSION STATE & CONTROLLER LOGIC
# ==============================================================================


def init_session_state():
    """Initializes persistent UI and editing states."""
    if "user_id" not in st.session_state:
        st.session_state.user_id = "default_user"
    if "editing_expense" not in st.session_state:
        st.session_state.editing_expense = None
    if "pending_delete_id" not in st.session_state:
        st.session_state.pending_delete_id = None
    if "ai_insights" not in st.session_state:
        st.session_state.ai_insights = ""


def reset_edit_state():
    st.session_state.editing_expense = None


# ==============================================================================
# 5. USER INTERFACE & PRESENTATION LAYER
# ==============================================================================


def main():
    st.set_page_config(
        page_title="Personal Expense Tracker",
        page_icon="💳",
        layout="wide",
    )
    init_session_state()

    st.title("💳 Personal Expense Tracker")
    st.caption("Cloud-persisted budgeting powered by Google Gemini AI & Firebase Firestore")

    # Verify Database connectivity early
    db = get_firestore_client()
    if db is None:
        st.error(
            "⚠️ Firebase is not connected. Please supply valid credentials in `.streamlit/secrets.toml` or environment variables."
        )
        st.stop()

    # Load records for current user
    try:
        expenses = FirestoreService.get_all(st.session_state.user_id)
    except Exception as e:
        st.error(f"Failed to fetch records from Firestore: {e}")
        expenses = []

    # Calculate High-Level Metrics
    total_spend = sum(float(e.get("amount", 0.0)) for e in expenses)
    total_tx = len(expenses)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Spending", f"{DEFAULT_CURRENCY} {total_spend:,.2f}")
    col2.metric("Total Transactions", total_tx)
    col3.metric("Configured Currency", DEFAULT_CURRENCY)

    st.divider()

    # --------------------------------------------------------------------------
    # Expense Form (Create / Edit)
    # --------------------------------------------------------------------------
    is_editing = st.session_state.editing_expense is not None
    form_title = "✏️ Edit Expense" if is_editing else "➕ Add New Expense"
    edit_data = st.session_state.editing_expense or {}

    with st.expander(form_title, expanded=is_editing or total_tx == 0):
        with st.form("expense_form", clear_on_submit=not is_editing):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                title_val = st.text_input(
                    "Title / Description *",
                    value=edit_data.get("title", ""),
                    placeholder="e.g. Office Lunch, Groceries",
                )
                amt_val = st.text_input(
                    f"Amount ({DEFAULT_CURRENCY}) *",
                    value=str(edit_data.get("amount", "")),
                    placeholder="e.g. 1500.00",
                )
            with f_col2:
                default_cat_idx = (
                    SUPPORTED_CATEGORIES.index(edit_data["category"])
                    if edit_data.get("category") in SUPPORTED_CATEGORIES
                    else 0
                )
                category_val = st.selectbox(
                    "Category *",
                    SUPPORTED_CATEGORIES,
                    index=default_cat_idx,
                )
                default_date = (
                    datetime.date.fromisoformat(edit_data["transaction_date"])
                    if "transaction_date" in edit_data
                    else datetime.date.today()
                )
                date_val = st.date_input("Transaction Date *", value=default_date)
            with f_col3:
                default_pm_idx = (
                    PAYMENT_METHODS.index(edit_data["payment_method"])
                    if edit_data.get("payment_method") in PAYMENT_METHODS
                    else 0
                )
                payment_val = st.selectbox(
                    "Payment Method *",
                    PAYMENT_METHODS,
                    index=default_pm_idx,
                )
                notes_val = st.text_input(
                    "Optional Notes",
                    value=edit_data.get("notes", ""),
                    placeholder="e.g. Team reimbursement pending",
                )

            submit_label = "Update Expense" if is_editing else "Save Expense"
            submitted = st.form_submit_button(submit_label, type="primary")

            if submitted:
                # Validation: Title
                if not title_val.strip():
                    st.error("Title cannot be empty.")
                    st.stop()

                # Validation: Amount Decimal Safety
                try:
                    dec_amt = Decimal(amt_val.strip())
                    if dec_amt <= Decimal("0"):
                        st.error("Amount must be greater than zero.")
                        st.stop()
                except (InvalidOperation, ValueError):
                    st.error("Please enter a valid numeric amount.")
                    st.stop()

                payload = {
                    "title": title_val.strip(),
                    "amount": float(dec_amt),
                    "currency": DEFAULT_CURRENCY,
                    "category": category_val,
                    "transaction_date": date_val.isoformat(),
                    "payment_method": payment_val,
                    "notes": notes_val.strip(),
                }

                try:
                    if is_editing:
                        FirestoreService.update(
                            st.session_state.user_id,
                            edit_data["id"],
                            payload,
                        )
                        st.success("Expense updated successfully.")
                        reset_edit_state()
                    else:
                        FirestoreService.create(st.session_state.user_id, payload)
                        st.success("Expense recorded successfully.")
                    st.rerun()
                except Exception as err:
                    st.error(f"Database error: {err}")

        if is_editing:
            if st.button("Cancel Edit"):
                reset_edit_state()
                st.rerun()

    # --------------------------------------------------------------------------
    # Deletion Confirmation Dialog
    # --------------------------------------------------------------------------
    if st.session_state.pending_delete_id:
        st.warning(f"⚠️ Are you sure you want to permanently delete this expense?")
        del_col1, del_col2, _ = st.columns([1, 1, 4])
        with del_col1:
            if st.button("Yes, Delete", type="primary"):
                try:
                    FirestoreService.delete(
                        st.session_state.user_id,
                        st.session_state.pending_delete_id,
                    )
                    st.session_state.pending_delete_id = None
                    st.success("Record deleted successfully.")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Failed to delete record: {ex}")
        with del_col2:
            if st.button("Cancel"):
                st.session_state.pending_delete_id = None
                st.rerun()

    # --------------------------------------------------------------------------
    # Search, Filter, and Expense Table
    # --------------------------------------------------------------------------
    st.subheader("📑 Expense Records")

    s_col1, s_col2, s_col3 = st.columns([2, 1, 1])
    with s_col1:
        search_kw = st.text_input("🔍 Search Title or Notes", placeholder="Filter by text...")
    with s_col2:
        filter_cat = st.selectbox("Category Filter", ["All"] + SUPPORTED_CATEGORIES)
    with s_col3:
        sort_choice = st.selectbox(
            "Sort By",
            ["Date (Newest)", "Date (Oldest)", "Amount (Highest)", "Amount (Lowest)"],
        )

    # Filter pipeline
    display_expenses = list(expenses)
    if filter_cat != "All":
        display_expenses = [e for e in display_expenses if e.get("category") == filter_cat]

    if search_kw.strip():
        kw = search_kw.strip().lower()
        display_expenses = [
            e
            for e in display_expenses
            if kw in str(e.get("title", "")).lower()
            or kw in str(e.get("notes", "")).lower()
        ]

    if sort_choice == "Date (Newest)":
        display_expenses.sort(key=lambda x: str(x.get("transaction_date", "")), reverse=True)
    elif sort_choice == "Date (Oldest)":
        display_expenses.sort(key=lambda x: str(x.get("transaction_date", "")))
    elif sort_choice == "Amount (Highest)":
        display_expenses.sort(key=lambda x: float(x.get("amount", 0.0)), reverse=True)
    elif sort_choice == "Amount (Lowest)":
        display_expenses.sort(key=lambda x: float(x.get("amount", 0.0)))

    if not display_expenses:
        st.info("No expense transactions match the current view criteria.")
    else:
        # Display Table with Action Columns
        for item in display_expenses:
            r_col1, r_col2, r_col3, r_col4, r_col5, r_col6 = st.columns(
                [1.5, 3, 1.5, 2, 1, 1]
            )
            r_col1.write(f"📅 `{item.get('transaction_date')}`")
            r_col2.write(f"**{item.get('title')}**\n*{item.get('notes') or '-'}*")
            r_col3.caption(f"🏷️ {item.get('category')}")
            r_col4.write(f"**{item.get('currency')} {float(item.get('amount', 0.0)):,.2f}**")

            with r_col5:
                if st.button("Edit", key=f"edit_{item['id']}"):
                    st.session_state.editing_expense = item
                    st.rerun()

            with r_col6:
                if st.button("Delete", key=f"del_{item['id']}"):
                    st.session_state.pending_delete_id = item["id"]
                    st.rerun()
            st.divider()

    # --------------------------------------------------------------------------
    # AI Spending Analysis (Gemini)
    # --------------------------------------------------------------------------
    st.subheader("🤖 AI Spending Insights")
    st.caption("Automated budget evaluation powered by Google Gemini.")

    if st.button("Generate AI Insights", type="secondary"):
        if not expenses:
            st.warning("Please record some expenses before generating spending insights.")
        else:
            with st.spinner("Analyzing spending distribution with Gemini..."):
                cat_breakdown: Dict[str, float] = {}
                for exp in expenses:
                    c = exp.get("category", "Other")
                    val = float(exp.get("amount", 0.0))
                    cat_breakdown[c] = cat_breakdown.get(c, 0.0) + val

                insights = GeminiAnalyticsService.generate_spending_insights(
                    currency=DEFAULT_CURRENCY,
                    total_spent=total_spend,
                    category_breakdown=cat_breakdown,
                    expense_count=len(expenses),
                )
                st.session_state.ai_insights = insights

    if st.session_state.ai_insights:
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; padding: 16px; border-radius: 8px; border-left: 4px solid #6366f1;">
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
