"""
Personal Expense Tracker
Built with Reflex, Cloud Firestore (Firebase Admin SDK), and Google Gemini (google-genai).
"""

from __future__ import annotations

import datetime
from decimal import Decimal, InvalidOperation
import json
import logging
import os
from typing import Any, Dict, List, Optional
import reflex as rx

# Configure standard structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ExpenseTracker")

# ==============================================================================
# 1. CONFIGURATION & SECRETS MANAGEMENT
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

DEFAULT_CURRENCY: str = os.getenv("APP_DEFAULT_CURRENCY", "PKR")
FIREBASE_CREDENTIALS_PATH: Optional[str] = os.getenv("FIREBASE_CREDENTIALS_PATH")
FIREBASE_CREDENTIALS_JSON: Optional[str] = os.getenv("FIREBASE_CREDENTIALS_JSON")
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
ENABLE_AI_FEATURES: bool = os.getenv("ENABLE_AI_FEATURES", "true").lower() == "true"

# ==============================================================================
# 2. FIREBASE FIRESTORE DATA PERSISTENCE LAYER
# ==============================================================================

_firestore_client = None


def get_firestore_client():
    """Initializes and returns the Firestore client singleton via Firebase Admin SDK."""
    global _firestore_client
    if _firestore_client is not None:
        return _firestore_client

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            cred = None
            if FIREBASE_CREDENTIALS_JSON:
                cred_dict = json.loads(FIREBASE_CREDENTIALS_JSON)
                cred = credentials.Certificate(cred_dict)
            elif FIREBASE_CREDENTIALS_PATH and os.path.exists(FIREBASE_CREDENTIALS_PATH):
                cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
            else:
                logger.warning(
                    "No explicit Firebase credentials found. Falling back to Application Default Credentials."
                )
                cred = credentials.ApplicationDefault()

            firebase_admin.initialize_app(cred)

        _firestore_client = firestore.client()
        logger.info("Firebase Firestore client initialized successfully.")
        return _firestore_client
    except Exception as e:
        logger.error(f"Failed to initialize Firestore client: {e}")
        return None


class FirestoreService:
    """Encapsulates all database operations for isolated user expense collections."""

    @staticmethod
    def _collection(user_id: str):
        db = get_firestore_client()
        if db is None:
            raise RuntimeError(
                "Database connection unavailable. Verify Firebase credentials."
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
        data["updated_at"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
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
            return "AI Insights are disabled via application configuration."

        if not GEMINI_API_KEY:
            return "Gemini API Key is missing. Set GEMINI_API_KEY in your environment."

        if expense_count == 0:
            return "No transactions available to generate spending insights."

        try:
            from google import genai

            client = genai.Client(api_key=GEMINI_API_KEY)

            # Aggregate sanitized prompt with zero PII/transaction descriptions
            prompt = f"""
You are an expert personal finance budgeting analyst.
Review the following aggregated spending summary and generate 3-4 concise, actionable insights or recommendations.
Keep tone professional, encouraging, and clear. Note that this is automated guidance, not formal financial advice.

Summary Data:
- Total Spending: {currency} {total_spent:,.2f}
- Transaction Count: {expense_count}
- Category Breakdown:
{json.dumps(category_breakdown, indent=2)}

Format with markdown bullet points. Highlight anomalies or top categories.
"""
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return response.text or "No insights returned from the model."
        except Exception as err:
            logger.error(f"Gemini API request failed: {err}")
            return f"Unable to generate spending insights at this time: {str(err)}"


# ==============================================================================
# 4. APPLICATION STATE & BUSINESS LOGIC
# ==============================================================================


class State(rx.State):
    """Reflex state managing UI forms, filters, active expenses, and AI operations."""

    # Active User Context
    current_user_id: str = "default_user"
    currency: str = DEFAULT_CURRENCY

    # Data collections
    expenses: List[Dict[str, Any]] = []
    is_loading: bool = False
    status_message: str = ""
    is_error: bool = False

    # Search & Filter Controls
    search_query: str = ""
    filter_category: str = "All"
    sort_by: str = "Date (Newest)"

    # Expense Form State (Create / Update)
    form_id: str = ""
    form_title: str = ""
    form_amount: str = ""
    form_category: str = SUPPORTED_CATEGORIES[0]
    form_date: str = datetime.date.today().isoformat()
    form_payment_method: str = PAYMENT_METHODS[0]
    form_notes: str = ""
    is_editing: bool = False

    # Delete Dialog State
    pending_delete_id: str = ""
    show_delete_modal: bool = False

    # AI Feature State
    ai_insights: str = ""
    is_generating_ai: bool = False

    # --------------------------------------------------------------------------
    # Lifecycle & Loaders
    # --------------------------------------------------------------------------

    def on_load(self):
        """Initial data fetch upon application mount."""
        self.load_expenses()

    def set_status(self, msg: str, error: bool = False):
        self.status_message = msg
        self.is_error = error

    def clear_status(self):
        self.status_message = ""
        self.is_error = False

    def load_expenses(self):
        """Fetches records from Firestore."""
        self.is_loading = True
        self.clear_status()
        try:
            self.expenses = FirestoreService.get_all(self.current_user_id)
        except Exception as e:
            self.set_status(f"Error loading records: {str(e)}", error=True)
        finally:
            self.is_loading = False

    # --------------------------------------------------------------------------
    # CRUD Operations
    # --------------------------------------------------------------------------

    def reset_form(self):
        self.form_id = ""
        self.form_title = ""
        self.form_amount = ""
        self.form_category = SUPPORTED_CATEGORIES[0]
        self.form_date = datetime.date.today().isoformat()
        self.form_payment_method = PAYMENT_METHODS[0]
        self.form_notes = ""
        self.is_editing = False

    def populate_edit_form(self, expense: Dict[str, Any]):
        self.form_id = str(expense.get("id", ""))
        self.form_title = str(expense.get("title", ""))
        self.form_amount = str(expense.get("amount", ""))
        self.form_category = str(expense.get("category", SUPPORTED_CATEGORIES[0]))
        self.form_date = str(expense.get("transaction_date", datetime.date.today().isoformat()))
        self.form_payment_method = str(expense.get("payment_method", PAYMENT_METHODS[0]))
        self.form_notes = str(expense.get("notes", ""))
        self.is_editing = True

    def save_expense(self):
        """Validates and persists a new or updated expense record."""
        # Validation 1: Title
        clean_title = self.form_title.strip()
        if not clean_title:
            return rx.window_alert("Validation Error: Title is required.")

        # Validation 2: Amount (Exact Decimal Handling)
        try:
            amt = Decimal(self.form_amount.strip())
            if amt <= Decimal("0"):
                return rx.window_alert("Validation Error: Amount must be greater than 0.")
        except (InvalidOperation, ValueError):
            return rx.window_alert("Validation Error: Amount must be a valid positive number.")

        # Validation 3: Category & Payment Method
        if self.form_category not in SUPPORTED_CATEGORIES:
            return rx.window_alert("Validation Error: Invalid category selection.")
        if self.form_payment_method not in PAYMENT_METHODS:
            return rx.window_alert("Validation Error: Invalid payment method.")

        # Validation 4: Transaction Date
        try:
            datetime.date.fromisoformat(self.form_date)
        except ValueError:
            return rx.window_alert("Validation Error: Date must follow YYYY-MM-DD format.")

        payload = {
            "title": clean_title,
            "amount": float(amt),
            "currency": self.currency,
            "category": self.form_category,
            "transaction_date": self.form_date,
            "payment_method": self.form_payment_method,
            "notes": self.form_notes.strip(),
        }

        self.is_loading = True
        try:
            if self.is_editing and self.form_id:
                FirestoreService.update(self.current_user_id, self.form_id, payload)
                self.set_status("Expense updated successfully.")
            else:
                FirestoreService.create(self.current_user_id, payload)
                self.set_status("Expense added successfully.")

            self.reset_form()
            self.load_expenses()
        except Exception as e:
            self.set_status(f"Database error: {str(e)}", error=True)
        finally:
            self.is_loading = False

    def confirm_delete(self, doc_id: str):
        self.pending_delete_id = doc_id
        self.show_delete_modal = True

    def cancel_delete(self):
        self.pending_delete_id = ""
        self.show_delete_modal = False

    def execute_delete(self):
        if not self.pending_delete_id:
            return
        self.is_loading = True
        try:
            FirestoreService.delete(self.current_user_id, self.pending_delete_id)
            self.set_status("Expense deleted successfully.")
            self.load_expenses()
        except Exception as e:
            self.set_status(f"Failed to delete: {str(e)}", error=True)
        finally:
            self.pending_delete_id = ""
            self.show_delete_modal = False
            self.is_loading = False

    # --------------------------------------------------------------------------
    # AI Dispatch
    # --------------------------------------------------------------------------

    def request_ai_insights(self):
        """Asynchronously requests summary insights from Gemini."""
        if not self.expenses:
            self.ai_insights = "Please add some expenses first to analyze your spending."
            return

        self.is_generating_ai = True
        self.ai_insights = "Contacting Gemini API to evaluate spending patterns..."

        category_totals: Dict[str, float] = {}
        total = 0.0
        for exp in self.expenses:
            cat = exp.get("category", "Other")
            val = float(exp.get("amount", 0.0))
            category_totals[cat] = category_totals.get(cat, 0.0) + val
            total += val

        insights = GeminiAnalyticsService.generate_spending_insights(
            currency=self.currency,
            total_spent=total,
            category_breakdown=category_totals,
            expense_count=len(self.expenses),
        )

        self.ai_insights = insights
        self.is_generating_ai = False

    # --------------------------------------------------------------------------
    # Reactive Computed Properties
    # --------------------------------------------------------------------------

    @rx.var
    def filtered_expenses(self) -> List[Dict[str, Any]]:
        """Filters and sorts expenses reactively."""
        data = list(self.expenses)

        # 1. Filter Category
        if self.filter_category != "All":
            data = [e for e in data if e.get("category") == self.filter_category]

        # 2. Search Query (Title or Description/Notes)
        q = self.search_query.strip().lower()
        if q:
            data = [
                e
                for e in data
                if q in str(e.get("title", "")).lower()
                or q in str(e.get("notes", "")).lower()
            ]

        # 3. Sort Order
        if self.sort_by == "Date (Newest)":
            data.sort(key=lambda x: str(x.get("transaction_date", "")), reverse=True)
        elif self.sort_by == "Date (Oldest)":
            data.sort(key=lambda x: str(x.get("transaction_date", "")))
        elif self.sort_by == "Amount (Highest)":
            data.sort(key=lambda x: float(x.get("amount", 0.0)), reverse=True)
        elif self.sort_by == "Amount (Lowest)":
            data.sort(key=lambda x: float(x.get("amount", 0.0)))

        return data

    @rx.var
    def total_spending(self) -> float:
        return sum(float(e.get("amount", 0.0)) for e in self.expenses)

    @rx.var
    def formatted_total_spending(self) -> str:
        return f"{self.currency} {self.total_spending:,.2f}"

    @rx.var
    def total_transactions_count(self) -> int:
        return len(self.expenses)


# ==============================================================================
# 5. USER INTERFACE COMPONENTS
# ==============================================================================


def header_component() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.vstack(
                rx.heading("Personal Expense Tracker", size="6", color="gray.900"),
                rx.text(
                    "Cloud-persisted budgeting powered by Google Gemini AI & Firestore",
                    color="gray.600",
                    font_size="sm",
                ),
                align_items="start",
                spacing="1",
            ),
            rx.spacer(),
            rx.badge(
                rx.text(f"Currency: {State.currency}"),
                color_scheme="blue",
                variant="surface",
                size="3",
            ),
            width="100%",
            align_items="center",
            padding_y="4",
        ),
        border_bottom="1px solid #E2E8F0",
        width="100%",
        margin_bottom="6",
    )


def summary_metrics() -> rx.Component:
    return rx.grid(
        rx.card(
            rx.vstack(
                rx.text("Total Spending", font_size="sm", color="gray.500"),
                rx.heading(State.formatted_total_spending, size="5", color="blue.600"),
                align_items="start",
            ),
            variant="surface",
        ),
        rx.card(
            rx.vstack(
                rx.text("Total Records", font_size="sm", color="gray.500"),
                rx.heading(State.total_transactions_count.to_string(), size="5", color="gray.800"),
                align_items="start",
            ),
            variant="surface",
        ),
        rx.card(
            rx.vstack(
                rx.text("AI Budgeting Status", font_size="sm", color="gray.500"),
                rx.cond(
                    State.is_generating_ai,
                    rx.badge("Analyzing...", color_scheme="orange"),
                    rx.badge("Ready", color_scheme="green"),
                ),
                align_items="start",
            ),
            variant="surface",
        ),
        columns="3",
        spacing="4",
        width="100%",
        margin_bottom="6",
    )


def expense_form() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.heading(
                rx.cond(State.is_editing, "Edit Expense Record", "Add New Expense"),
                size="4",
            ),
            rx.divider(),
            rx.grid(
                rx.vstack(
                    rx.text("Title / Description *", font_size="xs", font_weight="bold"),
                    rx.input(
                        placeholder="e.g. Office Lunch or Fuel",
                        value=State.form_title,
                        on_change=State.set_form_title,
                        width="100%",
                    ),
                    align_items="start",
                ),
                rx.vstack(
                    rx.text(f"Amount ({DEFAULT_CURRENCY}) *", font_size="xs", font_weight="bold"),
                    rx.input(
                        placeholder="e.g. 1500.00",
                        type="number",
                        value=State.form_amount,
                        on_change=State.set_form_amount,
                        width="100%",
                    ),
                    align_items="start",
                ),
                rx.vstack(
                    rx.text("Category *", font_size="xs", font_weight="bold"),
                    rx.select(
                        SUPPORTED_CATEGORIES,
                        value=State.form_category,
                        on_change=State.set_form_category,
                        width="100%",
                    ),
                    align_items="start",
                ),
                rx.vstack(
                    rx.text("Date *", font_size="xs", font_weight="bold"),
                    rx.input(
                        type="date",
                        value=State.form_date,
                        on_change=State.set_form_date,
                        width="100%",
                    ),
                    align_items="start",
                ),
                rx.vstack(
                    rx.text("Payment Method *", font_size="xs", font_weight="bold"),
                    rx.select(
                        PAYMENT_METHODS,
                        value=State.form_payment_method,
                        on_change=State.set_form_payment_method,
                        width="100%",
                    ),
                    align_items="start",
                ),
                rx.vstack(
                    rx.text("Notes (Optional)", font_size="xs", font_weight="bold"),
                    rx.input(
                        placeholder="e.g. Project meeting reimbursement",
                        value=State.form_notes,
                        on_change=State.set_form_notes,
                        width="100%",
                    ),
                    align_items="start",
                ),
                columns="3",
                spacing="4",
                width="100%",
            ),
            rx.hstack(
                rx.button(
                    rx.cond(State.is_editing, "Update Record", "Save Record"),
                    on_click=State.save_expense,
                    color_scheme="blue",
                    loading=State.is_loading,
                ),
                rx.cond(
                    State.is_editing,
                    rx.button(
                        "Cancel Edit",
                        on_click=State.reset_form,
                        variant="soft",
                        color_scheme="gray",
                    ),
                    rx.box(),
                ),
                spacing="3",
                margin_top="3",
            ),
            width="100%",
            spacing="3",
        ),
        width="100%",
        margin_bottom="6",
    )


def search_and_filter_bar() -> rx.Component:
    return rx.card(
        rx.hstack(
            rx.input(
                placeholder="Search description or notes...",
                value=State.search_query,
                on_change=State.set_search_query,
                width="40%",
            ),
            rx.select(
                ["All"] + SUPPORTED_CATEGORIES,
                value=State.filter_category,
                on_change=State.set_filter_category,
                width="25%",
            ),
            rx.select(
                [
                    "Date (Newest)",
                    "Date (Oldest)",
                    "Amount (Highest)",
                    "Amount (Lowest)",
                ],
                value=State.sort_by,
                on_change=State.set_sort_by,
                width="35%",
            ),
            width="100%",
            spacing="3",
        ),
        variant="surface",
        width="100%",
        margin_bottom="4",
    )


def render_expense_row(expense: Dict[str, Any]) -> rx.Component:
    return rx.table.row(
        rx.table.cell(expense["transaction_date"]),
        rx.table.cell(rx.text(expense["title"], font_weight="medium")),
        rx.table.cell(rx.badge(expense["category"], color_scheme="teal", variant="soft")),
        rx.table.cell(f"{expense['currency']} {float(expense['amount']):,.2f}"),
        rx.table.cell(expense["payment_method"]),
        rx.table.cell(rx.text(expense.get("notes", "-"), color="gray.500", font_size="sm")),
        rx.table.cell(
            rx.hstack(
                rx.button(
                    "Edit",
                    size="1",
                    variant="soft",
                    color_scheme="blue",
                    on_click=lambda: State.populate_edit_form(expense),
                ),
                rx.button(
                    "Delete",
                    size="1",
                    variant="soft",
                    color_scheme="red",
                    on_click=lambda: State.confirm_delete(expense["id"]),
                ),
                spacing="2",
            )
        ),
    )


def expense_list() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.heading("Expense Transactions", size="4"),
                rx.spacer(),
                rx.button("Refresh", size="2", variant="outline", on_click=State.load_expenses),
                width="100%",
            ),
            search_and_filter_bar(),
            rx.cond(
                State.filtered_expenses.length() == 0,
                rx.center(
                    rx.vstack(
                        rx.text("No matching expenses found.", color="gray.500"),
                        rx.text(
                            "Add an expense using the form above to get started.",
                            font_size="sm",
                            color="gray.400",
                        ),
                        padding="8",
                    ),
                    width="100%",
                ),
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            rx.table.column_header_cell("Date"),
                            rx.table.column_header_cell("Title"),
                            rx.table.column_header_cell("Category"),
                            rx.table.column_header_cell("Amount"),
                            rx.table.column_header_cell("Payment"),
                            rx.table.column_header_cell("Notes"),
                            rx.table.column_header_cell("Actions"),
                        )
                    ),
                    rx.table.body(
                        rx.foreach(State.filtered_expenses, render_expense_row)
                    ),
                    width="100%",
                    variant="surface",
                ),
            ),
            width="100%",
            spacing="3",
        ),
        width="100%",
        margin_bottom="6",
    )


def ai_insights_panel() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.vstack(
                    rx.heading("Gemini AI Financial Analyst", size="4"),
                    rx.text(
                        "Aggregated budget recommendations and spending pattern detection.",
                        font_size="sm",
                        color="gray.500",
                    ),
                    align_items="start",
                ),
                rx.spacer(),
                rx.button(
                    "Generate AI Insights",
                    color_scheme="purple",
                    on_click=State.request_ai_insights,
                    loading=State.is_generating_ai,
                ),
                width="100%",
                align_items="center",
            ),
            rx.divider(),
            rx.cond(
                State.ai_insights != "",
                rx.box(
                    rx.markdown(State.ai_insights),
                    padding="4",
                    background_color="#F8FAFC",
                    border_radius="md",
                    border="1px solid #E2E8F0",
                    width="100%",
                ),
                rx.text(
                    "Click 'Generate AI Insights' to submit aggregated numbers to Google Gemini.",
                    color="gray.400",
                    font_style="italic",
                ),
            ),
            rx.text(
                "Disclaimer: Insights are generated by AI based strictly on aggregated category numbers. Verify financial decisions independently.",
                font_size="xs",
                color="gray.400",
                margin_top="2",
            ),
            width="100%",
            spacing="3",
        ),
        width="100%",
        margin_bottom="8",
    )


def delete_confirmation_dialog() -> rx.Component:
    return rx.dialog.root(
        rx.dialog.content(
            rx.dialog.title("Confirm Expense Deletion"),
            rx.dialog.description(
                "Are you sure you want to permanently delete this expense? This action cannot be undone."
            ),
            rx.hstack(
                rx.button(
                    "Cancel",
                    variant="soft",
                    color_scheme="gray",
                    on_click=State.cancel_delete,
                ),
                rx.button(
                    "Confirm Delete",
                    color_scheme="red",
                    on_click=State.execute_delete,
                ),
                spacing="3",
                margin_top="4",
                justify="end",
            ),
        ),
        open=State.show_delete_modal,
    )


def status_banner() -> rx.Component:
    return rx.cond(
        State.status_message != "",
        rx.callout(
            State.status_message,
            icon="info",
            color_scheme=rx.cond(State.is_error, "red", "green"),
            margin_bottom="4",
            width="100%",
        ),
        rx.box(),
    )


def index() -> rx.Component:
    return rx.container(
        header_component(),
        status_banner(),
        summary_metrics(),
        expense_form(),
        expense_list(),
        ai_insights_panel(),
        delete_confirmation_dialog(),
        size="4",
        padding="4",
        on_mount=State.on_load,
    )


# ==============================================================================
# 6. APPLICATION BOOTSTRAP
# ==============================================================================

app = rx.App(
    theme=rx.theme(
        appearance="light",
        has_background=True,
        accent_color="blue",
    )
)
app.add_page(index, route="/", title="Personal Expense Tracker | Reflex & Firestore")
