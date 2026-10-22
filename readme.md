# Personal Expense Tracker (Streamlit + SQLite + Gemini)

A cloud-ready personal finance tracker built with **Python**, **Streamlit**, an embedded **SQLite** database (no external Firebase setup required), and the **Google Gemini API (`google-genai`)**.

---

## Key Features

- **Zero-Cloud Database Setup:** Uses a self-contained local SQLite database with automatic schema migrations.
- **Full Reactive CRUD Operations:** Create, review, update, and delete expenses directly within Streamlit.
- **Financial Precision:** Input amounts are validated using `decimal.Decimal` to avoid floating-point errors.
- **Configurable Default Currency:** Defaults to `PKR`, configurable via `.env` or Streamlit Secrets.
- **Dynamic Search & Filtering:** Filter expenses by title, notes, and category, with multi-option sorting.
- **Two-Step Deletion Guard:** Prevents accidental record loss with an inline confirmation modal.
- **AI-Powered Spending Summaries:** Evaluates aggregated category trends via `gemini-2.5-flash` without transmitting personal transaction notes.

---

## Prerequisites

- **Python:** 3.11+
- **Google Gemini API Key:** Obtainable from [Google AI Studio](https://aistudio.google.com/).

---

## Project Structure

```text
expense-tracker/
├── app.py              # Streamlit application, SQLite persistence, and UI logic
├── requirements.txt    # Direct dependencies
├── .env.example        # Environment variable template
├── .env                # Local secrets (ignored by Git)
└── README.md           # Documentation and runbook
