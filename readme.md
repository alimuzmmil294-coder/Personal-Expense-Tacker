# Personal Expense Tracker

A modern, cloud-native personal finance application built with **Reflex**, **Firebase Cloud Firestore**, and the **Google Gemini API (`google-genai`)**.

---

## Features

- **Full Reactive CRUD Operations:** Create, review, modify, and permanently remove expenses persisted immediately to Cloud Firestore.
- **Precise Financial Ledger:** Implements zero-loss decimal parsing (`decimal.Decimal`) for precision transaction records.
- **Configurable Base Currency:** Defaults to `PKR`, customizable via environment settings.
- **Search & Multi-Facet Filtering:** Real-time client queries filtering by category, notes, and description, with multi-order sorting (Date/Amount).
- **Deletion Safeguard:** Built-in modal confirmation dialog preventing inadvertent record deletion.
- **Gemini AI Spending Insights:** One-click automated spending analysis powered by `gemini-2.5-flash` using high-level aggregated numbers (no PII or individual descriptions exposed to external APIs).

---

## Architecture Overview
