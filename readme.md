# Personal Expense Tracker (Streamlit Edition)

A cloud-persisted personal expense management application built with **Python**, **Streamlit**, **Firebase Cloud Firestore**, and the **Google Gemini API (`google-genai`)**.

Designed specifically for single-click deployment on **Streamlit Community Cloud** or local execution.

---

## Architecture Overview

- **UI Framework:** Streamlit (`st.session_state` reactive architecture).
- **Persistence:** Firebase Cloud Firestore via the official `firebase-admin` SDK.
- **AI Intelligence:** Google Gemini (`gemini-2.5-flash`) via the `google-genai` SDK.
- **Security & Privacy:** Database credentials run purely server-side; prompt payloads are stripped of personal transaction titles and individual notes, sending only aggregated numbers.

---

## Prerequisites

- Python 3.11 or higher.
- A Firebase Project with Cloud Firestore enabled in Native Mode.
- A Google Gemini API Key from Google AI Studio.

---

## Local Setup

### 1. Clone & Set Up Virtual Environment

**Windows PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
