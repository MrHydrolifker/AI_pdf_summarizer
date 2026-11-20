# PDF Answering AI (Gemini version)

A Streamlit starter app that summarizes PDFs and answers questions with source page, paragraph, extracted line range, and supporting text. Uses Google's `google-genai` Python SDK.

## Setup

Use Python 3.10+.

```bash
python -m venv .venv
```

Windows:
```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Configure Gemini API key

Create a key in Google AI Studio. Store it as an environment variable; do not hard-code it in `app.py` or commit it to Git.

Windows PowerShell:
```powershell
$env:GEMINI_API_KEY="YOUR_NEW_GEMINI_API_KEY"
```

Windows Command Prompt:
```cmd
set GEMINI_API_KEY=YOUR_NEW_GEMINI_API_KEY
```

macOS/Linux:
```bash

```

Run:
```bash
streamlit run app.py
```

The default model field is `gemini-3.8-flash`, matching the model name supplied for this project. If that model is not available to your API account, select a currently supported Gemini model in the sidebar.

## Citation notes
- Page numbering starts at 1.
- Paragraph numbers are assigned to non-empty extracted text blocks on each page.
- Line numbers are assigned to non-empty extracted text lines, numbered from 1 on each page.
- These are app-generated line numbers, not necessarily printed line numbers in the PDF.
- Complex layouts, tables, headers, footnotes, and scanned pages can affect extraction. Always verify quoted evidence against the source PDF.
- OCR is not included in this starter version.

## Security
If an API key has been pasted into a chat, source code, screenshot, or public repository, revoke it and create a replacement. Keep secrets in environment variables or a secrets manager.
