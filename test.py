import logging
import os
import traceback
from pathlib import Path

import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure detailed console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pdf_answering_ai")

logger.info("[INIT] Starting PDF Answering AI script...")

ENV_PATH = Path(__file__).resolve().parent / ".env"
logger.info(f"[ENV] Loading .env from: {ENV_PATH} (exists={ENV_PATH.exists()})")
load_dotenv(dotenv_path=ENV_PATH, override=False)

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
logger.info(f"[ENV] GEMINI_API_KEY present: {bool(API_KEY)} (length={len(API_KEY)})")

try:
    from google import genai

    logger.info("[IMPORT] google.genai imported successfully.")
except Exception as e:
    genai = None
    logger.error(f"[IMPORT] Failed to import google.genai: {e}", exc_info=True)

client = None
if API_KEY and genai is not None:
    try:
        client = genai.Client(api_key=API_KEY)
        logger.info("[GENAI] Top-level genai.Client initialized.")
    except Exception as e:
        logger.error(f"[GENAI] Top-level genai.Client failed: {e}", exc_info=True)
else:
    logger.warning("[GENAI] Skipping top-level genai.Client init (missing API_KEY or genai module).")

try:
    st.set_page_config(page_title="PDF Answering AI", page_icon="📚", layout="wide")
    st.title("📚 PDF Answering AI")
    st.caption("Summarize a PDF and ask questions with page, paragraph, and exact-line citations.")
    logger.info("[UI] Page config, title, and caption rendered.")
except Exception as e:
    logger.error(f"[UI] Error setting up page config/header: {e}", exc_info=True)
    raise

st.info(
    "Citation line numbers are generated from the PDF's extracted text in reading order. "
    "Scanned/image-only PDFs may need OCR before they can be cited accurately."
)
logger.info("[UI] Info banner rendered.")


def extract_pdf(uploaded_file):
    logger.info(f"[EXTRACT_PDF] Start extracting file='{getattr(uploaded_file, 'name', 'unknown')}'")
    records = []
    try:
        raw_bytes = uploaded_file.getvalue()
        logger.info(f"[EXTRACT_PDF] Read {len(raw_bytes)} bytes from uploaded file.")
        with fitz.open(stream=raw_bytes, filetype="pdf") as doc:
            page_count = len(doc)
            logger.info(f"[EXTRACT_PDF] Opened PDF with {page_count} page(s).")

            for page_idx, page in enumerate(doc):
                page_num = page_idx + 1
                try:
                    page_dict = page.get_text("dict", sort=True)
                except Exception as e:
                    logger.error(f"[EXTRACT_PDF] Page {page_num}: get_text('dict', sort=True) failed: {e}", exc_info=True)
                    continue

                line_num = 0
                para_num = 0
                blocks = page_dict.get("blocks", [])
                logger.info(f"[EXTRACT_PDF] Page {page_num}: found {len(blocks)} block(s).")

                for block_idx, block in enumerate(blocks):
                    if "lines" not in block:
                        continue
                    block_lines = []
                    for line in block["lines"]:
                        text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                        if text:
                            block_lines.append(text)

                    if not block_lines:
                        continue

                    para_num += 1
                    para_text = " ".join(block_lines).strip()
                    para_start_line = line_num + 1
                    paragraph_lines = []
                    for local_i, line_text in enumerate(block_lines, start=1):
                        line_num += 1
                        record = {
                            "page": page_num,
                            "paragraph": para_num,
                            "line": line_num,
                            "paragraph_line": local_i,
                            "text": line_text,
                            "paragraph_text": para_text,
                            "paragraph_start_line": para_start_line,
                        }
                        records.append(record)
                        paragraph_lines.append(record)

                logger.info(
                    f"[EXTRACT_PDF] Page {page_num} complete: {para_num} paragraph(s), {line_num} line(s)."
                )

        logger.info(f"[EXTRACT_PDF] Done. Total records={len(records)}, pages={page_count}.")
        return records, page_count
    except Exception as e:
        logger.error(f"[EXTRACT_PDF] Fatal error during PDF extraction: {e}", exc_info=True)
        raise


def make_paragraphs(records):
    logger.info(f"[MAKE_PARAGRAPHS] Building paragraphs from {len(records)} record(s).")
    try:
        paras = {}
        for r in records:
            key = (r["page"], r["paragraph"])
            paras.setdefault(
                key,
                {
                    "page": r["page"],
                    "paragraph": r["paragraph"],
                    "start_line": r["paragraph_start_line"],
                    "end_line": r["line"],
                    "text": r["paragraph_text"],
                    "lines": [],
                },
            )
            paras[key]["end_line"] = max(paras[key]["end_line"], r["line"])
            paras[key]["lines"].append(r)
        result = list(paras.values())
        logger.info(f"[MAKE_PARAGRAPHS] Built {len(result)} paragraph object(s).")
        return result
    except Exception as e:
        logger.error(f"[MAKE_PARAGRAPHS] Error grouping paragraphs: {e}", exc_info=True)
        raise


def retrieve(question, paragraphs, top_k=5):
    logger.info(f"[RETRIEVE] Retrieving top_k={top_k} from {len(paragraphs)} paragraph(s) for question='{question[:80]}'")
    texts = [p["text"] for p in paragraphs]
    if not texts:
        logger.warning("[RETRIEVE] No paragraph texts available; returning empty list.")
        return []
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        matrix = vectorizer.fit_transform(texts + [question])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
        logger.info(f"[RETRIEVE] TF-IDF matrix shape={matrix.shape}; max_score={scores.max():.4f}")
    except ValueError as e:
        logger.warning(f"[RETRIEVE] TF-IDF ValueError (likely empty vocabulary/stop words): {e}")
        return []
    except Exception as e:
        logger.error(f"[RETRIEVE] Unexpected error in TF-IDF/cosine similarity: {e}", exc_info=True)
        return []

    order = scores.argsort()[::-1]
    results = []
    for idx in order[:top_k]:
        if scores[idx] <= 0:
            logger.info(f"[RETRIEVE] Stopping at index={idx} because score={scores[idx]:.4f} <= 0.")
            continue
        item = dict(paragraphs[idx])
        item["score"] = float(scores[idx])
        results.append(item)

    logger.info(f"[RETRIEVE] Returning {len(results)} matching passage(s).")
    return results


def gemini_response(prompt, model):
    logger.info(f"[GEMINI_RESPONSE] Called with model='{model}', prompt_chars={len(prompt)}")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = (model or "").strip()

    if not api_key:
        logger.warning("[GEMINI_RESPONSE] GEMINI_API_KEY is empty or missing.")
        return None
    if not model_name:
        logger.warning("[GEMINI_RESPONSE] Model name is empty.")
        return None
    if genai is None:
        logger.error("[GEMINI_RESPONSE] google.genai module is None (import failed earlier).")
        return None

    try:
        req_client = genai.Client(api_key=api_key)
        logger.info(f"[GEMINI_RESPONSE] Sending generate_content request to model='{model_name}'...")
        response = req_client.models.generate_content(model=model_name, contents=prompt)
        text = (getattr(response, "text", None) or "").strip() or None
        logger.info(f"[GEMINI_RESPONSE] Received response (non_empty={bool(text)}, chars={len(text) if text else 0}).")
        return text
    except Exception as e:
        logger.error(f"[GEMINI_RESPONSE] Error calling Gemini API (model='{model_name}'): {e}", exc_info=True)
        raise


def evidence_label(p):
    return f"Page {p['page']} · Paragraph {p['paragraph']} · Lines {p['start_line']}–{p['end_line']}"


def summarize(paragraphs, model):
    logger.info(f"[SUMMARIZE] Start summarize with {len(paragraphs)} paragraph(s), model='{model}'.")
    if not paragraphs:
        logger.warning("[SUMMARIZE] No paragraphs to summarize.")
        return "No selectable text was extracted. This may be a scanned PDF; run OCR first."

    source = "\n".join(
        f"[Page {p['page']}, Paragraph {p['paragraph']}, Lines {p['start_line']}-{p['end_line']}] {p['text']}"
        for p in paragraphs
    )
    logger.info(f"[SUMMARIZE] Built source string of {len(source)} characters.")

    prompt = (
        "Summarize ONLY the supplied PDF text. Do not add outside facts. "
        "Return: (1) short overview, (2) key points, (3) important terms/findings, "
        "(4) conclusion. For every factual bullet, add the supporting page, paragraph, "
        "and line range from the source labels. If the text is too long, prioritize the most informative parts.\n\n"
        f"PDF TEXT:\n{source}"
    )

    try:
        result = gemini_response(prompt, model)
        if result:
            logger.info("[SUMMARIZE] Gemini summary succeeded.")
            return result
        logger.warning("[SUMMARIZE] gemini_response returned None; falling back to extractive TF-IDF summary.")
    except Exception as e:
        logger.error(f"[SUMMARIZE] Gemini API error caught in summarize(): {e}", exc_info=True)
        st.error(f"Gemini API error: {e}")

    # No-key fallback: extractive summary based on TF-IDF centrality.
    logger.info("[SUMMARIZE] Running TF-IDF extractive fallback.")
    texts = [p["text"] for p in paragraphs]
    try:
        vec = TfidfVectorizer(stop_words="english")
        X = vec.fit_transform(texts)
        mean_vec = X.mean(axis=0)
        scores = (X @ mean_vec.T).toarray().flatten()
        chosen = sorted(scores.argsort()[-min(8, len(texts)) :], key=lambda i: scores[i], reverse=True)
        chosen = sorted(chosen, key=lambda i: (paragraphs[i]["page"], paragraphs[i]["paragraph"]))
        logger.info(f"[SUMMARIZE] Extractive fallback selected {len(chosen)} paragraph index(es): {chosen}")
        lines = [
            "### Extractive overview",
            "Gemini generation is unavailable. Set `GEMINI_API_KEY` to enable generated summaries. These are source excerpts selected for relevance:",
        ]
        for i in chosen:
            p = paragraphs[i]
            lines.append(f"- {p['text']} ({evidence_label(p)})")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"[SUMMARIZE] Extractive fallback failed: {e}", exc_info=True)
        return "\n".join(f"- {p['text']} ({evidence_label(p)})" for p in paragraphs[:8])


def answer_question(question, evidence, model):
    logger.info(f"[ANSWER_QUESTION] Start with question='{question[:80]}', evidence_count={len(evidence)}, model='{model}'.")
    if not evidence:
        logger.warning("[ANSWER_QUESTION] No evidence retrieved.")
        return (
            "I couldn't find a sufficiently relevant passage in the extracted PDF text. "
            "Try rephrasing the question or check whether the PDF is scanned."
        )

    context = "\n\n".join(
        f"[SOURCE {i+1}: Page {p['page']}, Paragraph {p['paragraph']}, Lines {p['start_line']}-{p['end_line']}]\n{p['text']}"
        for i, p in enumerate(evidence)
    )
    logger.info(f"[ANSWER_QUESTION] Built context of {len(context)} characters.")

    prompt = (
        "You are a careful PDF question-answering assistant. Answer ONLY from the source passages below. "
        "Do not use outside knowledge. If evidence is insufficient, say so. "
        "Cite each substantive claim using [Page X, Paragraph Y, Lines A-B]. "
        "Do not invent quotations or citations. Keep exact quotations verbatim.\n\n"
        f"QUESTION: {question}\n\nSOURCE PASSAGES:\n{context}"
    )

    try:
        result = gemini_response(prompt, model)
        if result:
            logger.info("[ANSWER_QUESTION] Gemini answer succeeded.")
            return result
        logger.warning("[ANSWER_QUESTION] gemini_response returned None; returning raw evidence passages.")
    except Exception as e:
        logger.error(f"[ANSWER_QUESTION] Gemini API error caught in answer_question(): {e}", exc_info=True)
        st.error(f"Gemini API error: {e}")

    output = [
        "**Gemini answer generation is off** (set `GEMINI_API_KEY` to enable it).",
        "Here are the closest matching passages from the PDF:",
    ]
    for p in evidence:
        output.append(f"- {p['text']} **[{evidence_label(p)}]**")
    return "\n".join(output)


try:
    with st.sidebar:
        st.header("Settings")
        model = st.text_input("Gemini model", value="gemini-3.6-flash")
        st.caption("Set GEMINI_API_KEY in .env or as an environment variable. Do not commit API keys to Git.")
        st.markdown("---")
        st.markdown("**Citation convention**")
        st.markdown("- Page: PDF page, starting at 1")
        st.markdown("- Paragraph: extracted text block on that page")
        st.markdown("- Lines: extracted non-empty text lines, numbered from 1 on each page")
    logger.info(f"[UI] Sidebar rendered with model='{model}'.")

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded:
        logger.info(f"[UI] File uploaded: name='{uploaded.name}', size={uploaded.size} bytes.")
        file_key = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("active_file_key") != file_key:
            logger.info(f"[STATE] New file detected ('{file_key}'). Clearing previous summary and Q&A state.")
            st.session_state["active_file_key"] = file_key
            st.session_state.pop("pdf_summary", None)
            st.session_state.pop("last_qa", None)

        with st.spinner("Extracting text and building citation locations..."):
            records, page_count = extract_pdf(uploaded)
            paragraphs = make_paragraphs(records)

        st.success(
            f"Loaded **{uploaded.name}** · {page_count} pages · {len(paragraphs)} text paragraphs · {len(records)} extracted lines"
        )
        logger.info("[UI] Upload success banner displayed.")

        tab_summary, tab_ask, tab_text = st.tabs(["📝 Summary", "💬 Ask the PDF", "🔎 Extracted text"])

        with tab_summary:
            if st.button("Generate summary", type="primary"):
                logger.info("[UI:SUMMARY] 'Generate summary' button clicked.")
                with st.spinner("Preparing summary..."):
                    st.session_state["pdf_summary"] = summarize(paragraphs, model)
                logger.info("[UI:SUMMARY] Summary stored in session_state.")
            if st.session_state.get("pdf_summary"):
                st.markdown(st.session_state["pdf_summary"])
            else:
                st.write("Click **Generate summary** to create an overview and key points.")

        with tab_ask:
            question = st.text_area(
                "Ask a question about this PDF",
                placeholder="e.g., What method did the authors propose, and what evidence supports it?",
            )
            top_k = st.slider("Number of evidence passages to retrieve", min_value=2, max_value=8, value=5)
            if st.button("Get answer", type="primary", disabled=not question.strip()):
                logger.info(f"[UI:ASK] 'Get answer' button clicked. question='{question.strip()[:80]}', top_k={top_k}")
                evidence = retrieve(question.strip(), paragraphs, top_k=top_k)
                with st.spinner("Finding evidence and answering with Gemini..."):
                    response = answer_question(question.strip(), evidence, model)
                st.session_state["last_qa"] = {"response": response, "evidence": evidence}
                logger.info("[UI:ASK] Q&A result stored in session_state.")

            last_qa = st.session_state.get("last_qa")
            if last_qa:
                st.markdown("### Answer")
                st.markdown(last_qa["response"])
                evidence = last_qa["evidence"]
                if evidence:
                    st.markdown("### Supporting evidence")
                    for p in evidence:
                        with st.expander(evidence_label(p), expanded=True):
                            st.write(p["text"])
                            st.caption("Exact extracted lines:")
                            for line in p["lines"]:
                                st.code(f"Page {line['page']} · Line {line['line']}: {line['text']}", language=None)

        with tab_text:
            if page_count > 0:
                page_choice = st.selectbox("Choose PDF page", options=list(range(1, page_count + 1)))
                logger.info(f"[UI:TEXT] Viewing extracted text for page={page_choice}.")
                page_records = [r for r in records if r["page"] == page_choice]
                if not page_records:
                    st.warning("No selectable text was found on this page. It may contain only a scanned image.")
                    logger.warning(f"[UI:TEXT] Page {page_choice} has 0 selectable text records.")
                else:
                    for pnum in sorted({r["paragraph"] for r in page_records}):
                        para = next(p for p in paragraphs if p["page"] == page_choice and p["paragraph"] == pnum)
                        st.markdown(f"**Paragraph {pnum} · Lines {para['start_line']}–{para['end_line']}**")
                        for line in para["lines"]:
                            st.text(f"Line {line['line']}: {line['text']}")
            else:
                st.warning("This PDF has 0 pages.")
                logger.warning("[UI:TEXT] Uploaded PDF has 0 pages.")
    else:
        logger.info("[UI] Waiting for PDF upload.")
        st.markdown("### What this app does")
        st.markdown(
            "- Summarizes uploaded PDFs\n"
            "- Answers questions using retrieved passages\n"
            "- Displays source page, paragraph, and extracted line numbers\n"
            "- Shows the exact supporting text so citations can be checked"
        )
        st.warning("For scanned PDFs, add OCR (for example Tesseract) as a next development step.")

except Exception as e:
    logger.critical(f"[FATAL] Unhandled exception in Streamlit app flow: {e}", exc_info=True)
    st.error(f"An unexpected error occurred: {e}")
    st.code(traceback.format_exc())