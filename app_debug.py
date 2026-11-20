import os
import logging

import fitz  # PyMuPDF
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------------------------------------
# Logging setup: messages appear in the terminal running Streamlit.
# Never log the actual API key.
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger("PDF_Answering_AI")

try:
    from google import genai
    logger.info("[STAGE 1 SUCCESS] Gemini SDK imported successfully.")
except Exception:
    genai = None
    logger.exception("[STAGE 1 FAILED] Could not import Gemini SDK.")


st.set_page_config(page_title="PDF Answering AI", page_icon="📚", layout="wide")
st.title("📚 PDF Answering AI")
st.caption("Summarize a PDF and ask questions with page, paragraph, and exact-line citations.")

st.info(
    "Citation line numbers are generated from the PDF's extracted text in reading order. "
    "Scanned/image-only PDFs may need OCR before they can be cited accurately."
)


def extract_pdf(uploaded_file):
    logger.info("[PDF] Starting extraction for uploaded file: %s", uploaded_file.name)
    try:
        doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
        records = []

        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            page_dict = page.get_text("dict", sort=True)
            line_num = 0
            para_num = 0

            for block in page_dict.get("blocks", []):
                if "lines" not in block:
                    continue

                block_lines = []
                for line in block["lines"]:
                    text = "".join(
                        span.get("text", "") for span in line.get("spans", [])
                    ).strip()
                    if text:
                        block_lines.append(text)

                if not block_lines:
                    continue

                para_num += 1
                para_text = " ".join(block_lines).strip()
                para_start_line = line_num + 1

                for local_i, line_text in enumerate(block_lines, start=1):
                    line_num += 1
                    records.append({
                        "page": page_num,
                        "paragraph": para_num,
                        "line": line_num,
                        "paragraph_line": local_i,
                        "text": line_text,
                        "paragraph_text": para_text,
                        "paragraph_start_line": para_start_line,
                    })

        logger.info(
            "[PDF SUCCESS] Extracted %d lines from %d pages.",
            len(records), len(doc)
        )
        if not records:
            logger.warning("[PDF WARNING] No selectable text found; PDF may be scanned.")
        return records, len(doc)

    except Exception:
        logger.exception("[PDF FAILED] Could not extract text from PDF.")
        raise


def make_paragraphs(records):
    paras = {}
    for r in records:
        key = (r["page"], r["paragraph"])
        paras.setdefault(key, {
            "page": r["page"],
            "paragraph": r["paragraph"],
            "start_line": r["paragraph_start_line"],
            "end_line": r["line"],
            "text": r["paragraph_text"],
            "lines": [],
        })
        paras[key]["end_line"] = max(paras[key]["end_line"], r["line"])
        paras[key]["lines"].append(r)
    logger.info("[PDF] Built %d paragraph records.", len(paras))
    return list(paras.values())


def retrieve(question, paragraphs, top_k=5):
    logger.info("[RETRIEVAL] Searching %d paragraphs; top_k=%d.", len(paragraphs), top_k)
    texts = [p["text"] for p in paragraphs]
    if not texts:
        logger.warning("[RETRIEVAL] No text available to search.")
        return []

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        matrix = vectorizer.fit_transform(texts + [question])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    except ValueError:
        logger.exception("[RETRIEVAL FAILED] Could not build search vectors.")
        return []

    order = scores.argsort()[::-1]
    results = []
    for idx in order[:top_k]:
        if scores[idx] <= 0:
            continue
        item = dict(paragraphs[idx])
        item["score"] = float(scores[idx])
        results.append(item)

    logger.info("[RETRIEVAL SUCCESS] Found %d relevant passages.", len(results))
    return results


def gemini_response(prompt, model):
    logger.info("=" * 65)
    logger.info("[STAGE 2] Starting Gemini request.")

    # Stage 2: API key availability
    logger.info("[STAGE 2] Checking GEMINI_API_KEY environment variable.")
   
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        logger.error("[STAGE 2 FAILED] GEMINI_API_KEY is missing or empty.")
        st.error(
            "API key missing. Set the GEMINI_API_KEY environment variable "
            "in the terminal that launches Streamlit."
        )
        return None

    logger.info("[STAGE 2 SUCCESS] API key exists (value hidden).")
    logger.debug("[STAGE 2] API key length: %d characters.", len(api_key))

    # Stage 3: SDK availability
    logger.info("[STAGE 3] Checking Gemini SDK.")
    if genai is None:
        logger.error("[STAGE 3 FAILED] Gemini SDK is unavailable.")
        st.error("Gemini SDK import failed. Check the terminal logs and installation.")
        return None
    logger.info("[STAGE 3 SUCCESS] Gemini SDK is available.")

    # Stage 4: Request details (do not log prompt content)
    logger.info("[STAGE 4] Preparing request.")
    logger.info("[STAGE 4] Model selected: %s", model)
    logger.info("[STAGE 4] Prompt length: %d characters.", len(prompt))

    try:
        # Stage 5: Client initialization
        logger.info("[STAGE 5] Initializing Gemini client.")
        client = genai.Client(api_key=api_key)
        logger.info("[STAGE 5 SUCCESS] Gemini client initialized.")

        # Stage 6: API call
        logger.info("[STAGE 6] Sending request to Gemini API.")
        response = client.models.generate_content(model=model, contents=prompt)
        logger.info("[STAGE 6 SUCCESS] Gemini API request completed.")

        # Stage 7: Response parsing
        logger.info("[STAGE 7] Extracting generated text.")
        result = (getattr(response, "text", None) or "").strip()

        if not result:
            logger.warning("[STAGE 7 WARNING] API returned no readable text.")
            logger.debug("[STAGE 7] Response type: %s", type(response).__name__)
            st.warning(
                "Gemini returned no readable text. Check terminal logs for details."
            )
            return None

        logger.info("[STAGE 7 SUCCESS] Received %d characters.", len(result))
        logger.info("=" * 65)
        return result

    except Exception as e:
        logger.exception(
            "[GEMINI API FAILED] %s: %s",
            type(e).__name__, str(e)
        )
        st.error(f"Gemini API failed ({type(e).__name__}): {e}")
        return None


def evidence_label(p):
    return (
        f"Page {p['page']} · Paragraph {p['paragraph']} · "
        f"Lines {p['start_line']}–{p['end_line']}"
    )


def summarize(paragraphs, model):
    logger.info("[SUMMARY] Starting summary process.")
    if not paragraphs:
        logger.warning("[SUMMARY] No paragraphs available.")
        return "No selectable text was extracted. This may be a scanned PDF; run OCR first."

    source = "\n".join(
        f"[Page {p['page']}, Paragraph {p['paragraph']}, "
        f"Lines {p['start_line']}-{p['end_line']}] {p['text']}"
        for p in paragraphs
    )
    prompt = (
        "Summarize ONLY the supplied PDF text. Do not add outside facts. "
        "Return: (1) short overview, (2) key points, (3) important terms/findings, "
        "(4) conclusion. For every factual bullet, add the supporting page, paragraph, "
        "and line range from the source labels. If the text is too long, prioritize "
        "the most informative parts.\n\n"
        f"PDF TEXT:\n{source}"
    )

    result = gemini_response(prompt, model)
    if result:
        logger.info("[SUMMARY SUCCESS] Gemini summary generated.")
        return result

    logger.warning("[SUMMARY FALLBACK] Using extractive summary because Gemini returned no result.")
    texts = [p["text"] for p in paragraphs]
    try:
        vec = TfidfVectorizer(stop_words="english")
        X = vec.fit_transform(texts)
        mean_vec = X.mean(axis=0)
        scores = (X @ mean_vec.T).toarray().flatten()
        chosen = sorted(
            scores.argsort()[-min(8, len(texts)):],
            key=lambda i: scores[i],
            reverse=True,
        )
        chosen = sorted(
            chosen,
            key=lambda i: (paragraphs[i]["page"], paragraphs[i]["paragraph"]),
        )
        lines = [
            "### Extractive overview",
            "Gemini generation is unavailable. Set `GEMINI_API_KEY` to enable generated "
            "summaries. These are source excerpts selected for relevance:",
        ]
        for i in chosen:
            p = paragraphs[i]
            lines.append(f"- {p['text']} ({evidence_label(p)})")
        return "\n".join(lines)
    except Exception:
        logger.exception("[SUMMARY FALLBACK FAILED] Returning first available paragraphs.")
        return "\n".join(
            f"- {p['text']} ({evidence_label(p)})" for p in paragraphs[:8]
        )


def answer_question(question, evidence, model):
    logger.info("[Q&A] Starting question-answering process.")
    if not evidence:
        logger.warning("[Q&A] No relevant evidence was retrieved.")
        return (
            "I couldn't find a sufficiently relevant passage in the extracted PDF text. "
            "Try rephrasing the question or check whether the PDF is scanned."
        )

    context = "\n\n".join(
        f"[SOURCE {i+1}: Page {p['page']}, Paragraph {p['paragraph']}, "
        f"Lines {p['start_line']}-{p['end_line']}]\n{p['text']}"
        for i, p in enumerate(evidence)
    )
    prompt = (
        "You are a careful PDF question-answering assistant. Answer ONLY from the "
        "source passages below. Do not use outside knowledge. If evidence is "
        "insufficient, say so. Cite each substantive claim using "
        "[Page X, Paragraph Y, Lines A-B]. Do not invent quotations or citations. "
        "Keep exact quotations verbatim.\n\n"
        f"QUESTION: {question}\n\nSOURCE PASSAGES:\n{context}"
    )

    result = gemini_response(prompt, model)
    if result:
        logger.info("[Q&A SUCCESS] Gemini answer generated.")
        return result

    logger.warning("[Q&A FALLBACK] Showing matching PDF passages instead of Gemini answer.")
    output = [
        "**Gemini answer generation is off** (set `GEMINI_API_KEY` to enable it).",
        "Here are the closest matching passages from the PDF:",
    ]
    for p in evidence:
        output.append(f"- {p['text']} **[{evidence_label(p)}]**")
    return "\n".join(output)


with st.sidebar:
    st.header("Settings")
    model = st.text_input("Gemini model", value="gemini-3.8-flash")
    st.caption(
        "Set GEMINI_API_KEY as an environment variable. "
        "Do not paste API keys into source code or commit them to Git."
    )
    st.markdown("---")
    st.markdown("**Citation convention**")
    st.markdown("- Page: PDF page, starting at 1")
    st.markdown("- Paragraph: extracted text block on that page")
    st.markdown("- Lines: extracted non-empty text lines, numbered from 1 on each page")

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded:
    try:
        with st.spinner("Extracting text and building citation locations..."):
            records, page_count = extract_pdf(uploaded)
            paragraphs = make_paragraphs(records)

        st.success(
            f"Loaded **{uploaded.name}** · {page_count} pages · "
            f"{len(paragraphs)} text paragraphs · {len(records)} extracted lines"
        )

        tab_summary, tab_ask, tab_text = st.tabs(
            ["📝 Summary", "💬 Ask the PDF", "🔎 Extracted text"]
        )

        with tab_summary:
            if st.button("Generate summary", type="primary"):
                with st.spinner("Preparing summary..."):
                    st.session_state["pdf_summary"] = summarize(paragraphs, model)
            if st.session_state.get("pdf_summary"):
                st.markdown(st.session_state["pdf_summary"])
            else:
                st.write("Click **Generate summary** to create an overview and key points.")

        with tab_ask:
            question = st.text_area(
                "Ask a question about this PDF",
                placeholder="e.g., What method did the authors propose, and what evidence supports it?",
            )
            top_k = st.slider(
                "Number of evidence passages to retrieve",
                min_value=2,
                max_value=8,
                value=5,
            )
            if st.button("Get answer", type="primary", disabled=not question.strip()):
                evidence = retrieve(question.strip(), paragraphs, top_k=top_k)
                with st.spinner("Finding evidence and answering with Gemini..."):
                    response = answer_question(question.strip(), evidence, model)
                st.markdown("### Answer")
                st.markdown(response)
                if evidence:
                    st.markdown("### Supporting evidence")
                    for p in evidence:
                        with st.expander(evidence_label(p), expanded=True):
                            st.write(p["text"])
                            st.caption("Exact extracted lines:")
                            for line in p["lines"]:
                                st.code(
                                    f"Page {line['page']} · Line {line['line']}: {line['text']}",
                                    language=None,
                                )

        with tab_text:
            page_choice = st.selectbox(
                "Choose PDF page", options=list(range(1, page_count + 1))
            )
            page_records = [r for r in records if r["page"] == page_choice]
            if not page_records:
                st.warning(
                    "No selectable text was found on this page. "
                    "It may contain only a scanned image."
                )
            else:
                for pnum in sorted(set(r["paragraph"] for r in page_records)):
                    para = next(
                        p for p in paragraphs
                        if p["page"] == page_choice and p["paragraph"] == pnum
                    )
                    st.markdown(
                        f"**Paragraph {pnum} · Lines "
                        f"{para['start_line']}–{para['end_line']}**"
                    )
                    for line in para["lines"]:
                        st.text(f"Line {line['line']}: {line['text']}")

    except Exception as e:
        logger.exception("[APP FAILED] Error while processing uploaded PDF.")
        st.error(f"Could not process the PDF: {type(e).__name__}: {e}")

else:
    st.markdown("### What this app does")
    st.markdown(
        "- Summarizes uploaded PDFs\n"
        "- Answers questions using retrieved passages\n"
        "- Displays source page, paragraph, and extracted line numbers\n"
        "- Shows the exact supporting text so citations can be checked"
    )
    st.warning("For scanned PDFs, add OCR (for example Tesseract) as a next development step.")
