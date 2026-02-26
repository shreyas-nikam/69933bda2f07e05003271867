import os
import io
import time
import json
import re

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Import all required functions / globals from source.py
from source import *  # noqa: F401,F403


# ============================================================
# App Config
# ============================================================
st.set_page_config(
    page_title="Portfolio-Level Data Extraction with RAG (QuantUniversity)",
    layout="wide",
)

# Light styling to keep the UI clean
st.markdown(
    """
<style>
/* tighten sidebar */
section[data-testid="stSidebar"] > div { padding-top: 1rem; }
.small-note { font-size: 0.9rem; opacity: 0.85; }
.codebox { background: #0b1020; padding: 1rem; border-radius: 0.75rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Session State
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "Introduction"

if "doc_dir" not in st.session_state:
    st.session_state.doc_dir = "proxy_filings"

if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 500

if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = 100

if "ingested" not in st.session_state:
    st.session_state.ingested = False

if "collection" not in st.session_state:
    st.session_state.collection = None

if "embedder" not in st.session_state:
    st.session_state.embedder = None

if "extracted_df" not in st.session_state:
    st.session_state.extracted_df = None

if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

if "elapsed_time" not in st.session_state:
    st.session_state.elapsed_time = 0.0

if "val_df" not in st.session_state:
    st.session_state.val_df = None

if "overall_accuracy" not in st.session_state:
    st.session_state.overall_accuracy = 0.0


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.image("https://www.quantuniversity.com/assets/img/logo5.jpg",
             use_container_width=True)
    st.divider()

    st.header("Navigation")
    page_selection = st.selectbox(
        "Choose a step:",
        [
            "Introduction",
            "1. Ingest Data",
            "2. Define Schema",
            "3. Filtered Retrieval",
            "4. Batch Extraction",
            "5. Validate Accuracy",
            "6. Actionable Insights",
            "7. ROI Analysis",
        ],
        index=[
            "Introduction",
            "1. Ingest Data",
            "2. Define Schema",
            "3. Filtered Retrieval",
            "4. Batch Extraction",
            "5. Validate Accuracy",
            "6. Actionable Insights",
            "7. ROI Analysis",
        ].index(st.session_state.page),
    )
    st.session_state.page = page_selection

    st.divider()
    st.markdown("**Portfolio (demo):**")
    st.write(pd.DataFrame(
        {"Ticker": list(PORTFOLIO.keys()), "Company": list(PORTFOLIO.values())}))
    st.caption(
        "This lab is an educational blueprint. Use your own filings and validation data for production.")


# ============================================================
# Helper utilities (App-side)
# ============================================================
def _df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    """Export a DataFrame to Excel in-memory for st.download_button."""
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return bio.getvalue()


def _safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        return float(str(x).strip())
    except Exception:
        return None


# ============================================================
# Main Layout
# ============================================================
st.title("Portfolio-Level Data Extraction with RAG")
st.caption(
    "A QuantUniversity blueprint for extracting structured, comparable metrics across a portfolio using Retrieval-Augmented Generation (RAG)."
)
st.divider()


# ============================================================
# Page: Introduction
# ============================================================
if st.session_state.page == "Introduction":
    st.header("Introduction: The Challenge of Manual Portfolio Review")

    st.markdown(
        """
Alex Chen is a CFA Charterholder and Portfolio Manager at Apex Capital Management.
Each earnings season, he must review multiple earnings releases and consolidated financial statements
across his portfolio to extract *the same* set of comparable financial metrics.

In practice, this work is:

- **Time-consuming**: each document can be long and dense, and the same metrics may be spread across sections.
- **Inconsistent**: analysts may interpret or record metrics differently across companies.
- **Hard to scale**: as the number of holdings grows, manual review becomes a bottleneck.

This application demonstrates a portfolio-level RAG pipeline that turns unstructured filings into a **structured, analyst-ready table**
in minutes — and then validates accuracy against a ground-truth sample before producing portfolio insights and ROI metrics.
"""
    )

    st.markdown("#### What you will build in this lab")
    st.markdown(
        """
1. **Ingest PDFs** into a metadata-rich vector store (ChromaDB).
2. **Define a strict extraction schema** (JSON) so outputs are consistent and comparable.
3. **Use company-filtered retrieval** to avoid cross-company information leakage.
4. **Batch extract** metrics across all companies using an LLM with structured JSON output.
5. **Validate** against ground truth and quantify accuracy.
6. **Generate insights** (comparisons, screening, and charts).
7. **Quantify ROI** vs. manual review.
"""
    )

    st.info(
        "Tip: Use the sidebar to follow the steps in order. Each step builds on the previous one."
    )

    st.divider()


# ============================================================
# Page: 1. Ingest Data
# ============================================================
elif st.session_state.page == "1. Ingest Data":
    st.header("1. Setting the Foundation: Ingesting Portfolio Documents")

    st.markdown(
        """
In a portfolio workflow, the hardest part is not “asking a question,” but building a **reliable index** of multiple documents.
For multi-document RAG, we need:

- **Chunking**: break long PDFs into manageable text units.
- **Embeddings**: map chunks into a vector space for semantic search.
- **Metadata**: attach fields like `ticker` so retrieval can be filtered by company.

This metadata is the guardrail that prevents a “silent catastrophic error” where the model mixes two companies.
"""
    )

    st.subheader("Configuration")

    c1, c2 = st.columns(2)
    with c1:
        st.session_state.chunk_size = st.number_input(
            "Chunk size", min_value=200, max_value=2000, value=int(st.session_state.chunk_size), step=50
        )
    with c2:
        st.session_state.chunk_overlap = st.number_input(
            "Chunk overlap", min_value=0, max_value=500, value=int(st.session_state.chunk_overlap), step=10
        )

    st.markdown("#### Optional: Download demo filings")
    download_files = st.checkbox(
        "Download demo earnings PDFs into the directory (from QuantUniversity S3)",
        value=True,
        help="Uses source.download_proxy_filings(). You can uncheck if you already have PDFs in place.",
    )

    if st.button("Ingest / Re-ingest Documents"):
        with st.spinner("Ingesting PDFs and building the vector store..."):
            try:
                os.makedirs(st.session_state.doc_dir, exist_ok=True)

                if download_files:
                    download_proxy_filings(st.session_state.doc_dir, PORTFOLIO)

                # Ingest into ChromaDB
                collection, embedder = ingest_proxy_statements(
                    doc_dir=st.session_state.doc_dir,
                    portfolio_map=PORTFOLIO,
                    chunk_size=int(st.session_state.chunk_size),
                    chunk_overlap=int(st.session_state.chunk_overlap),
                )

                st.session_state.collection = collection
                st.session_state.embedder = embedder
                st.session_state.ingested = bool(
                    collection is not None and embedder is not None)

                if st.session_state.ingested:
                    st.success("Data ingestion complete.")
                    st.markdown(
                        f"**Total chunks in collection:** {st.session_state.collection.count() if st.session_state.collection else 0}"
                    )
                else:
                    st.warning(
                        "Ingestion did not produce a usable collection. Check logs and your PDFs.")
            except Exception as e:
                st.session_state.ingested = False
                st.error(f"Error during ingestion: {e}")

    if st.session_state.ingested:
        st.info("Data has been ingested. Proceed to **2. Define Schema**.")
    else:
        st.warning("Please ingest data to proceed with other steps.")

    st.markdown("### Explanation of Execution")
    st.markdown(
        """
Under the hood, the ingestion step:
1. Reads each PDF, extracts text, and splits it into overlapping chunks.
2. Embeds each chunk using a sentence-transformer model.
3. Stores chunk text + embeddings in ChromaDB along with metadata (`ticker`, `company`, etc.).

Once complete, the portfolio becomes a searchable knowledge base that supports **company-filtered retrieval**.
"""
    )
    st.divider()


# ============================================================
# Page: 2. Define Schema
# ============================================================
elif st.session_state.page == "2. Define Schema":
    st.header("2. Defining the Extraction Schema: Structured Outputs")

    st.markdown(
        """
Alex doesn’t want free-form answers — he needs a **consistent table**.
So we define a strict JSON schema for extraction and instruct the model to return:

- **Only the required keys**
- **'N/A'** for missing information
- **No outside knowledge** (only what’s in retrieved excerpts)

This turns an LLM into a *structured data extraction engine* rather than a chatbot.
"""
    )

    st.subheader("Extraction Prompt (Schema)")
    st.code(EXTRACTION_PROMPT, language="python")

    st.markdown("### Why this matters")
    st.markdown(
        """
- **Comparability:** same keys across all companies enables direct comparison.
- **Auditability:** each value can be traced to retrieved excerpts.
- **Missingness is a signal:** "N/A" can itself highlight disclosure gaps.

In effect, the schema becomes Alex's **data model** for portfolio analytics.
"""
    )
    st.divider()


# ============================================================
# Page: 3. Filtered Retrieval
# ============================================================
elif st.session_state.page == "3. Filtered Retrieval":
    st.header("3. Precision in Retrieval: Company-Filtered Search")

    st.markdown(
        r"""
In multi-document RAG, retrieval must avoid *cross-company contamination*.
Even a small mix-up (e.g., pulling Alphabet numbers while analyzing Apple) can break the entire analysis.

We combine:
- **Semantic search** for relevance, and
- **Hard metadata filters** (`ticker`) for correctness.

A useful way to represent filtered retrieval is:

$$
R_{\text{company}} = \{ d_i \mid d_i \in \text{Collection} \land \text{metadata}(d_i).\text{ticker} = t \land S(q, d_i) \text{ is high} \}
$$

Where:
- $q$ is the query,
- $S(q, d_i)$ is similarity (e.g., cosine distance),
- $t$ is the target ticker.
"""
    )

    if not st.session_state.ingested:
        st.warning(
            "Please ingest data in **1. Ingest Data** to use this feature.")
    else:
        st.subheader("Test Company-Filtered Retrieval")
        test_query = st.text_input(
            "Enter a test query:",
            value="total revenue net sales for the quarter",
        )
        c1, c2 = st.columns(2)
        with c1:
            test_ticker = st.selectbox(
                "Select a company ticker:", list(PORTFOLIO.keys()))
        with c2:
            k = st.slider("Top-k chunks", min_value=1, max_value=10, value=5)

        if st.button("Retrieve Chunks"):
            with st.spinner(f"Retrieving chunks for {test_ticker}..."):
                try:
                    chunks = retrieve_for_company(
                        test_query,
                        st.session_state.collection,
                        st.session_state.embedder,
                        ticker=test_ticker,
                        k=int(k),
                    )
                    st.success(
                        f"Retrieved {len(chunks)} chunks for {test_ticker}.")
                    if not chunks:
                        st.warning(
                            "No chunks returned. Ensure PDFs were ingested and contain relevant content.")
                    else:
                        for i, c in enumerate(chunks):
                            with st.expander(f"Chunk {i+1} | distance={c.get('similarity', None)}"):
                                st.caption(
                                    f"Metadata: {c.get('metadata', {})}")
                                st.write(c.get("text", ""))
                except Exception as e:
                    st.error(f"Retrieval error: {e}")

        st.markdown("### Explanation of Execution")
        st.markdown(
            """
`retrieve_for_company()` performs semantic search inside ChromaDB, but adds a strict metadata filter:

- `where={"ticker": ticker}`

That filter ensures the retrieved context is **company-specific**, preserving the integrity of downstream extraction.
"""
        )

    st.divider()


# ============================================================
# Page: 4. Batch Extraction
# ============================================================
elif st.session_state.page == "4. Batch Extraction":
    st.header("4. Batch Extraction: Automated Portfolio Data Consolidation")

    st.markdown(
        r"""
Once retrieval is secure and the schema is fixed, Alex can extract structured metrics across the full portfolio.

To improve coverage, the pipeline uses **multiple targeted queries** per company, then unions the retrieved chunks:

$$
R_{\text{company}} = \bigcup_{q \in Q} \text{top-k}(q, t)
$$

Where $Q$ is a set of targeted questions (revenue, income, EPS, cash, outlook, risks, etc.) and $t$ is the company ticker.

Those retrieved excerpts become the context that the LLM must translate into a single JSON object per company.
"""
    )

    if not st.session_state.ingested:
        st.warning(
            "Please ingest data in **1. Ingest Data** before running extraction.")
        st.divider()
    else:
        st.subheader("OpenAI Configuration")
        st.caption(
            "Provide your API key (not stored). If blank, the app will use OPENAI_API_KEY from environment.")
        openai_api_key = st.text_input(
            "OpenAI API key", type="password", value="")
        llm_model = st.selectbox("Model", options=[
                                 "gpt-4o", "gpt-4.1-mini", "gpt-4o-mini", "gpt-3.5-turbo"], index=0)

        if st.button("Start Batch Extraction"):
            with st.spinner("Extracting across all portfolio companies..."):
                try:
                    # Build client (do not store key in session_state)
                    if openai_api_key.strip():
                        os.environ["OPENAI_API_KEY"] = openai_api_key.strip()

                    client_llm = OpenAI()

                    extracted_df, total_tokens, total_cost, elapsed_time = run_batch_extraction(
                        PORTFOLIO,
                        st.session_state.collection,
                        st.session_state.embedder,
                        client_llm,
                        model=llm_model,
                    )

                    st.session_state.extracted_df = extracted_df
                    st.session_state.total_tokens = int(total_tokens)
                    st.session_state.total_cost = float(total_cost)
                    st.session_state.elapsed_time = float(elapsed_time)

                    st.success("Batch extraction complete.")
                except Exception as e:
                    st.error(f"Batch extraction failed: {e}")

        if st.session_state.extracted_df is not None and not st.session_state.extracted_df.empty:
            st.subheader("Extracted Data Preview")
            st.dataframe(st.session_state.extracted_df)

            st.markdown("#### Export")
            st.download_button(
                label="Download extracted table (Excel)",
                data=_df_to_excel_bytes(
                    st.session_state.extracted_df, sheet_name="extracted"),
                file_name="portfolio_extracted_metrics.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.markdown("#### Run summary")
            st.markdown(
                f"""
- Companies: **{len(st.session_state.extracted_df)}**
- Total time: **{st.session_state.elapsed_time:.1f}s** (~{(st.session_state.elapsed_time/max(1, len(st.session_state.extracted_df))):.1f}s / company)
- Tokens: **{st.session_state.total_tokens:,}**
- Estimated LLM cost: **${st.session_state.total_cost:.4f}**
"""
            )

    st.markdown("### Explanation of Execution")
    st.markdown(
        """
The batch process loops over each company in the portfolio and:
1. Retrieves company-filtered context using multiple targeted queries.
2. Calls the LLM with `response_format={"type": "json_object"}` to enforce structured JSON output.
3. Builds a single DataFrame for analysis, validation, and reporting.

This converts an unstructured, manual workflow into a repeatable data pipeline.
"""
    )
    st.divider()


# ============================================================
# Page: 5. Validate Accuracy
# ============================================================
elif st.session_state.page == "5. Validate Accuracy":
    st.header("5. Trust, But Verify: Validation Against Ground Truth")

    st.markdown(
        r"""
Before Alex uses results for investment committee discussions or proxy-voting decisions,
he validates the extracted output against manually verified ground truth.

We compute an overall field-level accuracy metric (excluding fields where both sides are "N/A"):

$$
Accuracy_{\text{field}} = \frac{|exact| + |partial|}{|total\ non\text{-}N/A|}
$$

The validation function includes fuzzy matching for numerical values (e.g., different currency formatting).
"""
    )

    if st.session_state.extracted_df is None or st.session_state.extracted_df.empty:
        st.warning("Run **4. Batch Extraction** first to produce extracted data.")
        st.divider()
    else:
        st.subheader("Ground Truth (Sample)")
        ground_truth = get_ground_truth_data()
        st.dataframe(ground_truth)

        fields_to_check = ["total_revenue", "net_income",
                           "diluted_eps", "cash_and_equivalents"]
        st.caption(f"Fields checked: {', '.join(fields_to_check)}")

        if st.button("Validate Extracted Data"):
            with st.spinner("Validating extracted data..."):
                try:
                    val_df, acc = validate_extraction(
                        st.session_state.extracted_df,
                        ground_truth,
                        fields_to_check=fields_to_check,
                    )
                    st.session_state.val_df = val_df
                    st.session_state.overall_accuracy = float(acc)
                    st.success("Validation complete.")
                except Exception as e:
                    st.error(f"Validation failed: {e}")

        if st.session_state.val_df is not None:
            st.subheader("Validation Results")
            st.markdown(
                f"**Overall accuracy (excluding BOTH_NA): {st.session_state.overall_accuracy:.1%}**")
            st.dataframe(st.session_state.val_df)

            status_counts = st.session_state.val_df["status"].value_counts()
            st.markdown("**Status counts**")
            st.code(status_counts.to_string())

            # Plot (matplotlib, explicit figure)
            fig = plt.figure(figsize=(8, 4))
            ax = fig.add_subplot(111)
            ax.bar(status_counts.index.astype(str), status_counts.values)
            ax.set_title("Extraction Accuracy Status per Field")
            ax.set_xlabel("Status")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=35)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            st.pyplot(fig)

            mismatches = st.session_state.val_df[st.session_state.val_df["status"].isin([
                                                                                        "MISMATCH", "MISSING"])]
            if not mismatches.empty:
                st.warning(
                    f"Mismatches found: {len(mismatches)} rows. Review before using outputs operationally.")
                st.dataframe(mismatches)
            else:
                st.info("No mismatches detected in this sample ground-truth set.")

    st.markdown("### Explanation of Execution")
    st.markdown(
        """
Validation compares extracted values to ground truth using:
- Exact matches for identical strings
- Numeric parsing (for revenue / income / cash) and tolerant comparisons
- Special handling of "N/A" (BOTH_NA is excluded from accuracy)

This gives Alex quantified confidence and a clear error-review list.
"""
    )
    st.divider()


# ============================================================
# Page: 6. Actionable Insights
# ============================================================
elif st.session_state.page == "6. Actionable Insights":
    st.header("6. Actionable Insights: Comparison, Screening, and Charts")

    st.markdown(
        """
With structured extraction complete, Alex's goal is to convert raw fields into analyst-ready insights:

- A **comparison table** he can export to Excel.
- **Portfolio summary statistics** (medians, ranges).
- **Screening flags** that surface potential risks or data quality gaps.

This is the “last mile” that turns RAG output into a workflow artifact.
"""
    )

    if st.session_state.extracted_df is None or st.session_state.extracted_df.empty:
        st.warning("Run **4. Batch Extraction** first.")
        st.divider()
    else:
        # Prepare a clean display table + numeric columns for analysis
        display_cols = [
            "ticker",
            "company",
            "reporting_period",
            "total_revenue",
            "net_income",
            "diluted_eps",
            "cash_and_equivalents",
            "segment_revenue_detail",
            "guidance_or_outlook",
            "key_risks_forward_looking",
        ]
        df = st.session_state.extracted_df.copy()
        display_df = df[[c for c in display_cols if c in df.columns]].copy()

        # numeric parsing
        if "total_revenue" in display_df.columns:
            display_df["total_revenue_numeric"] = display_df["total_revenue"].apply(
                parse_dollars)
        if "net_income" in display_df.columns:
            display_df["net_income_numeric"] = display_df["net_income"].apply(
                parse_dollars)
        if "cash_and_equivalents" in display_df.columns:
            display_df["cash_and_equivalents_numeric"] = display_df["cash_and_equivalents"].apply(
                parse_dollars)
        if "diluted_eps" in display_df.columns:
            display_df["diluted_eps_numeric"] = display_df["diluted_eps"].apply(
                parse_eps)

        st.subheader("Portfolio Comparison Table")
        st.dataframe(display_df.drop(columns=[
                     c for c in display_df.columns if c.endswith("_numeric")], errors="ignore"))

        st.download_button(
            label="Download comparison table (Excel)",
            data=_df_to_excel_bytes(display_df, sheet_name="comparison"),
            file_name="portfolio_comparison.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.subheader("Portfolio Summary Statistics")
        c1, c2, c3, c4 = st.columns(4)

        def _metric_card(col, series, label, scale=1.0, suffix=""):
            s = series.dropna()
            if len(s) == 0:
                col.metric(label, "N/A")
                return
            col.metric(label, f"{(s.median()/scale):.2f}{suffix}")

        if "total_revenue_numeric" in display_df.columns:
            _metric_card(c1, display_df["total_revenue_numeric"],
                         "Median Revenue", scale=1e9, suffix="B")
        if "net_income_numeric" in display_df.columns:
            _metric_card(c2, display_df["net_income_numeric"],
                         "Median Net Income", scale=1e9, suffix="B")
        if "cash_and_equivalents_numeric" in display_df.columns:
            _metric_card(
                c3, display_df["cash_and_equivalents_numeric"], "Median Cash", scale=1e9, suffix="B")
        if "diluted_eps_numeric" in display_df.columns:
            _metric_card(c4, display_df["diluted_eps_numeric"],
                         "Median Diluted EPS", scale=1.0, suffix="")

        st.subheader("Quick Visuals")
        # Revenue bar chart (if available)
        if "total_revenue_numeric" in display_df.columns and display_df["total_revenue_numeric"].notna().any():
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(111)
            plot_df = display_df[["ticker", "total_revenue_numeric"]].dropna(
            ).sort_values("total_revenue_numeric", ascending=False)
            ax.bar(plot_df["ticker"].astype(str),
                   plot_df["total_revenue_numeric"] / 1e9)
            ax.set_title("Total Revenue (B)")
            ax.set_xlabel("Ticker")
            ax.set_ylabel("Revenue (Billions)")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            st.pyplot(fig)

        # Screening examples
        st.subheader("Screening Flags")
        flags = []

        # Example thresholds for this financial-metrics pipeline
        low_cash_threshold = 10_000_000_000  # $10B
        if "cash_and_equivalents_numeric" in display_df.columns:
            low_cash = display_df[display_df["cash_and_equivalents_numeric"].notna() & (
                display_df["cash_and_equivalents_numeric"] < low_cash_threshold)]
            for _, r in low_cash.iterrows():
                flags.append(
                    {
                        "ticker": r.get("ticker", ""),
                        "company": r.get("company", ""),
                        "flag": f"Cash < ${low_cash_threshold/1e9:.0f}B",
                        "value": r.get("cash_and_equivalents", "N/A"),
                    }
                )

        if "net_income_numeric" in display_df.columns:
            losses = display_df[display_df["net_income_numeric"].notna() & (
                display_df["net_income_numeric"] < 0)]
            for _, r in losses.iterrows():
                flags.append(
                    {
                        "ticker": r.get("ticker", ""),
                        "company": r.get("company", ""),
                        "flag": "Net income < 0",
                        "value": r.get("net_income", "N/A"),
                    }
                )

        # Missingness flag: 2+ missing core fields
        core_fields = [c for c in ["total_revenue", "net_income",
                                   "diluted_eps"] if c in display_df.columns]
        if core_fields:
            missing_count = display_df[core_fields].apply(
                lambda row: sum(str(x).strip().lower() in [
                                "n/a", "na", "none", ""] for x in row),
                axis=1,
            )
            missing_rows = display_df[missing_count >= 2]
            for _, r in missing_rows.iterrows():
                flags.append(
                    {
                        "ticker": r.get("ticker", ""),
                        "company": r.get("company", ""),
                        "flag": "Missing 2+ core metrics",
                        "value": " / ".join([str(r.get(f, "N/A")) for f in core_fields]),
                    }
                )

        if flags:
            st.dataframe(pd.DataFrame(flags))
        else:
            st.info(
                "No companies currently breach the screening rules configured in this demo.")

    st.divider()


# ============================================================
# Page: 7. ROI Analysis
# ============================================================
elif st.session_state.page == "7. ROI Analysis":
    st.header("7. ROI Analysis: Quantifying Time and Cost Savings")

    st.markdown(
        r"""
Alex needs to quantify the business value of the RAG pipeline.

A simple time model is:

```math
T_{\text{manual}} = N \times T_{\text{manual\_per\_doc}}
```

```math
\text{Time Reduction (\%)} = \left(1 - \frac{T_{\text{RAG}}}{T_{\text{manual}}}\right) \times 100
```

Where:
- $N$ is the number of companies,
- $T_{\text{manual\_per\_doc}}$ is a reasonable analyst estimate,
- $T_{\text{RAG}}$ is measured runtime of the pipeline.
"""
    )

    if st.session_state.extracted_df is None or st.session_state.extracted_df.empty:
        st.warning(
            "Run **4. Batch Extraction** first to measure runtime and cost.")
        st.divider()
    else:
        st.subheader("Assumptions")
        manual_time_per_doc_min = st.number_input(
            "Manual review time per company (minutes)", min_value=5, max_value=120, value=30, step=5)

        n = len(PORTFOLIO)
        manual_time_min = n * float(manual_time_per_doc_min)
        manual_time_hr = manual_time_min / 60.0

        rag_time_min = st.session_state.elapsed_time / 60.0
        savings_min = manual_time_min - rag_time_min
        savings_pct = (1.0 - (rag_time_min / manual_time_min)) * \
            100.0 if manual_time_min > 0 else 0.0

        st.markdown("### Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Manual time (hours)", f"{manual_time_hr:.1f}")
        c2.metric("RAG time (minutes)", f"{rag_time_min:.1f}")
        c3.metric("Time reduction", f"{savings_pct:.0f}%")

        st.markdown(
            f"""
- **Absolute savings:** {savings_min:.1f} minutes  
- **Estimated LLM cost:** ${st.session_state.total_cost:.4f}  
- **Tokens used:** {st.session_state.total_tokens:,}
"""
        )

        # Simple bar chart
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(111)
        ax.bar(["Manual Review", "RAG Pipeline"],
               [manual_time_min, rag_time_min])
        ax.set_title("Time Comparison (minutes)")
        ax.set_ylabel("Minutes")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        st.pyplot(fig)

    st.divider()


# ============================================================
# Footer
# ============================================================
st.write("© 2025 QuantUniversity. All Rights Reserved.")
st.caption(
    "The purpose of this demonstration is solely for educational use and illustration. "
    "Any reproduction requires prior written consent from QuantUniversity."
)
st.caption(
    "This lab was generated using the QuCreate platform. QuCreate relies on AI models for generating code, "
    "which may contain inaccuracies or errors."
)
