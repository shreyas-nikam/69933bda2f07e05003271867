
# Streamlit Application Specification: Portfolio-Level Governance RAG

## 1. Application Overview

### Purpose
This Streamlit application serves as a development blueprint for "Portfolio-Level Data Extraction with RAG." Its primary purpose is to demonstrate how CFA Charterholders and Investment Professionals, like Alex Chen, can leverage Retrieval-Augmented Generation (RAG) to automate the extraction and analysis of corporate governance data from multiple proxy statements (DEF 14A filings). The application aims to transform a time-consuming manual review process into an efficient, data-driven workflow, enabling rapid identification of governance "red flags," comparative analysis across portfolios, and quantifiable time and cost savings.

### High-Level Story Flow
Alex Chen, a Portfolio Manager, uses this application to streamline his governance review process.

1.  **Introduction**: The app greets Alex and sets the stage, highlighting the challenges of manual proxy statement review.
2.  **Ingest Data**: Alex specifies a directory containing proxy statements. The system then ingests these PDF documents, extracts text, splits it into chunks, and stores them with metadata in a ChromaDB vector store.
3.  **Define Schema**: The application displays the predefined JSON schema used for structured data extraction, emphasizing consistency and handling of "N/A" values.
4.  **Filtered Retrieval**: Alex tests the crucial company-filtered retrieval mechanism, ensuring that queries only retrieve information from a specified company's documents, preventing data mix-up.
5.  **Batch Extraction**: Alex initiates a batch extraction process across his entire portfolio. The system uses a multi-query strategy and an LLM (GPT-4o) to extract structured governance data for each company, compiling the results into a pandas DataFrame.
6.  **Validate Accuracy**: The extracted data is then validated against a manually verified ground truth, providing metrics on field-level, company-level completeness, and portfolio-level accuracy, giving Alex confidence in the AI's output.
7.  **Actionable Insights**: The application generates a comparative governance table, calculates portfolio-level statistics (median CEO pay, ESG adoption rates), and flags companies based on predefined governance thresholds. Visualizations aid in quick understanding.
8.  **ROI Analysis**: Finally, Alex views a summary of the time and cost savings achieved by using the RAG pipeline compared to manual review, demonstrating the tangible business value.

## 2. Code Requirements

### Imports

```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import time
import re # Used by parse_dollars
from source import * # Import all functions and global variables from source.py
```

### Streamlit Application Structure and Flow

The application will use a sidebar for navigation, simulating a multi-page experience. `st.session_state` will be used to maintain state across pages and interactions.

```python
# Initialize session state variables
if 'page' not in st.session_state:
    st.session_state.page = "Introduction"
if 'doc_dir' not in st.session_state:
    st.session_state.doc_dir = 'proxy_filings' # Default directory for PDFs
if 'ingested' not in st.session_state:
    st.session_state.ingested = False
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'embedder' not in st.session_state:
    st.session_state.embedder = None
if 'governance_df' not in st.session_state:
    st.session_state.governance_df = None
if 'total_tokens' not in st.session_state:
    st.session_state.total_tokens = 0
if 'total_cost' not in st.session_state:
    st.session_state.total_cost = 0
if 'elapsed_time' not in st.session_state:
    st.session_state.elapsed_time = 0
if 'val_df' not in st.session_state:
    st.session_state.val_df = None
if 'overall_accuracy' not in st.session_state:
    st.session_state.overall_accuracy = 0.0

# Sidebar for navigation
with st.sidebar:
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
            "7. ROI Analysis"
        ]
    )
    if page_selection:
        st.session_state.page = page_selection

# Main content area based on page_selection
st.title("Portfolio-Level Governance Data Extraction with RAG")

# --- Introduction Page ---
if st.session_state.page == "Introduction":
    st.header("Introduction: The Challenge of Manual Governance Review")
    st.markdown(f"Alex Chen, a CFA Charterholder and Portfolio Manager at Apex Capital Management, faces a significant challenge. His firm manages a diverse portfolio of publicly traded companies, and a critical part of his fiduciary duty involves monitoring the corporate governance practices of these holdings. Each year, he needs to review dozens of complex proxy statements (DEF 14A filings), which can be 50-150 pages long. This manual review is incredibly time-consuming, making it nearly impossible to conduct a comprehensive, consistent, and timely comparative analysis across the entire portfolio.")
    st.markdown(f"Alex needs a way to rapidly identify potential governance \"red flags\"—such as excessive CEO compensation, low board independence, or the absence of ESG metrics in executive incentive plans—and consolidate this information into an actionable format for investment committee discussions and proxy voting decisions. This application demonstrates how Alex leverages Retrieval-Augmented Generation (RAG) to transform hours of manual document review into minutes of AI-assisted, data-driven analysis, enabling him to fulfill his responsibilities more effectively and proactively.")
    st.markdown(f"---")

# --- 1. Ingest Data Page ---
elif st.session_state.page == "1. Ingest Data":
    st.header("1. Setting the Foundation: Ingesting Proxy Statements for Portfolio-Wide Analysis")
    st.markdown(f"Alex’s first step is to get the raw data—the proxy statements—into a format that the RAG system can process. Manually reading each PDF is tedious and error-prone. Instead, he will automate the extraction of text from these documents and store them in a searchable database. This process is crucial for multi-document RAG, as it creates an indexed, machine-readable repository of his portfolio's governance information. Each document is broken down into smaller, meaningful chunks, which are then converted into numerical representations (embeddings) for efficient similarity search. Critically, each chunk is tagged with metadata like the company ticker, ensuring that information from one company's proxy statement doesn't accidentally get mixed with another's during retrieval. This metadata-rich indexing is fundamental to scaling RAG from single-document Q&A to a comprehensive multi-company analysis.")

    st.subheader("Configuration")
    st.session_state.doc_dir = st.text_input("Enter directory for proxy PDF files:", value=st.session_state.doc_dir)

    if st.button("Ingest Proxy Statements"):
        with st.spinner("Ingesting data... This may take a few minutes depending on the number of documents."):
            try:
                # Function invocation from source.py
                st.session_state.collection, st.session_state.embedder = ingest_proxy_statements(doc_dir=st.session_state.doc_dir)
                st.session_state.ingested = True
                st.success("Data Ingestion Complete!")
                st.markdown(f"**Total chunks ingested:** {st.session_state.collection.count() if st.session_state.collection else 0} across {len(PORTFOLIO)} companies.")
            except Exception as e:
                st.error(f"Error during data ingestion: {e}")
                st.session_state.ingested = False
    
    if not st.session_state.ingested:
        st.warning("Please ingest data to proceed with other steps.")
    else:
        st.info("Data has been ingested. You can proceed to the next steps.")

    st.markdown(f"### Explanation of Execution")
    st.markdown(f"The process above initializes the embedding model and text splitter, then iterates through each proxy statement in Alex's portfolio. For each PDF, it extracts the text, splits it into smaller chunks, and creates a unique ID and relevant metadata (like `ticker` and `company_name`) for each chunk. These chunks are then embedded into a vector space and added to the `ChromaDB` collection.")
    st.markdown(f"This process transforms unstructured PDF documents into a structured, searchable knowledge base. Alex can now query this database efficiently, knowing that each piece of information is indexed with its source company. The output confirms how many total text chunks are available for retrieval across all portfolio companies, signaling readiness for detailed analysis.")
    st.markdown(f"---")

# --- 2. Define Schema Page ---
elif st.session_state.page == "2. Define Schema":
    st.header("2. Defining Our Governance Lens: Structured Data Extraction Schema")
    st.markdown(f"Alex isn't just looking for random facts; he needs specific, comparable governance data points to perform his analysis. He requires a structured output—a predefined set of fields for each company, ensuring consistency. For example, he needs 'CEO total compensation' in a numerical format and 'ESG in compensation' as a 'Yes/No'. This structured approach prevents the LLM from hallucinating or providing irrelevant information, allowing Alex to build a reliable comparative table. The instruction to use \"N/A\" for missing information is also critical, as the absence of a data point (e.g., ESG metrics in compensation) can itself be an important governance signal for Alex.")
    st.markdown(f"The extraction schema *is* Alex's data model. The fields he defines here directly dictate the metrics he can use for comparative analysis, governance screening, and ultimately, his investment decisions. This design is a financial decision, driven by what metrics matter for Apex Capital’s investment thesis and fiduciary obligations.")

    st.subheader("Extraction Schema Prompt")
    st.code(EXTRACTION_PROMPT, language='python')

    st.markdown(f"### Explanation of Execution")
    st.markdown(f"The `EXTRACTION_PROMPT` defines the precise JSON schema that Alex requires for his governance data. It lists 18 fields, covering quantitative details (like CEO compensation figures and board size) and qualitative information (such as ESG metric details and clawback policies). This prompt is critical for two reasons:")
    st.markdown(f"1.  **Consistency:** It ensures that the LLM extracts the *same* data points in the *same* format for every company, making direct comparisons possible.")
    st.markdown(f"2.  **Handling Missing Data:** The explicit instruction to use \"N/A\" for unavailable information is a vital governance signal. If a company's proxy statement doesn't mention ESG metrics in compensation, \"N/A\" indicates this lack of disclosure, which is a key insight for Alex, rather than a system failure. This output informs Alex's real-world decision-making about potential governance weaknesses or disclosure gaps.")
    st.markdown(f"---")

# --- 3. Filtered Retrieval Page ---
elif st.session_state.page == "3. Filtered Retrieval":
    st.header("3. Precision in Retrieval: Ensuring Company-Specific Insights")
    st.markdown(f"A critical challenge in multi-document RAG is preventing information leakage or \"mix-up\" across companies. If Alex queries for \"CEO total compensation,\" he must ensure that the retrieved chunks of text pertain *only* to the specific company he is analyzing (e.g., Apple) and not inadvertently include information from another company (e.g., Amazon). A mix-up could lead to catastrophic errors in his analysis, attributing Amazon's CEO pay to Apple, for instance. This company-filtered retrieval mechanism is a fundamental safeguard, guaranteeing the integrity and accuracy of Alex's governance data.")

    st.markdown(r"The underlying mathematical concept for similarity search in vector databases often involves distance metrics like cosine similarity. When embedding a query $q$ and comparing it to document chunks $d_i$, a similarity score $S(q, d_i)$ is computed. However, for company-filtered retrieval, Alex applies a categorical filter. This is an essential non-semantic filtering step that precedes or runs in conjunction with semantic search. The `where` clause acts as a hard constraint:")
    st.markdown(r"$$R_{\text{company}} = \{d_i \mid d_i \in \text{Collection} \land \text{metadata}(d_i).\text{ticker} = \text{ticker} \land S(q, d_i) \text{ is high}\}$$")
    st.markdown(r"where $R_{\text{company}}$ represents the set of retrieved chunks for a specific company, $d_i$ are document chunks, $\text{Collection}$ is the vector store, $\text{metadata}(d_i).\text{ticker}$ is the ticker metadata associated with chunk $d_i$, $S(q, d_i)$ is the similarity score between query $q$ and chunk $d_i$, and `is high` refers to chunks with high similarity scores.")
    st.markdown(r"This ensures that only chunks whose metadata explicitly match the target `ticker` are considered, eliminating cross-company contamination.")

    if st.session_state.ingested:
        st.subheader("Test Company-Filtered Retrieval")
        test_query = st.text_input("Enter a test query (e.g., 'CEO total compensation'):", "CEO total compensation salary bonus stock awards")
        test_ticker = st.selectbox("Select a company ticker:", list(PORTFOLIO.keys()))

        if st.button("Retrieve Chunks"):
            with st.spinner(f"Retrieving chunks for {test_ticker}..."):
                # Function invocation from source.py
                test_chunks = retrieve_for_company(test_query, st.session_state.collection, st.session_state.embedder, ticker=test_ticker, k=5)
                st.success(f"Retrieved {len(test_chunks)} chunks for {test_ticker}.")
                if test_chunks:
                    for i, c in enumerate(test_chunks[:3]):
                        st.text(f"[{c['similarity']:.3f}] (Ticker: {c['metadata'].get('ticker', 'N/A')}) {c['text'][:150]}...")
                else:
                    st.warning(f"No chunks retrieved for {test_ticker}. Ensure '{test_ticker}_proxy_2024.pdf' exists and was ingested.")
    else:
        st.warning("Please ingest data in '1. Ingest Data' to use this feature.")

    st.markdown(f"### Explanation of Execution")
    st.markdown(f"The `retrieve_for_company` function demonstrates the crucial company-filtered retrieval. When Alex searches for information, the `where={\"ticker\": ticker}}` clause acts as a strict filter, ensuring that the RAG system only considers text chunks that originate from the specified company's proxy statement. The output of the test query for '{test_ticker}' explicitly shows the `ticker` in the metadata for each retrieved chunk, confirming that the filter is working as intended.")
    st.markdown(f"This safeguard is paramount for Alex. Without it, the LLM might receive mixed signals from various company documents, leading to inaccurate and misleading governance data—a \"silent, catastrophic error\" that would undermine Apex Capital's analysis and decision-making process.")
    st.markdown(f"---")

# --- 4. Batch Extraction Page ---
elif st.session_state.page == "4. Batch Extraction":
    st.header("4. The Automated Analyst: Batch Extraction and Data Consolidation")
    st.markdown(f"With the schema defined and retrieval secured, Alex can now unleash the full power of the RAG pipeline: batch extraction across all portfolio companies. Instead of manually reviewing each proxy, he'll run an automated loop that extracts the structured governance data for every company. To ensure comprehensive coverage, he employs a **multi-query retrieval strategy**. A single query might miss information found in different sections of a proxy statement. By using several targeted queries (e.g., for compensation, board composition, ESG metrics, vesting, and say-on-pay), he can union the retrieved chunks to build a more complete context for the LLM. This not only makes the process significantly faster but also more consistent and thorough than any manual review.")

    st.markdown(r"The multi-query retrieval strategy for a given company is formulated as the union of top-$k$ relevant chunks from multiple queries $q \in Q$:")
    st.markdown(r"$$R_{\text{company}} = \bigcup_{q \in Q} \text{top-k}(q, \text{ticker})$$")
    st.markdown(r"where $R_{\text{company}}$ represents the comprehensive context for a specific company, $Q = \{q_{\text{comp}}, q_{\text{board}}, q_{\text{esg}}, q_{\text{vesting}}, q_{\text{vote}}\}$ represents the set of targeted queries covering different governance aspects, and $\text{top-k}(q, \text{ticker})$ retrieves the top-$k$ chunks for query $q$ filtered by `ticker`.")
    st.markdown(r"This approach ensures that the LLM receives a comprehensive context from which to extract all required fields, even if they are scattered across the document.")

    if st.session_state.ingested:
        st.subheader("Run Batch Extraction")
        if st.button("Start Batch Extraction"):
            with st.spinner("Extracting governance data for all portfolio companies... This may take several minutes."):
                results = []
                total_cost_temp = 0
                total_tokens_temp = 0
                start_time_temp = time.time()

                for ticker, name in PORTFOLIO.items():
                    st.write(f"Extracting: {ticker} ({name})...")
                    # Function invocation from source.py
                    data, tokens, cost = extract_governance_data(
                        ticker, name, st.session_state.collection, st.session_state.embedder, model='gpt-4o'
                    )
                    results.append(data)
                    total_tokens_temp += tokens
                    total_cost_temp += cost
                    st.write(f"  OK ({tokens:,} tokens, ${cost:.4f})")

                st.session_state.elapsed_time = time.time() - start_time_temp
                st.session_state.governance_df = pd.DataFrame(results)
                st.session_state.total_tokens = total_tokens_temp
                st.session_state.total_cost = total_cost_temp

                st.success("Batch Extraction Complete!")
                st.markdown(f"**Batch extraction complete:**")
                st.markdown(f"Companies: {len(results)}")
                st.markdown(f"Total time: {st.session_state.elapsed_time:.1f}s ({st.session_state.elapsed_time/len(results):.1f}s per company)")
                st.markdown(f"Total tokens: {st.session_state.total_tokens:,}")
                st.markdown(f"Total cost: ${st.session_state.total_cost:.4f}")
                st.subheader("Extracted Governance Data (First 5 Rows):")
                st.dataframe(st.session_state.governance_df.head())
        
        if st.session_state.governance_df is not None:
            st.subheader("Previously Extracted Governance Data (First 5 Rows):")
            st.dataframe(st.session_state.governance_df.head())
    else:
        st.warning("Please ingest data in '1. Ingest Data' to use this feature.")

    st.markdown(f"### Explanation of Execution")
    st.markdown(f"This section automates the data extraction for Alex's entire portfolio. The `extract_governance_data` function uses multiple, targeted queries to retrieve relevant chunks for each company, ensuring that the LLM has a comprehensive context. These chunks are then fed to the `gpt-4o` model, which extracts data according to the predefined JSON schema.")
    st.markdown(f"The output confirms the efficiency of this batch process: it shows the time taken and the tokens and cost incurred per company and for the entire portfolio. Alex can see at a glance that processing multiple proxies now takes minutes, not hours. The resulting `governance_df` DataFrame is the raw, structured data Alex needs to start his comparative analysis, a significant step forward from manual review. The `parse_dollars` function is crucial for converting textual compensation figures (e.g., \"$15.3 million\") into numerical values, enabling quantitative analysis and aggregation.")
    st.markdown(f"---")

# --- 5. Validate Accuracy Page ---
elif st.session_state.page == "5. Validate Accuracy":
    st.header("5. Trust, But Verify: Validating Extraction Accuracy")
    st.markdown(f"As a CFA Charterholder, Alex understands that even advanced AI systems like RAG are not infallible. Before presenting any findings to Apex Capital's investment committee, he must validate the accuracy of the extracted data against manually verified ground truth for a subset of the portfolio. This step is crucial for maintaining data integrity and building trust in the AI-assisted workflow. Without validation, potential errors (e.g., misparsed numbers or incorrect qualitative assessments) could lead to flawed investment decisions or misinformed proxy votes. Alex needs specific metrics to quantify this accuracy, including fuzzy matching for numerical values to account for slight variations in how they might be represented.")

    st.markdown(f"Alex uses three key metrics for validation:")
    st.markdown(f"1.  **Field-level accuracy:** This measures the correctness of individual data points.")
    st.markdown(r"$$Accuracy_{\text{field}} = \frac{\text{|exact match|} + \text{|partial match|}}{\text{|total non-N/A fields|}}$$")
    st.markdown(r"where `exact match` means values are identical, `partial match` accounts for numerical values that are logically equivalent but may differ in formatting (e.g., \"$15.3M\" vs. \"$15,300,000\"), and `total non-N/A fields` refers to fields where the ground truth is not 'N/A'.")
    st.markdown(f"2.  **Company-level completeness:** This assesses how many required fields were successfully extracted for each company (not marked \"N/A\").")
    st.markdown(r"$$Completeness_i = \frac{\text{|non-N/A fields extracted for company i|}}{\text{|total schema fields|}}$$")
    st.markdown(r"where $Completeness_i$ is the completeness for company $i$, `non-N/A fields extracted` are the fields successfully extracted, and `total schema fields` is the total number of fields defined in the extraction schema.")
    st.markdown(f"3.  **Portfolio-level accuracy:** This provides an aggregated view of accuracy across all companies and fields.")
    st.markdown(r"$$Accuracy_{\text{portfolio}} = \frac{\sum_i \sum_j [\text{correct}_{ij}]}{\sum_i \sum_j [\text{verifiable}_{ij}]}$$")
    st.markdown(r"where $[\text{correct}_{ij}]$ is 1 if field $j$ for company $i$ is correct (exact or partial match) and verifiable, and $[\text{verifiable}_{ij}]$ is 1 if the ground truth for that field is not N/A.")

    if st.session_state.governance_df is not None:
        st.subheader("Ground Truth Data (Sample):")
        st.dataframe(ground_truth) # ground_truth is globally available from source.py

        if st.button("Validate Extracted Data"):
            with st.spinner("Validating extracted data against ground truth..."):
                # Function invocation from source.py
                st.session_state.val_df, st.session_state.overall_accuracy = validate_extraction(st.session_state.governance_df, ground_truth, fields_to_check)
                st.success("Validation Complete!")

                st.subheader("Validation Results Summary:")
                st.markdown(f"**Overall accuracy (excluding 'BOTH_NA' fields): {st.session_state.overall_accuracy:.1%}")
                
                if st.session_state.val_df is not None:
                    status_counts = st.session_state.val_df['status'].value_counts()
                    st.markdown(f"**Status Counts:**")
                    st.text(status_counts.to_string())

                    mismatches = st.session_state.val_df[st.session_state.val_df['status'].isin(['MISMATCH', 'MISSING'])]
                    if not mismatches.empty:
                        st.markdown(f"**MISMATCHES ({len(mismatches)}):**")
                        for idx, row in mismatches.iterrows():
                            st.write(f" - {row['ticker']}.{row['field']}: truth='{row['truth']}' vs extracted='{row['extracted']}' (Status: {row['status']})")
                    else:
                        st.info("No mismatches found!")
                    
                    st.subheader("Extraction Accuracy Status per Field")
                    # Recreate the plot explicitly for Streamlit
                    plt.figure(figsize=(8, 5))
                    status_counts.plot(kind='bar', color=['skyblue', 'lightgreen', 'orange', 'salmon', 'lightcoral'])
                    plt.title('Extraction Accuracy Status per Field')
                    plt.xlabel('Status')
                    plt.ylabel('Count')
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    st.pyplot(plt.gcf()) # Display the plot
        
        if st.session_state.val_df is not None:
            st.subheader("Previous Validation Results:")
            st.markdown(f"**Overall accuracy (excluding 'BOTH_NA' fields): {st.session_state.overall_accuracy:.1%}")
            # Display chart if available from previous run
            status_counts = st.session_state.val_df['status'].value_counts()
            plt.figure(figsize=(8, 5))
            status_counts.plot(kind='bar', color=['skyblue', 'lightgreen', 'orange', 'salmon', 'lightcoral'])
            plt.title('Extraction Accuracy Status per Field')
            plt.xlabel('Status')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(plt.gcf())

    else:
        st.warning("Please run batch extraction in '4. Batch Extraction' to validate data.")

    st.markdown(f"### Explanation of Execution")
    st.markdown(f"The `validate_extraction` function compares the RAG-extracted data (`governance_df`) against a small, manually verified `ground_truth` dataset. It performs checks for exact matches, fuzzy matches for numerical values (e.g., \"$15.3 million\" vs. \"15,300,000\"), and partial matches for qualitative fields. A critical aspect is how it handles \"N/A\" values: if both ground truth and extraction return \"N/A,\" it's considered `BOTH_NA` and correctly excluded from the accuracy calculation, as Alex recognizes this as informative rather than an extraction error.")
    st.markdown(f"The output provides a detailed breakdown of `EXACT_MATCH`, `PARTIAL_MATCH`, `BOTH_NA`, and `MISMATCH` statuses, along with an overall accuracy score. Alex can quickly see where the RAG system performs well and where manual review might still be needed. The bar chart provides a clear visual summary, empowering Alex to make informed decisions about the trustworthiness of the extracted data for Apex Capital's critical governance analysis.")
    st.markdown(f"---")

# --- 6. Actionable Insights Page ---
elif st.session_state.page == "6. Actionable Insights":
    st.header("6. Actionable Insights: Comparative Table and Portfolio-Level Metrics")
    st.markdown(f"With validated and structured data in hand, Alex's primary objective is to generate actionable insights. The core deliverable for Apex Capital's investment committee is a clean, comparative table summarizing key governance metrics across all portfolio companies. This table allows Alex to quickly identify outliers or companies that deviate from Apex Capital's governance standards. Furthermore, calculating portfolio-level aggregated statistics (e.g., median CEO compensation, ESG incentive adoption rate) provides a high-level overview, useful for internal reporting and client communications on Apex Capital’s overall governance footprint. This transformation of raw data into an analyst-ready report is the \"last mile\" that makes the RAG pipeline a truly valuable business workflow.")
    st.markdown(f"Alex uses these insights to:")
    st.markdown(f"*   **Screen for Red Flags:** Flag companies that breach pre-defined governance thresholds (e.g., CEO compensation exceeding a certain percentile, board independence below a target).")
    st.markdown(f"*   **Inform Engagement:** Identify companies requiring focused engagement on governance issues.")
    st.markdown(f"*   **Support Proxy Voting:** Make data-driven proxy voting decisions.")

    if st.session_state.governance_df is not None:
        st.subheader("Portfolio Governance Comparison Table")
        
        # Logic to prepare display_df (copied from source.py notebook block)
        display_cols = [
            'ticker', 'company', 'ceo_name', 'ceo_total_compensation',
            'board_size', 'independent_directors', 'board_independence_pct',
            'esg_in_compensation', 'say_on_pay_approval', 'clawback_policy'
        ]
        display_df = st.session_state.governance_df[[c for c in display_cols if c in st.session_state.governance_df.columns]].copy()
        
        display_df['ceo_total_compensation_numeric'] = display_df['ceo_total_compensation'].apply(parse_dollars)
        display_df['board_size_numeric'] = pd.to_numeric(display_df['board_size'], errors='coerce')
        display_df['independent_directors_numeric'] = pd.to_numeric(display_df['independent_directors'], errors='coerce')

        display_df['board_independence_pct_computed'] = (
            display_df['independent_directors_numeric'] / display_df['board_size_numeric'] * 100
        ).fillna('N/A').apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x)

        display_df['board_independence_pct'] = display_df.apply(
            lambda row: row['board_independence_pct_computed'] if row['board_independence_pct'] == 'N/A' else row['board_independence_pct'],
            axis=1
        )
        
        final_display_df = display_df.drop(columns=['ceo_total_compensation_numeric', 'board_size_numeric', 'independent_directors_numeric', 'board_independence_pct_computed'])
        st.dataframe(final_display_df)

        st.download_button(
            label="Export Comparison Table to Excel",
            data=final_display_df.to_excel(index=False).encode('utf-8'),
            file_name="governance_comparison.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.subheader("Portfolio Governance Insights")
        # CEO Compensation Analysis
        ceo_comps_numeric = display_df['ceo_total_compensation_numeric'].dropna()
        if len(ceo_comps_numeric) > 0:
            st.markdown(f"**CEO Compensation:**")
            st.markdown(f"  Median: ${ceo_comps_numeric.median()/1e6:.1f}M")
            st.markdown(f"  Range: ${ceo_comps_numeric.min()/1e6:.1f}M - ${ceo_comps_numeric.max()/1e6:.1f}M")
            
            # Visualize CEO Compensation
            plt.figure(figsize=(10, 6))
            sns.barplot(x=ceo_comps_numeric.values / 1e6, y=display_df.loc[ceo_comps_numeric.index, 'ticker'], palette='viridis')
            plt.title('CEO Total Compensation Across Portfolio (in $M)')
            plt.xlabel('CEO Total Compensation ($M)')
            plt.ylabel('Company Ticker')
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(plt.gcf())
        else:
            st.markdown("No numerical CEO compensation data available for analysis.")

        # ESG in Compensation Analysis
        esg_rate = display_df['esg_in_compensation'].dropna()
        if len(esg_rate) > 0:
            esg_yes_count = (esg_rate.str.lower() == 'yes').sum()
            esg_percent = (esg_yes_count / len(esg_rate)) * 100
            st.markdown(f"**ESG in Compensation:**")
            st.markdown(f"  {esg_yes_count}/{len(esg_rate)} companies ({esg_percent:.0f}%) include ESG metrics in executive compensation.")
        else:
            st.markdown("No ESG in Compensation data available for analysis.")

        # Clawback Policy Analysis
        clawback_policy = display_df['clawback_policy'].dropna()
        if len(clawback_policy) > 0:
            cb_yes_count = (clawback_policy.str.lower() == 'yes').sum()
            cb_percent = (cb_yes_count / len(clawback_policy)) * 100
            st.markdown(f"**Clawback Policy:**")
            st.markdown(f"  {cb_yes_count}/{len(clawback_policy)} companies ({cb_percent:.0f}%) have a clawback policy.")
        else:
            st.markdown("No Clawback Policy data available for analysis.")

        st.subheader("Governance Screening - Companies Flagged")
        flagged_companies = pd.DataFrame()

        # Threshold 1: CEO compensation > $50M
        high_comp_threshold = 50_000_000
        high_comp_flags = display_df[display_df['ceo_total_compensation_numeric'] > high_comp_threshold].copy()
        if not high_comp_flags.empty:
            high_comp_flags['Flag_Reason'] = f"CEO Total Comp > ${high_comp_threshold/1e6:.0f}M"
            flagged_companies = pd.concat([flagged_companies, high_comp_flags[['ticker', 'company', 'ceo_total_compensation', 'Flag_Reason']]])

        # Threshold 2: Board independence < 75%
        low_board_independence_threshold = 75.0
        low_ind_flags = display_df[
            (pd.to_numeric(display_df['board_independence_pct'].str.replace('%', ''), errors='coerce') < low_board_independence_threshold) &
            (pd.to_numeric(display_df['board_independence_pct'].str.replace('%', ''), errors='coerce').notna())
        ].copy()
        if not low_ind_flags.empty:
            low_ind_flags['Flag_Reason'] = f"Board Independence < {low_board_independence_threshold:.0f}%"
            flagged_companies = pd.concat([flagged_companies, low_ind_flags[['ticker', 'company', 'board_independence_pct', 'Flag_Reason']]])

        # Threshold 3: Absence of ESG incentives
        no_esg_flags = display_df[display_df['esg_in_compensation'].str.lower() == 'no'].copy()
        if not no_esg_flags.empty:
            no_esg_flags['Flag_Reason'] = "No ESG Metrics in Compensation"
            flagged_companies = pd.concat([flagged_companies, no_esg_flags[['ticker', 'company', 'esg_in_compensation', 'Flag_Reason']]])

        if not flagged_companies.empty:
            flagged_companies = flagged_companies.drop_duplicates(subset=['ticker', 'Flag_Reason']).reset_index(drop=True)
            st.markdown("Companies requiring attention:")
            st.dataframe(flagged_companies)
        else:
            st.info("No companies currently breach defined governance thresholds.")

    else:
        st.warning("Please run batch extraction in '4. Batch Extraction' to generate insights.")

    st.markdown(f"### Explanation of Execution")
    st.markdown(f"This section is where the extracted data truly becomes valuable for Alex. The RAG pipeline generates a `PORTFOLIO GOVERNANCE COMPARISON TABLE`, a clear and concise summary of key metrics for each company. This table, which Alex can easily export to Excel, is ready for presentation to Apex Capital's investment committee, enabling data-driven discussions on corporate governance.")
    st.markdown(f"Beyond the table, the notebook calculates and displays **portfolio-level aggregated governance insights**, such as the median CEO compensation, the percentage of companies including ESG metrics in executive pay, and the prevalence of clawback policies. These high-level statistics provide Alex with a quick overview of the portfolio's governance landscape.")
    st.markdown(f"Finally, Alex applies **custom governance thresholds** to automatically flag companies that exhibit \"red flags.\" This immediate identification of companies with, for example, high CEO compensation or low board independence, allows Alex to prioritize his engagement efforts and inform targeted proxy voting decisions, directly addressing his real-world workflow challenges. The bar chart for CEO compensation provides a quick visual comparison of pay across the portfolio.")
    st.markdown(f"---")

# --- 7. ROI Analysis Page ---
elif st.session_state.page == "7. ROI Analysis":
    st.header("7. The ROI: Quantifying Time and Cost Savings")
    st.markdown(f"Alex understands that demonstrating the return on investment (ROI) of new technology is crucial for gaining continued support for AI initiatives at Apex Capital Management. He needs to quantify the tangible benefits of this RAG-powered governance analysis. By comparing the estimated time and cost of manual review against the actual figures from the RAG pipeline, Alex can clearly articulate the efficiency gains. This analysis provides concrete evidence of how AI transforms a time-intensive, low-leverage task into a rapid, high-value activity, freeing up his time for more strategic financial analysis and direct engagement with portfolio companies.")

    st.markdown(r"The time savings calculation highlights the sheer scale of efficiency:")
    st.markdown(r"*   **Manual Review Time:** $T_{\text{manual}} = N_{\text{companies}} \times T_{\text{manual\_per\_proxy}}$")
    st.markdown(r"    where $T_{\text{manual}}$ is the total estimated manual review time, $N_{\text{companies}}$ is the number of companies in the portfolio, and $T_{\text{manual\_per\_proxy}}$ is the estimated manual review time per proxy statement (e.g., 30 minutes).")
    st.markdown(r"*   **RAG Extraction Time:** $T_{\text{RAG}} = T_{\text{elapsed}}$ (from Section 4)")
    st.markdown(r"    where $T_{\text{RAG}}$ is the actual time taken by the RAG extraction process.")
    st.markdown(r"The percentage reduction in time is then:")
    st.markdown(r"$$ \text{Time Reduction (%)} = \left(1 - \frac{T_{\text{RAG}}}{T_{\text{manual}}}\right) \times 100 $$")
    st.markdown(r"where $T_{\text{RAG}}$ is the RAG extraction time and $T_{\text{manual}}$ is the manual review time.")

    if st.session_state.governance_df is not None:
        st.subheader("Time Savings Analysis")
        n_companies = len(PORTFOLIO)
        manual_time_per_proxy_min = 30 # Estimated 30 minutes per proxy for manual review
        manual_time_min = n_companies * manual_time_per_proxy_min
        manual_time_hours = manual_time_min / 60

        rag_time_min = st.session_state.elapsed_time / 60

        st.markdown(f"**Manual Review:**")
        st.markdown(f"  Estimated Manual Time: {manual_time_min:.1f} min ({manual_time_hours:.1f} hours) for {n_companies} companies")

        st.markdown(f"**RAG Extraction:**")
        st.markdown(f"  RAG Time: {rag_time_min:.1f} min")

        savings_min = manual_time_min - rag_time_min
        savings_percent = (1 - (rag_time_min / manual_time_min)) * 100 if manual_time_min > 0 else 0

        st.markdown(f"**Savings:**")
        st.markdown(f"  Absolute Savings: {savings_min:.1f} min")
        st.markdown(f"  Percentage Reduction: {savings_percent:.0f}% reduction")
        st.markdown(f"  Total Cost for RAG: ${st.session_state.total_cost:.4f}")

        # Visualize Time Savings
        labels = ['Manual Review', 'RAG Extraction']
        times_minutes = [manual_time_min, rag_time_min]

        plt.figure(figsize=(8, 5))
        sns.barplot(x=labels, y=times_minutes, palette=['lightcoral', 'skyblue'])
        plt.title('Time Savings: Manual Review vs. RAG Extraction (Minutes)')
        plt.xlabel('Method')
        plt.ylabel('Time (Minutes)')
        plt.text(0, times_minutes[0], f'{times_minutes[0]:.1f} min', ha='center', va='bottom', fontsize=10)
        plt.text(1, times_minutes[1], f'{times_minutes[1]:.1f} min\n(${st.session_state.total_cost:.4f})', ha='center', va='bottom', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(plt.gcf())

    else:
        st.warning("Please run batch extraction in '4. Batch Extraction' to calculate ROI.")
    
    st.markdown(f"### Explanation of Execution")
    st.markdown(f"This final section quantifies the operational benefits of the RAG pipeline. By comparing the estimated time for manual review (30 minutes per proxy, as per the CFA Institute's benchmark) against the actual RAG processing time, Alex can clearly demonstrate significant **time and cost savings**. The output shows the estimated hours saved and the substantial percentage reduction in time.")
    st.markdown(f"The bar chart visually reinforces this impact, providing a compelling narrative for Apex Capital's management. This concrete ROI data enables Alex to justify the adoption of AI tools, highlighting how they enhance efficiency, reduce operational costs, and ultimately allow portfolio managers to focus on higher-value tasks, thereby fulfilling their fiduciary duties more effectively.")
    st.markdown(f"---")

```
