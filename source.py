import fitz  # PyMuPDF for PDF parsing
import chromadb
from sentence_transformers import SentenceTransformer
# Assuming 'langchain-classic' is specifically used as per the original notebook.
# If a newer 'langchain' is intended, this would be 'from langchain.text_splitter import RecursiveCharacterTextSplitter'
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
import os
import json
import pandas as pd
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import requests  # For downloading files programmatically

# --- Global Configurations and Constants ---
# It's good practice to load API keys securely from environment variables.
# DO NOT hardcode API keys directly in code for production.
# The OpenAI client will be initialized within the run_financial_analysis_pipeline function.

PORTFOLIO = {
    'AAPL': 'Apple Inc.',
    'GOOGL': 'Alphabet Inc.',
    'TSLA': 'Tesla Inc.',
    'IR': 'Unilever Inc.'
}

EXTRACTION_PROMPT = """You are a financial analysis assistant specializing
in extracting key financial performance metrics from earnings releases
and consolidated financial statements.

Extract the following fields from the provided document excerpts and
output the information as a JSON object. For not applicable or missing
information, use "N/A".

Required fields:
{{
  "company": "Name of the company",
  "ticker": "Stock ticker",
  "reporting_period": "Reporting period covered (e.g., Q4 2025, FY 2025)",
  "total_revenue": "Total revenue / net sales / turnover for the reporting period (include currency + units if shown)",
  "net_income": "Net income for the reporting period (include currency + units if shown)",
  "diluted_eps": "Diluted earnings per share for the reporting period (include currency if shown)",
  "cash_and_equivalents": "Cash and cash equivalents at period end (include currency + units if shown)",
  "segment_revenue_detail": "Key segment revenue if explicitly stated (e.g., Google Cloud revenue); otherwise N/A",
  "guidance_or_outlook": "Any explicit guidance/outlook statements for next quarter/year; otherwise N/A",
  "key_risks_forward_looking": "Key risks mentioned in forward-looking statements (e.g., macro, FX, regulation); otherwise N/A"
}}

IMPORTANT: Extract ONLY from the provided text. Do NOT use outside knowledge.
If a field cannot be determined from the text, use "N/A".

DOCUMENT EXCERPTS:
{context}"""

# Helper to parse required fields from the prompt for N/A returns


def _get_required_fields_from_prompt(prompt: str) -> list[str]:
    match = re.search(
        r'Required fields:\n(\s*\{\{.*?\}\}\s*)', prompt, re.DOTALL)
    if match:
        json_str = match.group(1).replace('{{', '{').replace('}}', '}')
        try:
            return list(json.loads(json_str).keys())
        except json.JSONDecodeError:
            pass
    return []  # Fallback

# --- Helper Functions for Data Parsing ---


def parse_dollars(val: str) -> float | None:
    """
    Extract numeric value from strings like '$63.2 million'.
    Handles "N/A" or NaN values by returning None.
    """
    if val is None or str(val).strip().lower() == "n/a" or pd.isna(val) or not isinstance(val, str):
        return None

    val = val.strip()
    match = re.search(
        r'[\$€£]?([\d,.]+)\s*(million|billion|M|B)?', val, re.IGNORECASE)
    if match:
        num_str = match.group(1).replace(',', '')
        try:
            num = float(num_str)
        except ValueError:
            return None  # Handle cases like '1.2.3' which are not valid numbers
        unit = match.group(2)
        if unit:
            unit = unit.lower()
            if unit in ['million', 'm']:
                return num * 1e6
            elif unit in ['billion', 'b']:
                return num * 1e9
        return num  # Return as is if no unit or unit not matched
    return None  # Return None if no match found after trying to parse


def parse_eps(val: str) -> float | None:
    """
    Parses a string value to extract diluted EPS as a float.
    Handles "N/A" or NaN values by returning None.
    """
    if val is None or str(val).strip().lower() == "n/a" or pd.isna(val):
        return None
    if not isinstance(val, str):
        return None
    s = val.strip().replace('$', '').replace('€', '').replace('£', '')
    try:
        # Improved regex to handle potential leading/trailing non-numeric chars
        match = re.search(r'[-+]?\d*\.?\d+', s)
        if match:
            return float(match.group())
        return None
    except Exception:
        return None

# --- Data Handling Functions ---


def download_proxy_filings(doc_dir: str, portfolio_map: dict):
    """
    Downloads PDF files from specified URLs into the doc_dir.
    """
    os.makedirs(doc_dir, exist_ok=True)
    base_url = "https://qucoursify.s3.us-east-1.amazonaws.com/ai-for-cfa/Session+28/"
    files_to_download_map = {
        'TSLA': 'TSLA_Q4_2025_Earnings.pdf',
        'IR': 'IR_Q4_2025_Earnings.pdf',
        'GOOGL': 'GOOGL_Q4_2025_Earnings.pdf',
        'AAPL': 'AAPL_Q4_2025_Earnings.pdf'
    }

    print(f"\nDownloading files to {doc_dir}...")
    for ticker, filename in files_to_download_map.items():
        if ticker in portfolio_map:  # Only download if in current portfolio
            file_url = f"{base_url}{filename}"
            filepath = os.path.join(doc_dir, filename)
            if os.path.exists(filepath):
                print(f"  {filename} already exists. Skipping download.")
                continue
            try:
                print(f"  Downloading {filename} from {file_url}...")
                response = requests.get(file_url, stream=True, timeout=30)
                response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"  Downloaded {filename}")
            except requests.exceptions.RequestException as e:
                print(f"  Error downloading {filename} from {file_url}: {e}")


def ingest_proxy_statements(doc_dir: str, portfolio_map: dict, chunk_size: int = 500, chunk_overlap: int = 100):
    """
    Ingest multiple PDF earnings releases into a single vector store with company metadata
    for filtered retrieval.
    """
    print("\nInitializing embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model loaded.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ",]  # More robust separators
    )

    client = chromadb.Client()

    # It's good practice to delete and recreate for a fresh ingest in this kind of pipeline
    try:
        client.delete_collection(name="portfolio_proxies")
        print("Deleted existing collection 'portfolio_proxies'.")
    except Exception:  # ChromaDB raises generic exception if collection doesn't exist
        print("No existing collection 'portfolio_proxies' to delete or error deleting. Proceeding to create.")

    collection = client.create_collection(
        name="portfolio_proxies",
        # Using cosine similarity for SentenceTransformer embeddings
        metadata={"hnsw:space": "cosine"}
    )
    print("Created new collection 'portfolio_proxies'.")

    all_chunks = []
    os.makedirs(doc_dir, exist_ok=True)  # Ensure the directory exists

    print(f"\nIngesting earnings statements from '{doc_dir}'...")
    for ticker, name in portfolio_map.items():
        # Assuming filename convention: Ticker_Q4_2025_Earnings.pdf
        filepath = os.path.join(doc_dir, f'{ticker}_Q4_2025_Earnings.pdf')

        if not os.path.exists(filepath):
            print(
                f"WARNING: File not found for {ticker} at {filepath}. Skipping.")
            continue

        print(f"Processing {ticker} ({name})...")
        try:
            doc = fitz.open(filepath)
            text = "".join([page.get_text() for page in doc])
            doc.close()
        except Exception as e:
            print(f"Error reading PDF {filepath}: {e}. Skipping.")
            continue

        chunks = splitter.split_text(text)

        # Assuming 2025 fiscal year based on filenames
        fiscal_year = '2025'

        for i, chunk in enumerate(chunks):
            chunk_id = f"{ticker}_chunk_{i}"
            all_chunks.append({
                'id': chunk_id,
                'text': chunk,
                'metadata': {
                    'ticker': ticker,
                    'company': name,
                    'chunk_id': i,
                    'doc_type': 'Earnings_Release',
                    'fiscal_year': fiscal_year,
                }
            })
        print(f"  {ticker}: {len(chunks)} chunks from {len(text.split()):,} words")

    if not all_chunks:
        print("No chunks were processed. Ensure PDF files exist and are readable in the 'proxy_filings' directory.")
        return None, None

    texts = [c['text'] for c in all_chunks]
    if not texts:
        print("No text documents to embed. Please check the input files.")
        return collection, embedder

    print(f"\nEmbedding {len(texts)} chunks...")
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True)

    print("Adding chunks to ChromaDB collection...")
    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[c['metadata'] for c in all_chunks],
        ids=[c['id'] for c in all_chunks]
    )

    print(
        f"\nTotal: {collection.count()} chunks across {len(portfolio_map)} companies")
    return collection, embedder


def retrieve_for_company(query: str, collection: chromadb.Collection, embedder: SentenceTransformer, ticker: str, k: int = 8):
    """Retrieve relevant chunks ONLY from a specific company's documents."""
    query_embedding = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        where={"ticker": ticker},
        include=['documents', 'metadatas', 'distances']
    )

    chunks = []
    if results and results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            chunks.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity': results['distances'][0][i],
            })
    return chunks


def extract_governance_data(ticker: str, company_name: str, collection: chromadb.Collection, embedder: SentenceTransformer, llm_client: OpenAI, model: str = 'gpt-4o'):
    """Extract structured financial data for one company via RAG."""

    queries = [
        "total revenue total net sales turnover for the quarter full year",
        "net income profit attributable to shareholders for the quarter full year",
        "diluted earnings per share EPS diluted EPS for the quarter full year",
        "cash and cash equivalents balance sheet period end",
        "outlook guidance future expectations forward-looking statements key risks",
        "segment revenue detail if explicitly stated"
    ]

    all_chunks = []
    seen_texts_hashes = set()

    for query in queries:
        chunks = retrieve_for_company(
            query, collection, embedder, ticker=ticker, k=5)
        for c in chunks:
            # Deduplicate chunks based on text content hash
            chunk_text_hash = hash(c['text'].strip())
            if chunk_text_hash not in seen_texts_hashes:
                all_chunks.append(c)
                seen_texts_hashes.add(chunk_text_hash)

    # Assemble context (deduplicated, sorted by relevance - lower distance is higher relevance for cosine)
    all_chunks.sort(key=lambda x: x['similarity'], reverse=False)
    context = "\n---\n".join([c['text'] for c in all_chunks[:15]])

    required_fields = _get_required_fields_from_prompt(EXTRACTION_PROMPT)
    if not context:
        print(f"No context found for {ticker}.")
        return {field: "N/A" for field in required_fields}, 0, 0

    # Extract via LLM
    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a financial analysis assistant. Extract financial data precisely. Return only valid JSON."},
                {"role": "user", "content": EXTRACTION_PROMPT.format(
                    context=context)}
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=2000
        )

        data = json.loads(response.choices[0].message.content)
        data['ticker'] = ticker  # Ensure ticker is set explicitly
        data['company'] = company_name

        tokens_used = response.usage.prompt_tokens + response.usage.completion_tokens
        # Current GPT-4o pricing (illustrative, update to actual rates if necessary):
        # Input: $5.00 / 1M tokens, Output: $15.00 / 1M tokens
        cost = (response.usage.prompt_tokens * 5.00 +
                response.usage.completion_tokens * 15.00) / 1_000_000

        return data, tokens_used, cost
    except json.JSONDecodeError as e:
        print(
            f"JSON Decode Error for {ticker}: {e}. Raw response: {response.choices[0].message.content}")
        return {field: "N/A" for field in required_fields}, 0, 0
    except Exception as e:
        print(f"An unexpected error occurred for {ticker}: {e}")
        return {field: "N/A" for field in required_fields}, 0, 0


def run_batch_extraction(portfolio_map: dict, collection: chromadb.Collection, embedder: SentenceTransformer, llm_client: OpenAI, model: str = 'gpt-4o'):
    """Orchestrates extraction for all companies in the portfolio."""
    print("\nStarting batch extraction for portfolio companies...")
    results = []
    total_cost = 0
    total_tokens = 0
    start_time = time.time()

    for ticker, name in portfolio_map.items():
        print(f"Extracting: {ticker} ({name})...", end="")
        data, tokens, cost = extract_governance_data(
            ticker, name, collection, embedder, llm_client, model=model
        )
        results.append(data)
        total_tokens += tokens
        total_cost += cost
        print(f"OK ({tokens:,} tokens, ${cost:.4f})")

    elapsed_time = time.time() - start_time
    extracted_df = pd.DataFrame(results)

    print(f"\nBatch extraction complete:")
    print(f"Companies: {len(results)}")
    print(
        f"Total time: {elapsed_time:.1f}s ({elapsed_time/max(1, len(results)):.1f}s per company)")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total cost: ${total_cost:.4f}")

    return extracted_df, total_tokens, total_cost, elapsed_time

# --- Validation Functions ---


def get_ground_truth_data() -> pd.DataFrame:
    """
    Returns a DataFrame with ground-truth data for validation.
    Adjusted 'UL' to 'IR' to match the actual data in `PORTFOLIO`.
    Added placeholder for TSLA.
    """
    return pd.DataFrame([
        # Apple (AAPL Q4 FY2025)
        {
            'ticker': 'AAPL',
            'company': 'Apple Inc.',
            'total_revenue': '$416,161 million',
            'net_income': '$112,010 million',
            'diluted_eps': '$7.46',
            'cash_and_equivalents': '$35,934 million'
        },

        # Alphabet (GOOGL Q4 2025)
        {
            'ticker': 'GOOGL',
            'company': 'Alphabet Inc.',
            'total_revenue': '$113,828 million',
            'net_income': '$34,455 million',
            'diluted_eps': '$2.82',
            'cash_and_equivalents': '$30,708 million'
        },

        # Unilever (IR Q4 2025 - adjusted from UL for consistency with downloaded files)
        {
            'ticker': 'IR',
            'company': 'Unilever Inc.',
            'total_revenue': '€50.5 billion',
            'net_income': '€9,469 million',
            'diluted_eps': '€4.32',
            'cash_and_equivalents': 'N/A'
        },
        # Tesla (TSLA Q4 2025) - No ground truth provided in original notebook, adding placeholder
        {
            'ticker': 'TSLA',
            'company': 'Tesla Inc.',
            'total_revenue': '24,901 (million USD)',
            'net_income': '840 (million USD)',
            'diluted_eps': '0.24 USD',
            'cash_and_equivalents': 'N/A'
        }
    ])


def validate_extraction(extracted_df: pd.DataFrame, truth_df: pd.DataFrame, fields_to_check: list[str]) -> tuple[pd.DataFrame, float]:
    """
    Compare extracted values against ground truth with fuzzy matching for numerical fields.
    """
    validation_results = []

    for idx, truth_row in truth_df.iterrows():
        ticker = truth_row['ticker']
        ext_row_candidates = extracted_df[extracted_df['ticker'] == ticker]

        if len(ext_row_candidates) == 0:
            validation_results.append({
                'ticker': ticker, 'field': 'ALL',
                'truth': 'N/A', 'extracted': 'N/A',
                'status': 'MISSING', 'note': 'Company not extracted'
            })
            continue

        ext_row = ext_row_candidates.iloc[0]  # Take the first matched row

        for field in fields_to_check:
            truth_val = str(truth_row.get(field, 'N/A')).strip()
            ext_val = str(ext_row.get(field, 'N/A')).strip()

            status = 'MISMATCH'
            parsed_truth, parsed_ext = None, None

            # Parse numerical fields for comparison
            if field in ['total_revenue', 'net_income', 'cash_and_equivalents']:
                parsed_truth = parse_dollars(truth_val)
                parsed_ext = parse_dollars(ext_val)
            elif field == 'diluted_eps':
                parsed_truth = parse_eps(truth_val)
                parsed_ext = parse_eps(ext_val)

            # Case 1: Both are N/A (or None equivalents)
            if ((truth_val.lower() == 'n/a' or parsed_truth is None) and
                    (ext_val.lower() == 'n/a' or parsed_ext is None)):
                status = 'BOTH_NA'
            # Case 2: Exact string match (for non-numerical fields or exact numerical string)
            elif truth_val.lower() == ext_val.lower():
                status = 'EXACT_MATCH'
            # Case 3: Fuzzy matching for numerical fields
            elif parsed_truth is not None and parsed_ext is not None:
                # Allow for a small tolerance (e.g., 5%) for numerical comparison
                if abs(parsed_truth - parsed_ext) / max(abs(parsed_truth), 1e-9) < 0.05:
                    status = 'PARTIAL_MATCH_NUMERIC'
            # Case 4: Partial string match (for qualitative fields or if numbers are embedded in text)
            # This logic needs to be careful not to override a numeric mismatch if the field is expected to be numeric
            # Only for qualitative fields
            elif field not in ['total_revenue', 'net_income', 'diluted_eps', 'cash_and_equivalents']:
                if truth_val.lower() != 'n/a' and ext_val.lower() != 'n/a' and \
                   (truth_val.lower() in ext_val.lower() or ext_val.lower() in truth_val.lower()):
                    status = 'PARTIAL_MATCH_TEXT'

            validation_results.append({
                'ticker': ticker, 'field': field,
                'truth': truth_val, 'extracted': ext_val,
                'status': status
            })

    val_df = pd.DataFrame(validation_results)

    # Summary calculations
    n_total_verifiable = len(val_df[val_df['status'] != 'BOTH_NA'])
    n_correct = len(val_df[val_df['status'].isin(
        ['EXACT_MATCH', 'PARTIAL_MATCH_NUMERIC', 'PARTIAL_MATCH_TEXT'])])

    accuracy = n_correct / max(1, n_total_verifiable)  # Avoid division by zero

    print("\nVALIDATION RESULTS")
    print("=" * 55)
    print(val_df['status'].value_counts().to_string())
    print(f"Overall accuracy (excluding 'BOTH_NA' fields): {accuracy:.1%}")

    # Show mismatches for manual review
    mismatches = val_df[val_df['status'].isin(['MISMATCH', 'MISSING'])]
    if not mismatches.empty:
        print(f"\nMISMATCHES ({len(mismatches)}):")
        for idx, row in mismatches.iterrows():
            print(
                f" {row['ticker']}.{row['field']}: truth='{row['truth']}' vs extracted='{row['extracted']}' (Status: {row['status']})")
    else:
        print("\nNo mismatches found!")

    # Visualize accuracy validation results
    plt.figure(figsize=(8, 5))
    status_counts = val_df['status'].value_counts()
    sns.barplot(x=status_counts.index, y=status_counts.values,
                palette=['skyblue', 'lightgreen', 'orange', 'salmon', 'lightcoral'])
    plt.title('Extraction Accuracy Status per Field')
    plt.xlabel('Status')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return val_df, accuracy

# --- Reporting and Visualization Functions ---


def display_financial_comparison(governance_df: pd.DataFrame, excel_path: str = 'financial_comparison.xlsx') -> pd.DataFrame:
    """
    Generates and displays a financial comparison table and exports it to Excel.
    Adds numeric parsing for calculations later.
    """
    display_cols = [
        'ticker', 'company', 'reporting_period',
        'total_revenue', 'net_income', 'diluted_eps', 'cash_and_equivalents',
        'segment_revenue_detail', 'guidance_or_outlook', 'key_risks_forward_looking'
    ]

    display_df = governance_df[[
        c for c in display_cols if c in governance_df.columns]].copy()

    display_df['total_revenue_numeric'] = display_df.get(
        'total_revenue', pd.Series(dtype='float64')).apply(parse_dollars)
    display_df['net_income_numeric'] = display_df.get(
        'net_income', pd.Series(dtype='float64')).apply(parse_dollars)
    display_df['cash_and_equivalents_numeric'] = display_df.get(
        'cash_and_equivalents', pd.Series(dtype='float64')).apply(parse_dollars)
    display_df['diluted_eps_numeric'] = display_df.get(
        'diluted_eps', pd.Series(dtype='float64')).apply(parse_eps)

    print("\nPORTFOLIO FINANCIAL COMPARISON TABLE")
    print("=" * 110)
    drop_cols_for_display = [
        'total_revenue_numeric', 'net_income_numeric',
        'cash_and_equivalents_numeric', 'diluted_eps_numeric'
    ]
    print(display_df.drop(columns=[
          c for c in drop_cols_for_display if c in display_df.columns]).to_string(index=False))

    display_df.to_excel(excel_path, index=False)
    print(f"\nExported to {excel_path}")
    return display_df  # Return the display_df with numeric columns for further analysis


def analyze_portfolio_financials(display_df: pd.DataFrame):
    """
    Analyzes and prints portfolio-level financial insights based on parsed numeric data.
    """
    print("\nPORTFOLIO FINANCIAL INSIGHTS")
    print("=" * 55)

    def print_metric_stats(series, name, unit=""):
        cleaned_series = series.dropna()
        if not cleaned_series.empty:
            print(f"{name}:")
            print(
                f"  Median: {unit}{cleaned_series.median()/1e9:.2f}B" if unit else f"  Median: {cleaned_series.median():.2f}")
            print(f"  Range: {unit}{cleaned_series.min()/1e9:.2f}B - {unit}{cleaned_series.max()/1e9:.2f}B" if unit else f"  Range: {cleaned_series.min():.2f} - {cleaned_series.max():.2f}")
        else:
            print(f"No numerical {name.lower()} data available for analysis.")

    print_metric_stats(
        display_df['total_revenue_numeric'], "Total Revenue", "$")
    print_metric_stats(display_df['net_income_numeric'], "\nNet Income", "$")
    print_metric_stats(display_df['diluted_eps_numeric'], "\nDiluted EPS")
    print_metric_stats(
        display_df['cash_and_equivalents_numeric'], "\nCash & Cash Equivalents", "$")


def screen_companies(display_df: pd.DataFrame):
    """
    Flags companies based on predefined financial thresholds.
    """
    print("\nFINANCIAL SCREENING - COMPANIES FLAGGED")
    print("=" * 55)

    flagged_companies = pd.DataFrame()

    # Threshold 1: Low cash buffer (cash < $10B)
    low_cash_threshold = 10_000_000_000
    low_cash_flags = display_df[display_df['cash_and_equivalents_numeric'].notna() &
                                (display_df['cash_and_equivalents_numeric'] < low_cash_threshold)].copy()
    if not low_cash_flags.empty:
        low_cash_flags['Flag_Reason'] = f"Cash & Equivalents < ${low_cash_threshold/1e9:.0f}B"
        flagged_companies = pd.concat([flagged_companies,
                                       low_cash_flags[['ticker', 'company', 'cash_and_equivalents', 'Flag_Reason']]])

    # Threshold 2: Negative net income (loss)
    loss_flags = display_df[display_df['net_income_numeric'].notna() &
                            (display_df['net_income_numeric'] < 0)].copy()
    if not loss_flags.empty:
        loss_flags['Flag_Reason'] = "Net Income < 0 (Loss)"
        flagged_companies = pd.concat([flagged_companies,
                                       loss_flags[['ticker', 'company', 'net_income', 'Flag_Reason']]])

    # Threshold 3: Missing key metrics (too many N/A)
    key_fields = ['total_revenue', 'net_income', 'diluted_eps']
    # Use the original string columns for N/A check
    missing_count = display_df[key_fields].apply(lambda r: sum(
        str(x).strip().lower() in ['n/a', 'na', 'none', ''] for x in r), axis=1)
    missing_flags = display_df[missing_count >= 2].copy()
    if not missing_flags.empty:
        missing_flags['Flag_Reason'] = "Missing 2+ key metrics (Revenue/NI/EPS)"
        flagged_companies = pd.concat([flagged_companies,
                                       missing_flags[['ticker', 'company', 'total_revenue', 'net_income', 'diluted_eps', 'Flag_Reason']]])

    if not flagged_companies.empty:
        flagged_companies = flagged_companies.drop_duplicates(
            subset=['ticker', 'Flag_Reason']).reset_index(drop=True)
        print("Companies requiring attention:")
        print(flagged_companies.to_string(index=False))
    else:
        print("No companies currently breach defined financial thresholds.")


def analyze_time_savings(portfolio_size: int, rag_time_seconds: float, total_cost: float):
    """
    Performs and visualizes time savings analysis.
    """
    print("\nTime Savings Analysis")
    print("=" * 50)

    manual_time_per_proxy_min = 30  # Estimated 30 minutes per proxy for manual review
    manual_time_min = portfolio_size * manual_time_per_proxy_min
    manual_time_hours = manual_time_min / 60

    rag_time_min = rag_time_seconds / 60

    print(f"Manual Review:")
    print(
        f"  Estimated Manual Time: {manual_time_min:.1f} min ({manual_time_hours:.1f} hours) for {portfolio_size} companies")

    print(f"RAG Extraction:")
    print(f"  RAG Time: {rag_time_min:.1f} min")

    savings_min = manual_time_min - rag_time_min
    savings_percent = (1 - (rag_time_min / manual_time_min)
                       ) * 100 if manual_time_min > 0 else 0

    print(f"Savings:")
    print(f"  Absolute Savings: {savings_min:.1f} min")
    print(f"  Percentage Reduction: {savings_percent:.0f}% reduction")
    print(f"  Total Cost for RAG: ${total_cost:.4f}")

    # Visualize Time Savings
    labels = ['Manual Review', 'RAG Extraction']
    times_minutes = [manual_time_min, rag_time_min]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=labels, y=times_minutes, palette=['lightcoral', 'skyblue'])
    plt.title('Time Savings: Manual Review vs. RAG Extraction (Minutes)')
    plt.xlabel('Method')
    plt.ylabel('Time (Minutes)')
    # Adjusted text position to be above the bar.
    for i, time_val in enumerate(times_minutes):
        plt.text(i, time_val, f'{time_val:.1f} min' + (f'\n(${total_cost:.4f})' if i == 1 else ''),
                 ha='center', va='bottom', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# --- Main Pipeline Function ---


def run_financial_analysis_pipeline(
    doc_dir: str = 'proxy_filings',
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    llm_model: str = 'gpt-4o',
    download_files: bool = True,
    portfolio_companies: dict | None = None,
    openai_api_key: str | None = None  # Allows explicit API key pass or env var
) -> pd.DataFrame:
    """
    Runs the complete financial analysis pipeline: download, ingest, extract, validate, and report.

    Args:
        doc_dir (str): Directory to store PDF documents.
        chunk_size (int): Size of text chunks for RAG.
        chunk_overlap (int): Overlap between text chunks.
        llm_model (str): OpenAI model to use for extraction (e.g., 'gpt-4o', 'gpt-3.5-turbo').
        download_files (bool): Whether to download PDF files if not present.
        portfolio_companies (dict | None): A dictionary of ticker: company_name. Defaults to global PORTFOLIO.
        openai_api_key (str | None): OpenAI API key. If None, it will try to read from OPENAI_API_KEY env var.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted financial data, or an empty DataFrame on failure.
    """
    if portfolio_companies is None:
        portfolio_companies = PORTFOLIO

    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    elif "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY environment variable not set. Please set it or pass it via openai_api_key argument.")
        return pd.DataFrame()  # Return empty DataFrame on failure

    client_llm = OpenAI()  # Initialize OpenAI client after ensuring API key is set

    print("--- Starting Financial Analysis Pipeline ---")

    if download_files:
        download_proxy_filings(doc_dir, portfolio_companies)

    collection, embedder = ingest_proxy_statements(
        doc_dir, portfolio_companies, chunk_size, chunk_overlap)

    if collection is None or embedder is None:
        print("Ingestion failed, cannot proceed with extraction.")
        return pd.DataFrame()

    extracted_df, total_tokens, total_cost, elapsed_time = run_batch_extraction(
        portfolio_companies, collection, embedder, client_llm, model=llm_model
    )

    if extracted_df.empty:
        print("No data extracted, stopping pipeline.")
        return pd.DataFrame()

    print("\n--- Validation ---")
    ground_truth = get_ground_truth_data()
    fields_to_check = [
        'total_revenue', 'net_income', 'diluted_eps', 'cash_and_equivalents'
    ]
    # Filter ground truth to only include companies in the current portfolio
    ground_truth_filtered = ground_truth[ground_truth['ticker'].isin(
        portfolio_companies.keys())].copy()

    val_df, accuracy = validate_extraction(
        extracted_df, ground_truth_filtered, fields_to_check)

    print("\n--- Reporting & Insights ---")
    display_df = display_financial_comparison(extracted_df)
    analyze_portfolio_financials(display_df)
    screen_companies(display_df)

    print("\n--- Performance Analysis ---")
    analyze_time_savings(len(portfolio_companies), elapsed_time, total_cost)

    print("\n--- Financial Analysis Pipeline Complete ---")
    return extracted_df


if __name__ == "__main__":
    # Example usage when run as a script
    # Ensure OPENAI_API_KEY is set in your environment variables.
    # For testing, you can uncomment and set it here, but prefer environment variables.
    # os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

    # Define the portfolio (can be passed as an argument to the main function)
    my_portfolio = {
        'AAPL': 'Apple Inc.',
        'GOOGL': 'Alphabet Inc.',
        'TSLA': 'Tesla Inc.',
        'IR': 'Unilever Inc.'
    }

    final_extracted_data = run_financial_analysis_pipeline(
        doc_dir='proxy_filings',
        chunk_size=500,
        chunk_overlap=100,
        llm_model='gpt-4o',
        download_files=True,  # Set to False if files are already downloaded
        portfolio_companies=my_portfolio,
        # openai_api_key="sk-..." # Uncomment and set if you want to pass it directly
    )

    if not final_extracted_data.empty:
        print("\nFinal Extracted Data:")
        print(final_extracted_data.to_string())
