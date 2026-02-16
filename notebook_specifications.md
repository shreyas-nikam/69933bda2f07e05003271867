
# Portfolio Governance Screening & Comparative Analysis with RAG

**Persona:** Alex Chen, CFA, Portfolio Manager at Apex Capital Management
**Organization:** Apex Capital Management
**Case Study:** Automating the extraction and comparative analysis of corporate governance data from proxy statements to inform investment and proxy voting decisions.

## Introduction: The Challenge of Manual Governance Review

Alex Chen, a CFA Charterholder and Portfolio Manager at Apex Capital Management, faces a significant challenge. His firm manages a diverse portfolio of publicly traded companies, and a critical part of his fiduciary duty involves monitoring the corporate governance practices of these holdings. Each year, he needs to review dozens of complex proxy statements (DEF 14A filings), which can be 50-150 pages long. This manual review is incredibly time-consuming, making it nearly impossible to conduct a comprehensive, consistent, and timely comparative analysis across the entire portfolio.

Alex needs a way to rapidly identify potential governance "red flags"—such as excessive CEO compensation, low board independence, or the absence of ESG metrics in executive incentive plans—and consolidate this information into an actionable format for investment committee discussions and proxy voting decisions. This notebook demonstrates how Alex leverages Retrieval-Augmented Generation (RAG) to transform hours of manual document review into minutes of AI-assisted, data-driven analysis, enabling him to fulfill his responsibilities more effectively and proactively.

---

## 0. Setup: Installing Libraries and Importing Dependencies

Alex begins by setting up his environment, installing the necessary Python libraries for PDF processing, vector storage, natural language processing, and data manipulation.

```python
!pip install pymupdf chromadb sentence-transformers openai pandas matplotlib seaborn
```

```python
import fitz # PyMuPDF for PDF parsing
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import json
import pandas as pd
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Configure OpenAI API key from environment variable
# It's good practice to load API keys securely.
# For demonstration purposes, we assume OPENAI_API_KEY is set in your environment.
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY" # Uncomment and replace if not set
```

---

## 1. Setting the Foundation: Ingesting Proxy Statements for Portfolio-Wide Analysis

### Story + Context + Real-World Relevance

Alex’s first step is to get the raw data—the proxy statements—into a format that the RAG system can process. Manually reading each PDF is tedious and error-prone. Instead, he will automate the extraction of text from these documents and store them in a searchable database. This process is crucial for multi-document RAG, as it creates an indexed, machine-readable repository of his portfolio's governance information. Each document is broken down into smaller, meaningful chunks, which are then converted into numerical representations (embeddings) for efficient similarity search. Critically, each chunk is tagged with metadata like the company ticker, ensuring that information from one company's proxy statement doesn't accidentally get mixed with another's during retrieval. This metadata-rich indexing is fundamental to scaling RAG from single-document Q&A to a comprehensive multi-company analysis.

```python
# Define the portfolio companies and their full names
PORTFOLIO = {
    'AAPL': 'Apple Inc.',
    'AMZN': 'Amazon.com Inc.',
    'MSFT': 'Microsoft Corp.',
    'JPM': 'JPMorgan Chase & Co.',
    'JNJ': 'Johnson & Johnson',
}

def ingest_proxy_statements(doc_dir='proxy_filings',
                            chunk_size=500, chunk_overlap=100):
    """
    Ingest multiple proxy PDFs into a single vector store with company metadata
    for filtered retrieval.
    """
    print("Initializing embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model loaded.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ""] # More robust separators
    )

    client = chromadb.Client()
    
    # Attempt to get collection, create if it doesn't exist.
    try:
        collection = client.get_or_create_collection(
            name="portfolio_proxies",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Using existing collection 'portfolio_proxies'. Current chunks: {collection.count()}")
        # Optionally, clear existing collection if a fresh ingest is always desired
        # client.delete_collection(name="portfolio_proxies")
        # collection = client.create_collection(name="portfolio_proxies", metadata={"hnsw:space": "cosine"})
        # print("Cleared existing collection and created new one.")

    except Exception as e:
        print(f"Error getting/creating collection: {e}. Creating new one.")
        collection = client.create_collection(
            name="portfolio_proxies",
            metadata={"hnsw:space": "cosine"}
        )
        print("Created new collection 'portfolio_proxies'.")


    all_chunks = []
    
    # Ensure the directory exists
    os.makedirs(doc_dir, exist_ok=True)

    print(f"Ingesting proxy statements from '{doc_dir}'...")
    for ticker, name in PORTFOLIO.items():
        filepath = os.path.join(doc_dir, f'{ticker}_proxy_2024.pdf')
        
        if not os.path.exists(filepath):
            print(f"WARNING: File not found for {ticker} at {filepath}. Skipping.")
            continue

        print(f"Processing {ticker} ({name})...")
        doc = fitz.open(filepath)
        text = "".join([page.get_text() for page in doc])
        doc.close()

        chunks = splitter.split_text(text)
        
        # Add a placeholder for fiscal_year if not extracted dynamically
        fiscal_year = '2024' # Assuming 2024 proxies for this lab

        for i, chunk in enumerate(chunks):
            chunk_id = f"{ticker}_chunk_{i}"
            all_chunks.append({
                'id': chunk_id,
                'text': chunk,
                'metadata': {
                    'ticker': ticker,
                    'company': name,
                    'chunk_id': i,
                    'doc_type': 'DEF14A',
                    'fiscal_year': fiscal_year,
                }
            })
        print(f"  {ticker}: {len(chunks)} chunks from {len(text.split()):,} words")

    if not all_chunks:
        print("No chunks were processed. Ensure PDF files exist in the 'proxy_filings' directory.")
        return None, None # Return None if no chunks

    # Batch embed and insert
    texts = [c['text'] for c in all_chunks]
    
    # Handle cases where all_chunks might be empty if no files were found
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
    
    print(f"\nTotal: {collection.count()} chunks across {len(PORTFOLIO)} companies")
    return collection, embedder

# Execute the ingestion process
collection, embedder = ingest_proxy_statements()
```

### Explanation of Execution

The code above initializes the embedding model and text splitter, then iterates through each proxy statement in Alex's portfolio. For each PDF, it extracts the text, splits it into smaller chunks, and creates a unique ID and relevant metadata (like `ticker` and `company_name`) for each chunk. These chunks are then embedded into a vector space and added to the `ChromaDB` collection.

This process transforms unstructured PDF documents into a structured, searchable knowledge base. Alex can now query this database efficiently, knowing that each piece of information is indexed with its source company. The `collection.count()` output confirms how many total text chunks are available for retrieval across all portfolio companies, signaling readiness for detailed analysis.

---

## 2. Defining Our Governance Lens: Structured Data Extraction Schema

### Story + Context + Real-World Relevance

Alex isn't just looking for random facts; he needs specific, comparable governance data points to perform his analysis. He requires a structured output—a predefined set of fields for each company, ensuring consistency. For example, he needs 'CEO total compensation' in a numerical format and 'ESG in compensation' as a 'Yes/No'. This structured approach prevents the LLM from hallucinating or providing irrelevant information, allowing Alex to build a reliable comparative table. The instruction to use "N/A" for missing information is also critical, as the absence of a data point (e.g., ESG metrics in compensation) can itself be an important governance signal for Alex.

The extraction schema *is* Alex's data model. The fields he defines here directly dictate the metrics he can use for comparative analysis, governance screening, and ultimately, his investment decisions. This design is a financial decision, driven by what metrics matter for Apex Capital’s investment thesis and fiduciary obligations.

```python
# Initialize OpenAI client
client_llm = OpenAI()

# Extraction schema (adapted from CFA Automation Ahead report)
# This prompt guides the LLM to extract specific data points in a JSON format.
EXTRACTION_PROMPT = """You are a financial analysis assistant specializing
in extracting detailed executive compensation and governance data from
corporate proxy statements.

Extract the following fields from the provided proxy statement excerpts
and output the information as a JSON object. For not applicable or
missing information, use "N/A".

Required fields:
{{
"company": "Name of the company",
"ticker": "Stock ticker",
"fiscal_year": "Fiscal period covered",
"ceo_name": "Name of CEO",
"ceo_total_compensation": "Total compensation amount (e.g., $15.3 million)",
"ceo_base_salary": "Base salary amount",
"ceo_stock_awards": "Value of stock/equity awards",
"ceo_cash_bonus": "Annual cash incentive/bonus",
"board_size": "Total number of directors",
"independent_directors": "Number of independent directors",
"board_independence_pct": "Percentage independent (computed)",
"esg_in_compensation": "Yes/No - Are ESG metrics included in incentive plan?",
"esg_metrics_detail": "Specific ESG metrics if applicable (e.g., emissions reduction, DEI targets)",
"equity_vesting_schedule": "Details on time-based vesting",
"performance_vesting_detail": "Performance metrics and vesting conditions",
"say_on_pay_approval": "Percentage of shareholder say-on-pay approval",
"compensation_committee_independent": "Yes/No",
"clawback_policy": "Yes/No - Does the company have a clawback policy?"
}}

IMPORTANT: Extract ONLY from the provided text. Do NOT use outside knowledge.
If a field cannot be determined from the text, use "N/A".

PROXY STATEMENT EXCERPTS:
{context}"""

print("Defined the structured extraction prompt and initialized LLM client.")
```

### Explanation of Execution

The `EXTRACTION_PROMPT` defines the precise JSON schema that Alex requires for his governance data. It lists 18 fields, covering quantitative details (like CEO compensation figures and board size) and qualitative information (such as ESG metric details and clawback policies). This prompt is critical for two reasons:
1.  **Consistency:** It ensures that the LLM extracts the *same* data points in the *same* format for every company, making direct comparisons possible.
2.  **Handling Missing Data:** The explicit instruction to use "N/A" for unavailable information is a vital governance signal. If a company's proxy statement doesn't mention ESG metrics in compensation, "N/A" indicates this lack of disclosure, which is a key insight for Alex, rather than a system failure. This output informs Alex's real-world decision-making about potential governance weaknesses or disclosure gaps.

---

## 3. Precision in Retrieval: Ensuring Company-Specific Insights

### Story + Context + Real-World Relevance

A critical challenge in multi-document RAG is preventing information leakage or "mix-up" across companies. If Alex queries for "CEO total compensation," he must ensure that the retrieved chunks of text pertain *only* to the specific company he is analyzing (e.g., Apple) and not inadvertently include information from another company (e.g., Amazon). A mix-up could lead to catastrophic errors in his analysis, attributing Amazon's CEO pay to Apple, for instance. This company-filtered retrieval mechanism is a fundamental safeguard, guaranteeing the integrity and accuracy of Alex's governance data.

The underlying mathematical concept for similarity search in vector databases often involves distance metrics like cosine similarity. When embedding a query $q$ and comparing it to document chunks $d_i$, a similarity score $S(q, d_i)$ is computed. However, for company-filtered retrieval, Alex applies a categorical filter. This is an essential non-semantic filtering step that precedes or runs in conjunction with semantic search. The `where` clause acts as a hard constraint:

$$
\text{Retrieve}(q, \text{ticker}) = \{d_i \mid d_i \in \text{Collection} \land \text{metadata}(d_i).\text{ticker} = \text{ticker} \land S(q, d_i) \text{ is high}\}
$$

This ensures that only chunks whose metadata explicitly match the target `ticker` are considered, eliminating cross-company contamination.

```python
def retrieve_for_company(query, collection, embedder, ticker, k=8):
    """Retrieve relevant chunks ONLY from a specific company's proxy."""
    query_embedding = embedder.encode([query]).tolist()

    # CRITICAL: filter by company using the 'where' clause in ChromaDB
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

# Test: retrieve compensation info for Apple only
test_query = "CEO total compensation salary bonus stock awards"
test_ticker = 'AAPL'
test_chunks = retrieve_for_company(test_query, collection, embedder, ticker=test_ticker, k=5)

print(f"Retrieved {len(test_chunks)} chunks for {test_ticker}.")
if test_chunks:
    for i, c in enumerate(test_chunks[:3]):
        print(f"[{c['similarity']:.3f}] (Ticker: {c['metadata'].get('ticker', 'N/A')}) {c['text'][:80]}...")
else:
    print(f"No chunks retrieved for {test_ticker}. Please ensure '{test_ticker}_proxy_2024.pdf' exists and was ingested.")

```

### Explanation of Execution

The `retrieve_for_company` function demonstrates the crucial company-filtered retrieval. When Alex searches for information, the `where={"ticker": ticker}` clause acts as a strict filter, ensuring that the RAG system only considers text chunks that originate from the specified company's proxy statement. The output of the test query for 'AAPL' explicitly shows the `ticker` in the metadata for each retrieved chunk, confirming that the filter is working as intended.

This safeguard is paramount for Alex. Without it, the LLM might receive mixed signals from various company documents, leading to inaccurate and misleading governance data—a "silent, catastrophic error" that would undermine Apex Capital's analysis and decision-making process.

---

## 4. The Automated Analyst: Batch Extraction and Data Consolidation

### Story + Context + Real-World Relevance

With the schema defined and retrieval secured, Alex can now unleash the full power of the RAG pipeline: batch extraction across all portfolio companies. Instead of manually reviewing each proxy, he'll run an automated loop that extracts the structured governance data for every company. To ensure comprehensive coverage, he employs a **multi-query retrieval strategy**. A single query might miss information found in different sections of a proxy statement. By using several targeted queries (e.g., for compensation, board composition, ESG metrics, vesting, and say-on-pay), he can union the retrieved chunks to build a more complete context for the LLM. This not only makes the process significantly faster but also more consistent and thorough than any manual review.

The multi-query retrieval strategy for a given company is formulated as the union of top-$k$ relevant chunks from multiple queries $q \in Q$:

$$
R_{company} = \bigcup_{q \in Q} \text{top-k}(q, \text{ticker})
$$

where $Q = \{q_{comp}, q_{board}, q_{esg}, q_{vesting}, q_{vote}\}$ represents the set of targeted queries covering different governance aspects. This approach ensures that the LLM receives a comprehensive context from which to extract all required fields, even if they are scattered across the document.

```python
def parse_dollars(val):
    """
    Extract numeric value from strings like '$63.2 million'.
    Handles "N/A" or NaN values by returning None.
    """
    if val is None or val == "N/A" or pd.isna(val) or not isinstance(val, str):
        return None
    
    val = val.strip()
    match = re.search(r'[\$€£]?([\d,.]+)\s*(million|billion|M|B)?', val, re.IGNORECASE)
    if match:
        num_str = match.group(1).replace(',', '')
        num = float(num_str)
        unit = match.group(2)
        if unit:
            unit = unit.lower()
            if unit in ['million', 'm']:
                return num * 1e6
            elif unit in ['billion', 'b']:
                return num * 1e9
        return num # Return as is if no unit or unit not matched
    return None # Return None if no match found after trying to parse

def extract_governance_data(ticker, company_name, collection, embedder, model='gpt-4o'):
    """Extract structured governance data for one company via RAG."""
    
    # Retrieve relevant chunks using multiple queries for broad coverage
    queries = [
        "CEO total compensation salary bonus stock awards equity",
        "board of directors independent directors composition",
        "ESG sustainability metrics executive compensation incentive",
        "equity vesting schedule performance-based awards clawback",
        "say-on-pay shareholder vote approval compensation committee"
    ]
    
    all_chunks = []
    seen_texts = set()

    for query in queries:
        chunks = retrieve_for_company(query, collection, embedder, ticker=ticker, k=5)
        for c in chunks:
            # Deduplicate chunks based on text prefix to avoid redundancy
            if c['text'][:100] not in seen_texts:
                all_chunks.append(c)
                seen_texts.add(c['text'][:100])
    
    # Assemble context (deduplicated, sorted by relevance)
    # Sort by similarity to prioritize most relevant chunks
    all_chunks.sort(key=lambda x: x['similarity'], reverse=True)
    # Join up to 15 most relevant chunks to form the context for the LLM
    context = "\n\n---\n\n".join([c['text'] for c in all_chunks[:15]])

    # If no context is found, return N/A for all fields
    if not context:
        print(f"No context found for {ticker}.")
        return {field: "N/A" for field in json.loads(EXTRACTION_PROMPT.split('Required fields:\n')[1].split('}}')[0] + '}}').keys()}, 0, 0
    
    # Extract via LLM
    try:
        response = client_llm.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Extract governance data precisely. Return only valid JSON."},
                {"role": "user", "content": EXTRACTION_PROMPT.format(context=context)}
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=2000
        )

        data = json.loads(response.choices[0].message.content)
        data['ticker'] = ticker # Ensure ticker is set explicitly
        data['company'] = company_name

        tokens_used = response.usage.prompt_tokens + response.usage.completion_tokens
        # Assuming OpenAI pricing (e.g., $0.005/1K prompt tokens, $0.015/1K completion tokens for GPT-4o)
        # These rates are illustrative and should be updated to current OpenAI pricing
        cost = (response.usage.prompt_tokens * 0.005 + response.usage.completion_tokens * 0.015) / 1000
        
        return data, tokens_used, cost
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error for {ticker}: {e}. Raw response: {response.choices[0].message.content}")
        # Return N/A for all fields if JSON parsing fails
        return {field: "N/A" for field in json.loads(EXTRACTION_PROMPT.split('Required fields:\n')[1].split('}}')[0] + '}}').keys()}, 0, 0
    except Exception as e:
        print(f"An unexpected error occurred for {ticker}: {e}")
        return {field: "N/A" for field in json.loads(EXTRACTION_PROMPT.split('Required fields:\n')[1].split('}}')[0] + '}}').keys()}, 0, 0


# --- BATCH EXTRACTION LOOP ---
print("Starting batch extraction for portfolio companies...")
results = []
total_cost = 0
total_tokens = 0
start_time = time.time()

for ticker, name in PORTFOLIO.items():
    print(f"Extracting: {ticker} ({name})...", end="")
    data, tokens, cost = extract_governance_data(
        ticker, name, collection, embedder, model='gpt-4o' # Using GPT-4o model
    )
    results.append(data)
    total_tokens += tokens
    total_cost += cost
    print(f"OK ({tokens:,} tokens, ${cost:.4f})")

elapsed = time.time() - start_time

# Compile into DataFrame
governance_df = pd.DataFrame(results)

print(f"\nBatch extraction complete:")
print(f"Companies: {len(results)}")
print(f"Total time: {elapsed:.1f}s ({elapsed/len(results):.1f}s per company)")
print(f"Total tokens: {total_tokens:,}")
print(f"Total cost: ${total_cost:.4f}")

# Display the first few rows of the extracted data
print("\nExtracted Governance Data (First 5 Rows):")
print(governance_df.head().to_string())
```

### Explanation of Execution

This section automates the data extraction for Alex's entire portfolio. The `extract_governance_data` function uses multiple, targeted queries to retrieve relevant chunks for each company, ensuring that the LLM has a comprehensive context. These chunks are then fed to the `gpt-4o` model, which extracts data according to the predefined JSON schema.

The output confirms the efficiency of this batch process: it shows the time taken and the tokens and cost incurred per company and for the entire portfolio. Alex can see at a glance that processing multiple proxies now takes minutes, not hours. The resulting `governance_df` DataFrame is the raw, structured data Alex needs to start his comparative analysis, a significant step forward from manual review. The `parse_dollars` function is crucial for converting textual compensation figures (e.g., "$15.3 million") into numerical values, enabling quantitative analysis and aggregation.

---

## 5. Trust, But Verify: Validating Extraction Accuracy

### Story + Context + Real-World Relevance

As a CFA Charterholder, Alex understands that even advanced AI systems like RAG are not infallible. Before presenting any findings to Apex Capital's investment committee, he must validate the accuracy of the extracted data against manually verified ground truth for a subset of the portfolio. This step is crucial for maintaining data integrity and building trust in the AI-assisted workflow. Without validation, potential errors (e.g., misparsed numbers or incorrect qualitative assessments) could lead to flawed investment decisions or misinformed proxy votes. Alex needs specific metrics to quantify this accuracy, including fuzzy matching for numerical values to account for slight variations in how they might be represented.

Alex uses three key metrics for validation:
1.  **Field-level accuracy:** This measures the correctness of individual data points.
    $$Accuracy_{field} = \frac{|\text{exact match}| + |\text{partial match}|}{|\text{total non-N/A fields}|}$$
    Here, "partial match" accounts for numerical values that are logically equivalent but may differ in formatting (e.g., "$15.3M" vs. "$15,300,000").

2.  **Company-level completeness:** This assesses how many required fields were successfully extracted for each company (not marked "N/A").
    $$Completeness_i = \frac{|\text{non-N/A fields extracted for company i}|}{|\text{total schema fields}|}$$

3.  **Portfolio-level accuracy:** This provides an aggregated view of accuracy across all companies and fields.
    $$Accuracy_{portfolio} = \frac{\sum_i \sum_j [\text{correct}_{ij}]}{\sum_i \sum_j [\text{verifiable}_{ij}]}$$
    Where $[\text{correct}_{ij}]$ is 1 if field $j$ for company $i$ is correct (exact or partial match) and verifiable, and $[\text{verifiable}_{ij}]$ is 1 if the ground truth for that field is not N/A.

```python
# Manually verified ground-truth (for a subset of companies and fields)
# This would typically be created by a human analyst.
ground_truth = pd.DataFrame([
    {'ticker': 'AAPL', 'company': 'Apple Inc.', 'ceo_total_compensation': '$63.2 million', 'board_size': '8', 'independent_directors': '7', 'esg_in_compensation': 'No', 'clawback_policy': 'Yes'},
    {'ticker': 'MSFT', 'company': 'Microsoft Corp.', 'ceo_total_compensation': '$48.5 million', 'board_size': '12', 'independent_directors': '11', 'esg_in_compensation': 'Yes', 'clawback_policy': 'Yes'},
    {'ticker': 'AMZN', 'company': 'Amazon.com Inc.', 'ceo_total_compensation': '$29.2 million', 'board_size': '10', 'independent_directors': '9', 'esg_in_compensation': 'No', 'clawback_policy': 'Yes'},
    {'ticker': 'JPM', 'company': 'JPMorgan Chase & Co.', 'ceo_total_compensation': '$36.0 million', 'board_size': '16', 'independent_directors': '14', 'esg_in_compensation': 'Yes', 'clawback_policy': 'Yes'},
    {'ticker': 'JNJ', 'company': 'Johnson & Johnson', 'ceo_total_compensation': '$28.4 million', 'board_size': '12', 'independent_directors': '10', 'esg_in_compensation': 'Yes', 'clawback_policy': 'Yes'},
])

# Fields to check for accuracy validation
fields_to_check = [
    'ceo_total_compensation',
    'board_size',
    'independent_directors',
    'esg_in_compensation',
    'clawback_policy'
]

def validate_extraction(extracted_df, truth_df, fields_to_check):
    """Compare extracted values against ground truth with fuzzy matching for numerical fields."""
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
        
        # Take the first matched row if multiple exist
        ext_row = ext_row_candidates.iloc[0]

        for field in fields_to_check:
            truth_val = str(truth_row.get(field, 'N/A')).strip()
            ext_val = str(ext_row.get(field, 'N/A')).strip()
            
            status = 'MISMATCH'

            # Case 1: Both are N/A (or None equivalents)
            if (truth_val.lower() == 'n/a' or pd.isna(truth_val)) and \
               (ext_val.lower() == 'n/a' or pd.isna(ext_val)):
                status = 'BOTH_NA'
            # Case 2: Exact string match
            elif truth_val.lower() == ext_val.lower():
                status = 'EXACT_MATCH'
            # Case 3: Fuzzy matching for numerical fields (after dollar parsing)
            elif field in ['ceo_total_compensation', 'board_size', 'independent_directors']:
                parsed_truth = parse_dollars(truth_val) if field == 'ceo_total_compensation' else pd.to_numeric(truth_val, errors='coerce')
                parsed_ext = parse_dollars(ext_val) if field == 'ceo_total_compensation' else pd.to_numeric(ext_val, errors='coerce')

                if parsed_truth is not None and parsed_ext is not None:
                    # Allow for a small tolerance for numerical comparison
                    if abs(parsed_truth - parsed_ext) / max(abs(parsed_truth), 1e-6) < 0.05: # 5% tolerance
                        status = 'PARTIAL_MATCH_NUMERIC'
                elif (parsed_truth is None and (truth_val.lower() == 'n/a' or pd.isna(truth_val))) and \
                     (parsed_ext is None and (ext_val.lower() == 'n/a' or pd.isna(ext_val))):
                     status = 'BOTH_NA' # Re-check if both are essentially N/A after parsing

            # Case 4: Partial string match (e.g., "Yes" in "Yes, includes ESG") - for qualitative fields
            elif field in ['esg_in_compensation', 'clawback_policy']:
                if truth_val.lower() != 'n/a' and ext_val.lower() != 'n/a' and \
                   truth_val.lower() in ext_val.lower() or ext_val.lower() in truth_val.lower():
                    status = 'PARTIAL_MATCH_TEXT'
            
            validation_results.append({
                'ticker': ticker, 'field': field,
                'truth': truth_val, 'extracted': ext_val,
                'status': status
            })

    val_df = pd.DataFrame(validation_results)

    # Summary calculations
    n_total_verifiable = len(val_df[val_df['status'] != 'BOTH_NA'])
    n_correct = len(val_df[val_df['status'].isin(['EXACT_MATCH', 'PARTIAL_MATCH_NUMERIC', 'PARTIAL_MATCH_TEXT'])])
    
    accuracy = n_correct / max(1, n_total_verifiable) # Avoid division by zero

    print("VALIDATION RESULTS")
    print("=" * 55)
    print(val_df['status'].value_counts().to_string())
    print(f"\nOverall accuracy (excluding 'BOTH_NA' fields): {accuracy:.1%}")

    # Show mismatches for manual review
    mismatches = val_df[val_df['status'].isin(['MISMATCH', 'MISSING'])]
    if len(mismatches) > 0:
        print(f"\nMISMATCHES ({len(mismatches)}):")
        for idx, row in mismatches.iterrows():
            print(f" {row['ticker']}.{row['field']}: truth='{row['truth']}' vs extracted='{row['extracted']}' (Status: {row['status']})")
    else:
        print("\nNo mismatches found!")

    return val_df, accuracy

val_df, accuracy = validate_extraction(governance_df, ground_truth, fields_to_check)

# Visualize accuracy validation results
status_counts = val_df['status'].value_counts()
plt.figure(figsize=(8, 5))
status_counts.plot(kind='bar', color=['skyblue', 'lightgreen', 'orange', 'salmon', 'lightcoral'])
plt.title('Extraction Accuracy Status per Field')
plt.xlabel('Status')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

```

### Explanation of Execution

The `validate_extraction` function compares the RAG-extracted data (`governance_df`) against a small, manually verified `ground_truth` dataset. It performs checks for exact matches, fuzzy matches for numerical values (e.g., "$15.3 million" vs. "15,300,000"), and partial matches for qualitative fields. A critical aspect is how it handles "N/A" values: if both ground truth and extraction return "N/A," it's considered `BOTH_NA` and correctly excluded from the accuracy calculation, as Alex recognizes this as informative rather than an extraction error.

The output provides a detailed breakdown of `EXACT_MATCH`, `PARTIAL_MATCH`, `BOTH_NA`, and `MISMATCH` statuses, along with an overall accuracy score. Alex can quickly see where the RAG system performs well and where manual review might still be needed. The bar chart provides a clear visual summary, empowering Alex to make informed decisions about the trustworthiness of the extracted data for Apex Capital's critical governance analysis.

---

## 6. Actionable Insights: Comparative Table and Portfolio-Level Metrics

### Story + Context + Real-World Relevance

With validated and structured data in hand, Alex's primary objective is to generate actionable insights. The core deliverable for Apex Capital's investment committee is a clean, comparative table summarizing key governance metrics across all portfolio companies. This table allows Alex to quickly identify outliers or companies that deviate from Apex Capital's governance standards. Furthermore, calculating portfolio-level aggregated statistics (e.g., median CEO compensation, ESG incentive adoption rate) provides a high-level overview, useful for internal reporting and client communications on Apex Capital’s overall governance footprint. This transformation of raw data into an analyst-ready report is the "last mile" that makes the RAG pipeline a truly valuable business workflow.

Alex uses these insights to:
*   **Screen for Red Flags:** Flag companies that breach pre-defined governance thresholds (e.g., CEO compensation exceeding a certain percentile, board independence below a target).
*   **Inform Engagement:** Identify companies requiring focused engagement on governance issues.
*   **Support Proxy Voting:** Make data-driven proxy voting decisions.

```python
# Select key columns for the governance comparison table display
display_cols = [
    'ticker', 'company', 'ceo_name', 'ceo_total_compensation',
    'board_size', 'independent_directors', 'board_independence_pct',
    'esg_in_compensation', 'say_on_pay_approval', 'clawback_policy'
]

# Clean and format the display DataFrame
display_df = governance_df[[c for c in display_cols if c in governance_df.columns]].copy()

# Convert numerical fields to numeric type and handle "N/A"
display_df['ceo_total_compensation_numeric'] = display_df['ceo_total_compensation'].apply(parse_dollars)
display_df['board_size_numeric'] = pd.to_numeric(display_df['board_size'], errors='coerce')
display_df['independent_directors_numeric'] = pd.to_numeric(display_df['independent_directors'], errors='coerce')

# Calculate board_independence_pct if not already computed or if N/A
# Formula: (Independent Directors / Board Size) * 100
display_df['board_independence_pct_computed'] = (
    display_df['independent_directors_numeric'] / display_df['board_size_numeric'] * 100
).fillna('N/A').apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x)


# Overwrite 'board_independence_pct' if the extracted one is 'N/A' or if we want to use our computed one consistently
# Prioritize the RAG extracted value, but use computed if RAG failed or is N/A
display_df['board_independence_pct'] = display_df.apply(
    lambda row: row['board_independence_pct_computed'] if row['board_independence_pct'] == 'N/A' else row['board_independence_pct'],
    axis=1
)


print("PORTFOLIO GOVERNANCE COMPARISON TABLE")
print("=" * 80)
# Use to_string to ensure all rows/columns are displayed without truncation
print(display_df.drop(columns=['ceo_total_compensation_numeric', 'board_size_numeric', 'independent_directors_numeric', 'board_independence_pct_computed']).to_string(index=False))

# Export to Excel for IC presentation
excel_path = 'governance_comparison.xlsx'
display_df.to_excel(excel_path, index=False)
print(f"\nExported to {excel_path}")


# --- Portfolio-level governance metrics and insights ---
print("\nPORTFOLIO GOVERNANCE INSIGHTS")
print("=" * 50)

# CEO Compensation Analysis
ceo_comps_numeric = display_df['ceo_total_compensation_numeric'].dropna()
if len(ceo_comps_numeric) > 0:
    print(f"CEO Compensation:")
    print(f"  Median: ${ceo_comps_numeric.median()/1e6:.1f}M")
    print(f"  Range: ${ceo_comps_numeric.min()/1e6:.1f}M - ${ceo_comps_numeric.max()/1e6:.1f}M")
    
    # Visualize CEO Compensation
    plt.figure(figsize=(10, 6))
    sns.barplot(x=ceo_comps_numeric.values / 1e6, y=display_df.loc[ceo_comps_numeric.index, 'ticker'], palette='viridis')
    plt.title('CEO Total Compensation Across Portfolio (in $M)')
    plt.xlabel('CEO Total Compensation ($M)')
    plt.ylabel('Company Ticker')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
else:
    print("No numerical CEO compensation data available for analysis.")

# ESG in Compensation Analysis
esg_rate = display_df['esg_in_compensation'].dropna()
if len(esg_rate) > 0:
    esg_yes_count = (esg_rate.str.lower() == 'yes').sum()
    esg_percent = (esg_yes_count / len(esg_rate)) * 100
    print(f"\nESG in Compensation:")
    print(f"  {esg_yes_count}/{len(esg_rate)} companies ({esg_percent:.0f}%) include ESG metrics in executive compensation.")
else:
    print("\nNo ESG in Compensation data available for analysis.")

# Clawback Policy Analysis
clawback_policy = display_df['clawback_policy'].dropna()
if len(clawback_policy) > 0:
    cb_yes_count = (clawback_policy.str.lower() == 'yes').sum()
    cb_percent = (cb_yes_count / len(clawback_policy)) * 100
    print(f"\nClawback Policy:")
    print(f"  {cb_yes_count}/{len(clawback_policy)} companies ({cb_percent:.0f}%) have a clawback policy.")
else:
    print("\nNo Clawback Policy data available for analysis.")

# Flagging companies based on governance thresholds (hypothetical)
print("\nGOVERNANCE SCREENING - COMPANIES FLAGGED")
print("=" * 50)
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
    # Remove duplicate flags for the same company and reason if concat created them
    flagged_companies = flagged_companies.drop_duplicates(subset=['ticker', 'Flag_Reason']).reset_index(drop=True)
    print("Companies requiring attention:")
    print(flagged_companies.to_string(index=False))
else:
    print("No companies currently breach defined governance thresholds.")
```

### Explanation of Execution

This section is where the extracted data truly becomes valuable for Alex. The RAG pipeline generates a `PORTFOLIO GOVERNANCE COMPARISON TABLE`, a clear and concise summary of key metrics for each company. This table, which Alex can easily export to Excel, is ready for presentation to Apex Capital's investment committee, enabling data-driven discussions on corporate governance.

Beyond the table, the notebook calculates and displays **portfolio-level aggregated governance insights**, such as the median CEO compensation, the percentage of companies including ESG metrics in executive pay, and the prevalence of clawback policies. These high-level statistics provide Alex with a quick overview of the portfolio's governance landscape.

Finally, Alex applies **custom governance thresholds** to automatically flag companies that exhibit "red flags." This immediate identification of companies with, for example, high CEO compensation or low board independence, allows Alex to prioritize his engagement efforts and inform targeted proxy voting decisions, directly addressing his real-world workflow challenges. The bar chart for CEO compensation provides a quick visual comparison of pay across the portfolio.

---

## 7. The ROI: Quantifying Time and Cost Savings

### Story + Context + Real-World Relevance

Alex understands that demonstrating the return on investment (ROI) of new technology is crucial for gaining continued support for AI initiatives at Apex Capital Management. He needs to quantify the tangible benefits of this RAG-powered governance analysis. By comparing the estimated time and cost of manual review against the actual figures from the RAG pipeline, Alex can clearly articulate the efficiency gains. This analysis provides concrete evidence of how AI transforms a time-intensive, low-leverage task into a rapid, high-value activity, freeing up his time for more strategic financial analysis and direct engagement with portfolio companies.

The time savings calculation highlights the sheer scale of efficiency:
*   **Manual Review Time:** $T_{manual} = N_{companies} \times T_{manual\_per\_proxy}$
*   **RAG Extraction Time:** $T_{RAG} = T_{elapsed}$ (from Section 4)

The percentage reduction in time is then:
$$
\text{Time Reduction (\%)} = \left(1 - \frac{T_{RAG}}{T_{manual}}\right) \times 100
$$

```python
# --- Time savings analysis ---
print("\nTime Savings Analysis")
print("=" * 50)

n_companies = len(PORTFOLIO)
manual_time_per_proxy_min = 30 # Estimated 30 minutes per proxy for manual review
manual_time_min = n_companies * manual_time_per_proxy_min
manual_time_hours = manual_time_min / 60

# elapsed is from the batch extraction loop in Section 4 (in seconds)
rag_time_min = elapsed / 60

print(f"Manual Review:")
print(f"  Estimated Manual Time: {manual_time_min:.1f} min ({manual_time_hours:.1f} hours) for {n_companies} companies")

print(f"RAG Extraction:")
print(f"  RAG Time: {rag_time_min:.1f} min")

savings_min = manual_time_min - rag_time_min
savings_percent = (1 - (rag_time_min / manual_time_min)) * 100 if manual_time_min > 0 else 0

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
plt.text(0, times_minutes[0], f'{times_minutes[0]:.1f} min', ha='center', va='bottom', fontsize=10)
plt.text(1, times_minutes[1], f'{times_minutes[1]:.1f} min\n(${total_cost:.4f})', ha='center', va='bottom', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

### Explanation of Execution

This final section quantifies the operational benefits of the RAG pipeline. By comparing the estimated time for manual review (30 minutes per proxy, as per the CFA Institute's benchmark) against the actual RAG processing time, Alex can clearly demonstrate significant **time and cost savings**. The output shows the estimated hours saved and the substantial percentage reduction in time.

The bar chart visually reinforces this impact, providing a compelling narrative for Apex Capital's management. This concrete ROI data enables Alex to justify the adoption of AI tools, highlighting how they enhance efficiency, reduce operational costs, and ultimately allow portfolio managers to focus on higher-value tasks, thereby fulfilling their fiduciary duties more effectively.
