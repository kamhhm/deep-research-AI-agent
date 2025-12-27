"""
GenAI Adoption Classification (Description-Based)
==================================================
Author: Research Team
Date: December 2025
Purpose: Classify startups using Crunchbase descriptions (Short + Long) + Categories
         to detect GenAI adoption without web search.

Usage:
    python3 batch_processor_w14.py

Input:
    - Data: data/44k_crunchbase_startups.csv
    - Prompt: prompts/Jan_draft_prompt.txt

Output:
    - JSONL batch files for OpenAI API
    - Final CSV with flattened JSON results
"""

import os
import csv
import json
import time
import pandas as pd
import re
from datetime import datetime
from collections import Counter
from openai import OpenAI

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
BATCH_DIR = os.path.join(BASE_DIR, "batch_files")

INPUT_CSV = os.path.join(DATA_DIR, "44k_crunchbase_startups.csv")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "genai_classifications.csv")
PROMPT_FILE = os.path.join(PROMPTS_DIR, "Jan_draft_prompt.txt")

# OpenAI Config
API_KEY_PATH = os.path.join(os.path.dirname(BASE_DIR), "api_key.txt") # Root
if not os.path.exists(API_KEY_PATH):
    API_KEY_PATH = os.path.join(BASE_DIR, "api_key.txt")
if not os.path.exists(API_KEY_PATH):
    API_KEY_PATH = "api_key.txt" # Current dir

MODEL_NAME = "gpt-5-mini" # As requested
MAX_REQUESTS_PER_BATCH = 49000 # Safe limit under 50k
MAX_FILE_SIZE_MB = 190 # Safe limit under 200MB

# Ensure directories exist
for d in [RESULTS_DIR, BATCH_DIR, os.path.join(BATCH_DIR, "requests"), os.path.join(BATCH_DIR, "results"), os.path.join(BATCH_DIR, "ids")]:
    os.makedirs(d, exist_ok=True)

# ============================================================================
# SETUP
# ============================================================================

def get_api_key():
    """Load API key from file"""
    try:
        with open(API_KEY_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    return line
    except FileNotFoundError:
        print(f"[ERROR] API key file not found at {API_KEY_PATH}")
        return None
    return None

client = None
api_key = get_api_key()
if api_key:
    client = OpenAI(api_key=api_key)
else:
    print("[WARNING] OpenAI client not initialized (missing API key)")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_system_prompt():
    """Read the system prompt from file"""
    with open(PROMPT_FILE, "r") as f:
        return f.read().strip()

def format_user_message(row):
    """
    Format the startup profile for the LLM.
    """
    name = str(row.get('name', 'N/A')).strip()
    short_desc = str(row.get('short_description', 'N/A')).strip()
    long_desc = str(row.get('description', 'N/A')).strip()
    categories = str(row.get('category_list', 'N/A')).strip()
    
    # Handle empty/NaN values gracefully
    if pd.isna(row.get('description')): long_desc = "N/A"
    if pd.isna(row.get('short_description')): short_desc = "N/A"
    
    message = f"""Name: {name}
Short Description: {short_desc}
Long Description: {long_desc}
Categories: {categories}"""
    
    return message

def flatten_json_result(result_json, company_id, company_name):
    """
    Flatten the nested JSON response into a single dictionary for CSV row.
    """
    flat = {
        "company_id": company_id,
        "company_name": company_name,
        
        # Strict Mode
        "genai_strict_label": result_json.get("genai_adoption_strict", {}).get("label", "No"),
        "genai_strict_confidence": result_json.get("genai_adoption_strict", {}).get("confidence", "Low"),
        
        # Moderate Mode
        "genai_moderate_label": result_json.get("genai_adoption_moderate", {}).get("label", "No"),
        "genai_moderate_confidence": result_json.get("genai_adoption_moderate", {}).get("confidence", "Low"),
        
        # Lenient Mode
        "genai_lenient_label": result_json.get("genai_adoption_lenient", {}).get("label", "No"),
        "genai_lenient_confidence": result_json.get("genai_adoption_lenient", {}).get("confidence", "Low"),
        
        # Meta
        "primary_assessment": result_json.get("primary_assessment", "strict"),
        "no_evidence_flag": 1 if result_json.get("no_evidence_of_genai_use") else 0,
    }
    
    # Handle Functions List
    funcs = result_json.get("genai_functions", [])
    if funcs:
        flat["genai_functions_list"] = "; ".join([f.get("function_keyword", "") for f in funcs])
        flat["genai_functions_details"] = " | ".join([f"{f.get('function_keyword')}: {f.get('short_description')}" for f in funcs])
    else:
        flat["genai_functions_list"] = ""
        flat["genai_functions_details"] = ""

    # Handle Reasoning
    reasons = result_json.get("reasoning", [])
    flat["reasoning"] = " ".join(reasons) if isinstance(reasons, list) else str(reasons)
    
    return flat

# ============================================================================
# BATCH MANAGEMENT
# ============================================================================

def create_batches(limit=None):
    """
    Read input CSV and create JSONL batch files.
    Args:
        limit (int, optional): Limit number of records for testing.
    """
    print(f"[INFO] Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    if limit:
        print(f"[TEST MODE] Limiting to first {limit} records")
        df = df.head(limit)
        
    print(f"[INFO] Total records to process: {len(df)}")
    
    system_prompt = load_system_prompt()
    
    # Split into chunks based on COUNT and SIZE
    batch_files = []
    
    current_batch = []
    current_batch_size = 0
    batch_idx = 1
    
    print(f"[INFO] Processing {len(df)} records...")
    
    for idx, row in df.iterrows():
        # Use org_uuid as custom_id if available, else row index
        custom_id = str(row.get('org_uuid', f"row_{idx}"))
        user_msg = format_user_message(row)
        
        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                "response_format": {"type": "json_object"}
            }
        }
        
        json_line = json.dumps(request) + "\n"
        line_size = len(json_line.encode('utf-8'))
        
        # Check limits
        if (len(current_batch) >= MAX_REQUESTS_PER_BATCH) or \
           (current_batch_size + line_size > MAX_FILE_SIZE_MB * 1024 * 1024):
            
            # Write current batch
            filename = os.path.join(BATCH_DIR, "requests", f"batch_{batch_idx}.jsonl")
            with open(filename, "w") as f:
                for line in current_batch:
                    f.write(line)
            
            print(f"[OK] Created Batch {batch_idx}: {len(current_batch)} requests, {current_batch_size/1024/1024:.2f} MB")
            batch_files.append(filename)
            
            # Reset
            batch_idx += 1
            current_batch = []
            current_batch_size = 0
            
        current_batch.append(json_line)
        current_batch_size += line_size
        
    # Write remaining
    if current_batch:
        filename = os.path.join(BATCH_DIR, "requests", f"batch_{batch_idx}.jsonl")
        with open(filename, "w") as f:
            for line in current_batch:
                f.write(line)
        print(f"[OK] Created Batch {batch_idx}: {len(current_batch)} requests, {current_batch_size/1024/1024:.2f} MB")
        batch_files.append(filename)
        
    return batch_files

def upload_and_run_batch(batch_file):
    """Uploads and starts a batch, saving ID to file"""
    if not client:
        print("[ERROR] Cannot upload - Client not initialized")
        return None

    # Handle retry batch or regular batch
    filename = os.path.basename(batch_file)
    if filename == "batch_retry.jsonl":
        batch_num = "retry"
    else:
        batch_num = filename.split('_')[1].split('.')[0]
    
    id_file = os.path.join(BATCH_DIR, "ids", f"batch_{batch_num}_id.txt")
    
    # Check if already exists
    if os.path.exists(id_file):
        print(f"[INFO] Batch {batch_num} already exists (ID saved).")
        choice = input(f"Rerun Batch {batch_num}? (This costs money) [y/N]: ").strip().lower()
        if choice != 'y':
            with open(id_file, 'r') as f:
                return f.read().strip()

    print(f"[UPLOAD] Uploading {batch_file}...")
    with open(batch_file, "rb") as f:
        file_response = client.files.create(file=f, purpose="batch")
    
    file_id = file_response.id
    print(f"[UPLOAD] File ID: {file_id}")
    
    print(f"[START] Creating batch job...")
    batch_response = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    
    batch_id = batch_response.id
    print(f"[START] Batch ID: {batch_id}")
    
    # Save ID
    with open(id_file, "w") as f:
        f.write(batch_id)
        
    return batch_id

def get_batch_file_for_id(batch_id):
    """Finds the ID file associated with a given batch ID"""
    id_dir = os.path.join(BATCH_DIR, "ids")
    if not os.path.exists(id_dir): return None
    
    for f in os.listdir(id_dir):
        if f.endswith("_id.txt"):
            path = os.path.join(id_dir, f)
            try:
                with open(path, 'r') as file:
                    if file.read().strip() == batch_id:
                        return path
            except:
                continue
    return None

def monitor_batches(batch_ids):
    """
    Polls the status of submitted batches until completion.
    """
    if not batch_ids:
        return
        
    print(f"\n[MONITOR] Monitoring {len(batch_ids)} batches...")
    print("Press Ctrl+C to stop monitoring (batches will continue running on OpenAI).")
    
    completed_ids = set()
    batch_progress = {}  # Track progress to detect stuck batches
    stuck_warnings = {}  # Track how many times we've warned about stuck batches
    
    while len(completed_ids) < len(batch_ids):
        print(f"\n--- Status Check {datetime.now().strftime('%H:%M:%S')} ---")
        
        for bid in batch_ids:
            if bid in completed_ids:
                continue
                
            try:
                batch = client.batches.retrieve(bid)
                status = batch.status
                
                # Format counts if available
                counts = batch.request_counts
                if counts:
                    completed = counts.completed
                    failed = counts.failed
                    total = counts.total
                    remaining = total - completed - failed
                    progress = f"{completed}/{total} ({failed} failed, {remaining} remaining)"
                    
                    # Check if stuck
                    if bid in batch_progress:
                        prev_completed, prev_failed = batch_progress[bid]
                        if completed == prev_completed and failed == prev_failed:
                            stuck_count = stuck_warnings.get(bid, 0) + 1
                            stuck_warnings[bid] = stuck_count
                            
                            if stuck_count >= 10:  # 10 checks = ~5 minutes
                                print(f"   [WARNING] [STUCK] Batch {bid[-8:]} has made no progress for {stuck_count * 30 / 60:.0f} minutes")
                                print(f"      Consider: The batch may be retrying failed requests or stuck on problematic items")
                                print(f"      Action: You can cancel this batch and re-run only the remaining {remaining} items")
                        else:
                            # Progress made, reset stuck counter
                            stuck_warnings[bid] = 0
                    
                    batch_progress[bid] = (completed, failed)
                else:
                    progress = "Preparing..."
                
                print(f"Batch {bid[-8:]}: [{status.upper()}] - {progress}")
                
                if status in ['completed', 'failed', 'expired', 'cancelled']:
                    completed_ids.add(bid)
                    if status == 'completed':
                        failed_count = counts.failed if counts else 0
                        if failed_count > 0:
                            print(f"   > Batch {bid[-8:]} completed with {failed_count} failures")
                        else:
                            print(f"   > Batch {bid[-8:]} finished successfully!")
                        # Auto-download
                        id_file = get_batch_file_for_id(bid)
                        if id_file:
                            print(f"   > Auto-downloading results...")
                            process_results(id_file)
                        else:
                            print(f"   > [WARN] Could not find local ID file for {bid}")
                        
            except Exception as e:
                print(f"Error checking {bid}: {e}")
        
        if len(completed_ids) < len(batch_ids):
            time.sleep(30)
            
    print("\n[DONE] All batches completed/terminal.")
    
    # Auto-merge all batch outputs
    print("\n[MERGE] Combining all batch outputs into single file...")
    merge_all_batch_outputs()

def cancel_batch(batch_id):
    """Cancel a stuck or in-progress batch."""
    try:
        batch = client.batches.cancel(batch_id)
        print(f"[OK] Batch {batch_id[-8:]} cancellation requested")
        print(f"     Status: {batch.status}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to cancel batch: {e}")
        return False

def analyze_failed_requests(batch_id):
    """Analyze failed requests in a batch to understand failure reasons."""
    print(f"\n[ANALYZE] Analyzing batch {batch_id[-8:]}...")
    try:
        batch = client.batches.retrieve(batch_id)
        
        if hasattr(batch, 'request_counts') and batch.request_counts:
            failed = batch.request_counts.failed
            total = batch.request_counts.total
            completed = batch.request_counts.completed
            print(f"  Status: {batch.status}")
            print(f"  Total: {total:,} | Completed: {completed:,} | Failed: {failed:,}")
            
            if failed == 0:
                print("  [OK] No failures!")
                return
        
        # Try to get error details (may not always be available)
        if hasattr(batch, 'errors') and batch.errors:
            error_summary = {}
            errors_data = batch.errors.get('data', []) if isinstance(batch.errors, dict) else []
            for error in errors_data:
                error_code = error.get('code', 'unknown')
                error_summary[error_code] = error_summary.get(error_code, 0) + 1
            
            if error_summary:
                print(f"  Error breakdown:")
                for code, count in sorted(error_summary.items(), key=lambda x: x[1], reverse=True):
                    print(f"    - {code}: {count} failures")
        else:
            print("  [INFO] Detailed error breakdown not available from API")
            print("  [TIP] Check failure logs in results/ folder after downloading")
            
    except Exception as e:
        print(f"  [ERROR] Failed to analyze: {e}")

def process_results(batch_id_file):
    """
    Download and process results for a completed batch.
    """
    try:
        with open(batch_id_file, "r") as f:
            batch_id = f.read().strip()
    except FileNotFoundError:
        print(f"[ERROR] Batch ID file not found: {batch_id_file}")
        return

    # Load name mapping (UUID -> Name)
    print("[INFO] Loading company names for lookup...")
    try:
        name_df = pd.read_csv(INPUT_CSV, usecols=['org_uuid', 'name'])
        id_to_name = dict(zip(name_df['org_uuid'].astype(str), name_df['name']))
    except Exception as e:
        print(f"[WARN] Could not load names from CSV: {e}")
        id_to_name = {}

    print(f"[DOWNLOAD] Retrieve batch {batch_id}...")
    batch = client.batches.retrieve(batch_id)
    
    if batch.status != "completed":
        print(f"[ERROR] Batch not completed (Status: {batch.status})")
        # Still try to analyze failures if available
        if hasattr(batch, 'request_counts') and batch.request_counts.failed > 0:
            analyze_failed_requests(batch_id)
        return

    # Check for failures
    failed_count = batch.request_counts.failed if hasattr(batch, 'request_counts') else 0
    if failed_count > 0:
        print(f"[WARN] Batch completed with {failed_count} failed requests")
        analyze_failed_requests(batch_id)

    output_file_id = batch.output_file_id
    if not output_file_id:
        print("[ERROR] No output file ID found.")
        return

    print(f"[DOWNLOAD] Downloading content {output_file_id}...")
    content = client.files.content(output_file_id).content
    
    # Parse
    results = []
    failed_items = []
    json_errors = 0
    http_errors = 0
    
    for line in content.splitlines():
        try:
            item = json.loads(line)
            custom_id = item.get("custom_id", "")
            
            # Lookup Name
            company_name = id_to_name.get(custom_id, "Unknown ID")
            
            # Extract response
            response = item.get("response", {})
            status_code = response.get("status_code", 0)
            
            if status_code == 200:
                body = response.get("body", {})
                choices = body.get("choices", [])
                if choices:
                    content_str = choices[0].get("message", {}).get("content", "{}")
                    try:
                        json_content = json.loads(content_str)
                        # Flatten
                        flat = flatten_json_result(json_content, custom_id, company_name)
                        results.append(flat)
                    except json.JSONDecodeError as e:
                        json_errors += 1
                        failed_items.append({
                            'id': custom_id,
                            'name': company_name,
                            'error': 'Invalid JSON response',
                            'content_preview': content_str[:100] if content_str else 'Empty'
                        })
            else:
                http_errors += 1
                error_body = response.get("body", {})
                error_msg = error_body.get("error", {}).get("message", "Unknown error") if isinstance(error_body, dict) else str(error_body)
                failed_items.append({
                    'id': custom_id,
                    'name': company_name,
                    'error': f'HTTP {status_code}: {error_msg[:100]}',
                    'content_preview': ''
                })
                
        except Exception as e:
            failed_items.append({
                'id': 'unknown',
                'name': 'Unknown',
                'error': f'Parse error: {str(e)[:100]}',
                'content_preview': ''
            })
    
    # Report failures
    if failed_items:
        print(f"\n[WARN] Processing summary:")
        print(f"  - Successful: {len(results)}")
        print(f"  - JSON parse errors: {json_errors}")
        print(f"  - HTTP errors: {http_errors}")
        print(f"  - Total failed: {len(failed_items)}")
        
        # Save failure log
        batch_num = os.path.basename(batch_id_file).split('_')[1]
        failure_log = os.path.join(RESULTS_DIR, f"batch_{batch_num}_failures.csv")
        if failed_items:
            failure_df = pd.DataFrame(failed_items)
            failure_df.to_csv(failure_log, index=False)
            print(f"  - Failure log saved to: {failure_log}")

    # Save
    if results:
        # Determine output filename based on batch ID file
        batch_num = os.path.basename(batch_id_file).split('_')[1]
        output_csv = os.path.join(RESULTS_DIR, f"batch_{batch_num}_output.csv")
        
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"[SUCCESS] Saved {len(results)} results to {output_csv}")
        
        # --- STATISTICAL SUMMARY ---
        print(f"\n{'='*40}")
        print(f"BATCH {batch_num} SUMMARY")
        print(f"{'='*40}")
        total = len(df)
        print(f"Total Companies: {total}")
        
        strict_yes = (df['genai_strict_label'] == 'Yes').sum()
        mod_yes = (df['genai_moderate_label'] == 'Yes').sum()
        len_yes = (df['genai_lenient_label'] == 'Yes').sum()
        no_ev = (df['no_evidence_flag'] == 1).sum()
        
        print(f"Strict Adoption:   {strict_yes:>5} ({strict_yes/total*100:>5.1f}%)")
        print(f"Moderate Adoption: {mod_yes:>5} ({mod_yes/total*100:>5.1f}%)")
        print(f"Lenient Adoption:  {len_yes:>5} ({len_yes/total*100:>5.1f}%)")
        print(f"No Evidence:       {no_ev:>5} ({no_ev/total*100:>5.1f}%)")
        print(f"{'='*40}\n")
        
    else:
        print("[WARN] No valid results found.")

def download_all_completed():
    """Check all ID files and download results for completed ones"""
    id_files = [os.path.join(BATCH_DIR, "ids", f) for f in os.listdir(os.path.join(BATCH_DIR, "ids")) if f.endswith("_id.txt")]
    print(f"[INFO] Found {len(id_files)} batch ID files.")
    
    for idf in id_files:
        process_results(idf)

def extract_failed_ids_from_batches():
    """
    Extract all failed request IDs from completed batches.
    Strategy: Compare original request files (all IDs) with output files (successful IDs).
    The difference = failed IDs.
    """
    print("[ANALYZE] Extracting failed request IDs from all batches...")
    
    id_files = sorted([os.path.join(BATCH_DIR, "ids", f) for f in os.listdir(os.path.join(BATCH_DIR, "ids")) if f.endswith("_id.txt")])
    
    all_failed_ids = []
    all_successful_ids = set()
    failure_analysis = {
        'total_failed': 0,
        'http_errors': 0,
        'json_errors': 0,
        'parse_errors': 0,
        'error_types': Counter()
    }
    
    for idf in id_files:
        try:
            with open(idf, "r") as f:
                batch_id = f.read().strip()
        except:
            continue
        
        batch_num = os.path.basename(idf).split('_')[1]
        print(f"\n[ANALYZE] Processing batch {batch_num}...")
        
        batch = client.batches.retrieve(batch_id)
        
        if batch.status != "completed":
            print(f"  [SKIP] Batch not completed (status: {batch.status})")
            continue
        
        # Get failure count from batch metadata
        failed_count = batch.request_counts.failed if hasattr(batch, 'request_counts') else 0
        total_count = batch.request_counts.total if hasattr(batch, 'request_counts') else 0
        completed_count = batch.request_counts.completed if hasattr(batch, 'request_counts') else 0
        
        print(f"  Batch stats: {completed_count}/{total_count} completed, {failed_count} failed")
        
        # Step 1: Get ALL IDs from original request file
        request_file = os.path.join(BATCH_DIR, "requests", f"batch_{batch_num}.jsonl")
        all_submitted_ids = set()
        
        if os.path.exists(request_file):
            with open(request_file, "r") as f:
                for line in f:
                    try:
                        req = json.loads(line)
                        all_submitted_ids.add(req.get("custom_id", ""))
                    except:
                        pass
            print(f"  Found {len(all_submitted_ids)} submitted requests")
        else:
            print(f"  [WARN] Request file not found: {request_file}")
            continue
        
        # Step 2: Get successful IDs from output file
        output_file_id = batch.output_file_id
        successful_ids = set()
        json_parse_failures = []
        
        if output_file_id:
            content = client.files.content(output_file_id).content
            
            for line in content.splitlines():
                try:
                    item = json.loads(line)
                    custom_id = item.get("custom_id", "")
                    response = item.get("response", {})
                    status_code = response.get("status_code", 0)
                    
                    if status_code == 200:
                        # Check if JSON is valid
                        body = response.get("body", {})
                        choices = body.get("choices", [])
                        if choices:
                            content_str = choices[0].get("message", {}).get("content", "{}")
                            try:
                                json.loads(content_str)
                                successful_ids.add(custom_id)
                            except json.JSONDecodeError:
                                # Status 200 but invalid JSON - we'll retry these
                                json_parse_failures.append(custom_id)
                                failure_analysis['json_errors'] += 1
                    else:
                        # Non-200 status (shouldn't be in output, but check anyway)
                        failure_analysis['http_errors'] += 1
                except Exception as e:
                    failure_analysis['parse_errors'] += 1
        
        # Step 3: Calculate failed IDs = submitted - successful
        failed_ids = all_submitted_ids - successful_ids
        all_failed_ids.extend(failed_ids)
        all_failed_ids.extend(json_parse_failures)  # Add JSON parse failures too
        
        print(f"  Successful: {len(successful_ids)}")
        print(f"  Failed (HTTP): {len(failed_ids)}")
        print(f"  Failed (JSON parse): {len(json_parse_failures)}")
        print(f"  Total failed for this batch: {len(failed_ids) + len(json_parse_failures)}")
        
        failure_analysis['total_failed'] += len(failed_ids) + len(json_parse_failures)
        failure_analysis['http_errors'] += len(failed_ids)
    
    # Remove duplicates
    all_failed_ids = list(set(all_failed_ids))
    
    print(f"\n[ANALYZE] Overall Failure Summary:")
    print(f"  Total Failed Requests: {len(all_failed_ids)}")
    print(f"  HTTP Errors: {failure_analysis['http_errors']}")
    print(f"  JSON Parse Errors: {failure_analysis['json_errors']}")
    print(f"  Parse Errors: {failure_analysis['parse_errors']}")
    
    return {
        'failed_ids': all_failed_ids,
        'analysis': failure_analysis
    }

def create_retry_batch(failed_ids):
    """
    Create a new batch file with only the failed request IDs.
    """
    if not failed_ids:
        print("[ERROR] No failed IDs to retry")
        return None
    
    print(f"\n[RETRY] Creating retry batch for {len(failed_ids)} failed requests...")
    
    # Load original data
    df = pd.read_csv(INPUT_CSV)
    df['org_uuid'] = df['org_uuid'].astype(str)
    
    # Filter to only failed IDs
    failed_df = df[df['org_uuid'].isin(failed_ids)].copy()
    
    if len(failed_df) == 0:
        print("[ERROR] Could not find failed IDs in original dataset")
        return None
    
    print(f"[RETRY] Found {len(failed_df)} companies to retry")
    
    # Load system prompt
    system_prompt = load_system_prompt()
    
    # Create retry batch file
    retry_filename = os.path.join(BATCH_DIR, "requests", "batch_retry.jsonl")
    
    with open(retry_filename, "w") as f:
        for idx, row in failed_df.iterrows():
            custom_id = str(row.get('org_uuid', f"row_{idx}"))
            user_msg = format_user_message(row)
            
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg}
                    ],
                    "response_format": {"type": "json_object"}
                }
            }
            f.write(json.dumps(request) + "\n")
    
    file_size = os.path.getsize(retry_filename) / (1024 * 1024)
    print(f"[SUCCESS] Retry batch created: {retry_filename}")
    print(f"          Size: {file_size:.2f} MB")
    print(f"          Requests: {len(failed_df)}")
    
    return retry_filename

def merge_all_batch_outputs():
    """
    Merge all batch output CSVs into a single combined file.
    (Does NOT merge with original dataset - just combines batch outputs)
    """
    print(f"[MERGE] Finding batch result files in {RESULTS_DIR}...")
    result_files = sorted([os.path.join(RESULTS_DIR, f) for f in os.listdir(RESULTS_DIR) if f.startswith("batch_") and f.endswith("_output.csv")])
    
    if not result_files:
        print("[ERROR] No batch result files found.")
        return
    
    print(f"[MERGE] Loading {len(result_files)} batch result files...")
    results_dfs = []
    for rf in result_files:
        try:
            df = pd.read_csv(rf)
            results_dfs.append(df)
            print(f"  - {os.path.basename(rf)}: {len(df)} records")
        except Exception as e:
            print(f"[WARN] Failed to read {rf}: {e}")
            
    if not results_dfs:
        print("[ERROR] No valid result data loaded.")
        return
        
    all_results = pd.concat(results_dfs, ignore_index=True)
    print(f"[MERGE] Total records: {len(all_results)}")
    
    output_combined = os.path.join(RESULTS_DIR, "genai_classifications_combined.csv")
    all_results.to_csv(output_combined, index=False)
    
    print(f"[SUCCESS] Combined file saved to {output_combined}")
    
    # Print overall summary
    print(f"\n{'='*40}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*40}")
    total = len(all_results)
    print(f"Total Companies: {total}")
    
    strict_yes = (all_results['genai_strict_label'] == 'Yes').sum()
    mod_yes = (all_results['genai_moderate_label'] == 'Yes').sum()
    len_yes = (all_results['genai_lenient_label'] == 'Yes').sum()
    no_ev = (all_results['no_evidence_flag'] == 1).sum()
    
    print(f"Strict Adoption:   {strict_yes:>5} ({strict_yes/total*100:>5.1f}%)")
    print(f"Moderate Adoption: {mod_yes:>5} ({mod_yes/total*100:>5.1f}%)")
    print(f"Lenient Adoption:  {len_yes:>5} ({len_yes/total*100:>5.1f}%)")
    print(f"No Evidence:       {no_ev:>5} ({no_ev/total*100:>5.1f}%)")
    print(f"{'='*40}\n")

def merge_results_with_original():
    """
    Merge all batch output CSVs with the original dataset.
    """
    print(f"[MERGE] Reading original data from {INPUT_CSV}...")
    try:
        original_df = pd.read_csv(INPUT_CSV)
        # Ensure org_uuid is string for merging
        original_df['org_uuid'] = original_df['org_uuid'].astype(str)
    except Exception as e:
        print(f"[ERROR] Failed to read original CSV: {e}")
        return

    print(f"[MERGE] Finding batch result files in {RESULTS_DIR}...")
    result_files = [os.path.join(RESULTS_DIR, f) for f in os.listdir(RESULTS_DIR) if f.startswith("batch_") and f.endswith("_output.csv")]
    
    if not result_files:
        print("[ERROR] No batch result files found.")
        return
    
    print(f"[MERGE] Loading {len(result_files)} result files...")
    results_dfs = []
    for rf in result_files:
        try:
            df = pd.read_csv(rf)
            results_dfs.append(df)
        except Exception as e:
            print(f"[WARN] Failed to read {rf}: {e}")
            
    if not results_dfs:
        print("[ERROR] No valid result data loaded.")
        return
        
    all_results = pd.concat(results_dfs, ignore_index=True)
    print(f"[MERGE] Total classified records: {len(all_results)}")
    
    # Prepare for merge
    # The 'company_id' in results is the 'org_uuid' from original
    all_results['company_id'] = all_results['company_id'].astype(str)
    
    # Drop company_name from results if it's just "Lookup via ID" to avoid collision/mess
    if 'company_name' in all_results.columns:
        all_results = all_results.drop(columns=['company_name'])
        
    print("[MERGE] Merging with original data...")
    # Left join to keep all original records, or inner to keep only classified?
    # Usually we want to see the classifications attached to the original data.
    # Let's do a left join on original (preserves unclassified as NaN) OR inner (only what we processed).
    # If we ran a test (1k), we probably only want the 1k output.
    # If we ran full, we want full.
    # Safer to merge onto the RESULTS so we get the classified subset.
    
    merged_df = pd.merge(all_results, original_df, left_on='company_id', right_on='org_uuid', how='left')
    
    # Reorder columns to put classifications first? Or append?
    # Usually append. Let's keep it simple.
    
    output_merged = os.path.join(RESULTS_DIR, "genai_classifications_merged.csv")
    merged_df.to_csv(output_merged, index=False)
    
    print(f"[SUCCESS] Merged data saved to {output_merged}")
    print(f"          Total rows: {len(merged_df)}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("GenAI Adoption Classification Batch Processor")
    print("1. Create Batch Files (Full)")
    print("2. Create Batch Files (Test - First 1k)")
    print("3. Upload and Run (Requires API Key)")
    print("4. Download Results (All Completed)")
    print("5. Merge All Batch Outputs (Combined CSV)")
    print("6. Merge All Results with Original Data")
    print("7. Analyze Failed Requests")
    print("8. Cancel Stuck Batch")
    print("9. Extract Failed IDs & Create Retry Batch")
    
    choice = input("Select option (1-9): ").strip()
    
    if choice == "1":
        create_batches()
    elif choice == "2":
        create_batches(limit=1000)
    elif choice == "3":
        print("Run full (F), test 1k (T), or retry failed requests (R)?")
        sub_choice = input("Choice (F/T/R): ").strip().upper()
        
        if sub_choice == "R":
            # Upload retry batch
            retry_file = os.path.join(BATCH_DIR, "requests", "batch_retry.jsonl")
            if not os.path.exists(retry_file):
                print(f"[ERROR] Retry batch file not found: {retry_file}")
                print("       Run Option 9 first to create the retry batch.")
            else:
                print(f"[INFO] Uploading retry batch: {retry_file}")
                batch_id = upload_and_run_batch(retry_file)
                if batch_id:
                    print(f"\n[DONE] Retry batch submitted: {batch_id}")
                    monitor = input("Monitor progress now? (Y/n): ").strip().lower()
                    if monitor != 'n':
                        monitor_batches([batch_id])
        else:
            limit = 1000 if sub_choice == "T" else None
            
            print("[INFO] Starting Parallel Batch Submission...")
            batches = create_batches(limit=limit) # Ensure files exist
            batch_ids = []
            for bf in batches:
                bid = upload_and_run_batch(bf)
                if bid: batch_ids.append(bid)
            
            print(f"\n[DONE] Submitted {len(batch_ids)} batches.")
            
            if batch_ids:
                monitor = input("Monitor progress now? (Y/n): ").strip().lower()
                if monitor != 'n':
                    monitor_batches(batch_ids)
            else:
                print("No batches submitted.")
            
    elif choice == "4":
        download_all_completed()
        
    elif choice == "5":
        merge_all_batch_outputs()
        
    elif choice == "6":
        merge_results_with_original()
        
    elif choice == "7":
        print("\n[ANALYZE] Analyzing failed requests in all batches...")
        id_files = [os.path.join(BATCH_DIR, "ids", f) for f in os.listdir(os.path.join(BATCH_DIR, "ids")) if f.endswith("_id.txt")]
        for idf in sorted(id_files):
            try:
                with open(idf, "r") as f:
                    batch_id = f.read().strip()
                analyze_failed_requests(batch_id)
            except Exception as e:
                print(f"[ERROR] Failed to analyze {idf}: {e}")
                
    elif choice == "8":
        print("\n[CANCEL] Cancel a stuck batch")
        id_files = [os.path.join(BATCH_DIR, "ids", f) for f in os.listdir(os.path.join(BATCH_DIR, "ids")) if f.endswith("_id.txt")]
        if not id_files:
            print("[ERROR] No batch ID files found")
        else:
            print("Available batches:")
            for i, idf in enumerate(sorted(id_files), 1):
                batch_num = os.path.basename(idf).split('_')[1]
                try:
                    with open(idf, "r") as f:
                        batch_id = f.read().strip()
                    batch = client.batches.retrieve(batch_id)
                    status = batch.status
                    counts = batch.request_counts if hasattr(batch, 'request_counts') else None
                    if counts:
                        print(f"  {i}. Batch {batch_num} ({batch_id[-8:]}): {status} - {counts.completed}/{counts.total} completed, {counts.failed} failed")
                    else:
                        print(f"  {i}. Batch {batch_num} ({batch_id[-8:]}): {status}")
                except:
                    print(f"  {i}. Batch {batch_num}: (error reading)")
            
            try:
                sel = int(input("\nSelect batch number to cancel (or 0 to cancel): "))
                if 1 <= sel <= len(id_files):
                    with open(id_files[sel-1], "r") as f:
                        batch_id = f.read().strip()
                    confirm = input(f"Cancel batch {batch_id[-8:]}? (y/N): ").strip().lower()
                    if confirm == 'y':
                        cancel_batch(batch_id)
                else:
                    print("Cancelled")
            except ValueError:
                print("Invalid selection")
                
    elif choice == "9":
        print("\n[RETRY] Extracting failed IDs and creating retry batch...")
        print("\n[INFO] Why Retries Work:")
        print("  Since the data is identical to successful requests, failures are likely:")
        print("  ✓ Transient API issues (high retry success rate):")
        print("    - Network timeouts during processing")
        print("    - Temporary API overload/queue issues")
        print("    - Model returning invalid JSON (occasional glitches)")
        print("    - Processing timeouts for long descriptions")
        print("  ✗ Persistent issues (low retry success):")
        print("    - Content policy violations (rare, but consistent)")
        print("    - Account/quota limits (if you hit hard limits)")
        print("\n  Expected success rate: ~60-80% of retries should succeed")
        print("  (Most failures are transient API issues, not data problems)\n")
        
        result = extract_failed_ids_from_batches()
        
        if result['failed_ids']:
            print(f"\n[INFO] Found {len(result['failed_ids'])} failed requests")
            print(f"       Breakdown: {result['analysis']['http_errors']} HTTP errors, {result['analysis']['json_errors']} JSON parse errors")
            print(f"\n[ESTIMATE] Expected to recover: ~{int(len(result['failed_ids']) * 0.6)} - {int(len(result['failed_ids']) * 0.8)} requests")
            print(f"           (60-80% success rate expected since data is identical to successful requests)")
            
            proceed = input("\nCreate retry batch? (Y/n): ").strip().lower()
            if proceed != 'n':
                retry_file = create_retry_batch(result['failed_ids'])
                if retry_file:
                    print(f"\n[INFO] Retry batch created. Use Option 3, then select 'R' to upload it.")
                    print(f"       After completion, run Option 5 to merge with existing results.")
        else:
            print("[INFO] No failed requests found!")
        
    else:
        print("Invalid choice.")

