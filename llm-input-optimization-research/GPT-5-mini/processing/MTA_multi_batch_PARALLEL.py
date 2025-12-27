"""
Multi-Batch Startup Classification Script - PARALLEL UPLOAD VERSION
====================================================================
Week 10 Experiment - Optimized for Tier 4 API
Author: Research Team
Date: November 2025
Purpose: Upload ALL batches simultaneously to maximize Tier 4 capacity (1B tokens/day)

STRATEGY:
1. Create ALL batch files first (sequential)
2. Upload ALL batches at once (parallel queue)
3. Monitor ALL batches until complete
4. Download ALL results
5. Merge into single CSV

With Tier 4 (1B tokens/day), we can queue all ~19 batches simultaneously!
"""

# ============================================================================
# SECTION 1: IMPORTS AND SETUP
# ============================================================================

from openai import OpenAI
import pandas as pd
import json
import time
import os
import csv
import io
from datetime import datetime
import re

# ============================================================================
# SECTION 2: CONFIGURATION
# ============================================================================

# Read API key
with open("../api_key.txt", "r") as f:
    lines = f.readlines()
    api_key = None
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            api_key = line
            break
    if not api_key:
        raise ValueError("No API key found in api_key.txt")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Configuration
INPUT_CSV = "../../company_us_both_descriptions.csv"
OUTPUT_CSV = "classified_startups_short_only.csv"
SYSTEM_PROMPT_FILE = "../../system_prompt.txt"

# Batch API limits
MAX_REQUESTS_PER_BATCH = 50000
MAX_FILE_SIZE_MB = 100  # Target 100MB per batch

# Tier 4 limits
MAX_ENQUEUED_TOKENS_PER_DAY = 1_000_000_000  # 1 Billion
MAX_REQUESTS_PER_MINUTE = 10000
MAX_TOKENS_PER_MINUTE = 10_000_000
ESTIMATED_TOKENS_PER_REQUEST = 3200  # Short description only

MODEL_NAME = "gpt-5-mini"

print("="*70)
print("WEEK 10 - PARALLEL BATCH UPLOAD (TIER 4 OPTIMIZED)")
print("="*70)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Model: {MODEL_NAME}")
print(f"Strategy: Upload ALL batches simultaneously")
print("="*70)

# ============================================================================
# SECTION 3: HELPER FUNCTIONS
# ============================================================================

def extract_year_from_date(date_str):
    """Extract year from date string."""
    if pd.isna(date_str) or date_str == '' or date_str == 'N/A':
        return 'N/A'
    date_str = str(date_str).strip()
    year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
    if year_match:
        return year_match.group(0)
    return 'N/A'

def format_user_message(row):
    """Format startup data - SHORT DESCRIPTION ONLY (no Long Description)."""
    org_uuid_val = row.get('org_uuid', 'N/A')
    company_id = 'N/A' if pd.isna(org_uuid_val) else str(org_uuid_val).strip()
    
    name_val = row.get('name', 'N/A')
    company_name = 'N/A' if pd.isna(name_val) else str(name_val).strip()
    
    short_desc_val = row.get('short_description', '')
    short_desc = 'N/A' if (pd.isna(short_desc_val) or str(short_desc_val).strip() == '') else str(short_desc_val).strip()
    
    cat_list_val = row.get('category_list', '')
    category_list = '' if pd.isna(cat_list_val) else str(cat_list_val).strip()
    
    cat_groups_val = row.get('category_groups_list', '')
    category_groups = '' if pd.isna(cat_groups_val) else str(cat_groups_val).strip()
    
    if category_list and category_groups:
        keywords = f"{category_list}, {category_groups}".strip()
    elif category_list:
        keywords = category_list
    elif category_groups:
        keywords = category_groups
    else:
        keywords = 'N/A'
    
    founded_date = row.get('founded_date', 'N/A')
    year_founded = extract_year_from_date(founded_date)
    
    user_message = f"""INPUT:
CompanyID: {company_id}
CompanyName: {company_name}
Short Description: {short_desc}
Keywords: {keywords}
YearFounded: {year_founded}"""
    return user_message

# ============================================================================
# SECTION 4: BATCH CREATION FUNCTIONS
# ============================================================================

def calculate_batch_sizes():
    """Calculate optimal batch sizes for Tier 4."""
    print("\n[STEP 1] Calculating batch configuration...")
    
    df = pd.read_csv(INPUT_CSV)
    total_startups = len(df)
    print(f"[OK] Total startups: {total_startups:,}")
    
    # Calculate based on file size (100MB target)
    estimated_kb_per_request = 12.0  # Short description only
    estimated_mb_per_request = estimated_kb_per_request / 1024
    max_requests_per_batch = int(MAX_FILE_SIZE_MB / estimated_mb_per_request)
    
    print(f"\n[TIER 4 LIMITS]")
    print(f"  Daily token limit: {MAX_ENQUEUED_TOKENS_PER_DAY:,}")
    print(f"  TPM: {MAX_TOKENS_PER_MINUTE:,}")
    print(f"  RPM: {MAX_REQUESTS_PER_MINUTE:,}")
    
    print(f"\n[BATCH SIZING]")
    print(f"  Max per batch (file size): {max_requests_per_batch:,} requests")
    print(f"  Target file size: {MAX_FILE_SIZE_MB} MB")
    
    num_batches = (total_startups + max_requests_per_batch - 1) // max_requests_per_batch
    startups_per_batch = total_startups // num_batches
    
    total_tokens = total_startups * ESTIMATED_TOKENS_PER_REQUEST
    
    print(f"\n[RESULT]")
    print(f"  Number of batches: {num_batches}")
    print(f"  Startups per batch: ~{startups_per_batch:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  % of daily limit: {total_tokens/MAX_ENQUEUED_TOKENS_PER_DAY*100:.1f}%")
    
    if total_tokens < MAX_ENQUEUED_TOKENS_PER_DAY:
        print(f"\n[SUCCESS] Can upload all {num_batches} batches at once!")
        print(f"   Remaining capacity: {(MAX_ENQUEUED_TOKENS_PER_DAY - total_tokens):,} tokens")
    else:
        print(f"\n[WARNING] Total exceeds daily limit - will need multiple days")
    
    return num_batches, startups_per_batch

def create_all_batch_files(num_batches, startups_per_batch):
    """Create ALL batch files before uploading."""
    print(f"\n[STEP 2] Creating {num_batches} batch files...")
    
    # Load system prompt
    with open(SYSTEM_PROMPT_FILE, "r") as f:
        system_prompt = f.read().strip()
    
    # Load dataset
    df = pd.read_csv(INPUT_CSV)
    total_startups = len(df)
    
    created_files = []
    
    for batch_num in range(1, num_batches + 1):
        jsonl_filename = f"batch_files/batch_requests/batch_{batch_num}_requests.jsonl"
        
        # Skip if already exists
        if os.path.exists(jsonl_filename):
            file_size_mb = os.path.getsize(jsonl_filename) / (1024 * 1024)
            print(f"  [{batch_num}/{num_batches}] ✓ Already exists ({file_size_mb:.1f} MB)")
            created_files.append(jsonl_filename)
            continue
        
        # Calculate indices
        start_idx = (batch_num - 1) * startups_per_batch
        if batch_num == num_batches:
            end_idx = total_startups
        else:
            end_idx = start_idx + startups_per_batch
        
        batch_df = df.iloc[start_idx:end_idx]
        
        # Create JSONL
        with open(jsonl_filename, "w") as f:
            for idx, row in batch_df.iterrows():
                user_message = format_user_message(row)
                
                org_uuid_val = row.get('org_uuid', None)
                if pd.isna(org_uuid_val):
                    org_uuid = f'startup-{idx}'
                else:
                    org_uuid = str(org_uuid_val).strip()
                
                request = {
                    "custom_id": f"startup-{org_uuid}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL_NAME,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ]
                    }
                }
                
                f.write(json.dumps(request) + "\n")
        
        file_size_mb = os.path.getsize(jsonl_filename) / (1024 * 1024)
        print(f"  [{batch_num}/{num_batches}] ✓ Created ({file_size_mb:.1f} MB, {len(batch_df):,} startups)")
        created_files.append(jsonl_filename)
    
    print(f"\n[OK] All {num_batches} batch files ready!")
    return created_files

def upload_all_batches(num_batches):
    """Upload ALL batches simultaneously."""
    print(f"\n[STEP 3] Uploading {num_batches} batches to OpenAI...")
    print("[INFO] This will queue all batches at once for parallel processing")
    
    batch_ids = []
    
    for batch_num in range(1, num_batches + 1):
        jsonl_filename = f"batch_files/batch_requests/batch_{batch_num}_requests.jsonl"
        batch_id_file = f"batch_files/batch_ids/batch_{batch_num}_id.txt"
        
        # Skip if already uploaded
        if os.path.exists(batch_id_file):
            with open(batch_id_file, 'r') as f:
                batch_id = f.read().strip()
            print(f"  [{batch_num}/{num_batches}] ✓ Already uploaded (ID: {batch_id[:20]}...)")
            batch_ids.append((batch_num, batch_id))
            continue
        
        try:
            # Upload file
            with open(jsonl_filename, "rb") as f:
                batch_file = client.files.create(file=f, purpose="batch")
            
            # Create batch
            batch = client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            
            # Save batch ID
            with open(batch_id_file, "w") as f:
                f.write(batch.id)
            
            print(f"  [{batch_num}/{num_batches}] ✓ Uploaded (ID: {batch.id[:20]}...)")
            batch_ids.append((batch_num, batch.id))
            
            # Small delay to avoid rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"  [{batch_num}/{num_batches}] ✗ Failed: {e}")
            batch_ids.append((batch_num, None))
    
    print(f"\n[OK] {len([b for b in batch_ids if b[1]])} batches queued successfully!")
    return batch_ids

def monitor_all_batches(batch_ids):
    """Monitor all batches until completion."""
    print(f"\n[STEP 4] Monitoring {len(batch_ids)} batches...")
    print("[INFO] Checking every 60 seconds until all complete")
    print("[INFO] You can close your laptop - batches run on OpenAI servers\n")
    
    check_count = 0
    
    while True:
        check_count += 1
        print(f"\n{'='*70}")
        print(f"STATUS CHECK #{check_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        
        all_done = True
        completed = 0
        in_progress = 0
        failed = 0
        
        for batch_num, batch_id in batch_ids:
            if not batch_id:
                print(f"Batch {batch_num}: [ERROR] No batch ID")
                failed += 1
                continue
            
            try:
                batch = client.batches.retrieve(batch_id)
                status = batch.status
                
                if status == "completed":
                    completed += 1
                    progress = f"{batch.request_counts.completed}/{batch.request_counts.total}"
                    print(f"Batch {batch_num:2d}: COMPLETED ({progress})")
                elif status == "failed":
                    failed += 1
                    print(f"Batch {batch_num:2d}: ❌ FAILED")
                elif status == "in_progress":
                    in_progress += 1
                    all_done = False
                    progress = f"{batch.request_counts.completed}/{batch.request_counts.total}"
                    pct = (batch.request_counts.completed / batch.request_counts.total * 100) if batch.request_counts.total > 0 else 0
                    print(f"Batch {batch_num:2d}: 🔄 IN PROGRESS ({progress}, {pct:.1f}%)")
                else:
                    all_done = False
                    print(f"Batch {batch_num:2d}: {status.upper()}")
                    
            except Exception as e:
                print(f"Batch {batch_num:2d}: ❌ ERROR: {e}")
                all_done = False
        
        # Summary
        print(f"\nSummary: {completed} completed | {in_progress} in progress | {failed} failed")
        
        if all_done:
            print(f"\n{'='*70}")
            print("🎉 ALL BATCHES COMPLETED!")
            print(f"{'='*70}")
            break
        
        print(f"\nWaiting 60 seconds before next check...")
        time.sleep(60)

def download_all_results(batch_ids):
    """Download results from all completed batches."""
    print(f"\n[STEP 5] Downloading results...")
    
    for batch_num, batch_id in batch_ids:
        if not batch_id:
            print(f"  [{batch_num}] ✗ Skipped (no batch ID)")
            continue
        
        results_jsonl = f"batch_files/batch_results/batch_{batch_num}_results.jsonl"
        results_csv = f"batch_files/batch_outputs/batch_{batch_num}_output.csv"
        
        # Skip if already downloaded
        if os.path.exists(results_csv):
            print(f"  [{batch_num}] ✓ Already downloaded")
            continue
        
        try:
            batch = client.batches.retrieve(batch_id)
            
            if batch.status != "completed":
                print(f"  [{batch_num}] WARNING: Not completed (status: {batch.status})")
                continue
            
            # Download
            result = client.files.content(batch.output_file_id)
            
            # Save JSONL
            with open(results_jsonl, "wb") as f:
                f.write(result.content)
            
            # Parse to CSV
            parsed_results = []
            with open(results_jsonl, "r") as f:
                for line in f:
                    try:
                        response_obj = json.loads(line.strip())
                        ai_response = response_obj["response"]["body"]["choices"][0]["message"]["content"]
                        
                        # Parse CSV response
                        csv_reader = csv.reader(io.StringIO(ai_response.strip()))
                        rows = list(csv_reader)
                        if rows and len(rows[0]) >= 7:
                            parsed_results.append({
                                'CompanyID': rows[0][0].strip(),
                                'CompanyName': rows[0][1].strip(),
                                'AI_native': rows[0][2].strip(),
                                'Confidence_1to5': rows[0][3].strip(),
                                'Reasons_3_points': rows[0][4].strip(),
                                'Sources_used': rows[0][5].strip(),
                                'Verification_critique': rows[0][6].strip()
                            })
                    except:
                        pass
            
            # Save CSV
            if parsed_results:
                results_df = pd.DataFrame(parsed_results)
                results_df.to_csv(results_csv, index=False)
                print(f"  [{batch_num}] ✓ Downloaded ({len(parsed_results):,} results)")
            else:
                print(f"  [{batch_num}] WARNING: No valid results")
                
        except Exception as e:
            print(f"  [{batch_num}] ✗ Error: {e}")

def merge_all_results():
    """Merge all batch results into final CSV."""
    print(f"\n[STEP 6] Merging all results...")
    
    batch_files = sorted([
        os.path.join("batch_files/batch_outputs", f) 
        for f in os.listdir("batch_files/batch_outputs") 
        if f.startswith("batch_") and f.endswith("_output.csv")
    ])
    
    if not batch_files:
        print("[ERROR] No batch output files found!")
        return
    
    all_results = []
    for batch_file in batch_files:
        try:
            df = pd.read_csv(batch_file)
            all_results.append(df)
            batch_num = batch_file.split('_')[1]
            print(f"  [Batch {batch_num}] ✓ Loaded ({len(df):,} rows)")
        except Exception as e:
            print(f"  [Error] {batch_file}: {e}")
    
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(OUTPUT_CSV, index=False)
        
        print(f"\n{'='*70}")
        print("[SUCCESS] All batches processed successfully!")
        print(f"{'='*70}")
        print(f"Output: {OUTPUT_CSV}")
        print(f"Total rows: {len(final_df):,}")
        
        ai_native_count = (final_df['AI_native'].astype(str) == '1').sum()
        print(f"\nAI-Native: {ai_native_count:,} ({ai_native_count/len(final_df)*100:.1f}%)")
        print(f"Not AI-Native: {(len(final_df) - ai_native_count):,}")

# ============================================================================
# SECTION 5: MAIN WORKFLOW
# ============================================================================

def main():
    """Main workflow - parallel batch processing."""
    print("\n[STRATEGY] Parallel Batch Upload")
    print("  1. Create all batch files")
    print("  2. Upload all batches at once")
    print("  3. Monitor all batches")
    print("  4. Download all results")
    print("  5. Merge into single file\n")
    
    # Calculate
    num_batches, startups_per_batch = calculate_batch_sizes()
    
    # Create all files
    create_all_batch_files(num_batches, startups_per_batch)
    
    # Upload all at once
    batch_ids = upload_all_batches(num_batches)
    
    # Monitor until complete
    monitor_all_batches(batch_ids)
    
    # Download all
    download_all_results(batch_ids)
    
    # Merge
    merge_all_results()
    
    print(f"\n{'='*70}")
    print("🎉 WEEK 10 CLASSIFICATION COMPLETE!")
    print(f"{'='*70}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nNext: Run analysis/compare_classifications.py")

if __name__ == "__main__":
    main()

