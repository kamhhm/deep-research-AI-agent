"""
Helper Script: Check Status of All Batches
===========================================
Use this script to check the status of all batches without blocking.
Useful for checking progress when you come back to your computer.
"""

from openai import OpenAI
import os
from datetime import datetime

# Read API key
with open("../../api_key.txt", "r") as f:
    lines = f.readlines()
    api_key = None
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            api_key = line
            break

client = OpenAI(api_key=api_key)

print("="*70)
print("BATCH STATUS CHECK - WEEK 10 SHORT-ONLY EXPERIMENT")
print("="*70)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Find all batch ID files
batch_ids_dir = "../batch_files/batch_ids"

if not os.path.exists(batch_ids_dir):
    print("[ERROR] Batch IDs directory not found")
    print(f"Looking for: {batch_ids_dir}")
    exit(1)

batch_files = sorted([f for f in os.listdir(batch_ids_dir) if f.endswith('_id.txt')])

if not batch_files:
    print("[WARNING] No batch ID files found")
    print(f"Directory: {batch_ids_dir}")
    exit(0)

print(f"Found {len(batch_files)} batch(es)\n")

# Check status of each batch
total_completed = 0
total_in_progress = 0
total_failed = 0
total_pending = 0

for batch_file in batch_files:
    batch_num = batch_file.replace('batch_', '').replace('_id.txt', '')
    batch_id_path = os.path.join(batch_ids_dir, batch_file)
    
    with open(batch_id_path, 'r') as f:
        batch_id = f.read().strip()
    
    try:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        
        # Color-coded status
        if status == "completed":
            status_display = "COMPLETED"
            total_completed += 1
        elif status == "failed":
            status_display = "FAILED"
            total_failed += 1
        elif status == "in_progress":
            status_display = "IN PROGRESS"
            total_in_progress += 1
        else:
            status_display = f"{status.upper()}"
            total_pending += 1
        
        print(f"Batch {batch_num}: {status_display}")
        
        # Show progress if available
        if hasattr(batch, 'request_counts') and batch.request_counts:
            total = batch.request_counts.total
            completed = batch.request_counts.completed
            failed = batch.request_counts.failed
            
            if total > 0:
                progress_pct = (completed / total) * 100
                print(f"          Progress: {completed:,}/{total:,} ({progress_pct:.1f}%) | Failed: {failed:,}")
        
        # Show completion time if completed
        if status == "completed" and hasattr(batch, 'completed_at') and batch.completed_at:
            completed_time = datetime.fromtimestamp(batch.completed_at)
            print(f"          Completed: {completed_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print()
        
    except Exception as e:
        print(f"Batch {batch_num}: ❌ ERROR")
        print(f"          {str(e)}\n")
        total_failed += 1

# Summary
print("="*70)
print("SUMMARY")
print("="*70)
print(f"Total Batches:    {len(batch_files)}")
print(f"Completed:     {total_completed}")
print(f"🔄 In Progress:   {total_in_progress}")
print(f"Pending:       {total_pending}")
print(f"❌ Failed:        {total_failed}")

if total_completed == len(batch_files):
    print("\n🎉 All batches completed! Ready to run analysis.")
    print(f"\nNext step:")
    print(f"  cd ../../analysis")
    print(f"  python compare_classifications.py")
elif total_in_progress > 0 or total_pending > 0:
    print(f"\nStill processing... Check again later.")
else:
    print(f"\n[WARNING] Some batches failed. Review errors above.")

print("="*70)

