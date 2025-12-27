import json
import csv
import os
import pandas as pd
import sys

# Add current directory to path to allow importing if needed, 
# but I'll just redefine the flatten function for simplicity/stability
# to avoid circular dependency or side effects.

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
        "no_evidence_flag": result_json.get("no_evidence_of_genai_use", True),
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

def process_jsonl(input_file, output_file):
    print(f"Processing {input_file}...")
    
    results = []
    count = 0
    with open(input_file, 'r') as f:
        for line in f:
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} lines...", end='\r')
            try:
                item = json.loads(line)
                custom_id = item.get("custom_id", "")
                
                # Extract response
                response = item.get("response", {})
                if response.get("status_code") == 200:
                    body = response.get("body", {})
                    choices = body.get("choices", [])
                    if choices:
                        content_str = choices[0].get("message", {}).get("content", "{}")
                        try:
                            json_content = json.loads(content_str)
                            # Flatten
                            flat = flatten_json_result(json_content, custom_id, "Lookup via ID")
                            results.append(flat)
                        except json.JSONDecodeError:
                            print(f"[WARN] Failed to parse JSON content for {custom_id}")
                else:
                    print(f"[WARN] Error response for {custom_id}")
            except Exception as e:
                print(f"[WARN] Failed to parse line: {e}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"Saved {len(results)} records to {output_file}")
    else:
        print("No valid records found.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    jsonl_file = os.path.join(base_dir, "results", "jsonl", "batch_1_output.jsonl")
    csv_file = os.path.join(base_dir, "results", "batch_1_output.csv")
    
    process_jsonl(jsonl_file, csv_file)

