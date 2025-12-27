"""
GenAI Adoption Classification Dashboard
========================================
Generates an interactive HTML dashboard visualizing GenAI adoption classification results.
"""

import os
import pandas as pd
import json
from collections import Counter
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
INSIGHTS_DIR = os.path.join(BASE_DIR, "insights")
OUTPUT_FILE = os.path.join(INSIGHTS_DIR, "genai_adoption_dashboard.html")

def load_results():
    """Load all batch output CSVs or the combined file if it exists."""
    combined_file = os.path.join(RESULTS_DIR, "genai_classifications_combined.csv")
    
    if os.path.exists(combined_file):
        print(f"[INFO] Loading combined file: {combined_file}")
        return pd.read_csv(combined_file)
    
    # Otherwise, load all batch files
    batch_files = sorted([os.path.join(RESULTS_DIR, f) for f in os.listdir(RESULTS_DIR) 
                          if f.startswith("batch_") and f.endswith("_output.csv")])
    
    if not batch_files:
        raise FileNotFoundError("No result files found. Run batch processing first.")
    
    print(f"[INFO] Loading {len(batch_files)} batch files...")
    dfs = []
    for f in batch_files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"  - {os.path.basename(f)}: {len(df)} records")
    
    return pd.concat(dfs, ignore_index=True)

def generate_dashboard():
    """Generate the HTML dashboard."""
    print("="*60)
    print("Generating GenAI Adoption Classification Dashboard")
    print("="*60)
    
    # Load data
    df = load_results()
    total = len(df)
    print(f"\n[OK] Loaded {total:,} company classifications")
    
    # --- Calculate Statistics ---
    
    # Adoption Rates
    strict_yes = (df['genai_strict_label'] == 'Yes').sum()
    mod_yes = (df['genai_moderate_label'] == 'Yes').sum()
    len_yes = (df['genai_lenient_label'] == 'Yes').sum()
    no_evidence = (df['no_evidence_flag'] == 1).sum()
    
    strict_pct = (strict_yes / total * 100) if total > 0 else 0
    mod_pct = (mod_yes / total * 100) if total > 0 else 0
    len_pct = (len_yes / total * 100) if total > 0 else 0
    no_ev_pct = (no_evidence / total * 100) if total > 0 else 0
    
    # Confidence Distribution
    strict_conf = df[df['genai_strict_label'] == 'Yes']['genai_strict_confidence'].value_counts()
    mod_conf = df[df['genai_moderate_label'] == 'Yes']['genai_moderate_confidence'].value_counts()
    len_conf = df[df['genai_lenient_label'] == 'Yes']['genai_lenient_confidence'].value_counts()
    
    # GenAI Functions Analysis
    all_functions = []
    for func_list in df['genai_functions_list'].dropna():
        if func_list:
            all_functions.extend([f.strip() for f in str(func_list).split(';') if f.strip()])
    
    func_counts = Counter(all_functions)
    top_functions = dict(func_counts.most_common(10))
    
    # Sample Companies with GenAI
    genai_companies = df[df['genai_strict_label'] == 'Yes'][['company_name', 'genai_functions_list', 'genai_strict_confidence']].head(20)
    
    # Confidence by Adoption Level
    conf_by_level = {
        'Strict': df[df['genai_strict_label'] == 'Yes']['genai_strict_confidence'].value_counts().to_dict(),
        'Moderate': df[df['genai_moderate_label'] == 'Yes']['genai_moderate_confidence'].value_counts().to_dict(),
        'Lenient': df[df['genai_lenient_label'] == 'Yes']['genai_lenient_confidence'].value_counts().to_dict()
    }
    
    # Prepare data for charts (convert numpy types to native Python types for JSON)
    # Exclude "No Evidence" from the adoption chart
    adoption_labels = ['Strict', 'Moderate', 'Lenient']
    adoption_counts = [int(strict_yes), int(mod_yes), int(len_yes)]
    adoption_pcts = [float(strict_pct), float(mod_pct), float(len_pct)]
    
    # Confidence data
    conf_levels = ['Low', 'Medium', 'High']
    strict_conf_data = [int(strict_conf.get(level, 0)) for level in conf_levels]
    mod_conf_data = [int(mod_conf.get(level, 0)) for level in conf_levels]
    len_conf_data = [int(len_conf.get(level, 0)) for level in conf_levels]
    
    # Confidence statistics
    strict_total_conf = sum(strict_conf_data)
    mod_total_conf = sum(mod_conf_data)
    len_total_conf = sum(len_conf_data)
    
    strict_conf_pcts = [(x / strict_total_conf * 100) if strict_total_conf > 0 else 0 for x in strict_conf_data]
    mod_conf_pcts = [(x / mod_total_conf * 100) if mod_total_conf > 0 else 0 for x in mod_conf_data]
    len_conf_pcts = [(x / len_total_conf * 100) if len_total_conf > 0 else 0 for x in len_conf_data]
    
    # Functions data
    func_labels = list(top_functions.keys())
    func_values = [int(v) for v in top_functions.values()]
    
    # --- Generate HTML ---
    
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GenAI Adoption Statistical Summary Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --primary: #1d1d1f;
            --secondary: #86868b;
            --bg: #fbfbfd;
            --card: #FFFFFF;
            --text: #1d1d1f;
            --accent-1: #0071e3;
            --accent-2: #007aff;
            --accent-3: #5ac8fa;
            --success: #30d158;
            --warning: #ff9500;
            --danger: #ff3b30;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Helvetica Neue', Helvetica, Arial, sans-serif;
            background: var(--bg); 
            color: var(--text); 
            padding: 40px 20px;
            line-height: 1.47059;
            font-weight: 400;
            letter-spacing: -0.022em;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        
        header {{ 
            margin-bottom: 40px; 
            border-bottom: 3px solid var(--primary); 
            padding-bottom: 20px; 
        }}
        h1 {{ 
            color: var(--primary); 
            margin: 0; 
            font-size: 32px; 
            font-weight: 700;
        }}
        .subtitle {{ 
            color: #666; 
            margin-top: 10px; 
            font-size: 16px; 
        }}
        .timestamp {{
            color: #999;
            font-size: 14px;
            margin-top: 5px;
        }}

        .kpi-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 20px; 
            margin-bottom: 40px; 
        }}
        .kpi-card {{ 
            background: var(--card); 
            padding: 28px; 
            border-radius: 18px; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.08); 
            border: 1px solid rgba(0,0,0,0.05);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        .kpi-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        }}
        .kpi-card.strict {{ 
            border-top: 3px solid #0071e3;
            background: linear-gradient(135deg, #ffffff 0%, #f5f9ff 100%);
        }}
        .kpi-card.moderate {{ 
            border-top: 3px solid #007aff;
            background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%);
        }}
        .kpi-card.lenient {{ 
            border-top: 3px solid #5ac8fa;
            background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
        }}
        .kpi-card.no-evidence {{ 
            border-top: 3px solid #d2d2d7;
            background: linear-gradient(135deg, #ffffff 0%, #fafafa 100%);
        }}
        .kpi-val {{ 
            font-size: 36px; 
            font-weight: 700; 
            color: var(--primary); 
            margin-bottom: 5px;
        }}
        .kpi-label {{ 
            font-size: 13px; 
            text-transform: uppercase; 
            color: #666; 
            font-weight: 600; 
            letter-spacing: 0.5px;
        }}
        .kpi-pct {{
            font-size: 14px;
            color: #999;
            margin-top: 5px;
        }}
        
        .chart-grid {{ 
            display: grid; 
            grid-template-columns: repeat(2, 1fr); 
            gap: 30px; 
            margin-bottom: 40px; 
        }}
        .chart-card {{ 
            background: var(--card); 
            padding: 32px; 
            border-radius: 18px; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border: 1px solid rgba(0,0,0,0.05);
        }}
        .chart-card h3 {{ 
            margin-top: 0; 
            font-size: 18px; 
            color: var(--primary); 
            margin-bottom: 20px; 
            border-bottom: 2px solid #eee; 
            padding-bottom: 10px; 
            font-weight: 600;
        }}
        .full-width {{ grid-column: 1 / -1; }}
        .chart-container {{
            position: relative;
            height: 350px;
        }}
        
        .table-card {{
            background: var(--card);
            padding: 32px;
            border-radius: 18px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border: 1px solid rgba(0,0,0,0.05);
            margin-bottom: 40px;
        }}
        .table-card h3 {{
            margin-top: 0;
            font-size: 18px;
            color: var(--primary);
            margin-bottom: 20px;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
            font-weight: 600;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: var(--bg);
            font-weight: 600;
            color: var(--primary);
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        
        .summary-box {{
            background: linear-gradient(135deg, #1d1d1f 0%, #2d2d2f 100%);
            color: white;
            padding: 40px;
            border-radius: 20px;
            margin-bottom: 40px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.12);
        }}
        .summary-box h2 {{
            margin: 0 0 15px 0;
            font-size: 24px;
        }}
        .summary-box p {{
            margin: 10px 0;
            font-size: 16px;
            line-height: 1.8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>GENAI Adoption Statistical Summary Dashboard</h1>
            <div class="subtitle">Analysis of {total:,} Startups from Crunchbase Dataset</div>
        </header>
        
        <div class="summary-box">
            <h2>Executive Summary</h2>
            <p><strong>Dataset Size:</strong> {total:,} startups analyzed</p>
            <p><strong>Strict GenAI Adoption:</strong> {strict_yes:,} companies ({strict_pct:.1f}%) - Explicit evidence of GenAI use</p>
            <p><strong>Moderate GenAI Adoption:</strong> {mod_yes:,} companies ({mod_pct:.1f}%) - Strongly implied GenAI use</p>
            <p><strong>Lenient GenAI Adoption:</strong> {len_yes:,} companies ({len_pct:.1f}%) - Possible GenAI use</p>
            <p><strong>No Evidence:</strong> {no_evidence:,} companies ({no_ev_pct:.1f}%) - No clear GenAI adoption detected</p>
            <hr style="margin: 20px 0; border: none; border-top: 1px solid rgba(255,255,255,0.3);">
            <h3 style="margin-top: 20px; margin-bottom: 10px;">Confidence Analysis</h3>
            <p><strong>Strict Mode:</strong> {strict_conf_data[2]:,} High ({strict_conf_pcts[2]:.1f}%), {strict_conf_data[1]:,} Medium ({strict_conf_pcts[1]:.1f}%), {strict_conf_data[0]:,} Low ({strict_conf_pcts[0]:.1f}%)</p>
            <p><strong>Moderate Mode:</strong> {mod_conf_data[2]:,} High ({mod_conf_pcts[2]:.1f}%), {mod_conf_data[1]:,} Medium ({mod_conf_pcts[1]:.1f}%), {mod_conf_data[0]:,} Low ({mod_conf_pcts[0]:.1f}%)</p>
            <p><strong>Lenient Mode:</strong> {len_conf_data[2]:,} High ({len_conf_pcts[2]:.1f}%), {len_conf_data[1]:,} Medium ({len_conf_pcts[1]:.1f}%), {len_conf_data[0]:,} Low ({len_conf_pcts[0]:.1f}%)</p>
        </div>
        
        <div class="kpi-grid">
            <div class="kpi-card strict">
                <div class="kpi-val">{strict_yes:,}</div>
                <div class="kpi-label">Strict Adoption</div>
                <div class="kpi-pct">{strict_pct:.1f}% of total</div>
            </div>
            <div class="kpi-card moderate">
                <div class="kpi-val">{mod_yes:,}</div>
                <div class="kpi-label">Moderate Adoption</div>
                <div class="kpi-pct">{mod_pct:.1f}% of total</div>
            </div>
            <div class="kpi-card lenient">
                <div class="kpi-val">{len_yes:,}</div>
                <div class="kpi-label">Lenient Adoption</div>
                <div class="kpi-pct">{len_pct:.1f}% of total</div>
            </div>
            <div class="kpi-card no-evidence">
                <div class="kpi-val">{no_evidence:,}</div>
                <div class="kpi-label">No Evidence</div>
                <div class="kpi-pct">{no_ev_pct:.1f}% of total</div>
            </div>
        </div>
        
        <div class="chart-grid">
            <div class="chart-card">
                <h3>Adoption Rate Comparison</h3>
                <div class="chart-container">
                    <canvas id="adoptionChart"></canvas>
                </div>
            </div>
            
            <div class="chart-card">
                <h3>Confidence Distribution (Strict Mode)</h3>
                <div class="chart-container">
                    <canvas id="confidenceChart"></canvas>
                </div>
            </div>
            
            <div class="chart-card full-width">
                <h3>Confidence Comparison Across All Modes</h3>
                <div class="chart-container">
                    <canvas id="confidenceComparisonChart"></canvas>
                </div>
            </div>
            
            <div class="chart-card full-width">
                <h3>Top GenAI Business Functions</h3>
                <div class="chart-container">
                    <canvas id="functionsChart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="table-card">
            <h3>Confidence Statistics by Adoption Mode</h3>
            <table>
                <thead>
                    <tr>
                        <th>Adoption Mode</th>
                        <th>Total Companies</th>
                        <th>High Confidence</th>
                        <th>Medium Confidence</th>
                        <th>Low Confidence</th>
                        <th>High %</th>
                        <th>Medium %</th>
                        <th>Low %</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Strict</strong></td>
                        <td>{strict_total_conf:,}</td>
                        <td>{strict_conf_data[2]:,}</td>
                        <td>{strict_conf_data[1]:,}</td>
                        <td>{strict_conf_data[0]:,}</td>
                        <td>{strict_conf_pcts[2]:.1f}%</td>
                        <td>{strict_conf_pcts[1]:.1f}%</td>
                        <td>{strict_conf_pcts[0]:.1f}%</td>
                    </tr>
                    <tr>
                        <td><strong>Moderate</strong></td>
                        <td>{mod_total_conf:,}</td>
                        <td>{mod_conf_data[2]:,}</td>
                        <td>{mod_conf_data[1]:,}</td>
                        <td>{mod_conf_data[0]:,}</td>
                        <td>{mod_conf_pcts[2]:.1f}%</td>
                        <td>{mod_conf_pcts[1]:.1f}%</td>
                        <td>{mod_conf_pcts[0]:.1f}%</td>
                    </tr>
                    <tr>
                        <td><strong>Lenient</strong></td>
                        <td>{len_total_conf:,}</td>
                        <td>{len_conf_data[2]:,}</td>
                        <td>{len_conf_data[1]:,}</td>
                        <td>{len_conf_data[0]:,}</td>
                        <td>{len_conf_pcts[2]:.1f}%</td>
                        <td>{len_conf_pcts[1]:.1f}%</td>
                        <td>{len_conf_pcts[0]:.1f}%</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <div class="table-card">
            <h3>Sample Companies with GenAI Adoption (Strict Mode)</h3>
            <table>
                <thead>
                    <tr>
                        <th>Company Name</th>
                        <th>GenAI Functions</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Add table rows
    for _, row in genai_companies.iterrows():
        company_name = str(row['company_name']).replace("'", "&#39;")
        functions = str(row['genai_functions_list']).replace("'", "&#39;") if pd.notna(row['genai_functions_list']) else "N/A"
        confidence = str(row['genai_strict_confidence'])
        html_template += f"""
                    <tr>
                        <td>{company_name}</td>
                        <td>{functions}</td>
                        <td>{confidence}</td>
                    </tr>
"""
    
    html_template += f"""
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        // 1. Adoption Rate Chart
        new Chart(document.getElementById('adoptionChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(adoption_labels)},
                datasets: [{{
                    label: 'Number of Companies',
                    data: {json.dumps(adoption_counts)},
                    backgroundColor: [
                        '#0071e3',  // Strict - Apple Blue
                        '#007aff',  // Moderate - System Blue
                        '#5ac8fa'   // Lenient - Sky Blue
                    ],
                    borderRadius: 12
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                let label = context.dataset.label || '';
                                if (label) {{
                                    label += ': ';
                                }}
                                label += context.parsed.y.toLocaleString();
                                label += ' (' + ({json.dumps(adoption_pcts)})[context.dataIndex].toFixed(1) + '%)';
                                return label;
                            }}
                        }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        ticks: {{
                            callback: function(value) {{
                                return value.toLocaleString();
                            }}
                        }}
                    }}
                }}
            }}
        }});
        
        // 2. Confidence Distribution Chart
        new Chart(document.getElementById('confidenceChart'), {{
            type: 'doughnut',
            data: {{
                labels: {json.dumps(conf_levels)},
                datasets: [{{
                    data: {json.dumps(strict_conf_data)},
                    backgroundColor: [
                        '#d1d1d6',  // Low - Light Gray
                        '#8e8e93',  // Medium - Medium Gray
                        '#0071e3'   // High - Apple Blue
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'bottom'
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                let label = context.label || '';
                                if (label) {{
                                    label += ': ';
                                }}
                                label += context.parsed;
                                return label;
                            }}
                        }}
                    }}
                }}
            }}
        }});
        
        // 3. Confidence Comparison Chart
        new Chart(document.getElementById('confidenceComparisonChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(conf_levels)},
                datasets: [
                    {{
                        label: 'Strict Mode',
                        data: {json.dumps(strict_conf_data)},
                        backgroundColor: '#0071e3',
                        borderRadius: 12
                    }},
                    {{
                        label: 'Moderate Mode',
                        data: {json.dumps(mod_conf_data)},
                        backgroundColor: '#007aff',
                        borderRadius: 12
                    }},
                    {{
                        label: 'Lenient Mode',
                        data: {json.dumps(len_conf_data)},
                        backgroundColor: '#5ac8fa',
                        borderRadius: 12
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'top',
                        display: true
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                let label = context.dataset.label || '';
                                if (label) {{
                                    label += ': ';
                                }}
                                label += context.parsed.y.toLocaleString();
                                let total = 0;
                                if (context.dataset.label === 'Strict Mode') {{
                                    total = {strict_total_conf};
                                }} else if (context.dataset.label === 'Moderate Mode') {{
                                    total = {mod_total_conf};
                                }} else {{
                                    total = {len_total_conf};
                                }}
                                if (total > 0) {{
                                    let pct = (context.parsed.y / total * 100).toFixed(1);
                                    label += ' (' + pct + '%)';
                                }}
                                return label;
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        stacked: false,
                        title: {{
                            display: true,
                            text: 'Confidence Level'
                        }}
                    }},
                    y: {{
                        beginAtZero: true,
                        stacked: false,
                        title: {{
                            display: true,
                            text: 'Number of Companies'
                        }},
                        ticks: {{
                            callback: function(value) {{
                                return value.toLocaleString();
                            }}
                        }}
                    }}
                }}
            }}
        }});
        
        // 4. Functions Chart
        new Chart(document.getElementById('functionsChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(func_labels)},
                datasets: [{{
                    label: 'Number of Companies',
                    data: {json.dumps(func_values)},
                    backgroundColor: '#007aff',
                    borderRadius: 12
                }}]
            }},
            options: {{
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                return context.parsed.x.toLocaleString() + ' companies';
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        beginAtZero: true,
                        ticks: {{
                            callback: function(value) {{
                                return value.toLocaleString();
                            }}
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    # Ensure insights directory exists
    os.makedirs(INSIGHTS_DIR, exist_ok=True)
    
    # Write HTML file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"\n[SUCCESS] Dashboard generated: {OUTPUT_FILE}")
    print(f"[INFO] Open the file in your browser to view the dashboard")

if __name__ == "__main__":
    try:
        generate_dashboard()
    except Exception as e:
        print(f"[ERROR] Failed to generate dashboard: {e}")
        import traceback
        traceback.print_exc()

