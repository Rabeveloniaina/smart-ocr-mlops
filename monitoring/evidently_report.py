import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from loguru import logger

from monitoring.metrics_store import load_production_data_as_dataframe


def generate_reference_dataset(n_samples: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "confidence": np.clip(np.random.normal(loc=93.5, scale=4.0, size=n_samples), 60, 100),
        "text_length": np.clip(np.random.normal(loc=15.0, scale=6.0, size=n_samples), 3, 50),
        "num_words": np.clip(np.random.poisson(lam=2.5, size=n_samples), 1, 8),
        "latency_ms": np.clip(np.random.normal(loc=35.0, scale=8.0, size=n_samples), 15, 120)
    })


def run_drift_analysis(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_dir: str = "monitoring/reports"
) -> Dict[str, Any]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    drift_results = {}
    total_drift_score = 0.0

    for col in ["confidence", "text_length", "num_words", "latency_ms"]:
        if col in reference_data.columns and col in current_data.columns:
            ref_mean = float(reference_data[col].mean())
            cur_mean = float(current_data[col].mean())
            ref_std = float(reference_data[col].std())
            cur_std = float(current_data[col].std())

            shift = abs(cur_mean - ref_mean) / max(0.001, ref_std)
            is_drift = bool(shift > 1.5)

            drift_results[col] = {
                "reference_mean": round(ref_mean, 2),
                "current_mean": round(cur_mean, 2),
                "drift_score": round(shift, 3),
                "drift_detected": is_drift
            }
            if is_drift:
                total_drift_score += 1.0

    has_drift = total_drift_score >= 1.0
    summary = {
        "timestamp": timestamp,
        "reference_samples": len(reference_data),
        "current_samples": len(current_data),
        "data_drift_detected": has_drift,
        "features": drift_results
    }

    json_path = Path(output_dir) / f"drift_report_{timestamp}.json"
    latest_json_path = Path(output_dir) / "latest_drift_report.json"
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(latest_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Monitoring Report</title>
    <style>
        body {{ font-family: sans-serif; background: #0f172a; color: #f8fafc; margin: 2rem; }}
        .container {{ max-width: 900px; margin: 0 auto; background: #1e293b; padding: 2rem; border-radius: 8px; }}
        h1 {{ color: #38bdf8; font-size: 1.5rem; }}
        .status-box {{ padding: 0.8rem; border-radius: 6px; margin-bottom: 1.5rem; }}
        .status-drift {{ background: rgba(239, 68, 68, 0.2); border: 1px solid #ef4444; color: #fca5a5; }}
        .status-healthy {{ background: rgba(34, 197, 94, 0.2); border: 1px solid #22c55e; color: #86efac; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #0f172a; color: #94a3b8; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Monitoring Report</h1>
        <p>Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
        <div class="status-box {'status-drift' if has_drift else 'status-healthy'}">
            {"Data drift detected" if has_drift else "No drift detected"}
        </div>
        <table>
            <tr>
                <th>Feature</th>
                <th>Reference Mean</th>
                <th>Current Mean</th>
                <th>Drift Score</th>
                <th>Status</th>
            </tr>
            {"".join([f"<tr><td><b>{k}</b></td><td>{v['reference_mean']}</td><td>{v['current_mean']}</td><td>{v['drift_score']}</td><td>{'Drift' if v['drift_detected'] else 'Normal'}</td></tr>" for k, v in drift_results.items()])}
        </table>
    </div>
</body>
</html>"""
    
    html_path = Path(output_dir) / f"drift_report_{timestamp}.html"
    latest_html_path = Path(output_dir) / "latest_drift_report.html"
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    with open(latest_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Report saved in {output_dir}")
    return summary


def main():
    ref_df = generate_reference_dataset(200)
    cur_df = load_production_data_as_dataframe()
    
    if len(cur_df) < 5:
        np.random.seed(99)
        cur_df = pd.DataFrame({
            "confidence": np.clip(np.random.normal(loc=86.0, scale=6.0, size=50), 50, 100),
            "text_length": np.clip(np.random.normal(loc=22.0, scale=8.0, size=50), 5, 60),
            "num_words": np.clip(np.random.poisson(lam=4.0, size=50), 1, 10),
            "latency_ms": np.clip(np.random.normal(loc=48.0, scale=12.0, size=50), 20, 150)
        })

    summary = run_drift_analysis(ref_df, cur_df)
    logger.info(f"Drift status: {summary['data_drift_detected']}")


if __name__ == "__main__":
    main()
