import json
from pathlib import Path
from loguru import logger

from monitoring.evidently_report import main as generate_report
from src.retraining import AutoRetrainingPipeline


def check_and_alert():
    logger.info("Checking data drift...")
    generate_report()
    
    report_file = Path("monitoring/reports/latest_drift_report.json")
    if not report_file.exists():
        logger.error("Drift report not found.")
        return

    with open(report_file, "r", encoding="utf-8") as f:
        report = json.load(f)

    drift_detected = report.get("data_drift_detected", False)

    if drift_detected:
        logger.warning("Data drift detected, triggering retraining pipeline...")
        pipeline = AutoRetrainingPipeline()
        promoted = pipeline.trigger_full_cycle()
        if promoted:
            logger.info("New model promoted and deployed.")
        else:
            logger.info("Current model retained.")
    else:
        logger.info("No significant drift detected.")


if __name__ == "__main__":
    check_and_alert()
