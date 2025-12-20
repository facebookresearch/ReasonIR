import json
import os
import re
from datetime import datetime
from typing import Dict, Any, List, Optional


OUTPUTS_DIR = "outputs"
OUTPUT_FILENAME = "compiled_results.csv"


TIMESTAMP_PATTERN = re.compile(
    r"^(?:evaluation_results_)?(?P<model_safe>.+)_(?P<ts>\d{8}_\d{6})\.json$"
)


def _extract_timestamp(filename: str) -> Optional[datetime]:
    match = TIMESTAMP_PATTERN.match(filename)
    if not match:
        return None
    ts_str = match.group("ts")
    try:
        return datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def load_latest_results(outputs_dir: str) -> Dict[str, Dict[str, Any]]:
    """Return latest metrics per model from JSON results in outputs_dir."""

    latest_by_model: Dict[str, Dict[str, Any]] = {}

    for filename in os.listdir(outputs_dir):
        if not filename.endswith(".json"):
            continue

        ts = _extract_timestamp(filename)
        if ts is None:
            continue

        filepath = os.path.join(outputs_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        model = data.get("model")
        if not model:
            continue

        stored = latest_by_model.get(model)
        if stored is None or ts > stored["ts"]:
            latest_by_model[model] = {"metrics": data, "ts": ts}

    return {model: entry["metrics"] for model, entry in latest_by_model.items()}


def compile_results(latest_results: Dict[str, Dict[str, Any]]) -> List[str]:
    """Compile results into CSV lines (header + rows)."""

    all_metric_keys = set()
    for metrics in latest_results.values():
        all_metric_keys.update(key for key in metrics.keys() if key != "model")

    ordered_metrics = sorted(all_metric_keys)
    header = ["model"] + ordered_metrics
    lines = [",".join(header)]

    for model, metrics in sorted(latest_results.items()):
        row = [model]
        for key in ordered_metrics:
            value = metrics.get(key, "")
            row.append(str(value))
        lines.append(",".join(row))

    return lines


def main() -> None:
    if not os.path.isdir(OUTPUTS_DIR):
        raise FileNotFoundError(f"Outputs directory not found: {OUTPUTS_DIR}")

    latest_results = load_latest_results(OUTPUTS_DIR)
    if not latest_results:
        raise RuntimeError("No valid results found in outputs directory")

    csv_lines = compile_results(latest_results)

    output_path = os.path.join(OUTPUTS_DIR, OUTPUT_FILENAME)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines) + "\n")

    print(f"Compiled results written to {output_path}")


if __name__ == "__main__":
    main()
