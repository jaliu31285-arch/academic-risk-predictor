"""Demo client that exercises the Flask API at http://127.0.0.1:5000."""

import json
import sys
from urllib import request as _ur
from urllib.error import HTTPError, URLError

BASE_URL = "http://127.0.0.1:5000"
TIMEOUT = 30

RISK_MAP = {0: "Low", 1: "Medium", 2: "High"}


def _http(method, path, body=None):
    url = BASE_URL + path
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = _ur.Request(url, data=data, method=method, headers=headers)
    try:
        with _ur.urlopen(req, timeout=TIMEOUT) as resp:
            raw = resp.read()
            status = resp.getcode()
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if "json" in content_type and raw:
                try:
                    return status, json.loads(raw.decode("utf-8"))
                except ValueError:
                    pass
            return status, raw.decode("utf-8", errors="replace")
    except HTTPError as e:
        try:
            payload = e.read().decode("utf-8", errors="replace")
        except Exception:
            payload = ""
        try:
            return e.code, json.loads(payload)
        except ValueError:
            return e.code, payload
    except URLError as e:
        raise RuntimeError("Could not reach {0}: {1}".format(url, e))


def get(path):
    return _http("GET", path)


def post(path, body):
    return _http("POST", path, body)


def _ruler(widths, char="-"):
    return "+" + "+".join(char * (w + 2) for w in widths) + "+"


def _row(cells, widths):
    pieces = []
    for cell, w in zip(cells, widths):
        text = str(cell)
        pieces.append(" " + text + " " * (w - len(text)) + " ")
    return "|" + "|".join(pieces) + "|"


def print_table(header, rows):
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    print(_ruler(widths, "="))
    print(_row(header, widths))
    print(_ruler(widths, "="))
    for row in rows:
        print(_row(row, widths))
        print(_ruler(widths, "-"))


def print_bar(label, value, width=40, max_value=1.0):
    if max_value <= 0:
        filled = 0
    else:
        filled = int(round((value / max_value) * width))
    filled = max(0, min(width, filled))
    bar = "=" * filled + "-" * (width - filled)
    print(" {0:<40s} [{1}] {2:.4f}".format(label, bar, value))


def _risk_label(prediction):
    if isinstance(prediction, int) and prediction in RISK_MAP:
        return RISK_MAP[prediction]
    return str(prediction)


def _risk_column(response, level):
    probs = response.get("probabilities") or []
    if level < len(probs):
        return "{0:.4f}".format(probs[level])
    return "n/a"


def step1_status():
    print("\n=== Step 1: GET /api/status ===")
    status, body = get("/api/status")
    print("HTTP {0}".format(status))
    print(json.dumps(body, indent=2, sort_keys=True))


def step2_models():
    print("\n=== Step 2: GET /api/models ===")
    status, body = get("/api/models")
    print("HTTP {0}".format(status))
    print(json.dumps(body, indent=2, sort_keys=True))
    return body.get("available") or []


def step3_predict_all(models, label, features):
    print("\n=== Step {0}: POST /api/predict - {1} ===".format(label, features))
    rows = []
    for model in models:
        payload = {"model": model, "features": features}
        status, body = post("/api/predict", payload)
        if isinstance(body, dict) and body.get("success"):
            prediction = body.get("risk_level")
            rows.append([
                model,
                _risk_label(prediction),
                _risk_column(body, 0),
                _risk_column(body, 1),
                _risk_column(body, 2),
            ])
        else:
            rows.append([model, "error", "-", "-", "-"])
            print("  !! model={0} HTTP {1}: {2}".format(model, status, body))
    print_table(["Model", "Predicted", "Low", "Medium", "High"], rows)


def step5_random_forest():
    print("\n=== Step 5: POST /api/predict (model=random_forest) ===")
    features = {"f0": 1.5, "f1": -2.0, "f2": 0.8}
    payload = {"model": "random_forest", "features": features}
    status, body = post("/api/predict", payload)
    print("HTTP {0}".format(status))
    if isinstance(body, dict) and body.get("success"):
        print("Risk label : {0}".format(body.get("risk_label")))
        print("Risk level : {0}".format(body.get("risk_level")))
        print("Model label: {0}".format(body.get("model_label")))
        values = body.get("feature_values") or {}
        print("\nFeature values:")
        rows = [[k, "{0:.4f}".format(v)] for k, v in values.items()]
        print_table(["Feature", "Value"], rows)
        explanation = (body.get("explanation") or {}).get("key_factors") or []
        print("\nExplanation (key factors):")
        if explanation:
            rows = []
            for f in explanation:
                rows.append([
                    f.get("feature", ""),
                    "{0:.4f}".format(f.get("value", 0.0)),
                    "{0:.4f}".format(f.get("mean", 0.0)),
                    "{0:+.3f}".format(f.get("deviation", 0.0)),
                    f.get("direction", ""),
                ])
            print_table(["Feature", "Value", "Mean", "Deviation", "Direction"], rows)
        else:
            print("  (no strong deviations from the reference dataset mean)")
    else:
        print("Unexpected response: {0}".format(body))


def step6_feature_importance():
    print("\n=== Step 6: GET /api/feature_importance?model=random_forest ===")
    status, body = get("/api/feature_importance?model=random_forest")
    print("HTTP {0}".format(status))
    if isinstance(body, dict) and body.get("success"):
        features = body.get("features") or []
        importances = body.get("importances") or []
        pairs = list(zip(features, importances))
        pairs.sort(key=lambda p: p[1], reverse=True)
        top = pairs[:10]
        max_val = max((p[1] for p in top), default=1.0)
        print("\nTop 10 features (sorted by importance descending):\n")
        for name, imp in top:
            print_bar(name, imp, width=40, max_value=max_val)
        print()
        header = ["Rank", "Feature", "Importance"]
        rows = [[str(i + 1), name, "{0:.6f}".format(imp)] for i, (name, imp) in enumerate(top)]
        print_table(header, rows)
    else:
        print("Unexpected response: {0}".format(body))


def step7_pages():
    print("\n=== Step 7: GET HTML pages ===")
    paths = ["/", "/dashboard", "/predict", "/models", "/about"]
    rows = []
    for path in paths:
        status, _body = get(path)
        rows.append([path, str(status)])
    print_table(["Path", "HTTP Status"], rows)


def main():
    print("Demo API client - targeting {0}".format(BASE_URL))
    try:
        step1_status()
        models = step2_models()
        if not models:
            print("\nNo models reported by /api/models - skipping comparison steps.")
            return 1
        print("\nAvailable models: {0}".format(", ".join(models)))
        step3_predict_all(models, "3", {"f0": 1.5, "f1": -2.0, "f2": 0.8})
        step3_predict_all(models, "4", {"f0": 0.1, "f1": 0.2, "f2": 0.1})
        step5_random_forest()
        step6_feature_importance()
        step7_pages()
        print("\nAll demo steps finished successfully.")
        return 0
    except RuntimeError as exc:
        print("\nFATAL: {0}".format(exc))
        print("Make sure the Flask server is running at {0}".format(BASE_URL))
        return 2


if __name__ == "__main__":
    sys.exit(main())
