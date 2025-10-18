from flask import Flask, render_template, jsonify
import os, sys, json
from datetime import datetime

# allow importing from parent folder
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from smart_predictions import run_smart_predictions

app = Flask(__name__)

DATA_FILE = "daily_forecast.json"


def load_or_generate_forecast():
    """Load cached forecast or generate a new one if outdated."""
    if os.path.exists(DATA_FILE):
        mtime = datetime.fromtimestamp(os.path.getmtime(DATA_FILE))
        # reuse forecast if generated within 6 hours
        if (datetime.now() - mtime).total_seconds() < 6 * 3600:
            print("♻️ Using cached forecast file.")
            with open(DATA_FILE, "r") as f:
                return json.load(f)

    print("🔮 Generating fresh forecast...")
    data = run_smart_predictions()
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return data


@app.route("/")
def home():
    data = load_or_generate_forecast()

    # ensure older forecasts (without feels_like) still render correctly
    for city, cdata in data.items():
        for f in cdata.get("forecast", []):
            if "predicted_feels_like" not in f:
                f["predicted_feels_like"] = f.get("final_temperature", 0)

    return render_template("index.html", data=data)


@app.route("/api/forecast")
def api_forecast():
    data = load_or_generate_forecast()
    return jsonify(data)


if __name__ == "__main__":
    print("🚀 Starting Flask Weather Prediction Dashboard...")
    app.run(debug=True)
