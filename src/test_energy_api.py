
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# Project paths
ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
OUTPUT_DIR = ROOT / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "live_energy_data.csv"

# Load API key
load_dotenv(ENV_PATH)
API_KEY = os.getenv("ELECTRICITY_MAPS_API_KEY")

if not API_KEY:
    raise SystemExit(
        "ERROR: API key not found. Check the .env file."
    )

# Electricity Maps latest power-breakdown endpoint
URL = (
    "https://api.electricitymaps.com/v3/"
    "power-breakdown/latest"
)

HEADERS = {"auth-token": API_KEY}
PARAMS = {"zone": "IN-SO"}


def fetch_live_energy():
    print("Requesting electricity data for Southern India...")

    try:
        response = requests.get(
            URL,
            headers=HEADERS,
            params=PARAMS,
            timeout=30,
        )

        print("HTTP status:", response.status_code)

        if response.status_code != 200:
            print("API request failed.")
            print("Response:", response.text[:500])
            return

        data = response.json()

        timestamp = data.get("datetime")
        zone = data.get("zone")
        total_load = data.get("powerConsumptionTotal")
        breakdown = data.get("powerConsumptionBreakdown")

        print("\nAPI request successful!")
        print("Zone:", zone)
        print("Timestamp (UTC):", timestamp)
        print("Total load (MW):", total_load)

        if breakdown:
            print("\nPower consumption breakdown:")
            for source, value in breakdown.items():
                print(f"  {source}: {value} MW")

        # Prepare one record from this API response.
        record = {
            "timestamp": timestamp,
            "zone": zone,
            "total_load_mw": total_load,
        }

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Avoid appending the same timestamp twice.
        if OUTPUT_FILE.exists():
            existing = pd.read_csv(OUTPUT_FILE)

            if (
                "timestamp" in existing.columns
                and "zone" in existing.columns
                and (
                    (existing["timestamp"] == timestamp)
                    & (existing["zone"] == zone)
                ).any()
            ):
                print(
                    "\nThis timestamp and zone are already saved."
                )
                print("CSV:", OUTPUT_FILE)
                return

        pd.DataFrame([record]).to_csv(
            OUTPUT_FILE,
            mode="a",
            header=not OUTPUT_FILE.exists(),
            index=False,
        )

        print("\nReading saved successfully!")
        print("CSV path:", OUTPUT_FILE)

    except requests.RequestException as error:
        print("Network/API connection error:", error)

    except (ValueError, OSError) as error:
        print("Data processing or file error:", error)


if __name__ == "__main__":
    fetch_live_energy()
