"""Load readings from a Zigbee2MQTT-log CSV into the local SQLite database.

One-off tool for data collected before the logger was running. Columns:
timestamp (ISO 8601 with offset), device (friendly name), temperature_c,
humidity_pct, battery_pct, linkquality. Friendly names are mapped to IEEE
addresses using the devices table, so the logger must have seen the devices
(it fills the table from Zigbee2MQTT) before this is run.

Rows already in the database (same ts_utc and ieee) are skipped, so the script is
safe to run twice. The hourly upload job then uploads the backfilled hours.

    python tools/backfill.py readings.csv --dry-run
    python tools/backfill.py readings.csv
"""

import argparse
import csv
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from logger import open_db, stamp  # noqa: E402


def to_utc_stamp(text):
    t = dt.datetime.fromisoformat(text)
    if t.tzinfo is None:
        raise ValueError(f"timestamp has no UTC offset: {text}")
    return stamp(t.astimezone(dt.timezone.utc))


def number(text, kind):
    return None if text in ("", None) else kind(float(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("csv_path")
    parser.add_argument("--dry-run", action="store_true", help="report, but do not write")
    args = parser.parse_args()

    db = open_db()
    names = dict(db.execute("SELECT name, ieee FROM devices"))

    inserted = skipped = 0
    unknown = set()
    first = last = None
    with open(args.csv_path, newline="") as f:
        for row in csv.DictReader(f):
            ieee = names.get(row["device"])
            if ieee is None:
                unknown.add(row["device"])
                continue
            ts = to_utc_stamp(row["timestamp"])
            exists = db.execute(
                "SELECT 1 FROM readings WHERE ts_utc = ? AND ieee = ?", (ts, ieee)
            ).fetchone()
            if exists:
                skipped += 1
                continue
            db.execute(
                "INSERT INTO readings (ts_utc, ieee, temperature, humidity, battery, linkquality) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    ts,
                    ieee,
                    number(row["temperature_c"], float),
                    number(row["humidity_pct"], float),
                    number(row["battery_pct"], int),
                    number(row["linkquality"], int),
                ),
            )
            inserted += 1
            first = ts if first is None or ts < first else first
            last = ts if last is None or ts > last else last

    if unknown:
        db.rollback()
        sys.exit(f"no device with these names in the devices table: {sorted(unknown)}; nothing written")

    earliest_live = db.execute(
        "SELECT MIN(ts_utc) FROM readings WHERE ts_utc > ?", (last or "",)
    ).fetchone()[0]

    if args.dry_run:
        db.rollback()
    else:
        db.commit()

    print(f"{'would insert' if args.dry_run else 'inserted'} {inserted}, skipped {skipped} duplicates")
    if first:
        print(f"range {first} to {last} (UTC)")
        if earliest_live:
            print(f"note: the database already has readings after this range, from {earliest_live}")


if __name__ == "__main__":
    main()
