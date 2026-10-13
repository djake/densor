"""Export closed hours of readings to Parquet and upload them to an S3-compatible bucket.

Run hourly (see systemd/). Each hour is uploaded once, as an immutable object, and
recorded in synced_hours. A failed upload is retried on the next run.
"""

import datetime as dt
import json
import os
import tempfile

import boto3
import pyarrow as pa
import pyarrow.parquet as pq

from logger import open_db, utc_now

BUCKET = os.environ["S3_BUCKET"]
PREFIX = os.environ.get("S3_PREFIX", "densor").strip("/")
ENDPOINT = os.environ.get("S3_ENDPOINT")  # leave unset for AWS S3

SCHEMA = pa.schema([
    ("ts_utc", pa.string()),
    ("ieee", pa.string()),
    ("temperature", pa.float64()),
    ("humidity", pa.float64()),
    ("battery", pa.int64()),
    ("linkquality", pa.int64()),
])


def closed_hours(db, now):
    """Hours (YYYY-MM-DDTHH, UTC) that have ended and are not yet uploaded."""
    current = now.strftime("%Y-%m-%dT%H")
    rows = db.execute(
        "SELECT DISTINCT substr(ts_utc, 1, 13) FROM readings "
        "WHERE substr(ts_utc, 1, 13) < ? "
        "AND substr(ts_utc, 1, 13) NOT IN (SELECT hour FROM synced_hours) "
        "ORDER BY 1",
        (current,),
    )
    return [r[0] for r in rows]


def export_hour(db, hour, path):
    rows = db.execute(
        "SELECT ts_utc, ieee, temperature, humidity, battery, linkquality "
        "FROM readings WHERE substr(ts_utc, 1, 13) = ? ORDER BY ts_utc",
        (hour,),
    ).fetchall()
    columns = list(zip(*rows)) if rows else [[]] * len(SCHEMA)
    table = pa.Table.from_arrays(
        [pa.array(list(col), type=f.type) for col, f in zip(columns, SCHEMA)],
        schema=SCHEMA,
    )
    pq.write_table(table, path)
    return len(rows)


def object_key(hour):
    return f"{PREFIX}/readings/date={hour[:10]}/hour={hour[11:13]}.parquet"


def main():
    db = open_db()
    db.execute(
        "CREATE TABLE IF NOT EXISTS synced_hours "
        "(hour TEXT PRIMARY KEY, rows INTEGER, uploaded_at TEXT)"
    )
    s3 = boto3.client("s3", endpoint_url=ENDPOINT)

    devices = [
        {"ieee": i, "name": n, "location": loc}
        for i, n, loc in db.execute("SELECT ieee, name, location FROM devices")
    ]
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{PREFIX}/devices.json",
        Body=json.dumps(devices, indent=2).encode(),
        ContentType="application/json",
    )

    for hour in closed_hours(db, utc_now()):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "hour.parquet")
            count = export_hour(db, hour, path)
            s3.upload_file(path, BUCKET, object_key(hour))
        db.execute(
            "INSERT INTO synced_hours (hour, rows, uploaded_at) VALUES (?, ?, ?)",
            (hour, count, utc_now().isoformat()),
        )
        db.commit()
        print(f"uploaded {hour} ({count} rows)")


if __name__ == "__main__":
    main()
