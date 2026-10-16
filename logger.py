"""Log Zigbee2MQTT sensor readings from MQTT into a local SQLite database."""

import datetime as dt
import json
import os
import sqlite3

import paho.mqtt.client as mqtt

DB_PATH = os.environ.get("DENSOR_DB", "/var/lib/densor/densor.db")
MQTT_HOST = os.environ.get("MQTT_HOST", "localhost")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
BASE = os.environ.get("Z2M_BASE_TOPIC", "zigbee2mqtt")
FIELDS = ("temperature", "humidity", "battery", "linkquality")

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    ts_utc TEXT NOT NULL,          -- arrival time, UTC, fixed-width ISO 8601
    ieee TEXT NOT NULL,            -- permanent device ID
    temperature REAL,              -- degrees C
    humidity REAL,
    battery INTEGER,
    linkquality INTEGER
);
CREATE INDEX IF NOT EXISTS readings_ts ON readings (ts_utc);
CREATE TABLE IF NOT EXISTS devices (
    ieee TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT               -- free text from the Zigbee2MQTT device description
);
"""


def open_db(path=DB_PATH):
    db = sqlite3.connect(path, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.executescript(SCHEMA)
    migrate(db)
    return db


def migrate(db):
    """Rename devices.location to devices.description on databases created earlier."""
    cols = [row[1] for row in db.execute("PRAGMA table_info(devices)")]
    if "location" in cols and "description" not in cols:
        try:
            db.execute("ALTER TABLE devices RENAME COLUMN location TO description")
            db.commit()
        except sqlite3.OperationalError:
            pass  # another process migrated first


def utc_now():
    return dt.datetime.now(dt.timezone.utc)


def stamp(t):
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def update_devices(db, names, payload):
    """Handle the retained <base>/bridge/devices message."""
    for device in json.loads(payload):
        if device.get("type") == "Coordinator":
            continue
        ieee, name = device["ieee_address"], device["friendly_name"]
        # Top-level device description (set in the Zigbee2MQTT frontend), not the
        # model description under "definition". Zigbee2MQTT is authoritative, so an
        # empty description clears the stored value.
        description = (device.get("description") or "").strip() or None
        db.execute(
            "INSERT INTO devices (ieee, name, description) VALUES (?, ?, ?) "
            "ON CONFLICT(ieee) DO UPDATE SET name = excluded.name, "
            "description = excluded.description",
            (ieee, name, description),
        )
        names[name] = ieee
    db.commit()


def handle_message(db, names, topic, payload, now=None):
    if topic == f"{BASE}/bridge/devices":
        update_devices(db, names, payload)
        return
    name = topic[len(BASE) + 1:]
    if not topic.startswith(BASE + "/") or "/" in name:
        return  # bridge/*, <name>/availability, and so on
    ieee = names.get(name)
    if ieee is None:
        return
    try:
        data = json.loads(payload)
    except ValueError:
        return
    if not isinstance(data, dict):
        return
    values = [data.get(f) for f in FIELDS]
    if all(v is None for v in values):
        return
    db.execute(
        "INSERT INTO readings (ts_utc, ieee, temperature, humidity, battery, linkquality) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (stamp(now or utc_now()), ieee, *values),
    )
    db.commit()  # small and often: a power cut loses at most the last few readings


def main():
    db = open_db()
    names = dict(db.execute("SELECT name, ieee FROM devices"))

    def on_connect(client, userdata, flags, reason_code, properties):
        client.subscribe([(f"{BASE}/+", 0), (f"{BASE}/bridge/devices", 0)])

    def on_message(client, userdata, msg):
        handle_message(db, names, msg.topic, msg.payload)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=60)
    client.connect_async(MQTT_HOST, MQTT_PORT)
    client.loop_forever(retry_first_connection=True)


if __name__ == "__main__":
    main()
