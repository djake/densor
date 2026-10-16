# Densor

Code for a project to gather and analyze IoT sensor data: many low-cost wireless
temperature and humidity sensors in a home, logged on a Raspberry Pi and
uploaded to cloud storage for analysis.

- `logger.py`: subscribes to Zigbee2MQTT over MQTT and writes readings to a local SQLite database.
- `sync.py`: uploads each closed hour of readings to an S3-compatible bucket as Parquet.
- `tools/backfill.py`: one-off load of earlier readings (from a Zigbee2MQTT-log CSV) into the local database.
- `systemd/`: service and timer units that run both on the Pi.

Configuration is through environment variables (systemd reads `/etc/densor.env`):
`S3_BUCKET` (required for `sync.py`), `S3_PREFIX`, `S3_ENDPOINT`, `S3_ADDRESSING_STYLE`
(default `virtual`), `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `DENSOR_DB`,
`MQTT_HOST`, `MQTT_PORT`. See `densor.env.example`.

All rights reserved. See `LICENSE`.

`writeup/` contains the LaTeX source of the project writeup (`cd writeup && latexmk -pdf main.tex`).

## Exploring the data

`notebooks/explore.ipynb` reads the Parquet files from the bucket with DuckDB and plots them.

```
cd notebooks
cp densor-read.env.example .env      # then fill in a read-only Tigris key
uv venv && uv pip install -r requirements.txt
uv run jupyter lab explore.ipynb
```

Use a separate read-only key (Viewer role on the bucket), never the Pi's write key. `.env` is gitignored. Keep notebook outputs out of commits.
