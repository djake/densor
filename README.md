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

`notebooks/explore.py` is the notebook, stored as a plain Python file (jupytext "percent" format) so that running it never changes a tracked file. It reads the Parquet files for every site from the bucket with DuckDB and plots them. Each Pi uploads under its own prefix (`S3_PREFIX` on the Pi); the notebook lists them as `site=prefix` pairs in `S3_SITES`.

```
cd notebooks
cp densor-read.env.example .env      # then fill in a read-only Tigris key
uv venv && uv pip install -r requirements.txt
uv run jupytext --sync explore.py    # creates explore.ipynb (gitignored), paired with explore.py
uv run jupyter lab explore.ipynb
```

Edits made in JupyterLab are saved to both files; outputs only ever land in the ignored `.ipynb`. Edit `explore.py` and commit that.

Use a separate read-only key (Viewer role on the bucket), never the Pi's write key. `.env` and the generated `.ipynb` are gitignored.
