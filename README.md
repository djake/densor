# Densor

Code for a project to gather and analyze IoT sensor data: many low-cost wireless
temperature and humidity sensors in a home, logged on a Raspberry Pi and
uploaded to cloud storage for analysis.

- `logger.py`: subscribes to Zigbee2MQTT over MQTT and writes readings to a local SQLite database.
- `sync.py`: uploads each closed hour of readings to an S3-compatible bucket as Parquet.
- `systemd/`: service and timer units that run both on the Pi.

Configuration is through environment variables (systemd reads `/etc/densor.env`):
`S3_BUCKET` (required for `sync.py`), `S3_PREFIX`, `S3_ENDPOINT`, `S3_ADDRESSING_STYLE`
(default `virtual`), `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `DENSOR_DB`,
`MQTT_HOST`, `MQTT_PORT`. See `densor.env.example`.

All rights reserved. See `LICENSE`.

`writeup/` contains the LaTeX source of the project writeup (`cd writeup && latexmk -pdf main.tex`).
