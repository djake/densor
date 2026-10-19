# %% [markdown]
# # Densor: explore the readings
#
# Reads the hourly Parquet files and `devices.json` for every site straight from the Tigris bucket with DuckDB. Each site uploads under its own prefix (set in `S3_SITES`). Timestamps are stored in UTC and converted to local time here.
#
# Credentials come from `notebooks/.env` or the process environment. Use a read-only key.

# %%
import os
import duckdb
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()  # reads notebooks/.env if present; real environment variables also work

LOCAL_TZ = "America/New_York"
TEMP_UNIT = os.environ.get("TEMP_UNIT", "F").upper()  # "F" or "C"; readings are stored in Celsius
BUCKET = os.environ.get("S3_BUCKET", "densor")
# site=prefix pairs: each Pi uploads under its own prefix
SITES = dict(p.split("=", 1) for p in os.environ.get("S3_SITES", "ch=densor-ch,sr=densor-sr").split(","))
LOCAL_DIR = os.environ.get("DENSOR_LOCAL_DIR")

con = duckdb.connect()
if LOCAL_DIR:
    roots = {site: f"{LOCAL_DIR.rstrip('/')}/{prefix}" for site, prefix in SITES.items()}
else:
    endpoint = os.environ.get("S3_ENDPOINT", "t3.storage.dev").replace("https://", "")
    con.execute(f"""
        CREATE OR REPLACE SECRET tigris (
            TYPE s3,
            KEY_ID '{os.environ["AWS_ACCESS_KEY_ID"]}',
            SECRET '{os.environ["AWS_SECRET_ACCESS_KEY"]}',
            ENDPOINT '{endpoint}',
            URL_STYLE 'vhost',
            REGION '{os.environ.get("AWS_REGION", "auto")}'
        )
    """)
    roots = {site: f"s3://{BUCKET}/{prefix}" for site, prefix in SITES.items()}
roots


# %% [markdown]
# ## Load

# %%
COLUMNS = ["site", "ts_utc", "ieee", "name", "description",
           "temperature", "humidity", "pressure", "battery", "linkquality"]

def site_query(site, root):
    # r.* so that sites whose files predate a column (pressure) still load
    return f"""
        SELECT '{site}' AS site, r.*, d.name, d.description
        FROM read_parquet('{root}/readings/date=*/hour=*.parquet', union_by_name = true) r
        LEFT JOIN read_json('{root}/devices.json') d USING (ieee)
    """

frames = []
for site, root in roots.items():
    try:
        frames.append(con.execute(site_query(site, root)).df().reindex(columns=COLUMNS))
    except duckdb.IOException as e:  # a site with no files yet
        print(f"skipping {site}: {e}")
df = pd.concat(frames, ignore_index=True)

# Optional live tail: ask each Pi (over SSH) for readings newer than the last
# uploaded hour, so the newest hours show before the next hourly upload.
# DENSOR_LIVE="ch=densor@10.163.114.13,sr=densor@10.163.114.239"; unset to skip.
import json, shlex, subprocess

LIVE = dict(p.split("=", 1) for p in os.environ.get("DENSOR_LIVE", "").split(",") if p)
REMOTE_DB = os.environ.get("DENSOR_REMOTE_DB", "/var/lib/densor/densor.db")
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
            "-o", "ControlMaster=auto", "-o", "ControlPath=~/.ssh/cm-densor-%C",
            "-o", "ControlPersist=120"]

def live_rows(site, host, after):
    sql = ("SELECT r.ts_utc, r.ieee, d.name, d.description, r.temperature, r.humidity, "
           "r.pressure, r.battery, r.linkquality FROM readings r LEFT JOIN devices d USING (ieee) "
           f"WHERE r.ts_utc > '{after}' ORDER BY r.ts_utc")
    cmd = f"sqlite3 -readonly -json {shlex.quote(REMOTE_DB)} {shlex.quote(sql)}"
    out = subprocess.run(["ssh", *SSH_OPTS, host, cmd], capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or f"ssh exited {out.returncode}")
    rows = json.loads(out.stdout) if out.stdout.strip() else []
    return pd.DataFrame(rows).assign(site=site).reindex(columns=COLUMNS)

for site, host in LIVE.items():
    # Uploaded hours are complete, so anything newer than the newest bucket row is not in the bucket.
    after = df.loc[df["site"] == site, "ts_utc"].max() if (df["site"] == site).any() else "0000"
    try:
        extra = live_rows(site, host, after)
        print(f"live {site}: {len(extra)} readings newer than {after}")
        df = pd.concat([df, extra], ignore_index=True)
    except Exception as e:  # Pi unreachable, ssh denied, old schema, ...
        print(f"live {site}: skipped ({e})")
df = df.sort_values("ts_utc")

df["ts"] = pd.to_datetime(df["ts_utc"], utc=True)
df["ts_local"] = df["ts"].dt.tz_convert(LOCAL_TZ)
df["name"] = df["name"].fillna(df["ieee"])
df["temp"] = df["temperature"] * 9 / 5 + 32 if TEMP_UNIT == "F" else df["temperature"]  # `temperature` stays in Celsius
df.groupby(["site", "name"]).agg(rows=("ts", "size"), first=("ts_local", "min"), last=("ts_local", "max"))

# %% [markdown]
# ## Four series: temperature and humidity per sensor

# %%
px.scatter(df, x="ts_local", y="temp", color="name",
           title=f"Temperature (°{TEMP_UNIT})", labels={"ts_local": "", "temp": f"°{TEMP_UNIT}"}).show()

# %%
px.scatter(df, x="ts_local", y="humidity", color="name",
           title="Relative humidity (%)", labels={"ts_local": "", "humidity": "%"}).show()

# %% [markdown]
# ## Air pressure
#
# Only sensors with a barometer (SNZB-02M) report it, and files uploaded before pressure was added have no pressure column, so they read as empty.

# %%
p = df.dropna(subset=["pressure"])
if p.empty:
    print("No pressure readings yet")
else:
    px.scatter(p, x="ts_local", y="pressure", color="name",
               title="Air pressure (hPa)", labels={"ts_local": "", "pressure": "hPa"}).show()

# %% [markdown]
# ## Reporting behaviour
#
# The sensors report on change plus an hourly heartbeat, so a long gap means no change, not missing data.

# %%
df["gap_s"] = df.groupby("name")["ts"].diff().dt.total_seconds()
print(df.groupby("name")["gap_s"].describe(percentiles=[.5, .9, .99]).round(1))
px.histogram(df.dropna(subset=["gap_s"]), x="gap_s", color="name", log_y=True, nbins=60,
             title="Seconds between readings").show()

# %% [markdown]
# ## Battery and link quality

# %%
px.line(df, x="ts_local", y="battery", color="name", title="Battery (%)").show()
px.scatter(df, x="ts_local", y="linkquality", color="name", title="Link quality").show()

# %% [markdown]
# ## Your own queries
#
# `con` is a live DuckDB connection and `roots` maps each site to its location, so you can query the files directly, for example:
#
# ```python
# con.execute(f"SELECT date_trunc('hour', ts_utc::TIMESTAMP) h, ieee, avg(temperature) FROM read_parquet('{roots['ch']}/readings/*/*.parquet') GROUP BY ALL ORDER BY h").df()
# ```
