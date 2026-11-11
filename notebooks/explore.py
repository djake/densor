# %% [markdown]
# # Densor: explore the readings
#
# Reads the hourly Parquet files and `devices.json` for every site straight from the Tigris bucket with DuckDB. Each site uploads under its own prefix (set in `S3_SITES`). Timestamps are stored in UTC and converted to local time here.
#
# Credentials come from `notebooks/.env` or the process environment. Use a read-only key.
#
# Loading lives in `notebooks/densor_data.py`, which other notebooks import too (`import densor_data as dd`). It times every step; `dd.report()` prints where the time went. The bucket's hourly files are kept as a local copy so only new hours are downloaded, and live readings from the Pis are cached on disk for 10 minutes (`DENSOR_LIVE_TTL`).

# %%
# Pick up edits to densor_data.py and densor_closet.py without restarting the kernel.
# %load_ext autoreload
# %autoreload 2

import os
import pandas as pd
import plotly.express as px

import densor_data as dd  # data loading, shared with the other notebooks (notebooks/densor_data.py)

LOCAL_TZ = dd.LOCAL_TZ
TEMP_UNIT = dd.TEMP_UNIT
con, roots = dd.connect()
roots


# %% [markdown]
# ## Load

# %%
# Bucket plus an optional live tail from each Pi (DENSOR_LIVE), the live part cached on disk for
# DENSOR_LIVE_TTL seconds (default 600). Set DENSOR_REFRESH=1, or pass refresh=True, to skip the cache.
df = dd.load_readings(con, roots)
dd.report()  # where the time went
HOVER = ["description"]  # shown when hovering a point; names stay short

# One row per sensor, with the description from Zigbee2MQTT (max: a non-empty text sorts after "").
pd.set_option("display.max_colwidth", None)
df.groupby(["site", "name"]).agg(description=("description", "max"), rows=("ts", "size"),
                                 first=("ts_local", "min"), last=("ts_local", "max"))

# %% [markdown]
# ## Four series: temperature and humidity per sensor

# %%
px.scatter(df, x="ts_local", y="temp", color="name", hover_data=HOVER,
           title=f"Temperature (°{TEMP_UNIT})", labels={"ts_local": "", "temp": f"°{TEMP_UNIT}"}).show()

# %%
px.scatter(df, x="ts_local", y="humidity", color="name", hover_data=HOVER,
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
    px.scatter(p, x="ts_local", y="pressure", color="name", hover_data=HOVER,
               title="Air pressure (hPa)", labels={"ts_local": "", "pressure": "hPa"}).show()

# %% [markdown]
# ## Other notebooks
#
# `system_health.py` (reporting gaps, battery, voltage, link quality) and `experiment_1_closet.py` (the closet wall-proximity testbed) load the same data through `densor_data`.
#
# ## Your own queries
#
# `con` is a live DuckDB connection and `roots` maps each site to its location, so you can query the files directly, for example:
#
# ```python
# con.execute(f"SELECT date_trunc('hour', ts_utc::TIMESTAMP) h, ieee, avg(temperature) FROM read_parquet('{roots['ch']}/readings/*/*.parquet') GROUP BY ALL ORDER BY h").df()
# ```
