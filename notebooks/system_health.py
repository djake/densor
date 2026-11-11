# %% [markdown]
# # Densor: system health
#
# Whether the sensors and the network are behaving: how often each sensor reports, battery and voltage, and link quality. Data comes from the shared loader (`densor_data`), so the bucket copy and the live-readings cache are shared with the other notebooks.

# %%
import os
import pandas as pd
import plotly.express as px

import densor_data as dd

LOCAL_TZ = dd.LOCAL_TZ
TEMP_UNIT = dd.TEMP_UNIT
con, roots = dd.connect()

# %%
df = dd.load_readings(con, roots)
dd.report()  # where the time went
HOVER = ["description"]  # shown when hovering a point; names stay short
pd.set_option("display.max_colwidth", None)
df.groupby(["site", "name"]).agg(description=("description", "max"), rows=("ts", "size"),
                                 first=("ts_local", "min"), last=("ts_local", "max"))

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
px.line(df, x="ts_local", y="battery", color="name", hover_data=HOVER, title="Battery (%)").show()

# %%
# Voltage (mV) is the better discharge signal: the battery % is derived from it and moves in coarse steps.
v = df.dropna(subset=["voltage"])
if v.empty:
    print("No voltage readings yet")
else:
    px.line(v, x="ts_local", y="voltage", color="name", hover_data=HOVER, title="Battery voltage (mV)",
            labels={"ts_local": "", "voltage": "mV"}).show()

# %%
px.scatter(df, x="ts_local", y="linkquality", color="name", hover_data=HOVER, title="Link quality").show()
