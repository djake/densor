# %% [markdown]
# # Densor: system health
#
# Whether the sensors and the network are behaving: how often each sensor reports, battery and voltage, and link quality, plus a side-by-side spot check for comparing two sensors (calibration). Data comes from the shared loader (`densor_data`), so the bucket copy and the live-readings cache are shared with the other notebooks.

# %%
# Pick up edits to densor_data.py and densor_closet.py without restarting the kernel.
# %load_ext autoreload
# %autoreload 2

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

# %% [markdown]
# ## Spot check: two sensors side by side
#
# List each period when two sensors sat next to each other as `(sensor_a, sensor_b, start, end)` in local time. For each, the difference `a - b` is summarised for temperature, relative humidity and absolute humidity. Sonoff quotes ±0.2 °C and ±2 % RH per sensor, so two sensors can legitimately differ by up to about twice that (set in `SPEC_T_C` and `SPEC_RH`). A difference well inside that is not worth calibrating; one well outside it means a sensor is off or the two were not in the same air.

# %%
import densor_closet as dc

BIN = "5min"       # time grid; readings are held until the next one
MAX_HOLD = "2h"    # stop holding a reading this long after it was reported


def short(name):
    return name.split("-")[-1][-4:]


# (sensor a, sensor b, start, end) in local time; end None means up to the latest reading. Set end to the time they were moved.
SPOT_CHECKS = [
    # ("<new sensor>", "<existing sensor beside it>", "2026-11-10 15:00", "2026-11-10 17:00"),
]
SPEC_T_C = 0.2     # ±°C per sensor
SPEC_RH = 2.0      # ±% RH per sensor

# Built from the named sensors only, whatever their descriptions say.
sc_rows = df[df["name"].isin({n for a, b, *_ in SPOT_CHECKS for n in (a, b)})]
sc_t, sc_rh = (dc.on_grid(sc_rows, "temperature", BIN, MAX_HOLD), dc.on_grid(sc_rows, "humidity", BIN, MAX_HOLD)) if len(sc_rows) else (None, None)
rows, diffs = [], []
for a, b, start, end in SPOT_CHECKS:
    sl = slice(pd.Timestamp(start, tz=LOCAL_TZ), pd.Timestamp(end, tz=LOCAL_TZ) if end else None)
    if sc_t is None or a not in sc_t.columns or b not in sc_t.columns:
        print(f"Skipping {a} / {b}: no readings yet from one of them (check the name).")
        continue
    for label, g, scale, spec in [(f"temperature (°{TEMP_UNIT})", sc_t, 9 / 5 if TEMP_UNIT == "F" else 1, 2 * SPEC_T_C),
                                  ("relative humidity (% RH)", sc_rh, 1, 2 * SPEC_RH),
                                  ("absolute humidity (g/m³)", dc.absolute_humidity(sc_t, sc_rh), 1, None)]:
        d = ((g[a] - g[b]).loc[sl].dropna()) * scale
        if len(d) == 0:
            rows.append({"a": short(a), "b": short(b), "quantity": label, "n": 0})
            continue
        limit = spec * scale if spec else None
        rows.append({"a": short(a), "b": short(b), "quantity": label, "n": len(d), "mean a-b": round(d.mean(), 2),
                     "std": round(d.std(), 2), "worst-case spec": None if limit is None else round(limit, 2),
                     "within spec": None if limit is None else abs(d.mean()) <= limit})
        diffs.append(d.rename(f"{short(a)}-{short(b)} {label}").to_frame("diff").assign(series=f"{short(a)} - {short(b)}", quantity=label))
print("Sensors are named by their last four serial digits.")
spot = pd.DataFrame(rows)
if len(diffs):
    diff_df = pd.concat(diffs).rename_axis("ts").reset_index()
    diff_df["ts_local"] = diff_df["ts"].dt.tz_convert(LOCAL_TZ)
    px.line(diff_df, x="ts_local", y="diff", color="series", facet_row="quantity", height=650,
            labels={"ts_local": "", "diff": "a - b"}, title="Difference between side-by-side sensors").update_yaxes(matches=None).show()
spot
