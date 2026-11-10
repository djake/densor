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
           "temperature", "humidity", "pressure", "battery", "voltage", "linkquality"]

def site_query(site, root):
    # r.* so that sites whose files predate a column (pressure, voltage) still load
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
           "r.pressure, r.battery, r.voltage, r.linkquality FROM readings r LEFT JOIN devices d USING (ieee) "
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
df["description"] = df["description"].fillna("")
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
# ## Closet experiment (Experiment 1)
#
# The closet sensors drawn in 3-D, and on each of the six surfaces (top, floor, four walls) as seen from inside. Colour is the **deviation from the closet median** by default, since the wall effect is a few tenths of a degree on top of a shared swing; set `SHOW = "absolute"` for plain values. `QUANTITY` switches what is drawn: `"temperature"`, `"humidity"` (RH), `"abs_humidity"` (g/m³) or `"dew_point"`. Absolute humidity and dew point separate an air leak (outdoor air is drier, so they drop toward the outdoor value) from a cold surface (they stay the same while RH rises).
#
# Positions come from `notebooks/placements.csv` when it exists (see `placements.csv.example`: one row per placement, in cm, with optional start and end times so sensors can be moved). The closet's sensors are the ones listed in `CLOSET_MAP`, a hardcoded corner for each, which is approximate; descriptions are not used.
#
# A surface needs at least 3 sensors on it to be filled in (inverse-distance weighting); with fewer, only the sensor markers are shown. The fill is a rough interpolation, not a measurement between sensors.

# %%
import itertools
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CLOSET_SITE = os.environ.get("CLOSET_SITE", "ch")
PLACEMENTS = Path(os.environ.get("DENSOR_PLACEMENTS", "placements.csv"))
CLOSET_CM = os.environ.get("CLOSET_CM")                     # "x,y,z" in cm; default: from placements, else a unit box
EXTERIOR = ()                                               # e.g. ("north", "west"): marks those walls in the titles
QUANTITY = "temperature"                                    # "temperature", "abs_humidity" (g/m³), "dew_point" or "humidity" (RH %)
SHOW = "deviation"                                          # "deviation" (from the closet median) or "absolute"
CAL_WINDOW = None                                           # ("2026-11-07 18:00", "2026-11-08 18:00") local time, all sensors side by side
BIN = "5min"                                                # time grid; readings are held until the next one
MAX_HOLD = "2h"                                             # stop holding a reading this long after it was reported
ANIM_DAYS = 2                                               # length of the 3-D time slider
FRAME_EVERY = "1h"
AT = None                                                   # local time for the surface panels, e.g. "2026-11-08 07:30"; None = latest
AVG_HOURS = None                                            # e.g. 24: draw the trailing mean over this many hours instead of single readings
SURFACE_TOL = 0.2                                           # a sensor is "on" a surface within this fraction of the closet's size

# Positions: placements.csv where present, else this map of closet corners as fractions of the closet
# (x: 0 west .. 1 east, y: 0 south .. 1 north, z: 0 floor .. 1 top), from the Zigbee2MQTT descriptions.
CLOSET_MAP = {
    "densor-ch-02b-25494000601274": (0, 1, 0),  # northwest corner floor
    "densor-ch-02b-26300400159665": (1, 1, 0),  # northeast corner floor
    "densor-ch-02b-26300400477224": (1, 0, 0),  # southeast floor
    "densor-ch-02p-25494000023339": (0, 0, 0),  # southwest corner floor
    "densor-ch-02b-26300400139758": (1, 0, 1),  # southeast, top shelf
    "densor-ch-02b-26300400567826": (0, 0, 1),  # southwest corner top (description truncated in the list)
    "densor-ch-02b-26300400333849": (0, 1, 1),  # northwest corner ceiling (final position; see MOVES)
    "densor-ch-02b-26300400102426": (1, 1, 1),  # northeast corner ceiling (final position; see MOVES)
}

# Moves: (sensor, local time it was moved, (x, y, z) fractions as above). The latest move at or before a
# given time wins, so the 3-D view follows a sensor as it is relocated. Example:
#   MOVES = [("densor-ch-02b-26300400159665", "2026-11-10 15:00", (1, 1, 1))]
MOVES = [
    # Nov 10: the two ceiling-corner sensors start out on the floor beside the north floor sensors for a spot check.
    # When they go up, add ("<sensor>", "<local time they were moved>", (x, 1, 1)) for each, which puts them back at the ceiling.
    ("densor-ch-02b-26300400333849", "2026-11-10 13:40", (0, 1, 0)),
    ("densor-ch-02b-26300400102426", "2026-11-10 13:40", (1, 1, 0)),
]

cl = df[(df["site"] == CLOSET_SITE) & df["name"].isin(CLOSET_MAP)
        & (df["temperature"].notna() | df["humidity"].notna())]
names = sorted(cl["name"].unique())
silent = sorted(set(CLOSET_MAP) - set(names))
print(f"{len(names)} of {len(CLOSET_MAP)} sensors in CLOSET_MAP have readings.", f"No readings from: {silent}" if silent else "")

LABELS = {"temperature": ("temperature", f"°{TEMP_UNIT}"), "abs_humidity": ("absolute humidity", "g/m³"),
          "dew_point": ("dew point", f"°{TEMP_UNIT}"), "humidity": ("relative humidity", "% RH")}
qname, unit = LABELS[QUANTITY]

# One column per sensor on a regular UTC grid. Sensors report on change plus a heartbeat,
# so a reading holds until the next one (up to MAX_HOLD).
def on_grid(col, rows=None):
    rows = cl if rows is None else rows
    g = rows.assign(bin=rows["ts"].dt.floor(BIN)).pivot_table(index="bin", columns="name", values=col, aggfunc="mean")
    g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq=BIN))
    return g.ffill(limit=int(pd.Timedelta(MAX_HOLD) / pd.Timedelta(BIN)))

# Temperature and humidity are held separately, then combined (Magnus formula, temperature in °C).
t_c, rh = on_grid("temperature"), on_grid("humidity")
vapour = rh / 100 * 6.112 * np.exp(17.62 * t_c / (243.12 + t_c))        # hPa
if QUANTITY == "temperature":
    grid = t_c * 9 / 5 + 32 if TEMP_UNIT == "F" else t_c
elif QUANTITY == "humidity":
    grid = rh
elif QUANTITY == "abs_humidity":
    grid = 216.7 * vapour / (t_c + 273.15)
else:
    dp = 243.12 * np.log(vapour / 6.112) / (17.62 - np.log(vapour / 6.112))
    grid = dp * 9 / 5 + 32 if TEMP_UNIT == "F" else dp

offsets = pd.Series(0.0, index=grid.columns)
if CAL_WINDOW:
    w = grid.loc[pd.Timestamp(CAL_WINDOW[0], tz=LOCAL_TZ):pd.Timestamp(CAL_WINDOW[1], tz=LOCAL_TZ)]
    offsets = w.sub(w.median(axis=1), axis=0).mean()
corrected = grid - offsets
values = corrected.sub(corrected.median(axis=1), axis=0) if SHOW == "deviation" else corrected
# What the 3-D view and the surface panels draw: single readings, or a trailing mean to see past the noise.
shown = values.rolling(f"{AVG_HOURS}h", min_periods=1).mean() if AVG_HOURS else values
avg_note = f", {AVG_HOURS} h trailing mean" if AVG_HOURS else ""


# %%
placed = None
if PLACEMENTS.exists():
    placed = pd.read_csv(PLACEMENTS)
    for c in ("start_utc", "end_utc"):
        placed[c] = pd.to_datetime(placed[c], utc=True) if c in placed else pd.NaT
    placed = placed.rename(columns={"x_cm": "x", "y_cm": "y", "z_cm": "z"})

if CLOSET_CM:
    dims = tuple(float(v) for v in CLOSET_CM.split(","))
elif placed is not None:
    dims = tuple(float(max(placed[a].max(), 1)) for a in "xyz")
else:
    dims = (1.0, 1.0, 1.0)
W, D, H = dims
dim_of = dict(zip("xyz", dims))
units = "cm" if (CLOSET_CM or placed is not None) else "fraction of closet"

descriptions = cl.groupby("name")["description"].last()

def positions_at(t):
    """x, y, z per sensor at time t (UTC)."""
    rows = {}
    for name in names:
        row = None
        if placed is not None:
            p = placed[(placed["name"] == name)
                       & (placed["start_utc"].isna() | (placed["start_utc"] <= t))
                       & (placed["end_utc"].isna() | (t < placed["end_utc"]))]
            if len(p):
                row = (p.iloc[-1]["x"], p.iloc[-1]["y"], p.iloc[-1]["z"], "placements.csv")
        if row is None:
            moved = [(pd.Timestamp(when, tz=LOCAL_TZ), f) for n, when, f in MOVES if n == name]
            moved = sorted((w, f) for w, f in moved if w <= t)
            if moved:
                fx, fy, fz = moved[-1][1]
                row = (fx * W, fy * D, fz * H, "MOVES")
            else:
                fx, fy, fz = CLOSET_MAP.get(name, (0.5, 0.5, 0.5))
                row = (fx * W, fy * D, fz * H, "CLOSET_MAP" if name in CLOSET_MAP else "unmapped (centre)")
        rows[name] = row
    pos = pd.DataFrame(rows, index=["x", "y", "z", "source"]).T
    pos[["x", "y", "z"]] = pos[["x", "y", "z"]].astype(float)
    # Sensors drawn at the same spot would overlap: nudge them apart sideways.
    n_same = pos.groupby([pos["x"].round(3), pos["y"].round(3), pos["z"].round(3)]).cumcount()
    pos["x"] = pos["x"] + n_same * 0.08 * W
    return pos

def short(name):
    parts = name.split("-")
    return parts[-1][-4:]

# Which time to show in the surface panels.
valid = values.dropna(how="all")
at_utc = (pd.Timestamp(AT, tz=LOCAL_TZ).tz_convert("UTC") if AT else valid.index[-1])
at_utc = valid.index[valid.index.get_indexer([at_utc], method="nearest")[0]]

pos_now = positions_at(at_utc)
now = pd.DataFrame({
    "description": descriptions, "position from": pos_now["source"],
    "x": pos_now["x"].round(1), "y": pos_now["y"].round(1), "z": pos_now["z"].round(1),
    QUANTITY: grid.loc[at_utc].round(2), "offset": offsets.round(2), "shown": shown.loc[at_utc].round(2),
})
print(f"Showing {at_utc.tz_convert(LOCAL_TZ):%a %Y-%m-%d %H:%M} ({SHOW}; positions in {units}; offsets from CAL_WINDOW: {bool(CAL_WINDOW)})")
missing = [n for n in names if pd.isna(shown.loc[at_utc, n])]
if missing:
    print("No recent reading at this time (older than MAX_HOLD), so not drawn:", missing)
now


# %% [markdown]
# ### 3-D view, with a time slider

# %%
frame_times = shown.loc[at_utc - pd.Timedelta(days=ANIM_DAYS):].iloc[::max(1, pd.Timedelta(FRAME_EVERY) // pd.Timedelta(BIN))].index
all_vals = shown.loc[frame_times].to_numpy().ravel()
all_vals = all_vals[~np.isnan(all_vals)]
COLORSCALE = "RdBu_r" if QUANTITY in ("temperature", "dew_point") else "RdBu"   # red = warm, blue = wet
if SHOW == "deviation":
    r = max(float(np.percentile(np.abs(all_vals), 98)), 0.1) if len(all_vals) else 1.0
    color_axis = dict(colorscale=COLORSCALE, cmin=-r, cmax=r, cmid=0, colorbar=dict(title=f"{unit} from median"))
else:
    lo, hi = (np.percentile(all_vals, [2, 98]) if len(all_vals) else (0, 1))
    color_axis = dict(colorscale=COLORSCALE, cmin=float(lo), cmax=float(hi), colorbar=dict(title=unit))

def box_trace():
    corners = list(itertools.product((0, W), (0, D), (0, H)))
    xs, ys, zs = [], [], []
    for a, b in itertools.combinations(corners, 2):
        if sum(p != q for p, q in zip(a, b)) == 1:
            xs += [a[0], b[0], None]; ys += [a[1], b[1], None]; zs += [a[2], b[2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="gray", width=3), hoverinfo="skip", showlegend=False)

def sensors_trace(t):
    pos = positions_at(t)
    v = shown.loc[t].reindex(pos.index)
    ok = v.notna()
    return go.Scatter3d(
        x=pos.loc[ok, "x"], y=pos.loc[ok, "y"], z=pos.loc[ok, "z"], mode="markers+text",
        text=[short(n) for n in pos.index[ok]], textposition="top center",
        marker=dict(size=9, color=v[ok], coloraxis="coloraxis", line=dict(color="black", width=1)),
        customdata=np.column_stack([pos.index[ok], descriptions.reindex(pos.index[ok]), v[ok].round(2)]),
        hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>%{customdata[2]} " + unit + "<extra></extra>",
        showlegend=False)

def label(t):
    return f"{t.tz_convert(LOCAL_TZ):%a %m-%d %H:%M}"

fig3d = go.Figure(data=[box_trace(), sensors_trace(frame_times[0])],
                  frames=[go.Frame(data=[sensors_trace(t)], traces=[1], name=label(t)) for t in frame_times])
fig3d.update_layout(
    coloraxis=color_axis, height=650, title=f"Closet {qname} ({SHOW}{avg_note}, {unit})",
    scene=dict(xaxis_title="x (W→E)", yaxis_title="y (S→N)", zaxis_title="z (height)",
               xaxis=dict(range=[-0.1 * W, 1.1 * W]), yaxis=dict(range=[-0.1 * D, 1.1 * D]), zaxis=dict(range=[-0.1 * H, 1.1 * H]),
               aspectmode="manual",
               aspectratio=dict(x=W / max(dims), y=D / max(dims), z=H / max(dims))),
    updatemenus=[dict(type="buttons", direction="left", x=0.0, y=0.0, buttons=[
        dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=300, redraw=True), fromcurrent=True)]),
        dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0), mode="immediate")])])],
    sliders=[dict(active=0, currentvalue=dict(prefix="Time: "), pad=dict(t=30), steps=[
        dict(label=label(t), method="animate", args=[[label(t)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
        for t in frame_times])],
)
fig3d.show()


# %% [markdown]
# ### The six surfaces, seen from inside the closet

# %%
# title, axis the surface is perpendicular to, which side (0 or max), horizontal axis, vertical axis, flip horizontal
SURFACES = [
    ("Top", "z", 1, "x", "y", False), ("North wall", "y", 1, "x", "z", False), ("East wall", "x", 1, "y", "z", True),
    ("Floor", "z", 0, "x", "y", False), ("South wall", "y", 0, "x", "z", True), ("West wall", "x", 0, "y", "z", False),
]

def idw(px, py, pv, gx, gy, power=2):
    d2 = (gx[..., None] - px) ** 2 + (gy[..., None] - py) ** 2
    w = 1.0 / np.maximum(d2, 1e-9) ** (power / 2)
    return (w * pv).sum(-1) / w.sum(-1)

def surface_figure(t):
    pos = positions_at(t)
    v = shown.loc[t].reindex(pos.index)
    fig = make_subplots(rows=2, cols=3, vertical_spacing=0.16, horizontal_spacing=0.07,
                        subplot_titles=[s[0] for s in SURFACES])
    for i, (title, normal, side, hor, ver, flip) in enumerate(SURFACES):
        row, col = divmod(i, 3)
        row, col = row + 1, col + 1
        on = (pos[normal] - side * dim_of[normal]).abs() <= SURFACE_TOL * dim_of[normal]
        ok = on & v.notna()
        px, py, pv = pos.loc[ok, hor].to_numpy(), pos.loc[ok, ver].to_numpy(), v[ok].to_numpy()
        mark = " (exterior)" if title.split()[0].lower() in EXTERIOR else ""
        fig.layout.annotations[i].text = f"{title}{mark}: {int(ok.sum())} sensors"
        if len(pv) >= 3:
            gx = np.linspace(0, dim_of[hor], 40)
            gy = np.linspace(0, dim_of[ver], 40)
            z = idw(px, py, pv, *np.meshgrid(gx, gy))
            fig.add_trace(go.Heatmap(x=gx, y=gy, z=z, coloraxis="coloraxis", hoverinfo="skip"), row=row, col=col)
        if len(pv):
            fig.add_trace(go.Scatter(
                x=px, y=py, mode="markers+text", text=[f"{x:+.1f}" if SHOW == "deviation" else f"{x:.1f}" for x in pv],
                textposition="top center", showlegend=False,
                marker=dict(size=14, color=pv, coloraxis="coloraxis", line=dict(color="black", width=1)),
                customdata=[[n] for n in pos.index[ok]], hovertemplate="%{customdata[0]}<extra></extra>"),
                row=row, col=col)
        ph, pv_ = 0.12 * dim_of[hor], 0.12 * dim_of[ver]
        fig.update_xaxes(range=[dim_of[hor] + ph, -ph] if flip else [-ph, dim_of[hor] + ph], title_text=hor, row=row, col=col)
        fig.update_yaxes(range=[-pv_, dim_of[ver] + pv_], title_text=ver, row=row, col=col)
    fig.update_layout(coloraxis=color_axis, height=720,
                      title=f"Closet {qname} at {t.tz_convert(LOCAL_TZ):%a %m-%d %H:%M} ({SHOW}{avg_note}, {unit}); walls seen from inside")
    return fig

surface_figure(at_utc).show()


# %% [markdown]
# ### Over time, and humidity versus temperature
#
# Per-sensor deviation over time for the chosen `QUANTITY` (a leak should show up as the odd sensor dipping on particular nights; an offset is a constant gap). Below it, absolute-humidity deviation against temperature deviation: a tight line means the humidity pattern is mostly following temperature (sensor coupling or offsets), while a cloud means it carries separate information. Both use the `CAL_WINDOW` offsets if set.

# %%
last = valid.index[-1]
span = slice(last - pd.Timedelta(days=ANIM_DAYS), last)
ts_plot = shown.loc[span].iloc[::max(1, pd.Timedelta("15min") // pd.Timedelta(BIN))]
ts_plot = ts_plot.rename_axis("ts").reset_index().melt(id_vars="ts", var_name="name", value_name="value")
ts_plot["ts_local"] = ts_plot["ts"].dt.tz_convert(LOCAL_TZ)
ts_plot["description"] = ts_plot["name"].map(descriptions)
px.line(ts_plot, x="ts_local", y="value", color="name", hover_data=["description"],
        title=f"Closet {qname}, {SHOW}{avg_note} ({unit})", labels={"ts_local": "", "value": unit}).show()


def deviation(g):
    """A grid minus its CAL_WINDOW offsets (if set) and the closet median."""
    if CAL_WINDOW:
        w = g.loc[pd.Timestamp(CAL_WINDOW[0], tz=LOCAL_TZ):pd.Timestamp(CAL_WINDOW[1], tz=LOCAL_TZ)]
        g = g - w.sub(w.median(axis=1), axis=0).mean()
    return g.sub(g.median(axis=1), axis=0)

d_temp = deviation(t_c) * (9 / 5 if TEMP_UNIT == "F" else 1)
d_ah = deviation(216.7 * vapour / (t_c + 273.15))
pair = pd.concat({"d_temp": d_temp.loc[span], "d_ah": d_ah.loc[span]}, axis=1).iloc[::max(1, pd.Timedelta("30min") // pd.Timedelta(BIN))]
pair = pair.stack(level=1, future_stack=True).dropna().rename_axis(["ts", "name"]).reset_index()
print(f"Pooled correlation of absolute-humidity and temperature deviations: {pair['d_temp'].corr(pair['d_ah']):.2f}")
print("Per sensor, over time:")
print(pair.groupby("name").apply(lambda g: g["d_temp"].corr(g["d_ah"]), include_groups=False).round(2).to_string())
pair["description"] = pair["name"].map(descriptions)
px.scatter(pair, x="d_temp", y="d_ah", color="name", hover_data=["description"], opacity=0.5,
           title="Absolute humidity vs temperature, each as deviation from the closet median",
           labels={"d_temp": f"temperature deviation (°{TEMP_UNIT})", "d_ah": "absolute humidity deviation (g/m³)"}).show()


# %% [markdown]
# ### Spot check: two sensors side by side
#
# List each period when two sensors sat next to each other as `(sensor_a, sensor_b, start, end)` in local time. For each, the difference `a - b` is summarised for temperature, relative humidity and absolute humidity. Sonoff quotes ±0.2 °C and ±2 % RH per sensor, so two sensors can legitimately differ by up to about twice that (set in `SPEC_T_C` and `SPEC_RH`). A difference well inside that is not worth calibrating; one well outside it means a sensor is off or the two were not in the same air.

# %%
# (sensor a, sensor b, start, end) in local time; end None means up to the latest reading. Set end to the time they were moved.
SPOT_CHECKS = [
    ("densor-ch-02b-26300400333849", "densor-ch-02b-25494000601274", "2026-11-10 13:45", None),   # northwest: new 02B vs the 02P on the floor
    ("densor-ch-02b-26300400102426", "densor-ch-02b-26300400159665", "2026-11-10 13:45", None),   # northeast: new 02B vs the 02B on the floor
]
SPEC_T_C = 0.2     # ±°C per sensor
SPEC_RH = 2.0      # ±% RH per sensor

# Built from the named sensors only, whatever their descriptions say.
sc_rows = df[(df["site"] == CLOSET_SITE) & df["name"].isin({n for a, b, *_ in SPOT_CHECKS for n in (a, b)})]
sc_t, sc_rh = (on_grid("temperature", sc_rows), on_grid("humidity", sc_rows)) if len(sc_rows) else (None, None)
sc_vapour = sc_rh / 100 * 6.112 * np.exp(17.62 * sc_t / (243.12 + sc_t)) if len(sc_rows) else None
rows, diffs = [], []
for a, b, start, end in SPOT_CHECKS:
    sl = slice(pd.Timestamp(start, tz=LOCAL_TZ), pd.Timestamp(end, tz=LOCAL_TZ) if end else None)
    if sc_t is None or a not in sc_t.columns or b not in sc_t.columns:
        print(f"Skipping {a} / {b}: no readings yet from one of them at site {CLOSET_SITE!r} (check the name).")
        continue
    for label, g, scale, spec in [(f"temperature (°{TEMP_UNIT})", sc_t, 9 / 5 if TEMP_UNIT == "F" else 1, 2 * SPEC_T_C),
                                  ("relative humidity (% RH)", sc_rh, 1, 2 * SPEC_RH),
                                  ("absolute humidity (g/m³)", 216.7 * sc_vapour / (sc_t + 273.15), 1, None)]:
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
    dd = pd.concat(diffs).rename_axis("ts").reset_index()
    dd["ts_local"] = dd["ts"].dt.tz_convert(LOCAL_TZ)
    px.line(dd, x="ts_local", y="diff", color="series", facet_row="quantity", height=650,
            labels={"ts_local": "", "diff": "a - b"}, title="Difference between side-by-side sensors").update_yaxes(matches=None).show()
spot


# %% [markdown]
# ## Your own queries
#
# `con` is a live DuckDB connection and `roots` maps each site to its location, so you can query the files directly, for example:
#
# ```python
# con.execute(f"SELECT date_trunc('hour', ts_utc::TIMESTAMP) h, ieee, avg(temperature) FROM read_parquet('{roots['ch']}/readings/*/*.parquet') GROUP BY ALL ORDER BY h").df()
# ```
