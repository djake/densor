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
# The closet sensors drawn in 3-D, and on each of the six surfaces (top, floor, four walls) as seen from inside. Colour is the **deviation from the closet median** by default, since the wall effect is a few tenths of a degree on top of a shared swing; set `SHOW = "absolute"` for plain temperatures.
#
# Positions come from `notebooks/placements.csv` when it exists (see `placements.csv.example`: one row per placement, in cm, with optional start and end times so sensors can be moved). Sensors without a row fall back to their description ("southwest corner floor" and so on) and are drawn at the closet's corners, which is approximate.
#
# A surface needs at least 3 sensors on it to be filled in (inverse-distance weighting); with fewer, only the sensor markers are shown. The fill is a rough interpolation, not a measurement between sensors.

# %%
import itertools
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CLOSET_SITE = os.environ.get("CLOSET_SITE", "ch")
CLOSET_WORD = os.environ.get("CLOSET_WORD", "closet")      # sensors whose description mentions this
PLACEMENTS = Path(os.environ.get("DENSOR_PLACEMENTS", "placements.csv"))
CLOSET_CM = os.environ.get("CLOSET_CM")                     # "x,y,z" in cm; default: from placements, else a unit box
EXTERIOR = ()                                               # e.g. ("north", "west"): marks those walls in the titles
SHOW = "deviation"                                          # "deviation" (from the closet median) or "absolute"
CAL_WINDOW = None                                           # ("2026-11-07 18:00", "2026-11-08 18:00") local time, all sensors side by side
BIN = "5min"                                                # time grid; readings are held until the next one
MAX_HOLD = "2h"                                             # stop holding a reading this long after it was reported
ANIM_DAYS = 2                                               # length of the 3-D time slider
FRAME_EVERY = "1h"
AT = None                                                   # local time for the surface panels, e.g. "2026-11-08 07:30"; None = latest
SURFACE_TOL = 0.2                                           # a sensor is "on" a surface within this fraction of the closet's size

cl = df[(df["site"] == CLOSET_SITE) & df["temp"].notna()
        & df["description"].str.contains(CLOSET_WORD, case=False)]
names = sorted(cl["name"].unique())
unit = f"°{TEMP_UNIT}"

# One column per sensor on a regular UTC grid. Sensors report on change plus a heartbeat,
# so a reading holds until the next one (up to MAX_HOLD).
grid = cl.assign(bin=cl["ts"].dt.floor(BIN)).pivot_table(index="bin", columns="name", values="temp", aggfunc="mean")
grid = grid.reindex(pd.date_range(grid.index.min(), grid.index.max(), freq=BIN))
grid = grid.ffill(limit=int(pd.Timedelta(MAX_HOLD) / pd.Timedelta(BIN)))

offsets = pd.Series(0.0, index=grid.columns)
if CAL_WINDOW:
    w = grid.loc[pd.Timestamp(CAL_WINDOW[0], tz=LOCAL_TZ):pd.Timestamp(CAL_WINDOW[1], tz=LOCAL_TZ)]
    offsets = w.sub(w.median(axis=1), axis=0).mean()
corrected = grid - offsets
values = corrected.sub(corrected.median(axis=1), axis=0) if SHOW == "deviation" else corrected


# %%
# Positions: placements.csv where present, else the description (corners of the closet).
def parse_description(text):
    t = text.lower()
    def pick(low, high):
        return 1.0 if re.search(high, t) else 0.0 if re.search(low, t) else 0.5
    return (pick(r"west", r"east"), pick(r"south", r"north"),
            pick(r"\bfloor\b|\bbottom\b", r"\btop\b|\bshelf\b|\bceiling\b"))

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
            fx, fy, fz = parse_description(descriptions[name])
            row = (fx * W, fy * D, fz * H, "description")
        rows[name] = row
    pos = pd.DataFrame(rows, index=["x", "y", "z", "source"]).T
    pos[["x", "y", "z"]] = pos[["x", "y", "z"]].astype(float)
    # Sensors drawn at the same spot would overlap: nudge them apart sideways.
    n_same = pos.groupby([pos["x"].round(3), pos["y"].round(3), pos["z"].round(3)]).cumcount()
    pos["x"] = pos["x"] + n_same * 0.03 * W
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
    "temp": grid.loc[at_utc].round(2), "offset": offsets.round(2), "shown": values.loc[at_utc].round(2),
})
print(f"Showing {at_utc.tz_convert(LOCAL_TZ):%a %Y-%m-%d %H:%M} ({SHOW}; positions in {units}; offsets from CAL_WINDOW: {bool(CAL_WINDOW)})")
now


# %% [markdown]
# ### 3-D view, with a time slider

# %%
frame_times = values.loc[at_utc - pd.Timedelta(days=ANIM_DAYS):].iloc[::max(1, pd.Timedelta(FRAME_EVERY) // pd.Timedelta(BIN))].index
all_vals = values.loc[frame_times].to_numpy().ravel()
all_vals = all_vals[~np.isnan(all_vals)]
if SHOW == "deviation":
    r = max(float(np.percentile(np.abs(all_vals), 98)), 0.1) if len(all_vals) else 1.0
    color_axis = dict(colorscale="RdBu_r", cmin=-r, cmax=r, cmid=0, colorbar=dict(title=f"{unit} from median"))
else:
    lo, hi = (np.percentile(all_vals, [2, 98]) if len(all_vals) else (0, 1))
    color_axis = dict(colorscale="RdYlBu_r", cmin=float(lo), cmax=float(hi), colorbar=dict(title=unit))

def box_trace():
    corners = list(itertools.product((0, W), (0, D), (0, H)))
    xs, ys, zs = [], [], []
    for a, b in itertools.combinations(corners, 2):
        if sum(p != q for p, q in zip(a, b)) == 1:
            xs += [a[0], b[0], None]; ys += [a[1], b[1], None]; zs += [a[2], b[2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="gray", width=3), hoverinfo="skip", showlegend=False)

def sensors_trace(t):
    pos = positions_at(t)
    v = values.loc[t].reindex(pos.index)
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
    coloraxis=color_axis, height=650, title=f"Closet temperature ({SHOW}, {unit})",
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
    v = values.loc[t].reindex(pos.index)
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
                      title=f"Closet surfaces at {t.tz_convert(LOCAL_TZ):%a %m-%d %H:%M} ({SHOW}, {unit}); walls seen from inside")
    return fig

surface_figure(at_utc).show()


# %% [markdown]
# ## Your own queries
#
# `con` is a live DuckDB connection and `roots` maps each site to its location, so you can query the files directly, for example:
#
# ```python
# con.execute(f"SELECT date_trunc('hour', ts_utc::TIMESTAMP) h, ieee, avg(temperature) FROM read_parquet('{roots['ch']}/readings/*/*.parquet') GROUP BY ALL ORDER BY h").df()
# ```
