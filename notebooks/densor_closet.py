"""Helpers for the closet experiment notebooks: regular time grids, humidity maths, offsets and interpolation.

All functions are pure (no notebook state), so other notebooks can reuse them.
"""
import numpy as np
import pandas as pd


def on_grid(rows, col, bin="5min", max_hold="2h"):
    """One column per sensor on a regular UTC grid of `col`.

    Sensors report on change plus a heartbeat, so a reading holds until the next one, for at most `max_hold`.
    `rows` is a readings DataFrame with `ts` (UTC timestamps), `name` and `col`.
    """
    g = rows.assign(bin=rows["ts"].dt.floor(bin)).pivot_table(index="bin", columns="name", values=col, aggfunc="mean")
    g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq=bin))
    return g.ffill(limit=int(pd.Timedelta(max_hold) / pd.Timedelta(bin)))


# Magnus formula; temperature in °C, relative humidity in %.
def vapour_pressure_hpa(t_c, rh):
    return rh / 100 * 6.112 * np.exp(17.62 * t_c / (243.12 + t_c))


def absolute_humidity(t_c, rh):
    """Water vapour density, g/m³."""
    return 216.7 * vapour_pressure_hpa(t_c, rh) / (t_c + 273.15)


def dew_point_c(t_c, rh):
    ln = np.log(vapour_pressure_hpa(t_c, rh) / 6.112)
    return 243.12 * ln / (17.62 - ln)


def offsets_from_window(grid, window, tz):
    """Per-sensor offset from the group median over a time window when all sensors were side by side.

    `window` is (start, end) in local time `tz`; None gives zero offsets.
    """
    if not window:
        return pd.Series(0.0, index=grid.columns)
    w = grid.loc[pd.Timestamp(window[0], tz=tz):pd.Timestamp(window[1], tz=tz)]
    return w.sub(w.median(axis=1), axis=0).mean()


def deviation(grid, window, tz):
    """A grid minus its calibration-window offsets (if any) and then minus the median across sensors."""
    g = grid - offsets_from_window(grid, window, tz)
    return g.sub(g.median(axis=1), axis=0)


def idw(px, py, pv, gx, gy, power=2):
    """Inverse-distance-weighted interpolation of values `pv` at points (px, py) onto the grid (gx, gy)."""
    d2 = (gx[..., None] - px) ** 2 + (gy[..., None] - py) ** 2
    w = 1.0 / np.maximum(d2, 1e-9) ** (power / 2)
    return (w * pv).sum(-1) / w.sum(-1)


# ---------------------------------------------------------------------------------------------
# The closet experiment: sensor positions, the quantity grid, and the figures drawn from it.

import itertools
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# title, axis the surface is perpendicular to, which side (0 or max), horizontal axis, vertical axis, flip horizontal
SURFACES = [
    ("Upper (ceiling + shelf)", "z", 1, "x", "y", False), ("North wall", "y", 1, "x", "z", False),
    ("East wall", "x", 1, "y", "z", True), ("Floor", "z", 0, "x", "y", False),
    ("South wall", "y", 0, "x", "z", True), ("West wall", "x", 0, "y", "z", False),
]


def short(name):
    """A sensor's last four serial digits, for labels."""
    return name.split("-")[-1][-4:]


def load_placements(path):
    """placements.csv (name, x_cm, y_cm, z_cm, start_utc, end_utc) as a DataFrame, or None if the file is absent."""
    path = Path(path)
    if not path.exists():
        return None
    placed = pd.read_csv(path)
    for c in ("start_utc", "end_utc"):
        placed[c] = pd.to_datetime(placed[c], utc=True) if c in placed else pd.NaT
    return placed.rename(columns={"x_cm": "x", "y_cm": "y", "z_cm": "z"})


def _play_pause(y):
    return [dict(type="buttons", direction="left", x=0.0, y=y, buttons=[
        dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=300, redraw=True), fromcurrent=True)]),
        dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0), mode="immediate")])])]


class Closet:
    """The closet sensors on a regular time grid, with positions and the figures drawn from them.

    df: readings from densor_data.load_readings. closet_map: {sensor name: (x, y, z) fractions of the closet,
    x west to east, y south to north, z floor to top}. moves: [(sensor, local time, (x, y, z))]; the latest move at or
    before a time wins. quantity: "temperature", "abs_humidity", "dew_point" or "humidity". show: "absolute" or
    "deviation" (from the closet median). The rest are the notebook's settings of the same names, in lower case.
    """

    LABEL_UNITS = {"temperature": "temperature", "abs_humidity": "absolute humidity", "dew_point": "dew point",
                   "humidity": "relative humidity"}

    def __init__(self, df, closet_map, moves=(), *, tz, temp_unit="F", quantity="temperature", show="absolute",
                 cal_window=None, bin="5min", max_hold="2h", anim_days=2, frame_every="1h", at=None, abs_range=None,
                 avg_hours=None, surface_tol=0.2, exterior=(), placements=None, closet_cm=None):
        self.map, self.moves, self.tz, self.unit_t = dict(closet_map), list(moves), tz, temp_unit
        self.quantity, self.show, self.cal_window, self.bin = quantity, show, cal_window, bin
        self.anim_days, self.surface_tol, self.exterior, self.avg_hours = anim_days, surface_tol, tuple(exterior), avg_hours
        self.placed = load_placements(placements) if placements is not None else None

        # Closet size: explicit "x,y,z" in cm, else from placements, else a unit box (positions as fractions).
        if closet_cm:
            dims = tuple(float(v) for v in closet_cm.split(","))
        elif self.placed is not None:
            dims = tuple(float(max(self.placed[a].max(), 1)) for a in "xyz")
        else:
            dims = (1.0, 1.0, 1.0)
        self.dims = dims
        self.W, self.D, self.H = dims
        self.dim_of = dict(zip("xyz", dims))
        self.units = "cm" if (closet_cm or self.placed is not None) else "fraction of closet"

        self.rows = df[df["name"].isin(self.map) & (df["temperature"].notna() | df["humidity"].notna())]
        self.names = sorted(self.rows["name"].unique())
        self.silent = sorted(set(self.map) - set(self.names))
        self.descriptions = self.rows.groupby("name")["description"].last()
        self.qname = self.LABEL_UNITS[quantity]
        f = temp_unit == "F"
        self.unit = {"temperature": f"°{temp_unit}", "dew_point": f"°{temp_unit}", "abs_humidity": "g/m³", "humidity": "% RH"}[quantity]

        # Temperature and humidity are held separately on the grid, then combined.
        self.t_c = on_grid(self.rows, "temperature", bin, max_hold)
        self.rh = on_grid(self.rows, "humidity", bin, max_hold)
        if quantity == "temperature":
            grid = self.t_c * 9 / 5 + 32 if f else self.t_c
        elif quantity == "humidity":
            grid = self.rh
        elif quantity == "abs_humidity":
            grid = absolute_humidity(self.t_c, self.rh)
        else:
            dp = dew_point_c(self.t_c, self.rh)
            grid = dp * 9 / 5 + 32 if f else dp
        self.grid = grid
        self.offsets = offsets_from_window(grid, cal_window, tz)
        corrected = grid - self.offsets
        self.values = corrected.sub(corrected.median(axis=1), axis=0) if show == "deviation" else corrected
        self.shown = self.values.rolling(f"{avg_hours}h", min_periods=1).mean() if avg_hours else self.values
        self.avg_note = f", {avg_hours} h trailing mean" if avg_hours else ""

        # The time drawn in the surface table, and the frames of the time sliders.
        self.valid = self.values.dropna(how="all")
        at_utc = pd.Timestamp(at, tz=tz).tz_convert("UTC") if at else self.valid.index[-1]
        self.at_utc = self.valid.index[self.valid.index.get_indexer([at_utc], method="nearest")[0]]
        step = max(1, pd.Timedelta(frame_every) // pd.Timedelta(bin))
        self.frame_times = self.shown.loc[self.at_utc - pd.Timedelta(days=anim_days):].iloc[::step].index

        all_vals = self.shown.loc[self.frame_times].to_numpy().ravel()
        all_vals = all_vals[~np.isnan(all_vals)]
        colorscale = "RdBu_r" if quantity in ("temperature", "dew_point") else "RdBu"   # red = warm, blue = wet
        if show == "deviation":
            r = max(float(np.percentile(np.abs(all_vals), 98)), 0.1) if len(all_vals) else 1.0
            self.color_axis = dict(colorscale=colorscale, cmin=-r, cmax=r, cmid=0, colorbar=dict(title=f"{self.unit} from median"))
        else:
            if abs_range:
                lo, hi = abs_range
            elif quantity in ("temperature", "dew_point"):
                lo, hi = (40, 90) if f else (4, 32)
            else:
                lo, hi = np.percentile(all_vals, [2, 98]) if len(all_vals) else (0, 1)
            self.color_axis = dict(colorscale=colorscale, cmin=float(lo), cmax=float(hi), colorbar=dict(title=self.unit))

    # -- positions -------------------------------------------------------------------------
    def positions_at(self, t):
        """x, y, z per sensor at time t (UTC), plus where each position came from."""
        rows = {}
        for name in self.names:
            row = None
            if self.placed is not None:
                p = self.placed[(self.placed["name"] == name)
                                & (self.placed["start_utc"].isna() | (self.placed["start_utc"] <= t))
                                & (self.placed["end_utc"].isna() | (t < self.placed["end_utc"]))]
                if len(p):
                    row = (p.iloc[-1]["x"], p.iloc[-1]["y"], p.iloc[-1]["z"], "placements.csv")
            if row is None:
                moved = [(pd.Timestamp(when, tz=self.tz), f) for n, when, f in self.moves if n == name]
                moved = sorted((w, f) for w, f in moved if w <= t)
                if moved:
                    fx, fy, fz = moved[-1][1]
                    row = (fx * self.W, fy * self.D, fz * self.H, "MOVES")
                else:
                    fx, fy, fz = self.map.get(name, (0.5, 0.5, 0.5))
                    row = (fx * self.W, fy * self.D, fz * self.H, "CLOSET_MAP" if name in self.map else "unmapped (centre)")
            rows[name] = row
        pos = pd.DataFrame(rows, index=["x", "y", "z", "source"]).T
        pos[["x", "y", "z"]] = pos[["x", "y", "z"]].astype(float)
        # Sensors drawn at the same spot would overlap: nudge them apart sideways.
        n_same = pos.groupby([pos["x"].round(3), pos["y"].round(3), pos["z"].round(3)]).cumcount()
        pos["x"] = pos["x"] + n_same * 0.08 * self.W
        return pos

    def label(self, t):
        return f"{t.tz_convert(self.tz):%a %m-%d %H:%M}"

    # -- the table at one time --------------------------------------------------------------
    def describe(self):
        """Print a one-line status and return the per-sensor table at `at`."""
        print(f"{len(self.names)} of {len(self.map)} sensors in CLOSET_MAP have readings.",
              f"No readings from: {self.silent}" if self.silent else "")
        pos = self.positions_at(self.at_utc)
        table = pd.DataFrame({
            "description": self.descriptions, "position from": pos["source"],
            "x": pos["x"].round(1), "y": pos["y"].round(1), "z": pos["z"].round(1),
            self.quantity: self.grid.loc[self.at_utc].round(2), "offset": self.offsets.round(2),
            "shown": self.shown.loc[self.at_utc].round(2),
        })
        print(f"Showing {self.at_utc.tz_convert(self.tz):%a %Y-%m-%d %H:%M} ({self.show}; positions in {self.units}; "
              f"offsets from CAL_WINDOW: {bool(self.cal_window)})")
        missing = [n for n in self.names if pd.isna(self.shown.loc[self.at_utc, n])]
        if missing:
            print("No recent reading at this time (older than MAX_HOLD), so not drawn:", missing)
        return table

    # -- 3-D view ---------------------------------------------------------------------------
    def _box_trace(self):
        W, D, H = self.dims
        corners = list(itertools.product((0, W), (0, D), (0, H)))
        xs, ys, zs = [], [], []
        for a, b in itertools.combinations(corners, 2):
            if sum(p != q for p, q in zip(a, b)) == 1:
                xs += [a[0], b[0], None]; ys += [a[1], b[1], None]; zs += [a[2], b[2], None]
        return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="gray", width=3), hoverinfo="skip", showlegend=False)

    def _sensors_trace(self, t):
        pos = self.positions_at(t)
        v = self.shown.loc[t].reindex(pos.index)
        ok = v.notna()
        return go.Scatter3d(
            x=pos.loc[ok, "x"], y=pos.loc[ok, "y"], z=pos.loc[ok, "z"], mode="markers+text",
            text=[short(n) for n in pos.index[ok]], textposition="top center",
            marker=dict(size=9, color=v[ok], coloraxis="coloraxis", line=dict(color="black", width=1)),
            customdata=np.column_stack([pos.index[ok], self.descriptions.reindex(pos.index[ok]), v[ok].round(2)]),
            hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>%{customdata[2]} " + self.unit + "<extra></extra>",
            showlegend=False)

    # The three planes behind the default camera (south wall, west wall, floor), drawn as interpolated heatmaps.
    BACK_PLANES = ("South wall", "West wall", "Floor")

    def _plane_traces(self, t):
        """One Surface per back plane, always in the same order so frames can update them (NaN where under 3 sensors)."""
        data = {d["title"].split(":")[0].replace(" (exterior)", ""): d for d in self._surface_data(t)}
        names = [s[0] for s in SURFACES]
        traces = []
        for title in self.BACK_PLANES:
            _, normal, side, hor, ver, _ = SURFACES[names.index(title)]
            d = data[title]
            A, B = np.meshgrid(d["gx"], d["gy"])
            fixed = np.full_like(A, side * self.dim_of[normal] - 0.012 * self.dim_of[normal])  # just behind the sensors
            xyz = {normal: fixed, hor: A, ver: B}
            traces.append(go.Surface(x=xyz["x"], y=xyz["y"], z=xyz["z"], surfacecolor=d["z"], coloraxis="coloraxis",
                                     opacity=0.85, showscale=False, hoverinfo="skip", name=title,
                                     lighting=dict(ambient=1, diffuse=0, specular=0)))
        return traces

    def figure_3d(self, planes=True):
        """Sensors in 3-D with a time slider; with `planes`, the back south wall, west wall and floor are filled in."""
        W, D, H = self.dims
        ft = self.frame_times
        extra = self._plane_traces if planes else (lambda t: [])
        n_extra = len(extra(ft[0]))
        fig = go.Figure(data=[self._box_trace(), *extra(ft[0]), self._sensors_trace(ft[0])],
                        frames=[go.Frame(data=[*extra(t), self._sensors_trace(t)], traces=list(range(1, 2 + n_extra)),
                                         name=self.label(t)) for t in ft])
        fig.update_layout(
            coloraxis=self.color_axis, height=650,
            title=f"Closet {self.qname} ({self.show}{self.avg_note}, {self.unit})",
            scene=dict(xaxis_title="x (W→E)", yaxis_title="y (S→N)", zaxis_title="z (height)",
                       xaxis=dict(range=[-0.1 * W, 1.1 * W]), yaxis=dict(range=[-0.1 * D, 1.1 * D]),
                       zaxis=dict(range=[-0.1 * H, 1.1 * H]), aspectmode="manual",
                       aspectratio=dict(x=W / max(self.dims), y=D / max(self.dims), z=H / max(self.dims))),
            updatemenus=_play_pause(0.0),
            sliders=[dict(active=0, currentvalue=dict(prefix="Time: "), pad=dict(t=30), steps=[
                dict(label=self.label(t), method="animate",
                     args=[[self.label(t)], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) for t in ft])])
        return fig

    # -- the six surfaces -------------------------------------------------------------------
    def _surface_data(self, t):
        """Per surface: the sensors on it at time t, their values, and an interpolated plane when there are at least 3."""
        pos = self.positions_at(t)
        v = self.shown.loc[t].reindex(pos.index)
        out = []
        for title, normal, side, hor, ver, flip in SURFACES:
            on = (pos[normal] - side * self.dim_of[normal]).abs() <= self.surface_tol * self.dim_of[normal]
            ok = on & v.notna()
            px_, py_, pv = pos.loc[ok, hor].to_numpy(), pos.loc[ok, ver].to_numpy(), v[ok].to_numpy()
            gx, gy = np.linspace(0, self.dim_of[hor], 40), np.linspace(0, self.dim_of[ver], 40)
            z = idw(px_, py_, pv, *np.meshgrid(gx, gy)) if len(pv) >= 3 else np.full((40, 40), np.nan)
            mark = " (exterior)" if title.split()[0].lower() in self.exterior else ""
            out.append(dict(title=f"{title}{mark}: {int(ok.sum())} sensors", gx=gx, gy=gy, z=z, px=px_, py=py_, pv=pv,
                            text=[f"{x:+.1f}" if self.show == "deviation" else f"{x:.1f}" for x in pv],
                            names=list(pos.index[ok])))
        return out

    def _surface_traces(self, t):
        """Two traces per surface (plane, then sensors), always in the same order so frames can update them."""
        traces = []
        for d in self._surface_data(t):
            traces.append(go.Heatmap(x=d["gx"], y=d["gy"], z=d["z"], coloraxis="coloraxis", hoverinfo="skip"))
            traces.append(go.Scatter(
                x=d["px"], y=d["py"], mode="markers+text", text=d["text"], textposition="top center", showlegend=False,
                marker=dict(size=14, color=d["pv"], coloraxis="coloraxis", line=dict(color="black", width=1)),
                customdata=[[n] for n in d["names"]], hovertemplate="%{customdata[0]}<extra></extra>"))
        return traces

    def _surface_title(self, t):
        return (f"Closet {self.qname} at {t.tz_convert(self.tz):%a %m-%d %H:%M} "
                f"({self.show}{self.avg_note}, {self.unit}); walls seen from inside")

    def figure_surfaces(self, times=None):
        times = self.frame_times if times is None else times
        fig = make_subplots(rows=2, cols=3, vertical_spacing=0.16, horizontal_spacing=0.07,
                            subplot_titles=[s[0] for s in SURFACES])
        first = self._surface_traces(times[-1])
        for i, tr in enumerate(first):
            fig.add_trace(tr, row=i // 2 // 3 + 1, col=i // 2 % 3 + 1)
        for i, (title, normal, side, hor, ver, flip) in enumerate(SURFACES):
            row, col = divmod(i, 3)
            ph, pv_ = 0.12 * self.dim_of[hor], 0.12 * self.dim_of[ver]
            fig.update_xaxes(range=[self.dim_of[hor] + ph, -ph] if flip else [-ph, self.dim_of[hor] + ph],
                             title_text=hor, row=row + 1, col=col + 1)
            fig.update_yaxes(range=[-pv_, self.dim_of[ver] + pv_], title_text=ver, row=row + 1, col=col + 1)
        notes = [a.to_plotly_json() for a in fig.layout.annotations]

        def frame(t):
            ann = [dict(n, text=d["title"]) for n, d in zip(notes, self._surface_data(t))]
            return go.Frame(data=self._surface_traces(t), traces=list(range(len(first))), name=self.label(t),
                            layout=dict(annotations=ann, title=dict(text=self._surface_title(t))))

        frames = [frame(t) for t in times]
        fig.frames = frames
        fig.update_layout(
            annotations=frames[-1].layout.annotations, coloraxis=self.color_axis, height=760,
            title=self._surface_title(times[-1]), updatemenus=_play_pause(-0.08),
            sliders=[dict(active=len(times) - 1, currentvalue=dict(prefix="Time: "), pad=dict(t=30), y=-0.02, steps=[
                dict(label=self.label(t), method="animate",
                     args=[[self.label(t)], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) for t in times])])
        return fig

    # -- over time, and humidity versus temperature -----------------------------------------
    @property
    def span(self):
        last = self.valid.index[-1]
        return slice(last - pd.Timedelta(days=self.anim_days), last)

    def figure_timeseries(self):
        step = max(1, pd.Timedelta("15min") // pd.Timedelta(self.bin))
        ts = self.shown.loc[self.span].iloc[::step].rename_axis("ts").reset_index().melt(
            id_vars="ts", var_name="name", value_name="value")
        ts["ts_local"] = ts["ts"].dt.tz_convert(self.tz)
        ts["description"] = ts["name"].map(self.descriptions)
        return px.line(ts, x="ts_local", y="value", color="name", hover_data=["description"],
                       title=f"Closet {self.qname}, {self.show}{self.avg_note} ({self.unit})",
                       labels={"ts_local": "", "value": self.unit})

    def figure_humidity_vs_temperature(self):
        """Absolute-humidity deviation against temperature deviation per sensor; prints the correlations."""
        d_temp = deviation(self.t_c, self.cal_window, self.tz) * (9 / 5 if self.unit_t == "F" else 1)
        d_ah = deviation(absolute_humidity(self.t_c, self.rh), self.cal_window, self.tz)
        step = max(1, pd.Timedelta("30min") // pd.Timedelta(self.bin))
        pair = pd.concat({"d_temp": d_temp.loc[self.span], "d_ah": d_ah.loc[self.span]}, axis=1).iloc[::step]
        pair = pair.stack(level=1, future_stack=True).dropna().rename_axis(["ts", "name"]).reset_index()
        print(f"Pooled correlation of absolute-humidity and temperature deviations: {pair['d_temp'].corr(pair['d_ah']):.2f}")
        print("Per sensor, over time:")
        print(pair.groupby("name").apply(lambda g: g["d_temp"].corr(g["d_ah"]), include_groups=False).round(2).to_string())
        pair["description"] = pair["name"].map(self.descriptions)
        return px.scatter(pair, x="d_temp", y="d_ah", color="name", hover_data=["description"], opacity=0.5,
                          title="Absolute humidity vs temperature, each as deviation from the closet median",
                          labels={"d_temp": f"temperature deviation (°{self.unit_t})",
                                  "d_ah": "absolute humidity deviation (g/m³)"})
