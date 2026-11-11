"""Sensor health checks that apply to any sensors, not one experiment."""
import pandas as pd
import plotly.express as px

import densor_closet as dc


def spot_check(df, checks, *, tz, temp_unit="F", bin="5min", max_hold="2h", spec_t_c=0.2, spec_rh=2.0):
    """Compare pairs of sensors that sat side by side.

    checks: [(sensor_a, sensor_b, start, end)], local times; end None means up to the latest reading.
    spec_t_c, spec_rh: the manufacturer's +-accuracy per sensor, so two sensors may differ by twice that.
    Returns (summary table, figure of a - b over time or None).
    """
    names = {n for a, b, *_ in checks for n in (a, b)}
    rows_in = df[df["name"].isin(names)]
    t, rh = ((dc.on_grid(rows_in, "temperature", bin, max_hold), dc.on_grid(rows_in, "humidity", bin, max_hold))
             if len(rows_in) else (None, None))
    f = 9 / 5 if temp_unit == "F" else 1
    rows, diffs = [], []
    for a, b, start, end in checks:
        sl = slice(pd.Timestamp(start, tz=tz), pd.Timestamp(end, tz=tz) if end else None)
        if t is None or a not in t.columns or b not in t.columns:
            print(f"Skipping {a} / {b}: no readings yet from one of them (check the name).")
            continue
        for label, g, scale, spec in [(f"temperature (°{temp_unit})", t, f, 2 * spec_t_c),
                                      ("relative humidity (% RH)", rh, 1, 2 * spec_rh),
                                      ("absolute humidity (g/m³)", dc.absolute_humidity(t, rh), 1, None)]:
            d = ((g[a] - g[b]).loc[sl].dropna()) * scale
            sa, sb = dc.short(a), dc.short(b)
            if len(d) == 0:
                rows.append({"a": sa, "b": sb, "quantity": label, "n": 0})
                continue
            limit = spec * scale if spec else None
            rows.append({"a": sa, "b": sb, "quantity": label, "n": len(d), "mean a-b": round(d.mean(), 2),
                         "std": round(d.std(), 2), "worst-case spec": None if limit is None else round(limit, 2),
                         "within spec": None if limit is None else abs(d.mean()) <= limit})
            diffs.append(d.rename("diff").to_frame().assign(series=f"{sa} - {sb}", quantity=label))
    print("Sensors are named by their last four serial digits.")
    fig = None
    if diffs:
        out = pd.concat(diffs).rename_axis("ts").reset_index()
        out["ts_local"] = out["ts"].dt.tz_convert(tz)
        fig = px.line(out, x="ts_local", y="diff", color="series", facet_row="quantity", height=650,
                      labels={"ts_local": "", "diff": "a - b"}, title="Difference between side-by-side sensors")
        fig.update_yaxes(matches=None)
    return pd.DataFrame(rows), fig
