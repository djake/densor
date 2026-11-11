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
