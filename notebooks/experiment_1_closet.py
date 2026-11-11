# %% [markdown]
# # Experiment 1: closet wall-proximity testbed
#
# Data comes from the shared loader (`densor_data`), so the bucket copy and the live-readings cache are shared with the other notebooks.
#
# The closet sensors drawn in 3-D, and on each of the six surfaces (upper, floor, four walls) as seen from inside; "upper" covers both the ceiling and the top shelf, which sit about a foot apart. Colour shows plain values by default; set `SHOW = "deviation"` for the **deviation from the closet median**, which brings out the wall effect (a few tenths of a degree) on top of the shared swing. `QUANTITY` switches what is drawn: `"temperature"`, `"humidity"` (RH), `"abs_humidity"` (g/m³) or `"dew_point"`. Absolute humidity and dew point separate an air leak (outdoor air is drier, so they drop toward the outdoor value) from a cold surface (they stay the same while RH rises).
#
# Positions come from `notebooks/placements.csv` when it exists (see `placements.csv.example`: one row per placement, in cm, with optional start and end times so sensors can be moved). The closet's sensors are the ones listed in `CLOSET_MAP`, a hardcoded corner for each, which is approximate; descriptions are not used.
#
# A surface needs at least 3 sensors on it to be filled in (inverse-distance weighting); with fewer, only the sensor markers are shown. The fill is a rough interpolation, not a measurement between sensors.

# %%
# Pick up edits to densor_data.py and densor_closet.py without restarting the kernel.
# %load_ext autoreload
# %autoreload 2

import os
from pathlib import Path

import densor_closet as dc
import densor_data as dd

LOCAL_TZ = dd.LOCAL_TZ
TEMP_UNIT = dd.TEMP_UNIT
con, roots = dd.connect()
df = dd.load_readings(con, roots)
dd.report()  # where the time went

PLACEMENTS = Path(os.environ.get("DENSOR_PLACEMENTS", "placements.csv"))
CLOSET_CM = os.environ.get("CLOSET_CM")                     # "x,y,z" in cm; default: from placements, else a unit box
EXTERIOR = ()                                               # e.g. ("north", "west"): marks those walls in the titles
QUANTITY = "temperature"                                    # "temperature", "abs_humidity" (g/m³), "dew_point" or "humidity" (RH %)
SHOW = "absolute"                                           # "deviation" (from the closet median) or "absolute"
CAL_WINDOW = None                                           # ("2026-11-07 18:00", "2026-11-08 18:00") local time, all sensors side by side
BIN = "5min"                                                # time grid; readings are held until the next one
MAX_HOLD = "2h"                                             # stop holding a reading this long after it was reported
ANIM_DAYS = 2                                               # length of the 3-D time slider
FRAME_EVERY = "1h"
AT = None                                                   # local time for the surface panels, e.g. "2026-11-08 07:30"; None = latest
ABS_RANGE = None                                            # colour range for SHOW = "absolute", e.g. (40, 90); None = 40-90 °F (4-32 °C) for temperatures, automatic otherwise
AVG_HOURS = None                                            # e.g. 24: draw the trailing mean over this many hours instead of single readings
SURFACE_TOL = 0.2                                           # a sensor is "on" a surface within this fraction of the closet's size

# Positions: placements.csv where present, else this map of closet corners as fractions of the closet
# (x: 0 west .. 1 east, y: 0 south .. 1 north, z: 0 floor .. 1 top), from the Zigbee2MQTT descriptions.
SHELF_Z = 0.875  # top shelf height as a fraction of the closet's height (floor 0, ceiling 1); about 1 ft below the ceiling, so 1 - 30 / closet height in cm
CLOSET_MAP = {
    "densor-ch-02p-25494000601274": (0, 1, 0),  # northwest corner floor
    "densor-ch-02b-26300400159665": (1, 1, 0),  # northeast corner floor
    "densor-ch-02b-26300400477224": (1, 0, 0),  # southeast floor
    "densor-ch-02p-25494000023339": (0, 0, 0),  # southwest corner floor
    "densor-ch-02b-26300400139758": (1, 0, SHELF_Z),  # southeast, top shelf
    "densor-ch-02b-26300400567826": (0, 0, SHELF_Z),  # southwest corner, top shelf
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
    # Moved up to the ceiling corners around 13:57 (time approximate).
    ("densor-ch-02b-26300400333849", "2026-11-10 13:57", (0, 1, 1)),
    ("densor-ch-02b-26300400102426", "2026-11-10 13:57", (1, 1, 1)),
]

closet = dc.Closet(
    df, CLOSET_MAP, MOVES, tz=LOCAL_TZ, temp_unit=TEMP_UNIT, quantity=QUANTITY, show=SHOW, cal_window=CAL_WINDOW,
    bin=BIN, max_hold=MAX_HOLD, anim_days=ANIM_DAYS, frame_every=FRAME_EVERY, at=AT, abs_range=ABS_RANGE,
    avg_hours=AVG_HOURS, surface_tol=SURFACE_TOL, exterior=EXTERIOR, placements=PLACEMENTS, closet_cm=CLOSET_CM)
closet.describe()  # one row per sensor at the time shown


# %% [markdown]
# ### 3-D view, with a time slider

# %%
closet.figure_3d().show()


# %% [markdown]
# ### The six surfaces, seen from inside the closet, with a time slider

# %%
closet.figure_surfaces().show()


# %% [markdown]
# ### Over time, and humidity versus temperature
#
# Per-sensor deviation over time for the chosen `QUANTITY` (a leak should show up as the odd sensor dipping on particular nights; an offset is a constant gap). Below it, absolute-humidity deviation against temperature deviation: a tight line means the humidity pattern is mostly following temperature (sensor coupling or offsets), while a cloud means it carries separate information. Both use the `CAL_WINDOW` offsets if set.

# %%
closet.figure_timeseries().show()
closet.figure_humidity_vs_temperature().show()
