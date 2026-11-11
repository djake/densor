"""Shared data loading for the Densor notebooks.

Reads the hourly Parquet files and devices.json for every site from the Tigris bucket (or a local
directory) with DuckDB, optionally adds a live tail from each Pi over SSH, and returns one pandas
DataFrame. Every step is timed; call `report()` (or `timings()`) after loading to see where the time went.

    import densor_data as dd
    con, roots = dd.connect()
    df = dd.load_readings(con, roots)
    dd.report()

Configuration is by environment variable (or notebooks/.env): S3_BUCKET, S3_ENDPOINT, S3_SITES, AWS_*,
DENSOR_LOCAL_DIR, DENSOR_LIVE, DENSOR_REMOTE_DB, TEMP_UNIT, and for the caches DENSOR_CACHE_DIR
(default ~/.cache/densor), DENSOR_LIVE_TTL (seconds, default 600), DENSOR_MIRROR (1 = keep a local copy of the
bucket's hourly files, the default; 0 = read the bucket directly) and DENSOR_REFRESH_BUCKET=1 (re-download
everything).
"""
import json
import os
import shlex
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))  # notebooks/.env next to this file
load_dotenv()                                  # and the usual search from the working directory

LOCAL_TZ = "America/New_York"
TEMP_UNIT = os.environ.get("TEMP_UNIT", "F").upper()  # "F" or "C"; readings are stored in Celsius
COLUMNS = ["site", "ts_utc", "ieee", "name", "description",
           "temperature", "humidity", "pressure", "battery", "voltage", "linkquality"]
REMOTE_DB = os.environ.get("DENSOR_REMOTE_DB", "/var/lib/densor/densor.db")
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
            "-o", "ControlMaster=auto", "-o", "ControlPath=~/.ssh/cm-densor-%C",
            "-o", "ControlPersist=120"]


# ---------------------------------------------------------------- timing

_TIMINGS = []


@contextmanager
def timed(step, **info):
    """Time a block. Extra keyword arguments are recorded with it; the block may add more through the dict it yields."""
    record = {"step": step, **info}
    start = time.perf_counter()
    try:
        yield record
    except Exception as e:
        record["error"] = (str(e).strip().splitlines() or [type(e).__name__])[-1][:120]
        raise
    finally:
        record["seconds"] = round(time.perf_counter() - start, 3)
        _TIMINGS.append(record)


def reset_timings():
    _TIMINGS.clear()


def timings():
    """Every timed step so far, as a DataFrame (step, seconds, and whatever detail the step recorded)."""
    cols = ["step", "seconds"]
    df = pd.DataFrame(_TIMINGS)
    if df.empty:
        return pd.DataFrame(columns=cols)
    return df[cols + [c for c in df.columns if c not in cols]]


def report():
    """Print the timings, slowest first, and the total."""
    df = timings()
    if df.empty:
        print("no timings recorded")
        return
    total = df.loc[df["step"] == "load_readings total", "seconds"]
    print(df.sort_values("seconds", ascending=False).to_string(index=False))
    if len(total):
        print(f"\ntotal: {total.iloc[-1]:.2f} s")


# ---------------------------------------------------------------- connection

def sites():
    """site -> bucket prefix, from S3_SITES ("ch=densor-ch,sr=densor-sr")."""
    return dict(p.split("=", 1) for p in os.environ.get("S3_SITES", "ch=densor-ch,sr=densor-sr").split(","))


def connect():
    """A DuckDB connection with the Tigris secret installed, and `roots`: site -> location of its files."""
    with timed("connect"):
        con = duckdb.connect()
        local_dir = os.environ.get("DENSOR_LOCAL_DIR")
        if local_dir:
            roots = {site: f"{local_dir.rstrip('/')}/{prefix}" for site, prefix in sites().items()}
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
            bucket = os.environ.get("S3_BUCKET", "densor")
            roots = {site: f"s3://{bucket}/{prefix}" for site, prefix in sites().items()}
    return con, roots


# ---------------------------------------------------------------- bucket

def _site_query(site, readings_root, devices_root):
    # r.* so that sites whose files predate a column (pressure, voltage) still load
    return f"""
        SELECT '{site}' AS site, r.*, d.name, d.description
        FROM read_parquet('{readings_root}/readings/date=*/hour=*.parquet', union_by_name = true) r
        LEFT JOIN read_json('{devices_root}/devices.json') d USING (ieee)
    """


def mirror_dir():
    return cache_dir() / "bucket"


def sync_bucket(con, roots, workers=16, rebuild=False):
    """Keep a local copy of every site's hourly Parquet files and return {site: local root}.

    The uploader writes each closed hour once and never changes it, so a file already on disk never needs
    fetching again: only hours that are new in the bucket are downloaded, in parallel. `rebuild=True` deletes
    the local copy first. Files that fail to download are skipped and retried on the next run.
    """
    base = mirror_dir()
    local_roots, todo = {}, []
    with timed("sync total") as total:
        for site, root in roots.items():
            site_dir = base / site
            if rebuild and site_dir.exists():
                shutil.rmtree(site_dir)
            local_roots[site] = str(site_dir)
            with timed("list bucket", site=site) as t:
                remote = [r[0] for r in con.execute(
                    f"SELECT file FROM glob('{root}/readings/date=*/hour=*.parquet')").fetchall()]
                t["files"] = len(remote)
            for src in remote:
                dest = site_dir / src[len(root) + 1:]
                if not dest.exists():
                    todo.append((src, dest))

        def fetch(job):
            src, dest = job
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".tmp")
            cur = con.cursor()  # one cursor per thread; a connection is not thread safe
            try:
                cur.execute(f"COPY (SELECT * FROM read_parquet('{src}')) TO '{tmp}' (FORMAT PARQUET)")
                os.replace(tmp, dest)
                return None
            except Exception as e:
                tmp.unlink(missing_ok=True)
                return f"{src}: {e}"
            finally:
                cur.close()

        failed = []
        if todo:
            with timed("download", files=len(todo), workers=workers):
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    failed = [r for r in pool.map(fetch, todo) if r]
        total.update(downloaded=len(todo) - len(failed), failed=len(failed))
        for msg in failed[:3]:
            print("could not download", msg)
    return local_roots


def load_bucket(con, roots, mirror=None, rebuild=False):
    """The uploaded readings for every site, one row per reading.

    With `mirror` on (the default unless DENSOR_LOCAL_DIR is set; DENSOR_MIRROR=0 turns it off) the hourly
    files are read from the local copy kept by `sync_bucket`; devices.json is always read fresh.
    """
    if mirror is None:
        mirror = os.environ.get("DENSOR_MIRROR", "0" if os.environ.get("DENSOR_LOCAL_DIR") else "1") != "0"
    rebuild = rebuild or os.environ.get("DENSOR_REFRESH_BUCKET") == "1"
    readings_roots = roots
    if mirror:
        try:
            readings_roots = sync_bucket(con, roots, rebuild=rebuild)
        except Exception as e:
            print(f"local copy unavailable ({e}); reading the bucket directly")
    frames = []
    for site, root in roots.items():
        with timed("read", site=site, source="local copy" if readings_roots is not roots else "bucket") as t:
            try:
                f = con.execute(_site_query(site, readings_roots[site], root)).df().reindex(columns=COLUMNS)
                t["rows"] = len(f)
                frames.append(f)
            except duckdb.IOException as e:  # a site with no files yet
                t["rows"] = 0
                print(f"skipping {site}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


# ---------------------------------------------------------------- live tail

def live_hosts():
    """site -> ssh host, from DENSOR_LIVE ("ch=densor@10.0.0.1,sr=densor@10.0.0.2"); empty when unset."""
    return dict(p.split("=", 1) for p in os.environ.get("DENSOR_LIVE", "").split(",") if p)


def _ssh_rows(site, host, after):
    sql = ("SELECT r.ts_utc, r.ieee, d.name, d.description, r.temperature, r.humidity, "
           "r.pressure, r.battery, r.voltage, r.linkquality FROM readings r LEFT JOIN devices d USING (ieee) "
           f"WHERE r.ts_utc > '{after}' ORDER BY r.ts_utc")
    cmd = f"sqlite3 -readonly -json {shlex.quote(REMOTE_DB)} {shlex.quote(sql)}"
    out = subprocess.run(["ssh", *SSH_OPTS, host, cmd], capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or f"ssh exited {out.returncode}")
    rows = json.loads(out.stdout) if out.stdout.strip() else []
    return pd.DataFrame(rows).assign(site=site).reindex(columns=COLUMNS), len(out.stdout)


def cache_dir():
    path = Path(os.environ.get("DENSOR_CACHE_DIR", "~/.cache/densor")).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_read(site, after, ttl):
    """Cached live rows for this site if they are younger than `ttl` seconds and were fetched for the same `after`."""
    meta_path = cache_dir() / f"{site}.live.json"
    try:
        meta = json.loads(meta_path.read_text())
        if meta["after"] != str(after) or time.time() - meta["fetched_at"] > ttl:
            return None
        if meta["rows"] == 0:
            return pd.DataFrame(columns=COLUMNS)
        return pd.read_parquet(cache_dir() / f"{site}.live.parquet").reindex(columns=COLUMNS)
    except (OSError, ValueError, KeyError):
        return None


def _cache_write(site, after, frame):
    d = cache_dir()
    if len(frame):
        tmp = d / f"{site}.live.parquet.tmp"
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, d / f"{site}.live.parquet")
    tmp = d / f"{site}.live.json.tmp"
    tmp.write_text(json.dumps({"fetched_at": time.time(), "after": str(after), "rows": len(frame)}))
    os.replace(tmp, d / f"{site}.live.json")


def live_rows(site, host, after, ttl=None, refresh=False):
    """Readings newer than `after` from one Pi, from the on-disk cache when it is fresh enough.

    The cache is reused only while it is younger than `ttl` seconds (DENSOR_LIVE_TTL, default 600) and was
    fetched for the same `after`, so a new hourly upload in the bucket triggers a fresh pull. Failures are
    not cached. `refresh=True` (or DENSOR_REFRESH=1) skips the cache.
    """
    ttl = float(os.environ.get("DENSOR_LIVE_TTL", 600)) if ttl is None else ttl
    refresh = refresh or os.environ.get("DENSOR_REFRESH") == "1"
    with timed("live", site=site) as t:
        cached = None if refresh else _cache_read(site, after, ttl)
        if cached is not None:
            t.update(source="cache", rows=len(cached))
            return cached
        frame, nbytes = _ssh_rows(site, host, after)
        t.update(source="ssh", rows=len(frame), bytes=nbytes)
        _cache_write(site, after, frame)
        return frame


# ---------------------------------------------------------------- everything together

def load_readings(con, roots, live=True, ttl=None, refresh=False, mirror=None, rebuild=False):
    """All readings as one DataFrame, with ts (UTC), ts_local, name, temp (in TEMP_UNIT) and description added."""
    reset_timings()
    with timed("load_readings total"):
        df = load_bucket(con, roots, mirror=mirror, rebuild=rebuild)
        if live:
            for site, host in live_hosts().items():
                # Uploaded hours are complete, so anything newer than the newest bucket row is not in the bucket.
                has = (df["site"] == site).any()
                after = df.loc[df["site"] == site, "ts_utc"].max() if has else "0000"
                try:
                    extra = live_rows(site, host, after, ttl=ttl, refresh=refresh)
                    print(f"live {site}: {len(extra)} readings newer than {after} ({_TIMINGS[-1]['source']})")
                    df = pd.concat([df, extra], ignore_index=True)
                except Exception as e:  # Pi unreachable, ssh denied, old schema, ...
                    print(f"live {site}: skipped ({e})")
        with timed("postprocess"):
            df = df.sort_values("ts_utc")
            df["ts"] = pd.to_datetime(df["ts_utc"], utc=True)
            df["ts_local"] = df["ts"].dt.tz_convert(LOCAL_TZ)
            df["name"] = df["name"].fillna(df["ieee"])
            df["temp"] = df["temperature"] * 9 / 5 + 32 if TEMP_UNIT == "F" else df["temperature"]  # `temperature` stays in Celsius
            df["description"] = df["description"].fillna("")
    return df
