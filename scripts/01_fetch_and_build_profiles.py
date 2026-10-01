"""Step 1 (PRD §10): fetch ACN-Data session metadata and build 1-minute
demand profiles. Metadata-only path (AT-1a verdict: no time-series served).

Outputs (schemas in PRD §3.5):
  outputs/sessions_meta_caltech_<date>.csv
  outputs/demand_minute_caltech_<date>.parquet

Demand construction (documented assumption, metadata-only): a session draws
average power = requested_kWh / (deadline - connection) across its connection
window, capped at the 7.2 kW L2 pilot limit; requested_kWh = first
userInputs.kWhRequested if present, else kWhDelivered (realized proxy, PRD
§3.5). All times converted to the site-local wall clock (America/Los_Angeles);
t_start_min is minutes since local midnight of the eval date.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = os.path.join(os.path.dirname(__file__), "..")
EVSE_CAP_KW = 7.2
LOCAL_TZ = "America/Los_Angeles"


def to_local(ts) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_convert(LOCAL_TZ)


def fetch_day(client, day: str) -> list[dict]:
    start = datetime.fromisoformat(day)
    rows = []
    for s in client.get_sessions_by_time(
            site="caltech", start=start, end=start + timedelta(days=1)):
        rows.append(s)
    return rows


def build_frames(rows: list[dict], day: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d0 = pd.Timestamp(day, tz=LOCAL_TZ)
    meta, draws = [], []
    for s in rows:
        try:
            conn = to_local(s["connectionTime"])
            disc = to_local(s["disconnectTime"])
        except (TypeError, ValueError):
            continue
        done = s.get("doneChargingTime")
        done = to_local(done) if done else None
        delivered = float(s.get("kWhDelivered") or 0.0)
        req = None
        ui = s.get("userInputs")
        if isinstance(ui, list) and ui and isinstance(ui[0], dict):
            r = ui[0].get("kWhRequested")
            req = float(r) if r is not None else None
        requested = req if req is not None else delivered
        deadline = done if done is not None else disc
        meta.append({
            "sessionID": s.get("sessionID"), "stationID": s.get("stationID"),
            "spaceID": s.get("spaceID"),
            "connectionTime": conn.isoformat(), "disconnectTime": disc.isoformat(),
            "doneChargingTime": done.isoformat() if done is not None else "NA",
            "kWhDelivered": delivered,
            "requested_kWh": requested if requested is not None else np.nan,
            "deadline_source": "done" if done is not None else "disconnect",
            "timezone": s.get("timezone"),
        })
        # clip the charging window to the eval day; uniform-rate draw (doc'd)
        a = max(conn, d0)
        b = min(deadline if deadline is not None else disc, d0 + pd.Timedelta(days=1))
        minutes = int((b - a).total_seconds() // 60)
        if minutes <= 0 or requested <= 0:
            continue
        kw = min(EVSE_CAP_KW, requested / (minutes / 60.0))
        i0 = int((a - d0).total_seconds() // 60)
        draws.append((i0, minutes, kw, s.get("stationID")))
    meta_df = pd.DataFrame(meta)

    n = np.zeros(1440)
    kw = np.zeros(1440)
    for i0, minutes, power, _sid in draws:
        for m in range(i0, min(i0 + minutes, 1440)):
            n[m] += 1
            kw[m] += power
    dem = pd.DataFrame({
        "minute_idx": np.arange(1440),
        "ts": pd.date_range(d0, periods=1440, freq="min"),
        "n_active_sessions": n.astype(int),
        "pilot_kw_total": kw,
        "energy_delivered_kw_total": kw,   # metadata-only proxy (same draw)
    })
    return meta_df, dem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+",
                    default=["2019-10-15", "2019-10-16", "2019-10-17",
                             "2019-10-18", "2019-10-19"])
    args = ap.parse_args()
    load_dotenv(os.path.join(ROOT, ".env"))
    from acnportal.acndata import DataClient
    client = DataClient(api_token=os.environ["ACN_API_TOKEN"])
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    for day in args.days:
        rows = fetch_day(client, day)
        meta, dem = build_frames(rows, day)
        meta.to_csv(os.path.join(ROOT, "outputs",
                                 f"sessions_meta_caltech_{day}.csv"), index=False)
        dem.to_parquet(os.path.join(ROOT, "outputs",
                                    f"demand_minute_caltech_{day}.parquet"))
        print(f"{day}: {len(meta)} sessions | peak demand "
              f"{dem.pilot_kw_total.max():.1f} kW at minute "
              f"{int(dem.pilot_kw_total.idxmax())}")


if __name__ == "__main__":
    main()
