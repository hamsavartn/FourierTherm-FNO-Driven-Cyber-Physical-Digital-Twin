"""Step 2 (PRD §10): build the job-level dataset from sessions_meta CSVs
(PRD §3.5 jobs schema) and run AT-2: >= 99% of jobs have deadline >=
connection time; report slack statistics."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
LOCAL_TZ = "America/Los_Angeles"


def build_jobs(day: str) -> pd.DataFrame:
    meta = pd.read_csv(os.path.join(ROOT, "outputs",
                                    f"sessions_meta_caltech_{day}.csv"))
    d0 = pd.Timestamp(day, tz=LOCAL_TZ)
    rows = []
    for _, s in meta.iterrows():
        conn = pd.Timestamp(s.connectionTime)
        disc = pd.Timestamp(s.disconnectTime)
        deadline = (pd.Timestamp(s.doneChargingTime)
                    if s.doneChargingTime != "NA" else disc)
        req = float(s.requested_kWh)
        if req <= 0.5:                       # PRD §5.2: drop trivial sessions
            continue
        # Day-overlap rule (AT-2 fix): the API filters by UTC day, so sessions
        # that connected AND left during the previous LOCAL evening appear in
        # this query. Keep a session iff its charging window overlaps the
        # eval day [d0, d0+24h); clip start/deadline into [0, day end].
        conn_min = (conn - d0).total_seconds() / 60.0
        dl_min = (deadline - d0).total_seconds() / 60.0
        if dl_min <= 0:                      # charging ended before midnight
            continue
        if conn_min >= 1440:                 # starts after the day ends
            continue
        t_start = int(max(conn_min, 0.0))
        t_deadline = int(min(dl_min, 1559))
        slack = t_deadline - t_start
        rows.append({
            "job_id": f"{day[:10]}-{len(rows):03d}",
            "sessionID": s.sessionID, "stationID": s.stationID,
            "t_start_min": int(t_start), "t_deadline_min": int(t_deadline),
            "energy_requested_kwh": req,
            "energy_delivered_kwh": float(s.kWhDelivered),
            "slack_min": int(slack),
            "feasible": bool(slack > 0 and req > 0),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+",
                    default=["2019-10-15", "2019-10-16", "2019-10-17",
                             "2019-10-18", "2019-10-19"])
    args = ap.parse_args()
    all_bad, all_jobs = 0, 0
    for day in args.days:
        jobs = build_jobs(day)
        jobs.to_csv(os.path.join(ROOT, "outputs", f"jobs_caltech_{day}.csv"),
                    index=False)
        bad = int((jobs.slack_min < 0).sum())
        all_bad += bad
        all_jobs += len(jobs)
        print(f"{day}: {len(jobs)} jobs | deadline<connection: {bad} | "
              f"slack p50 {jobs.slack_min.median():.0f} min, "
              f"p10 {jobs.slack_min.quantile(0.1):.0f} min | "
              f"mean request {jobs.energy_requested_kwh.mean():.1f} kWh")
    ratio = 1.0 - all_bad / max(all_jobs, 1)
    verdict = "PASS" if ratio >= 0.99 else "FAIL"
    print(f"\nAT-2: {ratio:.2%} deadlines >= connection ({all_bad}/{all_jobs} "
          f"violations) -> {verdict}")


if __name__ == "__main__":
    main()
