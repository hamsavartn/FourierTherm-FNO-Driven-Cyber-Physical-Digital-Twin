# Data Findings — ACN-Data API (live tests, 2026-09-19)

## AT-1a: Does `timeseries=True` return charging telemetry? — **FAIL (definitive)**

**Test code** (token from `.env`, never committed):
```python
from acnportal.acndata import DataClient
client = DataClient(api_token=os.environ["ACN_API_TOKEN"])
for s in client.get_sessions_by_time(site="caltech",
        start=datetime(2019,10,15), end=datetime(2019,10,16), timeseries=True):
    if s.get("timeSeries"): ...
```

**Results:**

| Date range | Sessions | With timeSeries |
|---|---|---|
| 2019-10-15 (post-timezone-fix) | 32 | **0** |
| 2019-11-01 | 32 | **0** |
| 2023-06-10 | 0 sessions | — |
| 2026-06-10 | 0 sessions | — |

**Conclusion:** the `timeseries` parameter documented at
acnportal.readthedocs.io/en/latest/acndata/data_client.html ("If True return the
time-series of charging rates and pilot signals") is **not served by the current
API** for any date range tested. The v1 team's empirical finding
(metadata-only sessions) is **vindicated**; the PRD v3.0 hypothesis ("v1 likely
used the default `timeseries=False`") is **refuted** — v1 appears to have been
correct that the API is metadata-only, regardless of flag.

**Consequence (per PRD §3.2):** proceed with the **metadata-only design**
(FR1 fallback path, which was always fully valid):
- demand profiles built from connectionTime/deadline/energy at pilot power
- Playwright web-UI export remains an *optional* telemetry enhancement (Phase
  9-era task), not a dependency
- no changes needed to the validated Steps 3–5 pipeline (it never consumed
  time-series)

Session fields confirmed live for 2019-10-15: 32 sessions with the documented
metadata keys (sessionID, stationID, connectionTime, disconnectTime,
doneChargingTime, kWhDelivered, userInputs, timezone).

## AT-1b: Web Interface JSON export — **FAIL (definitive, closes the final branch)**

**Test (2026-09-20, live browser automation of ev.caltech.edu/dataset#interface):**
filters Site=Caltech, 2019-10-15 00:00 → 2019-10-16 00:00; form POST replicated
in-page (CSRF + session cookies). Response: `application/json`, HTTP 200,
**31 sessions, 0 with `timeSeries`** — key set identical to the API
(`_id, clusterID, connectionTime, disconnectTime, doneChargingTime,
kWhDelivered, sessionID, siteID, spaceID, stationID, timezone, userID,
userInputs`).

**Conclusion:** the Web Interface reads the same backend as the REST API.
Time-series telemetry does not exist in ACN-Data for this period on any access
path (API default, API `timeseries=True`, Web UI). The metadata-only design is
**triple-confirmed** as the only viable path; the uniform-rate demand
reconstruction (FR2) is the documented and final approach. No further
data-provenance branches remain open.
