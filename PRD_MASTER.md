# PRD MASTER v3.6 — Neural-Operator Dynamic Thermal Rating (DTR) + Safe Smart EV Charging for an 11 kV XLPE Underground Cable

**Project ID:** DTR-EV-2026 · **Owner:** Hamsavarthan H (24BEE0191) · **Type:** EEE + ML + Control, software-only
**Timeline:** 10 weeks (solo) · **Supersedes:** PRD v1 (PINN), PRD v2 (v2 change-log preserved in §15)
**Primary goal:** Capstone-level system + strong demo + (optional) provisional-patent viability check
**Secondary goal:** Publishable-quality evaluation pipeline (baselines, metrics, reproducibility)

---

## §0 — HOW TO USE THIS DOCUMENT (read this first if you are an AI agent)

You are expected to build the complete system from scratch using only this document. Rules:

1. **Build in runbook order (§10).** Every step has an acceptance test (AT-n). Do not proceed past a failing AT; record the failure in `outputs/phase_summary.csv` and the project log instead.
2. **Never fabricate numbers.** Every quantitative claim in any report/README you produce must be traceable to an artifact under `outputs/` produced by a script in `scripts/`. If a result is bad, report it — negative results are part of the deliverable.
3. **Claims discipline:** the project's own claims were audited (see §14 and `VERIFICATION.md`). Maintain that standard: when you cite a paper, cite the DOI/arXiv ID you actually fetched; when you cite a standard, name the clause you relied on or mark it "to be confirmed against the standard text."
4. **Open empirical questions are marked 🧪 and must be resolved by experiment, not by assertion.** The most important is AT-1a (§3.2): the v1 team believed the ACN API returns no time-series; the official docs contradict that. Test it first.
5. **If an environment dependency is missing, follow §8 fallbacks before installing anything new.**
6. **Definition of done (global):** every AT in §10 green, `scripts/run_all.sh` reproduces all artifacts from a fresh venv, demo runs, §6 experiment tables populated, README contains results + honest limitations section.

Working directory: `C:\Users\ASUS\Desktop\ML_Project` (Windows 11, Git Bash shell, `py` launcher available).

---

## §1 — IDENTITY, GOALS, NON-GOALS

### 1.1 Problem statement
Simultaneous EV charging raises feeder loading on an 11 kV underground XLPE cable. Cable insulation ages exponentially with conductor temperature; the industry-safe continuous limit for XLPE is **90°C** (verified: §14/B1). A *static* rating (worst-case ambient + rated ampacity) is conservative: on most days the cable can safely carry more energy than the static cap allows if temperature is actively managed. Build a closed-loop controller that exploits this headroom **safely**.

### 1.2 Product statement
A closed-loop EV-charging control system that:
1. estimates/predicts cable core temperature from EV load + ambient conditions via a physics-based digital twin;
2. enforces the hard constraint **T_core(t) ≤ 90 °C** at every minute via a receding-horizon safety shield;
3. maximizes delivered EV charging energy subject to (2);
4. allocates the allowed power across charging sessions **deadline-aware and fairly**;
5. replaces most twin rollouts inside the controller with a **neural-operator surrogate (1-D FNO)** trained on the twin — with the twin retained as a per-step verifier (guarded hybrid), so safety never depends on the network;
6. exposes everything through a Streamlit dashboard and a one-command reproducible pipeline.

### 1.3 Goals (must-have)
- **G1 Safety:** dynamic controller has **0 violations** (minutes with T_core > 90 °C + 1e-6 ε) across all evaluation days.
- **G2 Performance:** E_dynamic > E_static-cap on ≥ 1 of ≥ 5 evaluation days; report % improvement per day.
- **G3 Deadline + fairness:** job-level (per-session) scheduling with deadline awareness; report completion ratio and Jain fairness index.
- **G4 ML:** train the FNO surrogate on twin data; demonstrate **measurable benefit**: horizon-evaluation wall-time reduced ≥ 30% (p50) **or** violation-free operation under parameter noise with a smaller safety margin than the twin-only controller.
- **G5 Demo:** Streamlit dashboard + plots + reproducible scripts.

### 1.4 Non-goals (explicit)
- NG1: No hardware/sensors (optional extension only).
- NG2: No continuous-action DRL, no Transformers — compute and stability risk out of scope.
- NG3: No "first ever" claims. Novelty framing: *measurable integration* of neural-operator surrogate + verified safety shield for cable DTR under EV load, with prior-art analysis (§12).
- NG4: No claims of measured vs simulated temperatures — ground truth is the twin; say so everywhere.

### 1.5 Protection hierarchy and deployment gap (added v3.1, ISSUE-02 response)
Within this project the controller acts **only on a simulated feeder**; NG1 stands for the capstone. For any real deployment, the software layer (shield + allocator) is a **secondary optimization/dispatch layer**: primary thermal protection must remain a certified deterministic device (thermal overload relay / PLC trip per utility protection practice, e.g. IEC 60255-149-type relays) with its own measurement chain. The simulator implements the software analog of that hierarchy — a **watchdog fail-safe**: if the controller raises an exception or exceeds its per-minute latency budget, that minute reverts to the static-cap setpoint and logs `mode=failsafe` (§5.6). Reports must state this hierarchy explicitly; claims of the form "the software enforces safety in the field" are prohibited.

---

## §2 — DOMAIN PRIMER (EE fundamentals the implementer must know)

### 2.1 Cable thermal physics
Heat generated in the conductor (I²R Joule loss + dielectric loss) flows radially through: conductor → XLPE insulation → sheath → soil → ambient. Each layer has a **thermal resistance** (K·m/W) and **heat capacity** (J/K·m). This maps to a ladder RC network — the basis of IEC 60287 (steady-state) and IEC 60853 (cyclic/transient ratings).

**Verified material constants (IEC 60287 values, §14/B1–B2, §14/D1):**
| Layer | Thermal resistivity ρ_th |
|---|---|
| XLPE insulation | **3.5 K·m/W** |
| PVC/MDPE sheath | 5.0 (≤3 kV) – 6.0 (>3 kV) K·m/W |
| Soil (moist, typical) | 0.8–1.2 K·m/W (design value often 1.0) |

**Temperature limits (verified 2026-09-15, corrected v3.1):** XLPE max continuous conductor temperature **90 °C** — consistent across IEC 60502 and IS 7098 conventions. **Emergency overload** ratings differ by regional practice, not by material failure physics: **105 °C** per IEC/US practice (Cableizer reference table; EPRI found 105 °C used for ~80% of real cables; typical rule ≤ 72 h/yr), while **130 °C, ≤ 100 h/yr** appears in GB/regional and some manufacturer practice (some sources attribute it to IS 7098 as well — pin the exact clause from the standard text before citing; see §15 open items). **Short-circuit (adiabatic, ≈5 s): 250 °C** for Cu conductors (IEC 60949-style calculation; Al slightly lower). **Design impact: none** — the controller targets 90 °C continuous and deliberately does not exploit the emergency band (conservative choice; contingency/emergency rating is out of scope, listed as an extension).

### 2.2 Loss model (per unit cable length)
- Joule (**per-phase, v3.2** — phase-imbalance fix): AC Level-2 EVSEs are single-phase devices, so the aggregated feeder is **not** balanced by construction. Losses are computed per phase: `P_joule(t) = R_ac · (I_a² + I_b² + I_c²) + P_extra`, where `I_a,b,c` are per-phase RMS currents from the allocator's station→phase assignment (simulated round-robin by station index — documented assumption; ACN-Data does not publish site wiring; skew sensitivity in E7) and `P_extra = (λ_zs − 1) · R_ac · Σ I_i²` is a configurable armour/sheath add-on for unbalance-induced eddy losses beyond the balanced-load λ factors of IEC 60287 (`λ_zs ∈ [1.00, 1.05]`, 🧪 calibrate; default 1.03). Key physics: for fixed total power, `Σ I_i² ≥ 3·Ī²` — **the imbalance penalty emerges automatically** from per-phase computation; no blanket fudge factor. MVP fallback if per-phase bookkeeping is deferred: single aggregate current with factor `κ = (I_a²+I_b²+I_c²)/(3·Ī²) ≥ 1`. V_ll = 11 kV, pf ≈ 0.95 (representative; calibrate), R_ac = DC resistance × skin/proximity factor ≈ 1.02–1.05 (calibrate to datasheet, e.g. 185 mm² Al ≈ 0.164 Ω/km DC at 20 °C — **verify against a manufacturer datasheet**, §14/D1).
- Dielectric (**temperature-dependent, v3.4**): `W_d(t) = ω · C · U₀² · tanδ(T_c)` with `tanδ(T_c) = tanδ_20 · exp(β·(T_c − 20 °C))`, `{tanδ_20 ≈ 0.001–0.01, β ≈ 0.01–0.02 /°C}` (🧪 calibrate; Arrhenius-type rise toward the operating limit), ω = 2π·50, C ≈ 0.3–0.5 µF/km (datasheet), U₀ = 6.35 kV. **Materiality, quantified:** at 11 kV, `W_d ≈ 0.02–0.03 W/m` versus Joule `≈ 20 W/m` at rated load — dielectric heating is ~0.1% of total losses, so even a 3× tanδ excursion near 90 °C adds ≲ 0.3%, far below the λ_zs and margin conservatisms; the exponential tanδ runaway is a *critical* mechanism at EHV (thick insulation) and is included here because it is free, physically correct, and future-proofs the twin for higher voltage classes.

### 2.3 Lumped RC ladder (the digital twin)
**Per-core ladder (v3.6 — 3-core averaging fix):** state vector `x = [T_a, T_b, T_c, T_i, T_s, T_x, s]` — one node per conductor core (star-coupled through the common insulation/filler node; `R_core = 3·R1` preserves the balanced-load limit), filler/insulation node, sheath, near-soil, and the soil dry-fraction. A single conductor node would average the cores and hide the hottest one under phase imbalance (measured spread ≈ 13 °C at 200/50/50 A — Round-7 finding, trap 27); the shield enforces **max(T_a, T_b, T_c)**. W_d is Van Wormer-split (half to the three cores, half to the filler node); P_extra (sheath/armour eddy, λ_zs) deposits in the sheath node:

```
C_k·dT_k/dt = R_ac·I_k(t)² + W_d(T_k)/6 − (T_k − T_i)/R_core,   k ∈ {a,b,c}
C_i·dT_i/dt = Σ_k (T_k − T_i)/R_core + W_d(T̄)/2 − (T_i − T_s)/R2
C_s·dT_s/dt = (T_i − T_s)/R2 + P_extra − (T_s − T_x)/R3
C_x·dT_x/dt = (T_s − T_x)/R3 − (T_x − T_∞(t))/R4(s)
```
- R1 = insulation thermal resistance (per IEC 60287 T1, ~0.5–0.7 K·m/W for 185 mm² 11 kV class — calibrate), R2 = sheath/bedding, R3 = serving, R4 = **external soil resistance**: `R4 = ρ_soil/(2π) · ln(2L/D_ext)` (+mutual-heating terms if multi-cable; L = burial depth to cable axis, D_ext = external diameter) — IEC 60287-2-1 form. 🧪 Calibrate all R/C against a published worked example or datasheet before use; log the chosen table in `configs/twin.yaml` with its source.
- **Representative starting values (marked 🧪 calibrate):** put the cable in steady state at a nominal load (e.g. 200 A) with T_c ≈ 60–70 °C, T_amb(soil) = 20 °C; choose C_c ≈ 3–8e4 J/K·m, C_x large (soil) so its time constant is hours. Document every chosen number + source.

**Soil dry-out (added v3.1 — replaces the static-R4 assumption; ISSUE-01 response):** soil thermal resistivity is **state-dependent** under sustained high heat flux: moisture migrates away from the cable interface (vapor-phase transport — the Philip & de Vries 1957 mechanism), and the IEC 60287-2-1 two-zone "partial drying" treatment applies. Twin implementation: while the soil-interface temperature stays below the critical rise, use `rho_wet`; beyond it the effective external resistance gains an annular dry-zone term proportional to `(rho_dry − rho_wet)`. Concretely: `rho_soil(T_x) = rho_wet` for `T_x ≤ T_crit`, else ramped toward `rho_dry` over `ΔT_ramp`, with defaults `{rho_wet = 1.0, rho_dry = 2.5 (dry sand approaches 3.0), T_crit = 50 °C, ΔT_ramp = 15 °C}` in `configs/twin.yaml` — all 🧪 calibrate. This is the IEC-sanctioned approximation; full coupled heat–moisture PDEs (Philip–de Vries) are **out of scope** (no field data to calibrate; documented limitation §13-12). Consequences woven through the doc: scenario grid includes dry-out trajectories (§5.1), E4 stress-tests them (§6), the surrogate's parameter channels must include `rho_soil` state so the operator learns the feedback (§5.3a), and no field-safety claim is made without it.

**Moisture hysteresis (v3.3 — Round-4 fix, replaces the memoryless mapping above):** the `rho_soil(T_x)` mapping is symmetric — cool below `T_crit` and the trench instantly "re-wets", which is physically wrong: drying is heat-driven over hours, but re-wetting by capillary action takes days–weeks. The two-zone model therefore governs the *steady envelope*, while a **dry-fraction state** `s(t) ∈ [0,1]` governs transitions with asymmetric time constants: `ds/dt = σ(k·(T_x − T_crit))·(1−s)/τ_dry − σ(k·(T_crit − T_x))·s/τ_wet` — smooth sigmoid switching (v3.5: a hard if/else derivative degrades adaptive-solver efficiency during solve_ivp cross-validation; `k = 1 °C⁻¹` config, transition width ≈ ±3 °C, documented vs the safety margin), i.e. drying while hot, re-wetting while cool (natural re-wetting only; rain events out of scope), with `τ_wet ≥ 10·τ_dry` and defaults `{τ_dry ≈ 6 h, τ_wet = 120 h}` (🧪 calibrate), and `rho_soil = rho_wet + s·(ρ_dry − ρ_wet)`. `s` is a twin state **and** a surrogate input channel (§5.3a). Evaluation consequence: **the 5 evaluation days are consecutive and state-chained** (§5.6) — Day-2 ratings must reflect Day-1 drying, as in a real trench (§13 trap 19).

**Soil boundary condition (added v3.2 — non-physical-volatility fix):** the ladder's outer boundary is the **deep-soil far-field temperature `T_∞(t)`**, not air temperature. At burial depths 0.75–1.2 m the soil heavily damps and phase-shifts weather: diurnal air swings are filtered to near-zero at 1 m, leaving the **annual** wave (attenuated ≈ 0.6–0.7×, lagged ≈ 1–2 months) plus slow multi-day trends. `T_∞` is therefore generated by the **Kusuda–Achenbach (1965, NBS Report 8972)** annual soil model — `T_∞(t) = T_m − A_s·exp(−z/√(365·α/π))·cos((2π/365)·(t − t_shift − (z/2)·√(365/(π·α))))` — with `{T_m ≈ 18–22 °C, A_s ≈ 10–15 °C, z = burial depth, α_soil ≈ 0.5–1.0e-6 m²/s}` in `configs/twin.yaml` (🧪 calibrate; citation verified 2026-09-15 — frequently misspelled "Kasuda" in literature). Feeding minute-scale air noise into this boundary would inject non-physical heat-flux gradients and destabilize surrogate training; air temperature is retained only as an optional weak channel for shallow/duct installations. Scenario sampling now varies `{T_m, A_s, t_shift}` — slow and physical (§5.1).

### 2.4 Dynamic Thermal Rating (DTR)
DTR = the maximum feeder power at time t such that the twin predicts max_t∈[t, t+H] T_core ≤ 90 °C given the forecast load and ambient. The shield computes an achievable fraction α ∈ [0,1] of requested demand; §5.4 gives the algorithm.

### 2.5 Mutual heating (stretch-goal theory)
Neighboring cables raise each other's temperature (NYU ladder-type soil model w/ mutual heating, §14/C3). In graph form: cables = nodes, thermal couplings = edges → motivates the MeshGraphNets-style GNN stretch (§5.3c).

---

## §3 — DATA SPECIFICATION

### 3.1 Source: ACN-Data (Caltech EV charging sessions) — https://ev.caltech.edu/dataset
Public research dataset of real EV charging at the Caltech ACN (50 EVSEs). Requires a free **API token**: register at ev.caltech.edu → profile → API key. Store it in `.env` as `ACN_API_TOKEN=...` (never commit; `.env` is gitignored).

**Verified API usage (from official docs, §14/D3 — trust this over memory):**
```python
from acnportal.acndata import DataClient          # module acnportal.acndata
client = DataClient(api_token=os.environ["ACN_API_TOKEN"])   # url defaults to https://ev.caltech.edu/api/v1/
# Generator methods (yield dicts):
client.get_sessions(site="caltech", cond=None, sort=None, timeseries=False)
client.get_sessions_by_time(site="caltech",
                            start=datetime(2019,10,15),   # "sessions which began after start"
                            end=datetime(2019,10,16),     # "sessions which began before end"
                            min_energy=None,              # kWhDelivered >= min_energy
                            timeseries=False, count=False)
client.count_sessions(site="caltech", cond=None)  # returns int
```
`acnportal==0.3.3` (v1-verified installable via pip).

### 3.2 🧪 AT-1a (HIGHEST-PRIORITY falsifiable test — resolves a v1 contradiction)
- **v1 empirical finding:** API sessions are metadata-only; no `timeSeries` field observed.
- **Official docs (fetched 2026-09-15):** both generator methods accept **`timeseries=True`** — "If True return the time-series of charging rates and pilot signals. Default False."
- **Likely explanation:** v1 used the default (`timeseries=False`).
- **Test:** run `get_sessions_by_time(site="caltech", start=..., end=..., timeseries=True)` for a recent date; PASS iff ≥1 yielded session dict has `timeSeries` with `len(timeSeries) > 0` and each element contains a charging-rate/pilot field. If PASS → time-series telemetry comes free (skip Playwright web export entirely). If FAIL → metadata-only path (still fully valid; FR1 fallback).
- Record the outcome + the exact code + output sample in `docs/data_findings.md`.

### 3.3 Session schema (v1-observed; re-confirm at runtime)
Keys: `_id, userInputs, userID, sessionID, stationID, spaceID, siteID, clusterID, connectionTime, disconnectTime, doneChargingTime, kWhDelivered, timezone`.
- `userInputs` (list or null): contains per-request fields including **kWhRequested** and **minutesAvailable** (per ACN-Data docs; 🧪 verify presence at runtime, may be null when the driver didn't set a preference).
- `doneChargingTime` may be null (EV finished but stayed plugged) → fallback deadline = disconnectTime.

### 3.4 Known data trap (from ACN portal, carried from v1)
The ACN **web UI** had a timezone-handling bug before 2019-10-10; the Python API is correct. → Use API dates ≥ 2019-10-15. If using web-UI JSON export (only if AT-1a FAILs), treat web telemetry as enhancement-only and prefer post-2019-10 dates.

### 3.5 Derived artifacts (exact schemas — build these, tests assert them)
**`outputs/sessions_meta_<site>_<date>.csv`** — one row per session:
`sessionID:str, stationID:str, spaceID:str, connectionTime:UTC ISO8601, disconnectTime:UTC ISO8601, doneChargingTime:UTC ISO8601|NA, kWhDelivered:float, requested_kWh:float|NA, deadline_source:{done|disconnect}, timezone:str`

**`outputs/demand_minute_<site>_<date>.parquet`** — one row per minute of the day:
`minute_idx:int(0..1439), ts:UTC, n_active_sessions:int, pilot_kw_total:float, energy_delivered_kw_total:float`
(MVP aggregates all EVSEs; per-station column set `pilot_kw_<stationID>` added for the allocator.)
Demand construction (documented assumption): a session draws power from `connectionTime` until its energy request is met (or deadline), at pilot power; if userInputs.kWhRequested exists use it, else use kWhDelivered as the *realized* proxy (documented in README — v1 did the same).

**`outputs/jobs_<site>_<date>.csv`** — one row per charging job (input to scheduler):
`job_id:str, sessionID:str, stationID:str, t_start_min:int (minutes since day start), t_deadline_min:int, energy_requested_kwh:float, energy_delivered_kwh:float, slack_min:int (= t_deadline − t_start), feasible:bool (slack>0 & energy>0)`

### 3.6 Scope
Primary day: pick 5 distinct weekdays (e.g. 2019-10-15..) as the evaluation set; 1 day must also work for the demo scenario. All dates ≥ 2019-10-15 (§3.4).

---

## §4 — SYSTEM OVERVIEW

```
ACN-Data ──► demand_minute.parquet ──► jobs.csv ─┐
                                                 ▼
                              ┌────────────── closed-loop simulator (per minute) ──────────────┐
twin config ──► digital twin  │  demand request(t) ─► SAFETY SHIELD (bisection over α)        │
(scenario grid)│  (RC ladder) │      shield ─► allowed power(t) ─► ALLOCATOR ─► setpoints     │
               │              │      setpoints ─► twin integrates ─► T_core(t), state(t+1)    │
               ▼              └───────────────────────────────────────────────────────────────┘
        scenario data ──► FNO surrogate trainer ──► surrogate (used inside shield, guarded)
                                     │
                          evaluation: metrics.py → phase_summary.csv, plots, Streamlit app
```
Controllers compared (FR6): **(a)** no-control (serve all demand), **(b)** static cap (constant safe kW from steady-state rating), **(c)** dynamic shield (twin-only), **(d)** dynamic guarded-hybrid (FNO + twin verify) — (d) is the v2 contribution; (c) is its reference.

---

## §5 — MODULE SPECIFICATIONS (interfaces + pseudocode)

### 5.1 Thermal digital twin — `src/dtr/twin/`
- `rc_model.py`: implements §2.3 ODEs. Params from `configs/twin.yaml`: `{R1,R2,R3,R4 (K·m/W), C_c,C_i,C_s,C_x (J/K·m), R_ac (Ω/m), W_d (W/m), V_ll, pf}`. Function: `step(x, P_feeder_kw, T_amb, dt) -> x_next`; `rollout(x0, P_series, T_amb_series, dt) -> dict of arrays` (returns T_core etc.).
- `integrator.py`: RK4 fixed-step (dt=60 s, sub-steppable to 15 s for E7(j)) AND `scipy.integrate.solve_ivp` (rtol 1e-6) — they must agree (AT-6); RK4 is used in the loop for speed.
- `twin_torch.py` (**v3.5, differentiable twin**): the same RC + hysteresis model in torch — autograd-exact sensitivities for the Sobolev targets, uniform torch stack, and an enabler for gradient-based experiments. Cross-validation: `twin_torch` vs numpy RK4 must agree within 1e-3 °C over 24 h (extends AT-6). **Scope guard (v3.5):** differentiability serves training targets and experiments — the 1-D α search **remains bisection**, which is provably optimal for a monotone feasibility threshold (9 evaluations, deterministic); gradient ascent on α adds step-size risk on a safety-critical boundary for zero benefit in one dimension. Gradient/QP methods enter only through the multi-dimensional allocator (cvxpy, §5.5).
- **Analytical validation (AT-5):** 2-node special case (C_c,C_x; R,R4) has closed-form solution `T_c(t) = T_ss + (T_c0 − T_ss)·e^(−t/τ)` with `τ = C_eq·(R∥R4…)` — derive exactly for the constant-power case; assert RK4 matches within 0.5% after 3τ.
- `scenario_grid.py` (feeds §5.3): sample N≈4000 scenarios: seasonal boundary parameters `{T_m ~ U(14,26) °C, A_s ~ U(8,15) °C, t_shift ~ U(0,365 d)}` driving the Kusuda–Achenbach T∞ (§2.3, v3.2), baseload P0 ~ U(0.2,1.0)·P_rated, EV pulse profiles (random duty cycles), and parameter perturbations `R_i,C_i × lognormal(0, σ=0.1–0.25)`, **plus explicit soil-dry-out trajectories** (sustained overload until `T_x` crosses `T_crit` and the dry-fraction `s` evolves per the hysteresis ODE, §2.3), **and hysteresis draws `{τ_dry, τ_wet, s(0)}`** → save `outputs/scenarios.parquet`. **All scenarios and evaluation days are thermally preconditioned before their windows enter training or metrics** (48 h spin-up, §5.6 v3.2). with schema `{scenario_id, t (0..1799 min), P_kw[1800], T_amb[1800], params[8], T_core[1800]}`. Why: the surrogate must be robust to twin-parameter uncertainty (FR7 v2), not just memorize one cable.

### 5.2 Job builder — `src/dtr/data/job_builder.py`
Input `sessions_meta.csv` → output `jobs.csv` (schema §3.5). Rules: drop sessions with `kWhDelivered < 0.5`; `t_deadline = doneChargingTime if not null else disconnectTime`; `energy_requested = userInputs.kWhRequested if present else kWhDelivered` (documented proxy); clip deadlines to end-of-day+1h. **AT-2: ≥99% of jobs have deadline ≥ connection time; report violations explicitly.**

### 5.3 Surrogate — `src/dtr/surrogate/`
**(a) 1-D FNO (primary, v2).** Task: map horizon window → future T_core trajectory.
- Tensors: input `X ∈ R^(B × C_in × T_ctx)` with channels `C_in = [P_kw (normalized), T∞, T_c, T_i, T_s, T_x, s (soil dry-fraction, v3.3), params (R1–R4, C_c–C_x, ρ_wet, ρ_dry, tanδ_20, β — broadcast), α]`, `T_ctx = 60` context minutes; output `Y ∈ R^(B × T_out)` = T_core for next `T_out = 30` minutes. dt = 1 min.
- Architecture (defaults; tune on val): 4 Fourier layers, width 32–64, max modes 16 (1-D), spectral dropout 0.0–0.1, GELU; final pointwise head to 1 channel. Implementation: **custom torch** (`fno_1d.py`, ~60 lines: FFT → truncate modes → complex linear → iFFT → skip+nonlinearity). Custom is chosen deliberately: `neuraloperator`'s 1-D ergonomics were not confirmed (§14/A6) and the op is tiny. **Off-grid output mechanism (specified v3.1 — ISSUE-03 response):** all Fourier layers operate on the `T_ctx = 60` context grid; with FFT size 60 the Nyquist limit is 30 modes, so truncating at 16 modes is aliasing-free by construction. The future window `[t, t+30)` lies **off** the input grid, so the output is produced by evaluating the learned band-limited representation at the 30 future timestamps: linear interpolation of the post-layer feature map to the output positions, then the pointwise head. This is the same discretization-invariant evaluation that gives FNO its zero-shot super-resolution property (Li et al.) — the spectral layer preserves grid resolution, but the *operator* is evaluated wherever needed; it does not "fail dimensional checks." Configurable alternative: pad to a uniform 90-point grid (context ∥ future slots), run layers at 90, mask the loss to the future 30. **Unit tests:** (i) shape `(B, C_in, 60) → (B, 30)`; (ii) resolution transfer — train at ctx=60, evaluate at ctx=120 with no code change; (iii) no NaN under extreme inputs. Loss (**v3.3**, Round-4 fix): `L = MSE_tail + λ_peak·|max-T error| + λ_spec·Spectral(residual) + λ_sob·‖∂T̂/∂α − ∂T/∂α‖²`. The spectral term (magnitude + phase penalty on the FFT of the time-domain residual) guards the known low-frequency/spectral-smoothing bias of mode-truncated operators on sharp load steps; the Sobolev term supervises input-sensitivity along the α axis against twin sensitivities. **Implementation note (v3.5 — phantom-bottleneck correction):** sensitivity targets are **precomputed offline** during scenario-grid generation — the twin is never called during training, so labels and `∂T/∂α` columns are dataset artifacts and there is **no training-time cost**; with the differentiable torch twin (§5.1 v3.5) they come from exact autograd, else central differences (2 extra *offline* rollouts per scenario — dataset generation is CPU-hours, not a bottleneck). Bisection queries exactly the α direction, which is why this term matters (Sobolev-training principle, arXiv:1706.04859 — fetch before citing in the report). All λ swept in E5; training keeps mild switching noise in pulse profiles while E7(j) holds out the extreme cases, so the pulse test measures generalization, not memorization.
- Training: Adam 1e-3, cosine decay, batch 64, ≤200 epochs, early stop val-RMSE patience 20, seeds {0,1,2}; splits by scenario_id (no leakage). Hardware: CPU sufficient (expect minutes–1 h); Colab T4 optional.
- Baselines to beat/ablate (E5): (i) MLP on flattened window; (ii) PINN-style variant = same MLP/FNO + residual loss `L = MSE + λ·‖dT/dt − f_RC(x,u)‖²` (λ sweep {0, 1e-3, 1e-2}) — keeps the v1 methodology alive as an ablation; (iii) persistence (T_core constant).
- Eval (`evaluate.py`): trajectory RMSE, max-T-core error distribution (this is what the shield cares about), wall-clock per-rollout vs twin (p50/p95, AT-7), and parameter-noise suite (E4).
- **Acceptance (G4/AT-8):** p50 horizon-eval speedup ≥ 30% **or** under lognormal(0,0.1) parameter noise the guarded-hybrid controller stays violation-free with ≥ 20% smaller mean margin than twin-only. If neither → report honestly; twin-only controller remains the shipped result (project still passes G1–G3).

**(b) Integration — guarded hybrid (`control/guarded_hybrid.py`).**
```python
def choose_alpha(state, demand, amb, H=30, dt=1, T_max=90.0, margin=m):
    # 1) surrogate ENSEMBLE (3 seeds) runs the entire bisection (~9 steps x 3 passes)
    a, That_max, spread = surrogate_ensemble.rollout(state, a*demand, amb, H, dt)
    # 2) twin verifies ONCE -- TWO-SIDED (v3.2): safety AND temperature consistency
    twin_maxT = twin.rollout(state, a*demand, amb, H, dt).max_T_core
    bias_ewma.update(twin_maxT - That_max)
    # 3) twin takes over on ANY of: unsafe | temp mismatch | timid alpha | ensemble split
    if twin_maxT > T_max - margin:
        return bisect(twin, ...), "fallback"
    if (abs(twin_maxT - That_max) > dT_tol         # surrogate temperature error, either sign
        or twin_maxT <= T_max - K_RECOV*margin     # needlessly timid (>= 3x margin unused)
        or abs(a - a_prev) > 0.10                  # implausible setpoint jump
        or spread > spread_tol):                   # ensemble members disagree
        return bisect(twin, ...), "recovered"
    return a, "surrogate_ok"
```
Log per-step: `mode ∈ {surrogate_ok, recovered, fallback, failsafe}`; metrics: `fallback_rate` (target < 5%), `recovery_rate`, NEW `bias_ewma` (exponentially weighted twin-minus-surrogate peak-temperature error), `mismatch_rate`, `mean_ens_spread`. Defaults: `dT_tol = 1.5 °C` (two-sided temperature consistency — the Round-3 fix), `spread_tol` config; both swept in E3. Margin schedule (§5.4) consumes `fallback_rate` AND `|bias_ewma|`. **Drift alarm (v3.2):** `|bias_ewma| > 2 °C` for 15 consecutive minutes latches **twin-only mode** until recalibration — the system degrades to safe-and-slow rather than silently wrong. E3 speedup accounting **must** include twin re-solves on `recovered`/`fallback` steps and the ensemble's 3 forward passes per bisection step (honest speedup = total twin-rollout time saved). Safety argument (report): the twin remains the final authority each minute ⇒ violation-freedom reduces to the twin-only case, independent of surrogate error; the v3.1 escalation closed the one-sided-verification gap, and the v3.2 two-sided ΔT check additionally catches **systematic surrogate bias in either direction** (overprediction now triggers `recovered` even when slack is modest), while `bias_ewma` makes performance collapse visible inside the shield, not only in offline evaluation. Optional belt-and-braces: periodic exact twin bisection every 15 min (config flag). Cite precedent (§14/A7). **Monotonicity watchdog (v3.4 — Round-5 fix):** for the twin, `T_max(α)` is provably monotone non-decreasing (every horizon step is α-affine; the max of increasing functions is increasing), so bisection's prefix property holds exactly for the twin but is not guaranteed for a neural surrogate. The surrogate bisection therefore logs every `(α, T̂_max)` query and, after convergence, asserts near-isotonicity over the query history (tolerance `ε_mono = 0.2 °C`; near-free — it reuses queries already made, no extra forward passes). Violation → `mode="nonmono"` → twin re-solve that minute; `nonmono_rate` joins the surrogate-health metrics. Precision note: the reviewer's "infinite loop" consequence is impossible — bisection always terminates in `iters` steps; the true risks are false convergence (an accepted α beyond the true threshold) and cross-minute oscillation, both caught by the twin verification plus this watchdog.

**(c) GNN stretch (`gnn_thermal.py`, Week 7).** Nodes = RC ladder nodes × cables; edges = intra-cable resistances + inter-cable mutual-heating couplings `ΔT_ij` from IEC-style geometry (NYU ladder model, §14/C3). Same encode–process–decode message passing as MeshGraphNets (§14/A4; read the full paper for architectural details before citing specifics — §14/A5). Train on multi-cable twin variants. **Validation roles are separated (v3.2):** (i) **temporal stability — the primary gate** — validated on a synthetic multi-cable transient suite generated by the multi-cable twin itself: ≥ 72 continuous hours under highly variable loads, ≥ 40 parameter draws, reporting per-step drift and late-horizon (hour 14+) hallucination; (ii) **spatial/construction generalization** — the Mendeley CYMCAP dataset (440 steady-state ampacity cases, §14/C4) checks steady-state temperature/ampacity agreement only and is explicitly labeled as such (validating a temporal solver against static cases cannot expose drift — that mismatch is a category error the review correctly flagged). This is the publication-flavored differentiator; ship only if it beats the FNO on cross-parameter/cross-construction eval, else document as negative result. **Guardrails (v3.1 — ISSUE-06 response):** learned models are never rolled out beyond the 30-step shield horizon and are always twin-verified (§5.3b), so the documented MGN autoregressive error-accumulation failure mode cannot reach the 90 °C boundary; training keeps MGN's noise injection; E6 adds a per-step drift metric (error vs twin as a function of rollout step index). Continuous-time graph-ODE variants (e.g. the MeshODENet suggested in review): v3.3 runs this as an **evidence-gated A/B inside E6** — discrete MGN (30-step cap) vs graph-ODE on the 72 h suite, adopted only if drift-per-compute-hour is strictly better (adjoint solves are CPU-heavy). The Round-4 claim of "mathematically guaranteed long-term temporal stability" is **rejected**: a Neural ODE is still an approximate learned dynamical system with its own error accumulation and solver tolerances — no such guarantee exists in the literature. Governing invariant regardless of architecture: **no learned model is trusted beyond the 30-step horizon or without twin verification** (§5.3b).

### 5.4 Safety shield — `control/shield.py`
Receding horizon H = 30 min, dt = 1 min. At each minute t: given state x(t), demand forecast D(t..t+H) (from jobs: planned setpoints), ambient forecast (persistence = today's earlier values):
```
α = bisect(lo=0, hi=1, iters=9):            # resolution 1/512 of demand — sufficient
    feasible(a) := max_twin_rollout(x(t), Allocator(a·D), amb, H) ≤ 90 − margin
                  # v3.6 ROUTING FIX: bisection must verify the profile the allocator
                  # will actually dispatch — a·D never reaches the twin directly.
margin: start 2.0 °C; adapts on TWO signals (v3.2): `fallback_rate > 5%` over the last 60 steps → +0.5 (max 4); `|bias_ewma| > 1 °C` → +0.5; 0 fallbacks and `|bias_ewma| < 0.3 °C` for 240 steps → −0.25 (min 1.0). **Conformal Predictive Control (first-class, v3.5):** the scalar margin is replaced by a per-state conformal quantile of held-out twin-vs-surrogate peak-temperature residuals (split conformal; distribution-free finite-sample coverage at the chosen level — 95–99% practical; targeting 99.9% honestly requires ~10³–10⁴ calibration scenarios, which the scenario grid provides). Composition rule: `margin_eff = max(base, conformal, black-start-inflated)` — the safety mechanisms stack, never cancel.
allowed_power(t) = α · Σ demand
```
Invariants (pytest, AT-4): shield output α ∈ [0,1]; twin re-check of chosen α with margin **never** infeasible (property test over 200 random states); violations == 0 on eval days (AT-3).

### 5.5 Allocator — `control/allocator.py`
Inputs: allowed_power(t), active jobs. Weights: `urgency_i = rem_energy_i / max(rem_time_i, 5)` (kWh/min), `fair_i = 1 − delivered_i / (Σ delivered_j / n)` clipped ≥ 0.1 (anti-starvation), `w_i = β·norm(urgency_i) + (1−β)·norm(fair_i)`, β=0.6 (sweep {0.4,0.6,0.8} in E1). Setpoints `p_i = allowed_power · w_i / Σw`, cap per EVSE at 7.2 kW (L2 pilot limit), respect per-job deadline: if `t_deadline − t < 30 min` → urgency ×5. Output: per-minute per-station setpoints parquet.
Sanity test: Jain index of a no-scarcity day must be ≥ 0.95 (fairness shouldn't distort when not needed). **Phase-aware allocation (stretch, v3.2):** each EVSE is a single-phase device on a known/simulated phase (§2.2); as a secondary objective the allocator curtails the most-loaded phase first, minimizing the imbalance index `Imb = (I_a² + I_b² + I_c²)/(3·Ī²)`. Compare E1 runs with and without — turns the imbalance vulnerability (§13 trap 16) into an allocation feature. **Routing consistency (v3.6, structural):** the allocator is invoked **inside** the shield's feasibility check (§5.4) — the twin always verifies the dispatched per-phase profile, never a proportional abstraction of it. **Optional sim mode:** a documented `switchable_evse_fraction` (3-phase-capable chargers that can re-route phase in software — rare hardware, assumption must be flagged) allows the allocator to re-assign phases; ablate in E1. **V2G phase-balancing (stretch, v3.5, gated to Week 7+):** simulated bidirectional EVSEs (ISO 15118-20; a documented fraction of the fleet assumed V2G-capable — ACN-Data carries no capability labels) may discharge on under-loaded phases to offset over-loaded ones, directly minimizing `P_extra`. New allocator mode + E-case with converter and onboard-energy limits documented. This is the project's most novel differentiator — *EVs as active thermal balancers* — and is built only after the core pipeline passes AT-1..8.

### 5.6 Closed-loop simulator — `scripts/03_phase11_closed_loop.py`
Per minute: compute demand from jobs → shield α → allocator setpoints → twin step → log row. Log schema (`outputs/sim_<controller>_<date>.parquet`): `minute_idx, demand_kw, allowed_kw, alpha, setpoints_per_station(json), T_core, T_sheath, T_amb, mode, wall_time_shield_ms, wall_time_step_ms, violations_so_far`. Controllers: `no_control, static_cap, twin_only, guarded_hybrid`. **Watchdog fail-safe (v3.1, §1.5):** any controller exception or wall-time budget breach reverts that minute to the static-cap setpoint and logs `mode=failsafe`. **Thermal preconditioning (v3.2 — cold-start trap):** every evaluation day and training scenario is initialized from a **48 h spin-up** replaying the prior day's demand profile + baseload; where no prior day exists, initialize from the analytic steady state under a representative pre-load (the IEC 60853 cyclic-rating convention assumes a defined pre-load condition). Cost is negligible (RC ODE). Metrics are computed only after preconditioning (§13 trap 14). Sanity test: after preconditioning, a constant-baseload day must show |ΔT_core| < 0.5 °C over 24 h. **Chained multi-day evaluation (v3.3):** the 5 evaluation days are consecutive calendar days; twin state — including the soil dry-fraction `s` — carries across day boundaries, and spin-up runs once before Day 1. Multi-day chaining is what makes the hysteresis model testable: Day-2 ratings must reflect Day-1 drying (§13 trap 19). **Black-start recovery (v3.4 — telemetry-vacuum fix):** if the preconditioning inputs (prior-day profile / trailing history) are missing or corrupt, initialize at the **pessimistic envelope**: analytic steady state under the maximum continuous power the shield can command (static-cap load) at worst-case `T∞` — for this linear, monotone system that state provably upper-bounds any state reachable under shield-limited load during the unobserved window. During recovery the margin inflates and decays: `margin(t) = margin_base + Δ_bs·2^(−t/T_half)` with `{Δ_bs = 2 °C, T_half = 12 h}` (config), relaxing as live verification re-accumulates. No 48 h charging halt, no guessed benign state. E7(l) asserts the protocol end-to-end.

### 5.7 Baselines
- no-control: serve full demand (expect violations on stressed days — that's the point).
- static_cap: constant P_static = max power keeping steady-state T_core = 90 at T_amb = 35 °C (compute once from twin steady-state; document the number).

---

## §6 — METRICS & EVALUATION PROTOCOL (formulas, then experiments)

**Metrics (`evaluation/metrics.py`):**
- `violations = Σ_t 1[T_core(t) > 90 + 1e-6]` (minutes; day-level)
- `E_delivered` = ∫ allowed·served kWh; `improvement% = (E_dyn − E_static)/E_static × 100`
- `completion_ratio = #jobs meeting requested kWh by deadline / #jobs` (tolerance: 95% of request counts as met — document choice)
- `Jain = (Σ x_i)² / (n · Σ x_i²)` over per-job delivered energy (1 = perfectly fair)
- `speedup = t_twin_rollout / t_surrogate_rollout` (p50, p95 over ≥1000 rollouts)
- step-time p50/p95 (controller latency budget: < 2 s/min on CPU)
- Surrogate health (v3.2/v3.4): `bias_ewma`, `mismatch_rate`, `recovery_rate`, ensemble `spread`, `nonmono_rate`, transition-window RMSE; allocator: phase-imbalance index `Imb`

**Experiments (all seeds fixed; ≥3 seeds where stochastic; bootstrap 95% CIs over days):**
- **E1 Main comparison (5 days × 4 controllers):** violations, E_delivered, completion_ratio, Jain, step-time → `outputs/phase_summary.csv` + plot set P1 (temperature traces), P2 (energy bars), P3 (fairness/deadline scatter).
- **E2 Surrogate accuracy:** RMSE + max-T error quantiles (p50/p95/p99) on held-out scenarios, plus **transition-window error** (v3.4): metrics conditioned on windows where the dry-fraction state is actively evolving (`|ds/dt| > 0`) — the genuinely non-linear regime. **Capacity ladder (v3.4):** if transition-window error exceeds threshold, step width 32→64→96 and modes 16→24 (pre-registered policy, not silent auto-tuning).
- **E3 Speed microbenchmark:** twin vs surrogate rollout wall-clock; fallback_rate of guarded hybrid.
- **E4 Parameter-noise robustness:** σ ∈ {0.05, 0.1, 0.2} lognormal on twin params, 200 episodes each: violations + mean margin, guarded vs twin-only — **plus soil dry-out events** (§2.3 two-zone model; ρ_dry ∈ {1.5, 2.0, 3.0} K·m/W, T_crit ∈ {45, 50, 55} °C): the controller must remain violation-free as the effective R4 degrades (the red-team "thermal runaway" scenario, now inside the test suite).
- **E5 Ablations:** MLP vs PINN-residual variant vs FNO; noise-injection/long-context ablation optional.
- **E6 (stretch) GNN:** cross-construction generalization + CYMCAP external validation (roles per §5.3c v3.2).
**E7 Adversarial scenario bank (v3.2 — tricky-case suite):** (a) cold-start vs preconditioned day (must be near-identical — cold-start exploit detector); (b) mid-day soil dry-out event; (c) phase-skewed EV cluster (all stations on one phase); (d) heatwave day (`T_m + A_s` at design worst case); (e) EV surge immediately after a sustained high-load block (transient stacking); (f) out-of-distribution ambient for the surrogate (drift alarm must fire); (g) deadline-clash day (allocator stress); (h) induced controller latency spike (watchdog must emit `failsafe`); (i) seasonal boundary ramp mid-week (`T∞` drift); (j) **high-frequency pulse bank** (v3.3, design-corrected v3.5) — demand switching between 100% and 10% at periods {3, 4 min}, **strictly above the 2·dt Nyquist limit** (the v3.3 "every 2 min" wording sat exactly at Nyquist for the 1-min grid and is replaced); an optional 2-min extreme case is integrated by the twin at sub-minute steps (`dt_sub = 15 s`) with 1-min box-averaged demand fed to the FNO context (the anti-alias filter). Physics note: the RC ladder is an **overdamped diffusion system — "resonance" cannot occur**, and its thermal low-pass action attenuates 2-min load ripple by more than an order of magnitude (conductor τ is tens of minutes), so input aliasing cannot produce "catastrophically incorrect thermal integration" — the genuine risks being tested are surrogate input-distribution shift on high-frequency content and bisection resolution against peak-dominated demand profiles; (k) **engineered backfill trench** (v3.3) — fluidized thermal backfill (ρ ≈ 0.8 K·m/W, higher heat capacity) vs native soil: the controller must adapt ratings upward safely, proving it serves optimized modern trenches as well as degraded native ones; (l) **black-start telemetry vacuum** (v3.4): delete the prior-day demand file and corrupt the trailing buffer — assert pessimistic-envelope initialization, margin decay per §5.6, and 0 violations across the recovery window. Each case carries an explicit PASS condition recorded in phase_summary. (m) **V2G phase-balancing** (stretch, v3.5): skewed-phase day with a V2G-capable fraction — report `Imb` and `P_extra` with vs without V2G discharge.

**Acceptance criteria (hard):** AT-3 (G1: dynamic violations == 0 on all eval days), AT-8/E1 (G2: improvement > 0 on ≥1 day), G3 reported with AT-2 sanity, G4 via AT-8 rule, G5 demo lists all three "wow" elements: job-level urgency priority, 0-violation temperature trace, 4-way comparison.

---

## §7 — ENVIRONMENT & INSTALL

```bash
py -3.12 -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install numpy pandas pyarrow scipy matplotlib python-dateutil requests
pip install acnportal==0.3.3 torch --index-url https://download.pytorch.org/whl/cpu
pip install streamlit pyyaml pytest python-dotenv
pytest                          # from repo root — all tests must pass before E1
```
Python **3.12** (verified available via `py -3.12` on this machine; 3.13 fallback). torch CPU wheel (no GPU needed). `acnportal` pulled from PyPI. Colab alternative: same pip lines in a notebook, artifacts persisted to Drive (only for surrogate training if CPU too slow — optional).

## §8 — AGENT ENVIRONMENT: MCPs, SKILLS, PLUGINS, CONNECTORS (verified inventory)

**MCP servers (all six configured in `~/.zcode/cli/config.json`, each with `timeoutMs:60000`; a ZCode restart is required for them to appear as `mcp__<server>__*` tools in a NEW session):**
| Server | Use in this project | If unavailable (fallback) |
|---|---|---|
| `playwright` | ACN web-UI JSON export + timeSeries check (only if AT-1a fails) | `browser-use:control-browser` skill |
| `fetch` | docs/papers | native WebFetch tool |
| `context7` | library docs (acnportal, torch, streamlit) | native web search |
| `sequential-thinking` | long-horizon planning checklists | none needed |
| `filesystem` / `desktop-commander` | file+shell for other MCP clients | native Read/Write/Edit/Bash tools |

**Skills (already installed — no installs needed):** `pdf` (IEEE-format report), `pptx` (panel presentation), `docx` (if Word required), `gpt-researcher` (deep patent sweep), `browser-use:control-browser` (Playwright fallback), `skill-creator` (optional: wrap the runbook into a custom skill).
**Plugins:** existing set suffices; **optional** marketplace plugin `github` only if managing a GitHub remote with issues/PRs (needs owner's PAT).
**Connectors:** none required (public data, local compute).
**Anti-hallucination note for agents:** the two MCP tools always available in-session regardless of config are `4_5v_mcp.analyze_image` and `web_reader.webReader` — do not invent others; check your actual tool list.

## §9 — REPOSITORY MANIFEST
See `PLAN.md` §1 (scaffolded and authoritative): `configs/{twin,scenario_grid,controller}.yaml`, `src/dtr/{data,twin,surrogate,control,baselines,evaluation,viz}`, `scripts/01..04 + run_all.sh`, `tests/`, `outputs/`, `notebooks/`, `app/streamlit_app.py`, `docs/{patent_search,research_matrix,report}`. Every script writes to `outputs/`; every experiment has a YAML config copied into `configs/experiment_*.yaml` at run time.

## §10 — RUNBOOK (ordered, with acceptance tests)

| Step | Action | Acceptance test |
|---|---|---|
| 0a | Create venv, install §7 deps, `pytest` on the empty scaffold | test suite runs green (0 tests ok) |
| 0b | **AT-1a:** test `get_sessions_by_time(..., timeseries=True)` (§3.2) | documented outcome in `docs/data_findings.md` |
| 1 | `scripts/01_fetch_and_build_profiles.py` — fetch 5 eval days | `sessions_meta_*.csv` + `demand_minute_*.parquet` exist w/ §3.5 schemas; row counts > 0 |
| 2 | `scripts/02_build_jobs.py` | `jobs_*.csv`; **AT-2** deadline≥connection ≥99%; slack stats reported |
| 3 | Twin + integrator + analytical validation | **AT-5** RK4 vs closed-form ≤0.5% error at 3τ; **AT-6** RK4 vs solve_ivp max|ΔT| ≤ 0.1 °C over 24 h |
| 4 | `scripts/03_phase11_closed_loop.py` — 4 controllers × 5 days | **AT-3** guarded-hybrid & twin-only violations == 0 all days; **AT-4** property tests pass; E_dyn > E_static ≥1 day; phase_summary.csv written |
| 5 | Scenario grid + FNO training (`scripts/04_train_surrogate.py`) | E2 tables; **AT-7** speed measured; **AT-8** G4 rule decided (pass/fail honestly) |
| 6 | Ablations E5, noise E4, (stretch E6 GNN) | tables + CIs in phase_summary |
| 7 | `app/streamlit_app.py` — sliders (day, β, margin), temperature trace w/ 90°C line, energy bars, fairness/deadline panel, surrogate-vs-twin panel | demo runs `streamlit run app/streamlit_app.py` |
| 8 | Patent search (§12) → `docs/patent_search/` | **AT-9** ≥10 patents, each with patent number + ≥1 independent-claim element + overlap note |
| 9 | IEEE report (`docs/report/`, via pdf skill) + README + `run_all.sh` clean-venv rerun | **AT-10** one-command reproduction from fresh venv; every number in report traceable to `outputs/` |

## §11 — CHAIN-OF-VERIFICATION SUITE (living; re-run before any report/deadline)
1. timeSeries via API? (AT-1a) 2. deadlines sane? (AT-2) 3. violations == 0? (AT-3) 4. shield invariants hold? (AT-4) 5. twin matches analytics? (AT-5/6) 6. surrogate benefit real? (AT-7/8) 7. prior art cited with claim elements? (AT-9) 8. reproduction clean? (AT-10) 9. every README number ∈ outputs? 10. every citation fetched (not remembered)?

## §12 — PATENT SEARCH PROTOCOL (Phase 9)
Databases: Google Patents (patents.google.com), Espacenet (worldwide.espacenet.com). Search strings (v2-updated): "dynamic thermal rating underground cable" AND (machine learning | neural network | physics-informed | neural operator | Fourier neural operator | graph neural network); "cable ampacity" + (real-time | prediction | surrogate); CPC hints: G06N (ML) × H02G/H01B (cables), G01K (temperature estimation). **Deliverable (AT-9):** ≥10 closest patents, each: patent number, assignee/year, 1-line method summary, overlap note citing **≥1 independent claim element** verbatim (translated), differentiation sentence (technical-effect framing — physical thermal protection/control, NOT a business method; v1 trap §13). Log search strings + date + result counts for reproducibility.

## §13 — RISKS & TRAPS (v1 lessons + v2 audit additions)
1. Paywalled/messy data (IEX) → ACN-Data (solved, v1). 2. §3(k) business-method patent risk → physical-effect framing (solved strategy, v1). 3. Compute trap (PPO/SAC/Transformers) → shield + operator surrogate (solved strategy). 4. **Overconfidence trap** — never report unmeasured %s (v1 replaced fake numbers; keep it). 5. API expectation trap → **re-opened as AT-1a** (timeseries=True may make it moot). 6. Environment trap → pinned venv + lockfile (v1). 7. **NEW — surrogate drift near 90 °C:** guarded hybrid + margin schedule (§5.3b/5.4) + never trust surrogate for the final feasibility decision. 8. **NEW — twin mis-calibration:** every R/C value needs a source or explicit "representative" label; sensitivity check in E4 covers residual error. 9. Timezone bug (web UI <2019-10-10) → API dates ≥ 2019-10-15. 10. Colab session death → checkpoints to Drive; CPU-first design. 11. **NEW v3.1 — one-sided verification:** a bare twin check accepts safe-but-timid surrogate outputs, throttling energy (G2) → verify-and-grade escalation logic (§5.3b) + `recovery_rate` metric. 12. **NEW v3.1 — soil desiccation blindness:** static soil resistivity hides dry-out thermal runaway → state-dependent two-zone ρ_soil (§2.3) + E4 dry-out suite; full moisture physics out of scope, documented. 13. **NEW v3.1 — protection-hierarchy gap:** software-only safety is a field-deployment anti-pattern → §1.5 hierarchy statement + simulator watchdog; NG1 unchanged (capstone scope). 14. **NEW v3.2 — cold-start thermal vacuum:** uniform-ambient initialization gives the deep soil artificial headroom for 12–18 h and inflates G2 metrics → 48 h spin-up preconditioning for eval days and scenarios (§5.6, §5.1). 15. **NEW v3.2 — non-physical boundary volatility:** minute-scale air noise at the deep-soil boundary violates soil thermal damping → Kusuda–Achenbach annual T∞ boundary (§2.3). 16. **NEW v3.2 — phase imbalance:** single-phase EVSE aggregation underestimates losses → per-phase loss computation + λ_zs armour add-on + optional phase-aware allocator (§2.2, §5.5). 17. **NEW v3.2 — static-vs-transient validation mismatch:** steady-state datasets cannot certify a temporal solver → temporal gate = synthetic 72 h multi-cable suite; CYMCAP relabeled steady-state-only (§5.3c). 18. **NEW v3.3 — spectral smoothing of load steps:** mode-truncated operators can flatten sharp transient peaks (low-frequency bias) → spectral + Sobolev sensitivity losses and peak-weighted loss; guarded architecture caps the blast radius (§5.3a). 19. **NEW v3.3 — symmetric dry-out hysteresis:** memoryless two-zone re-wets instantly → dry-fraction state with `τ_wet ≥ 10·τ_dry` + chained multi-day evaluation (§2.3, §5.6). 20. **NEW v3.3 — "guaranteed stability" overclaims:** Neural ODEs carry no mathematical long-horizon guarantee; all learned models stay horizon-capped and twin-verified; graph-ODE adopted only via evidence-gated A/B (§5.3c). 21. **NEW v3.4 — boundary dielectric under-prediction:** static tanδ underestimates heat generation near 90 °C → temperature-dependent tanδ(T_c) in the twin; materiality at 11 kV quantified (~0.1% of losses — an EHV-critical mechanism, included because it is free and future-proofs the twin) (§2.2). 22. **NEW v3.4 — non-monotonic surrogate bisection:** neural surrogates carry no monotonicity guarantee → query-history isotonicity watchdog + twin fail-over (§5.3b). 23. **NEW v3.4 — telemetry-vacuum restart:** missing preconditioning history must force neither a charging halt nor a guessed benign state → pessimistic-envelope black start with decaying margin (§5.6). 24. **NEW v3.5 — exact-Nyquist test design:** a 2-min square wave sampled at 1 min sits exactly at Nyquist — adversarial test signals must sit strictly above 2·dt, and FNO inputs are box-averaged demand by construction (§6 E7-j). 25. **NEW v3.5 — phantom training-time twin cost:** the twin is never called during training; Sobolev targets are precomputed dataset columns — "differentiable twin or crippled training" is a false dilemma; the torch twin is adopted for exactness and gradient experiments, not speed necessity (§5.3a, §5.1). 26. **NEW v3.5 — discontinuous hysteresis derivative:** hard if/else in `ds/dt` degrades adaptive-solver efficiency → smooth sigmoid switching with documented transition width (§2.3). 27. **NEW v3.6 — 3-core averaging:** a single conductor node reports the *mean* core temperature and hides the hottest core under phase imbalance (≈13 °C spread at 200/50/50 A) → per-core nodes with star filler coupling; shield enforces max core (§2.3); the reviewer's `3·max(I)²` envelope rejected (overestimates losses ~2.7×). 28. **NEW v3.6 — shield–allocator routing mismatch:** verifying α·D while dispatching allocator weights means the twin verified a fiction → allocator composed inside the feasibility check (§5.4/§5.5). 29. **NEW v3.6 — W_d placement:** dielectric loss belongs in the insulation volume, not the conductor → Van Wormer 50/50 split (materiality ~0.003 °C at 11 kV, documented) (§2.3). FEA-referee pilot (FEniCSx, WSL, gated Week 7+) validates the RC twin externally; phase-switching allocation is sim-only behind a hardware-assumption flag.

## §14 — SOURCES (status-honest; "verified" = fetched/checked by an agent on 2026-09-14/15)

**A. Neural-operator methods (all verified):** Li et al., *FNO*, arXiv:2010.08895 (ICLR 2021) — mappings between function spaces; ≤3-orders speedup vs traditional PDE solvers; Burgers/Darcy/NS experiments (1-D-ness of Burgers: full text, not abstract). Pfaff et al., *MeshGraphNets*, arXiv:2010.03409 (ICLR 2021) — mesh-GNN, aero/structural/cloth, 1–2 orders speedup (encode-process-decode/noise-injection: full text only). Sanchez-Gonzalez et al., arXiv:2002.01666 (ICML 2020). Brandstetter et al., arXiv:2202.03376 (ICLR 2022). Library: github.com/neuraloperator/neuraloperator (official, MIT, PyTorch; 1-D support unconfirmed → custom FNO chosen).
**B. Thermal standards/physics (verified):** XLPE 90 °C normal / 100–105 °C emergency (elek.com; cableizer.com); IEC 60853-1/-2 cyclic rating factor M (IEC sample PDF); IEC 60287 T1–T4 method + ρ_th table: XLPE 3.5, PVC 5.0–6.0 K·m/W (cableizer.com/documentation/rho_i); soil 1.0–1.2 K·m/W (datasheets, e.g. Hellenic 185 mm² @1.2); NYU mutual-heating ladder model (research.engineering.nyu.edu).
**C. Prior-art landscape (verified searches):** PINN-cable/line/transformer papers plentiful (MDPI Energies 19(16):3788 layer-aware XLPE HVDC PINN; IJEPES PINN-DLR; EAAI 2-D temp-field PINN; transformer PINN). GNN-adjacent but not overlapping: line-GNN probabilistic DLR (arXiv 2512.04369); GNN cable-grid reliability (Applied Energy 2025, van Nooten); graph cable-temp preprint 2025; GNN cable-joint digital twin. **No paper found combining thermal-graph GNN + multi-cable mutual heating + safety-shield DTR** (NG3-compliant gap). CYMCAP dataset: data.mendeley.com/datasets/dmp355brwk (440 cases).
**D. API/infra (verified):** acnportal docs (DataClient signatures incl. `timeseries` flag — acnportal.readthedocs.io/en/latest/acndata/data_client.html); v1 empirical session-field list (trusted, re-verify runtime); ACN timezone caveat (portal warning, v1); environment probed: py 3.11/3.12/3.13 present, torch absent (install per §7).
**E. Unverified-by-agent (carry from v1, cite carefully):** the seven DTR cable DOIs (Brakelmann & Anders 2021 10.1109/TPWRD.2020.3026779; Enescu 2021 10.3390/en14092591; Sedaghat 2018 10.1109/TPWRD.2018.2841054; Olsen 2012 10.1109/PESGM.2012.6345324; Aras & Biçen 2010 10.1002/cae.20497; Petrović 2023 10.1016/j.epsr.2022.108916; Atoccsa 2024 10.3390/en17051023) and PINN refs (Raissi 2019; Karniadakis 2021; Gokhale 2022; Chen 2023; Li 2025; Zheng 2025 — DOIs in v1) — fetch each before citing in the final report.

## §15 — CHANGE-LOG & OPEN ITEMS
**v1→v2:** PINN → FNO primary + GNN stretch (G4, FR7, §8 Q&A, stack, sources); guarded-hybrid integration; parameter-channel robustness training; patent strings updated; surrogate metrics + demo panel; surrogate-drift trap.
**v2→v3 (this doc):** verified API signatures incl. **`timeseries=True`** (new AT-1a); grounded cable parameter table w/ calibration mandate; exact derived schemas; full pseudocode for shield/allocator/hybrid; AT-1..10; agent-environment inventory + fallbacks; anti-hallucination rules (§0); honest scope notes on every verified/unverified source.
**Open items:** AT-1a outcome; cable datasheet calibration (one 185 mm² 11 kV 3-core datasheet: R_ac, C, dimensions → R1..R4, C_i); ACN token; pick 5 eval dates; β, K_RECOV, dT_tol, spread_tol, margin final values from E1/E3; pin the IS 7098 emergency-overload figure from the standard text itself (105 °C vs 130 °C — IEC/US practice is 105 °C per Cableizer/EPRI-type rules; 130 °C appears in GB/manufacturer practice; quote the clause in the report); Kusuda–Achenbach site parameters (T_m, A_s, α_soil — cite NBS Report 8972, verified 2026-09-15); document the EVSE→phase assignment assumption in the report; calibrate hysteresis constants `{τ_dry, τ_wet}` and backfill properties against published trench studies; fetch the Sobolev-training citation (arXiv:1706.04859) before use; sweep `{λ_peak, λ_spec, λ_sob}` in E5; calibrate `{tanδ_20, β}` and document the 11 kV materiality calculation; tune `ε_mono` and black-start `{Δ_bs, T_half}`; choose sigmoid steepness `k` and verify its smearing width vs the safety margin; document the V2G-capable fleet fraction assumption.

**v3.0 → v3.1 (red-team response, 2026-09-15):** soil dry-out modeled via state-dependent two-zone ρ_soil + E4 dry-out suite (ISSUE-01); §1.5 protection hierarchy + simulator watchdog, NG1 retained with justification (ISSUE-02); FNO off-grid output mechanism specified + resolution-transfer test, "cannot compile" claim rejected (ISSUE-03); guarded hybrid upgraded from one-sided verification to verify-and-grade with escalation/recovery, K_RECOV + recovery_rate added (ISSUE-04); XLPE limits corrected to the trinary picture 90 / 105-vs-130 / 250 with standards divergence documented (ISSUE-05); GNN guardrails + drift metric, graph-ODE demoted to optional exploration (ISSUE-06).

**v3.1 → v3.2 (red-team Round 3 response, 2026-09-15 — commercial-hardening pack):** two-sided surrogate verification (safety + ΔT ≤ 1.5 °C temperature consistency) + 3-seed ensemble disagreement trigger + `bias_ewma` / `mismatch_rate` metrics + drift alarm latching twin-only mode (§5.3b); margin schedule consumes bias + optional conformal per-state margin (§5.4); 48 h thermal preconditioning for all eval days and scenarios (§5.6, §5.1); Kusuda–Achenbach annual T∞ deep-soil boundary replacing air-temperature noise, citation verified NBS Report 8972 (§2.3); per-phase loss computation + λ_zs armour add-on replacing the blanket 3-phase assumption, imbalance penalty emergent from Σ I_i² physics (§2.2); phase-aware allocation stretch (§5.5); validation roles separated — synthetic 72 h multi-cable transient suite (temporal gate) vs CYMCAP (steady-state spatial only) (§5.3c); adversarial scenario bank E7 (§6); surrogate-health + imbalance metrics (§9); traps 14–17 (§13).

**v3.2 → v3.3 (red-team Round 4 response, 2026-09-15 — SciML hardening):** spectral + Sobolev-sensitivity losses and peak-weighted loss guarding against mode-truncation smoothing of load steps (§5.3a); soil-moisture **hysteresis** — dry-fraction state `s(t)` with `τ_wet ≥ 10·τ_dry`, replacing the memoryless two-zone mapping; `s` added to twin state and surrogate channels; **chained multi-day evaluation** (consecutive days, state carried across boundaries) (§2.3, §5.1, §5.6); MeshODENet demoted to evidence-gated A/B in E6 with the "guaranteed stability" claim rejected (§5.3c); E7 extended with high-frequency pulse bank and engineered-backfill cases; traps 18–20 (§13).

**v3.3 → v3.4 (red-team Round 5 response, 2026-09-15 — extreme-boundary + deployment hardening):** temperature-dependent dielectric loss `tanδ(T_c)` with the 11 kV materiality calculation made explicit (§2.2); **monotonicity watchdog** — query-history isotonicity check on surrogate bisection with twin fail-over (§5.3b); **black-start pessimistic-envelope recovery** with decaying margin (§5.6); transition-window surrogate metrics + pre-registered capacity ladder (§6 E2); E7(l) black-start case; surrogate-health metrics extended (`nonmono_rate`) (§9); traps 21–23 (§13).

**v3.4 → v3.5 (build-phase adjudication of the second agent's implementation review, 2026-09-15):** E7(j) redefined off exact-Nyquist + twin sub-stepping + box-averaged FNO context — aliasing observation accepted, "catastrophic integration" severity rejected (thermal low-pass attenuation quantified) (D1); Sobolev targets clarified as offline-precomputed dataset columns — the "training-time gradient bottleneck" rejected as a category error, while the **differentiable torch twin** is adopted for exact autograd sensitivities and gradient experiments, with bisection retained for 1-D α (provably optimal) (D2/P1, §5.1/§5.3a); hysteresis switching smoothed to a sigmoid (D3, §2.3); **Conformal Predictive Control promoted to first-class** with honest finite-sample coverage and the stacking composition rule (P2, §5.4); **V2G phase-balancing accepted as a gated stretch** with E-case (P3, §5.5/§6); torch-twin cross-validation extends AT-6; traps 24–26 (§13).

**v3.5 → v3.6 (Round 7 response, 2026-09-16 — per-core physics + control-routing fixes):** twin expanded to **three conductor nodes** star-coupled through the filler node (`R_core = 3·R1`, balanced-limit preserving); shield enforces `max(T_a,T_b,T_c)`; Van Wormer W_d split (§2.3); **allocator composed inside the shield's feasibility check** — the twin verifies the dispatched profile (§5.4/§5.5); phase-switching accepted only as a hardware-flagged sim mode; FEA (FEniCSx) accepted as a gated *referee* study, not a training source; traps 27–29 (§13).
