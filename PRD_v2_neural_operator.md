# PRD v2 — Neural-Operator DTR + Safe Smart EV Charging for 11 kV XLPE Underground Cable

**Owner:** Hamsavarthan H (24BEE0191)
**Type:** EEE + ML + Control (software-only)
**Timeline:** 10 weeks (solo)
**Primary Goal:** Capstone-level system + strong demo + (optional) provisional patent viability check
**Secondary Goal:** Publishable-quality evaluation pipeline (baselines, metrics, reproducibility)

> **v2 change:** The PINN surrogate (v1) is replaced by a **neural-operator surrogate** —
> **FNO (Fourier Neural Operator)** as the primary architecture, with a **MeshGraphNets-style
> GNN** as the research stretch goal. Everything else (physics twin, safety shield, allocator,
> baselines, metrics, demo) is unchanged. A full change-log is at the bottom.

---

## 1) Background / Problem

### 1.1 Problem
Simultaneous EV charging increases feeder loading. Underground 11 kV XLPE cable temperature
must remain below the safe insulation limit (target 90°C) to reduce accelerated aging/failure
risk. Static ratings are conservative; dynamic operation can safely deliver more energy if
thermal constraints are enforced.

### 1.2 Product statement
Build a closed-loop EV charging control system that:
- predicts/estimates cable core temperature from loads + ambient conditions
- enforces a hard safety constraint `T_core(t) ≤ 90°C`
- maximizes delivered EV charging energy
- allocates charging fairly and deadline-aware across sessions
- adds an ML component: a **neural-operator surrogate (FNO)** trained on a thermal digital
  twin to speed and stabilize horizon predictions *(stretch: graph operator for multi-cable feeders)*
- provides a Streamlit demo

## 2) Users & Stakeholders — unchanged from v1

## 3) Goals / Non-goals

### 3.1 Goals (must-have)
- **G1. Safety:** dynamic controller achieves 0 thermal violations (minutes where T > 90°C).
- **G2. Performance:** dynamic controller delivers more kWh than a safe static baseline.
- **G3. Deadline + fairness:** deadline-aware scheduling (job/session-level) + fairness metric.
- **G4. ML:** train an **FNO surrogate** on the digital twin and show measurable benefit
  (horizon-eval speedup ≥ 30% **or** robustness under parameter noise). *(v2: replaces PINN)*
- **G5. Demo:** Streamlit dashboard + plots + reproducible run scripts.

### 3.2 Non-goals (explicit)
- NG1. Not a hardware/sensor project (sensors optional extension).
- NG2. Not continuous-action DRL / Transformers (compute and stability risk).
- NG3. No "first ever" claims — novelty framed as measurable integration + safety shield mechanism.

## 4) Key Requirements

### 4.1 Functional Requirements
- **FR1. Data ingestion (ACN-Data):** session metadata (connection/disconnect/doneChargingTime/
  kWhDelivered); optional time-series via web export (Playwright), else metadata-only.
- **FR2. Demand model:** 1-minute profiles — EVSE-aggregated (MVP) + session/job-level (scheduling).
- **FR3. Thermal digital twin:** RC lumped ODE (IEC 60853-style) + numeric integrator (unchanged).
  *v2: expose the twin as a 1-D state/field generator so its trajectories can train the operator.*
- **FR4. Controller (safety shield):** receding horizon H = 30 min, dt = 1 min; feasibility via
  bisection on scaling factor α or power cap; explicit robustness margin (unchanged).
  *v2: bisection evaluates the **FNO surrogate first**; the final chosen α is **verified once
  against the numeric twin** (guarded-hybrid pattern, see FR7).*
- **FR5. Allocator:** hybrid urgency (deadline) + fairness (anti-starvation) objective (unchanged).
- **FR6. Baselines:** no-control / static cap / dynamic shield (unchanged).
- **FR7. Neural-operator surrogate (v2, replaces PINN):**
  - **Input (function channels over the horizon window):** current thermal state (T_core, T_sheath,
    T_soil or RC-node temps), candidate load profile α·P(t) for t..t+30, ambient temperature profile,
    and **parameter channels** (thermal resistances/capacities, soil resistivity — enables
    robustness-to-uncertainty training).
  - **Output:** T_core trajectory over the 30-min horizon (and max T_core for the feasibility test).
  - **Architecture:** FNO with 1-D Fourier layers over the time axis (tiny model, CPU-friendly);
    grid = the 30-step horizon (+ context window). *(Stretch: MeshGraphNets-style GNN whose nodes
    are RC thermal nodes / multiple cables, edges = thermal resistances + mutual-heating couplings.)*
  - **Training data:** scenario grid from the digital twin (vary T_amb, baseload, thermal params,
    load profiles) — labels are twin trajectories (supervised), optional physics-residual
    regularization retained as an ablation.
  - **Integration pattern (guarded hybrid):** surrogate proposes the safe α (fast); numeric twin
    verifies the final candidate once per control step (safety-airtight, still ≥ 30% fewer
    numeric rollouts — bisection runs entirely on the surrogate).
  - **Acceptance:** horizon-eval speedup ≥ 30% **or** robust performance under parameter noise
    (violation-free with smaller margin than numeric-only baseline).
- **FR8. Evaluation:** phase_summary.csv + plots + reproducible config (unchanged).
- **FR9. Demo dashboard:** temperature trace, demand vs delivered, violations = 0,
  fairness/deadline outcomes (unchanged) — **plus** a "surrogate vs twin" speed/accuracy panel (v2).

## 5) Data — unchanged from v1 (ACN-Data verified findings, timezone caveat, timeSeries acceptance test)

## 6) Phases — status unchanged; Phase 11 PASSED (0 violations with margin search)
*Phase 11 rerun in v2 must use the **guarded-hybrid** controller (see FR4/FR7) and reproduce the 0-violation result.*

## 7) Technical Architecture

### 7.1 Modules
1. Data ingestion — ACN metadata via acnportal; optional web export via Playwright
2. Job builder — session → job (start = connectionTime; deadline = doneChargingTime, fallback
   disconnectTime; energy = userInputs request, else documented kWhDelivered proxy)
3. Thermal digital twin — RC ODE + RK4/solve_ivp (also the surrogate's label generator)
4. Safety shield — receding-horizon bisection; **FNO inside the loop, numeric twin as final verifier**
5. Allocator — urgency + fairness weights → per-job power setpoints
6. **Neural-operator surrogate (v2)** — FNO trainer + scenario-grid data generator + eval harness
   *(stretch: MeshGraphNets-style GNN variant for multi-cable / parameter generalization)*
7. Evaluation & plotting
8. Streamlit demo

## 8) ML Definition (for professor Q&A — v2 wording)
- **What learning is used?** Supervised operator learning: an FNO maps (thermal state, candidate
  load profile, ambient profile, cable parameters) → 30-min temperature trajectory. Labels come
  from the IEC-style RC digital twin (not measured core temperature). Optional physics-residual
  regularization is an ablation, not the core mechanism.
- **Why an operator instead of a PINN?** The task is "profile → trajectory" (function → function):
  exactly what neural operators learn. One FNO forward pass replaces a 30-step ODE rollout per
  bisection iteration; parameter channels give robustness to thermal-parameter uncertainty for free.
- **Safety then?** The surrogate never has final authority: the numeric twin verifies the chosen
  α each control step. Surrogate = speed; twin = truth; shield = enforcement.
- **Not used (by design):** RL, unsupervised clustering, Transformers.
- **Stretch story:** MeshGraphNets-style GNN generalizes the operator across cable constructions
  and multi-cable mutual heating — PINN v1 could not express this.

## 9) Metrics & Acceptance Criteria — unchanged (violations == 0; E_dyn > E_static; completion ratio;
Jain fairness; step-time p50/p95) **plus v2 surrogate metrics:** horizon-eval speedup ≥ 30%,
trajectory RMSE vs twin, violations under parameter noise.

## 10) Demo Requirements — unchanged (3 wow-factors) + surrogate speed/accuracy panel

## 11) Tech Stack
- Python 3.12 venv; acnportal, pandas, numpy, python-dateutil, requests; pyarrow; matplotlib
- **ML: torch (FNO, ~1-D Fourier layers — custom impl. or `neuraloperator` lib), scipy (solve_ivp)**
- cvxpy optional (QP allocator); streamlit demo
- **Zcode MCP mapping (v2, updated for installed servers):**
  `mcp__playwright__*` ACN web UI JSON download + timeSeries verification;
  `mcp__fetch__*` docs/papers; `mcp__context7__*` library docs (acnportal, torch, streamlit,
  neuraloperator); `mcp__sequential-thinking__*` planning checklists. *(All four are configured;
  require ZCode restart to go live.)*

## 12) Sources — v1 list retained, plus neural-operator additions
- **Li et al. (2021), Fourier Neural Operator** — https://arxiv.org/abs/2010.08895 (ICLR 2021)
- **Pfaff et al. (2021), MeshGraphNets** — https://arxiv.org/abs/2010.03409 (ICLR 2021)
- Sanchez-Gonzalez et al. (2020), Graph Network-based Simulations — https://arxiv.org/abs/2002.01666 (ICML 2020)
- Brandstetter et al. (2022), Message Passing Neural PDE Solvers — https://arxiv.org/abs/2202.03376 (ICLR 2022)
- PINN background retained for the ablation/related-work section (Raissi 2019; Karniadakis 2021;
  Gokhale 2022; Chen 2023; Li 2025; Zheng 2025 — DOIs as in v1)

## 13) Traps — v1 list retained; amended:
- Compute trap note now reads: "MPC-style shield + **neural-operator surrogate (FNO)**"
- New trap added: *surrogate drift near the 90°C boundary* → mitigated by guarded-hybrid
  verification + margin search (FR4/FR7).

## 14) Runbook (execution checklist — v2)
- **Step 0:** MCP availability (post-restart): `mcp__playwright__*`, `mcp__fetch__*` list OK.
- **Step 1:** Data pipeline (metadata) → `outputs/demand_minute_*.parquet`, `outputs/sessions_meta_*.csv`
  *(needs ACN token or v1 outputs restored into `ML_Project/outputs/`)*
- **Step 2:** Job dataset `jobs_caltech_YYYY-MM-DD.csv` (connectionTime, deadlines, energy requests)
- **Step 3:** Controller + baselines closed-loop day sim; **dynamic (guarded-hybrid) = 0 violations**
- **Step 4 (v2):** **Surrogate training:** twin scenario-grid generation → FNO training (torch) →
  speed/accuracy/noise-robustness eval vs numeric solver → ablation table (vs PINN-style MLP baseline
  and, optionally, physics-residual variant). Stretch: GNN variant.
- **Step 5:** Streamlit demo (unchanged + surrogate panel)
- **Step 6:** Patent search (Phase 9, pending) — **v2 search strings added:** "dynamic thermal rating
  neural operator", "FNO cable temperature", "graph neural network cable rating", "machine learning
  cable ampacity"; deliver 10 closest patents + claim-overlap notes citing patent numbers + ≥1
  independent claim element each.

## 15) Chain-of-verification tasks — v1 retained with G4/FR7 wording swapped to the surrogate criteria
(timeSeries JSON check; deadline ≥ connection for >99% jobs; dynamic violations == 0;
**surrogate benefit: ≥30% horizon-eval speedup or robust performance under parameter noise**;
patent prior art with claim-element citations)

---

## Change-log v1 → v2
| # | v1 | v2 |
|---|----|----|
| 1 | PINN surrogate (§1.2, G4, FR7, §7.1, §8, §11, §13, §14-4) | FNO neural-operator surrogate; MeshGraphNets-style GNN as stretch |
| 2 | Bisection evaluates numeric twin each iteration | Guarded hybrid: surrogate proposes, twin verifies final α once per step |
| 3 | PINN training set | Twin scenario-grid **with parameter channels** (robustness to thermal-parameter uncertainty) |
| 4 | Sources: PINN only | + FNO / MeshGraphNets / GNS / MP-PDE papers; PINN kept as related work + ablation |
| 5 | Patent strings generic | + neural-operator DTR search strings |
| 6 | — | Surrogate metrics in §9; surrogate panel in demo §10; surrogate-drift trap in §13 |

**Unchanged:** physics twin, safety shield mechanics, allocator, baselines, data pipeline,
metrics (§9), demo wow-factors (§10), all verification tests except the PINN-specific one.
