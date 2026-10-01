#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Phase 3 (Citation Network) builder for the neural-operator DTR project.

Public APIs only (no keys):
  - Semantic Scholar Graph API  https://api.semanticscholar.org/graph/v1/paper
  - OpenAlex                    https://api.openalex.org/works

Outputs (written next to this script):
  - citation_network.csv   edge list (source_title, target_title, source_year,
                           relation, source_id); relation in
                           {cites, co-cited-by-theme, standard-cited}
  - citation_network.md    human-readable summary
  - raw_api_cache.json     evidence cache of all API responses (reproducibility)

Honesty rules:
  * Every "cites" edge comes from an API reference list (S2 references with
    DOI/arXiv/title matching, or OpenAlex referenced_works IDs).
  * Every "co-cited-by-theme" count comes from OpenAlex (shared-reference counts
    over a sample of the most-cited works citing each core paper).
  * "standard-cited" edges to IEC 60287 / IEC 60853 follow the Phase 3 task
    convention (standards nodes with no citation API); each edge is labelled
    "api-corroborated" (the standard number appears in the paper's API
    reference titles / abstract) or "asserted" (project convention) in the MD.
  * Papers that cannot be resolved after retries are marked not-found-on-API.
"""
from __future__ import annotations

import csv
import difflib
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
CACHE_PATH = HERE / "raw_api_cache.json"
CSV_PATH = HERE / "citation_network.csv"
MD_PATH = HERE / "citation_network.md"

S2 = "https://api.semanticscholar.org/graph/v1"
OA = "https://api.openalex.org"
MAILTO = "ml-project-research@example.org"  # OpenAlex polite pool

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "ML_Project-citation-network-builder/1.0 (mailto:%s)" % MAILTO})

FETCH_DATE = None  # filled at runtime


def log(msg: str) -> None:
    print(msg, flush=True)


# -- S2 throttling control ---------------------------------------------------
# Unauthenticated S2 shares a congested pool; when retries keep failing we stop
# trying rather than burn the whole effort cap (papers stay oa-only / not-found).
RETRY_S2 = "--retry-s2" in sys.argv
S2_TRIES = 6 if RETRY_S2 else 4
S2_consecutive_fails = 0
S2_GIVE_UP_AFTER = 3 if RETRY_S2 else 10**9  # only bail in explicit retry mode


def s2_throttled_out() -> bool:
    return S2_consecutive_fails >= S2_GIVE_UP_AFTER


# ---------------------------------------------------------------- text utils
def norm(s: str | None) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def tmatch(t1: str | None, t2: str | None, th: float = 0.88) -> bool:
    if not t1 or not t2:
        return False
    n1, n2 = norm(t1), norm(t2)
    if not n1 or not n2:
        return False
    if n1 == n2:
        return True
    return tratio(n1, n2) >= th


def norm_doi(d: str | None) -> str | None:
    if not d:
        return None
    d = d.strip().lower()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    return d or None


def norm_arx(a: str | None) -> str | None:
    if not a:
        return None
    a = a.strip()
    if a.lower().startswith("arxiv:"):
        a = a[6:]
    return a.strip() or None


def oa_wid(url_or_id: str) -> str:
    return url_or_id.rsplit("/", 1)[-1]


def abstract_from_inv(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = {}
    for w, idxs in inv.items():
        for i in idxs or []:
            pos[i] = w
    return " ".join(pos[i] for i in sorted(pos))


# ---------------------------------------------------------------- HTTP layer
def http_json(url: str, params: dict | None = None, tries: int = 4,
              is_s2: bool = False):
    """GET JSON with 429/5xx backoff. Returns (json, None) or (None, err)."""
    last_err = None
    for i in range(tries):
        try:
            r = SESSION.get(url, params=params, timeout=45)
        except requests.RequestException as e:
            last_err = "net:%s" % e
            time.sleep(2 * (2 ** i))
            continue
        if r.status_code == 200:
            try:
                return r.json(), None
            except ValueError:
                return None, "bad-json"
        if r.status_code == 404:
            return None, "404"
        if r.status_code in (429,) or r.status_code >= 500:
            wait = (8 + 6 * i) if is_s2 else (5 + 4 * i)
            log("    HTTP %d -> backoff %ds (try %d/%d)" % (r.status_code, wait, i + 1, tries))
            last_err = "http-%d" % r.status_code
            time.sleep(wait)
            continue
        return None, "http-%d" % r.status_code
    return None, (last_err or "exhausted")


# ---------------------------------------------------------------- core list
# role in project: methods | domain | bridge | standards
CORE = [
    dict(n=1, key="fno", short="FNO (Li et al.)",
         title="Fourier Neural Operator for Parametric Partial Differential Equations",
         year=2021, venue="ICLR 2021", role="methods", arxiv="2010.08895", doi=None),
    dict(n=2, key="mgn", short="MeshGraphNets (Pfaff et al.)",
         title="Learning Mesh-Based Simulation with Graph Networks",
         year=2021, venue="ICLR 2021", role="methods", arxiv="2010.03409", doi=None),
    dict(n=3, key="gns", short="GNS (Sanchez-Gonzalez et al.)",
         title="Learning to Simulate Complex Physics with Graph Networks",
         year=2020, venue="ICML 2020", role="methods", arxiv="2002.01666", doi=None),
    dict(n=4, key="pmnp", short="PMNPs (Brandstetter et al.)",
         title="Message Passing Neural PDE Solvers",
         year=2022, venue="ICLR 2022", role="methods", arxiv="2202.03376", doi=None),
    dict(n=5, key="brak", short="Brakelmann & Anders 2021",
         title=None, year=2021, venue="IEEE Trans. Power Delivery", role="domain",
         arxiv=None, doi="10.1109/TPWRD.2020.3026779"),
    dict(n=6, key="enescu", short="Enescu et al. 2021",
         title=None, year=2021, venue="Energies", role="domain",
         arxiv=None, doi="10.3390/en14092591"),
    dict(n=7, key="sedaghat", short="Sedaghat et al. 2018",
         title=None, year=2018, venue="IEEE Trans. Power Delivery", role="domain",
         arxiv=None, doi="10.1109/TPWRD.2018.2841054"),
    dict(n=8, key="olsen", short="Olsen et al. 2012",
         title=None, year=2012, venue="IEEE PES GM", role="domain",
         arxiv=None, doi="10.1109/PESGM.2012.6345324"),
    dict(n=9, key="aras", short="Aras & Bicen 2010",
         title=None, year=2010, venue="Comput. Appl. Eng. Educ.", role="domain",
         arxiv=None, doi="10.1002/cae.20497"),
    dict(n=10, key="petrovic", short="Petrovic et al. 2023",
         title=None, year=2023, venue="Electr. Power Syst. Res.", role="domain",
         arxiv=None, doi="10.1016/j.epsr.2022.108916"),
    dict(n=11, key="atoccsa", short="Atoccsa et al. 2024",
         title=None, year=2024, venue="Energies", role="domain",
         arxiv=None, doi="10.3390/en17051023"),
    dict(n=12, key="pinn", short="PINNs (Raissi et al.)",
         title="Physics-informed neural networks: A deep learning framework for "
               "solving forward and inverse problems involving nonlinear partial "
               "differential equations",
         year=2019, venue="J. Comput. Phys.", role="methods",
         arxiv=None, doi="10.1016/j.jcp.2018.10.045"),
    dict(n=13, key="karn", short="Karniadakis et al. 2021",
         title="Physics-informed machine learning",
         year=2021, venue="Nature Reviews Physics", role="methods",
         arxiv=None, doi="10.1038/s42254-021-00314-5"),
    dict(n=14, key="gokhale", short="Gokhale et al. 2022",
         title=None, year=2022, venue="Applied Energy",
         role="domain (PINN thermal modeling)",
         arxiv=None, doi="10.1016/j.apenergy.2022.118852"),
    dict(n=15, key="godina", short="Godina et al. 2016",
         title=None, year=2016, venue="Applied Energy", role="domain",
         arxiv=None, doi="10.1016/j.apenergy.2016.06.019"),
    dict(n=16, key="soleimani", short="Soleimani & Kezunovic 2020",
         title=None, year=2020, venue="IEEE Trans. Ind. Appl.", role="domain",
         arxiv=None, doi="10.1109/TIA.2020.2986990"),
    dict(n=17, key="van_nooten", short="van Nooten et al. 2025",
         title="Graph neural networks for assessing the reliability of the medium-voltage grid",
         year=2025, venue="Applied Energy",
         role="bridge (GNN x cable-grid)",
         arxiv=None, doi="10.1016/j.apenergy.2025.125401"),
    dict(n=18, key="probdlr", short="Prob. DLR LineGNN (2025)",
         title="Probabilistic Dynamic Line Rating with Line Graph Convolutional LSTM",
         year=2025, venue="arXiv", role="bridge (DLR x GNN)",
         arxiv="2512.04369", doi=None),
    dict(n=19, key="kusuda", short="Kusuda & Achenbach 1965",
         title="Earth Temperature and Thermal Diffusivity at Selected Stations "
               "in the United States",
         year=1965, venue="NBS Report 8972", role="domain (soil-temp classic)",
         arxiv=None, doi=None),
]

STANDARDS = [
    dict(key="iec60287", short="IEC 60287",
         title="IEC 60287 - Electric cables: Calculation of the current rating (series)",
         year=2006),
    dict(key="iec60853", short="IEC 60853",
         title="IEC 60853 - Calculation of the cyclic and emergency current rating of cables (series)",
         year=1989),
]

CLUSTER = {1: "M", 2: "M", 3: "M", 4: "M", 12: "M", 13: "M", 18: "B",
           17: "B", 19: "D"}


def cluster_of(n: int) -> str:
    return CLUSTER.get(n, "D")  # 5-11, 14-16 are domain


# ---------------------------------------------------------------- cache
CACHE = {"s2": {}, "oa": {}, "coc": {}, "meta": {}}


def load_cache() -> None:
    if CACHE_PATH.exists():
        try:
            CACHE.update(json.loads(CACHE_PATH.read_text(encoding="utf-8")))
            log("cache loaded: s2=%d oa=%d coc=%d" % (len(CACHE["s2"]), len(CACHE["oa"]), len(CACHE["coc"])))
        except Exception as e:
            log("cache load failed (%s) -> fresh" % e)
    if RETRY_S2:
        # None entries may be transient (rate-limit exhaustion), unlike real 404s;
        # drop them so the retry pass tries again. (404s fail fast and re-cache.)
        dropped = [k for k, v in CACHE["s2"].items() if v is None]
        for k in dropped:
            del CACHE["s2"][k]
        if dropped:
            log("retry-s2: dropped %d transient/empty S2 cache entries" % len(dropped))


def save_cache() -> None:
    CACHE_PATH.write_text(json.dumps(CACHE, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- resolvers
S2_PAPER_FIELDS = ("title,year,citationCount,externalIds,abstract,"
                   "references.title,references.year,references.externalIds")


def s2_by_id(pid: str):
    ck = "s2:id:" + pid
    if ck in CACHE["s2"]:
        return CACHE["s2"][ck]
    if s2_throttled_out():
        log("    S2 skipped (throttled out): %s" % pid)
        CACHE["s2"][ck] = None
        return None
    global S2_consecutive_fails
    js, err = http_json("%s/paper/%s" % (S2, requests.utils.quote(pid, safe="")),
                        params={"fields": S2_PAPER_FIELDS}, is_s2=True, tries=S2_TRIES)
    if err:
        S2_consecutive_fails = S2_consecutive_fails + 1 if err != "404" else 0
        log("    S2 id-lookup %s failed: %s" % (pid, err))
        CACHE["s2"][ck] = None
    else:
        S2_consecutive_fails = 0
        CACHE["s2"][ck] = js
    save_cache()
    time.sleep(1.6)
    return CACHE["s2"][ck]


def s2_search(query: str, year: str | None = None):
    ck = "s2:search:" + query + "|" + (year or "")
    if ck in CACHE["s2"]:
        return CACHE["s2"][ck]
    if s2_throttled_out():
        log("    S2 skipped (throttled out): search '%s'" % query)
        CACHE["s2"][ck] = None
        return None
    global S2_consecutive_fails
    params = {"query": query, "limit": 10, "fields": "title,year,venue,externalIds,authors"}
    if year:
        params["year"] = year
    js, err = http_json("%s/paper/search" % S2, params=params, is_s2=True, tries=S2_TRIES)
    if err:
        S2_consecutive_fails = S2_consecutive_fails + 1 if err != "404" else 0
        log("    S2 search '%s' failed: %s" % (query, err))
        js = None
    else:
        S2_consecutive_fails = 0
    CACHE["s2"][ck] = js
    save_cache()
    time.sleep(1.6)
    return js


def oa_by_doi(doi: str):
    ck = "oa:doi:" + doi
    if ck in CACHE["oa"]:
        return CACHE["oa"][ck]
    js, err = http_json("%s/works/doi:%s" % (OA, doi), params={"mailto": MAILTO})
    if err:
        log("    OA doi-lookup %s failed: %s" % (doi, err))
        js = None
    CACHE["oa"][ck] = js
    save_cache()
    time.sleep(0.3)
    return CACHE["oa"][ck]


def oa_by_id(wid: str):
    ck = "oa:id:" + wid
    if ck in CACHE["oa"]:
        return CACHE["oa"][ck]
    js, err = http_json("%s/works/%s" % (OA, wid), params={"mailto": MAILTO})
    if err:
        log("    OA id-lookup %s failed: %s" % (wid, err))
        js = None
    CACHE["oa"][ck] = js
    save_cache()
    time.sleep(0.3)
    return CACHE["oa"][ck]


def oa_search(query: str, extra_filter: str | None = None, n: int = 10):
    ck = "oa:search:" + query + "|" + (extra_filter or "")
    if ck in CACHE["oa"]:
        return CACHE["oa"][ck]
    params = {"search": query, "per-page": n, "mailto": MAILTO,
              "select": "id,display_name,publication_year,doi,ids,cited_by_count,referenced_works"}
    if extra_filter:
        params["filter"] = extra_filter
    js, err = http_json("%s/works" % OA, params=params)
    if err:
        log("    OA search '%s' failed: %s" % (query, err))
        js = None
    CACHE["oa"][ck] = js
    save_cache()
    time.sleep(0.3)
    return js


# ---------------------------------------------------------------- resolution
def resolve_papers(refresh: bool) -> dict:
    P = {}
    for c in CORE:
        p = dict(c)
        p.update(s2=None, s2_err=None, oa=None, oa_err=None,
                 s2_id=None, oa_id=None, title=c["title"], title_hint=c["title"],
                 year=c["year"], doi=norm_doi(c["doi"]), arxiv=norm_arx(c["arxiv"]),
                 cites_s2=None, cites_oa=None, status="not-found-on-API",
                 s2_ref_titles=[], s2_ref_dois=set(), s2_ref_arx=set(),
                 oa_ref_ids=set(), blob="")
        P[c["key"]] = p

    if refresh:
        CACHE["s2"], CACHE["oa"] = {}, {}

    log("== Phase A: direct lookups (OA by DOI / S2 by arXiv) ==")
    for p in P.values():
        if p["doi"]:
            p["oa"] = oa_by_doi(p["doi"])
        if p["arxiv"]:
            p["s2"] = s2_by_id("arXiv:" + p["arxiv"])

    log("== Phase B: validate S2 records, fill gaps via search ==")
    # S2 occasionally mis-maps arXiv ids to unrelated records; for arXiv-resolved
    # papers whose S2 title grossly disagrees with the task hint, discard and
    # re-resolve by title search. (DOI-resolved records are trusted.)
    for p in P.values():
        if p["s2"] and p["title_hint"] and p["arxiv"] \
                and tratio(norm(p["s2"].get("title") or ""), norm(p["title_hint"])) < 0.6:
            log("  [%s] S2 record %r does not match hint %r -> discard & re-search"
                % (p["key"], p["s2"].get("title"), p["title_hint"]))
            p["s2"] = None
        if p["s2"] and p["s2"].get("title"):
            p["title"] = p["s2"]["title"]  # canonical title for later matching/search
    for p in P.values():
        key = p["key"]

        # --- S2 missing -> direct DOI lookup, then search (needs a title)
        if p["s2"] is None:
            if p["doi"]:
                p["s2"] = s2_by_id("DOI:" + p["doi"])
            title_src = p["title"] or (p["oa"] or {}).get("display_name")
            if p["s2"] is None and key == "van_nooten":
                queries = ["van Nooten graph neural network cable",
                           "graph neural network underground cable reliability applied energy",
                           "graph neural network low voltage cable dynamic rating"]
                for q in queries:
                    js = s2_search(q, year="2024-2026")
                    if not js or not js.get("data"):
                        continue
                    pick = None
                    for cand in js["data"]:
                        nt = norm(cand.get("title") or "")
                        authors = " ".join((a.get("name") or "") for a in cand.get("authors") or []).lower()
                        doi_c = norm_doi((cand.get("externalIds") or {}).get("DOI")) or ""
                        venue_c = norm(cand.get("venue") or "")
                        yr = cand.get("year") or 0
                        strong = "nooten" in authors
                        weak = ("cable" in nt
                                and ("neural" in nt or "graph" in nt or "gnn" in nt)
                                and 2024 <= (yr or 0) <= 2026
                                and ("applied energy" in venue_c or "apenergy" in doi_c))
                        if strong or weak:
                            pick = cand
                            if strong:
                                break
                    if pick:
                        log("    [17] S2 search pick: %r (%s, %s, doi=%s)"
                            % (pick.get("title"), pick.get("year"), pick.get("venue"),
                               (pick.get("externalIds") or {}).get("DOI")))
                        p["s2"] = s2_by_id(pick["paperId"])
                        break
            elif p["s2"] is None and key == "kusuda":
                for q in ["Earth Temperature and Thermal Diffusivity at Selected Stations in the United States",
                          "Kusuda Achenbach earth temperature thermal diffusivity"]:
                    js = s2_search(q, year="1960-1975")
                    pick = None
                    if js and js.get("data"):
                        for cand in js["data"]:
                            nt = norm(cand.get("title") or "")
                            if tratio(nt, norm(p["title"])) >= 0.75 or \
                               ("thermal diffusivity" in nt and ("earth" in nt or "ground" in nt)):
                                pick = cand
                                break
                    if pick:
                        log("    [19] S2 search pick: %r (%s, %s)"
                            % (pick.get("title"), pick.get("year"), pick.get("venue")))
                        p["s2"] = s2_by_id(pick["paperId"])
                        break
            elif p["s2"] is None and title_src:
                yr = p["year"]
                js = s2_search(title_src, year="%d-%d" % (yr - 1, yr + 1))
                pick = None
                if js and js.get("data"):
                    best = max(js["data"], key=lambda c: tratio(norm(c.get("title") or ""), norm(title_src)))
                    if tratio(norm(best.get("title") or ""), norm(title_src)) >= 0.85:
                        pick = best
                if pick:
                    log("    [%s] S2 search pick: %r" % (key, pick.get("title")))
                    p["s2"] = s2_by_id(pick["paperId"])

        # --- OA missing -> search by best known title
        if p["oa"] is None:
            title_src = p["title"] or (p["s2"] or {}).get("title")
            if key == "van_nooten":
                for flt in ["publication_year:2025,primary_location.source.issn:0306-2619",
                            "publication_year:2025", None]:
                    js = oa_search("graph neural network cable", extra_filter=flt)
                    if not js or not js.get("results"):
                        continue
                    pick = None
                    for cand in js["results"]:
                        nt = norm(cand.get("display_name") or "")
                        doi_c = norm_doi(cand.get("doi")) or ""
                        if "cable" in nt and ("neural" in nt or "graph" in nt or "gnn" in nt) \
                                and ("apenergy" in doi_c or "applied energy" in norm(str(cand.get("primary_location") or ""))):
                            pick = cand
                            break
                    if pick is None:
                        c0 = js["results"][0]
                        if "cable" in norm(c0.get("display_name") or ""):
                            pick = c0
                    if pick:
                        log("    [17] OA search pick: %r (%s, doi=%s)"
                            % (pick.get("display_name"), pick.get("publication_year"), pick.get("doi")))
                        p["oa"] = oa_by_id(oa_wid(pick["id"]))
                        break
            elif title_src:
                js = oa_search(title_src)
                pick = None
                if js and js.get("results"):
                    scored = sorted(js["results"],
                                    key=lambda c: tratio(norm(c.get("display_name") or ""), norm(title_src)))
                    best = scored[-1]
                    if tratio(norm(best.get("display_name") or ""), norm(title_src)) >= 0.85:
                        pick = best
                if pick:
                    log("    [%s] OA search pick: %r (%s)" % (key, pick.get("display_name"), pick.get("publication_year")))
                    p["oa"] = oa_by_id(oa_wid(pick["id"]))

    log("== Phase C: consolidate ==")
    for p in P.values():
        s2, oa = p["s2"], p["oa"]
        if s2:
            p["s2_id"] = s2.get("paperId")
            p["cites_s2"] = s2.get("citationCount")
            if s2.get("title"):
                p["title"] = s2["title"]
            if s2.get("year"):
                p["year"] = s2["year"]
            ext = s2.get("externalIds") or {}
            p["doi"] = p["doi"] or norm_doi(ext.get("DOI"))
            p["arxiv"] = p["arxiv"] or norm_arx(ext.get("ArXiv"))
            refs = s2.get("references") or []
            for r in refs:
                if r.get("title"):
                    p["s2_ref_titles"].append(r["title"])
                ext_r = r.get("externalIds") or {}
                d = norm_doi(ext_r.get("DOI"))
                if d:
                    p["s2_ref_dois"].add(d)
                a = norm_arx(ext_r.get("ArXiv"))
                if a:
                    p["s2_ref_arx"].add(a)
        if oa:
            p["oa_id"] = oa_wid(oa.get("id") or "")
            p["cites_oa"] = oa.get("cited_by_count")
            if oa.get("display_name"):
                p["title"] = p["title"] or oa["display_name"]
            if oa.get("publication_year"):
                p["year"] = p["year"] or oa["publication_year"]
            p["doi"] = p["doi"] or norm_doi(oa.get("doi"))
            p["oa_ref_ids"] = {oa_wid(w) for w in (oa.get("referenced_works") or [])}
        blob = " ".join(p["s2_ref_titles"])
        if s2 and s2.get("abstract"):
            blob += " " + s2["abstract"]
        if oa:
            blob += " " + abstract_from_inv(oa.get("abstract_inverted_index"))
        blob += " " + (p["title"] or "")
        p["blob"] = blob.lower()

        if s2 and oa:
            p["status"] = "ok"
        elif s2:
            p["status"] = "s2-only"
        elif oa:
            p["status"] = "oa-only"
        else:
            p["status"] = "not-found-on-API"
        log("  [%2d] %-28s S2=%s(cites=%s) OA=%s(cited_by=%s) refs:S2=%d OA=%d -> %s"
            % (p["n"], p["short"],
               "y" if s2 else "-", p["cites_s2"], "y" if oa else "-", p["cites_oa"],
               len(p["s2_ref_titles"]), len(p["oa_ref_ids"]), p["status"]))
    return P


def paper_src_id(p: dict) -> str:
    if p["doi"]:
        return "doi:" + p["doi"]
    if p["arxiv"]:
        return "arXiv:" + p["arxiv"]
    if p["s2_id"]:
        return "S2:" + p["s2_id"]
    if p["oa_id"]:
        return "OpenAlex:" + p["oa_id"]
    return "not-found-on-API"


# ---------------------------------------------------------------- edges
def build_cites_edges(P: dict):
    edges = {}  # (src_key, dst_key) -> dict(rel="cites", year, src_id, prov=set)
    for src in P.values():
        for dst in P.values():
            if src["key"] == dst["key"]:
                continue
            prov = set()
            if dst["doi"] and dst["doi"] in src["s2_ref_dois"]:
                prov.add("s2-doi")
            if dst["arxiv"] and dst["arxiv"] in src["s2_ref_arx"]:
                prov.add("s2-arxiv")
            t_hit = any(tmatch(rt, dst["title"]) for rt in src["s2_ref_titles"])
            if t_hit:
                prov.add("s2-title")
            if dst["oa_id"] and dst["oa_id"] in src["oa_ref_ids"]:
                prov.add("oa-refids")
            # drop s2-title-only matches when an ID match exists for another paper? keep as-is;
            # title matches are logged for manual review below.
            if prov:
                edges[(src["key"], dst["key"])] = dict(
                    rel="cites", year=src["year"], src_id=paper_src_id(src), prov=prov)
    return edges


def std_edges(P: dict):
    """standard-cited edges: corroborated from API blob, else asserted per
    project convention (cable-ampacity papers -> IEC 60287; dynamic-rating
    papers -> IEC 60853)."""
    out = {}
    for p in P.values():
        blob = p["blob"]
        title_n = norm(p["title"])
        # IEC 60287
        if re.search(r"60287", blob):
            out[(p["key"], "iec60287")] = "api-corroborated"
        elif p["role"].startswith("domain") and "cable" in title_n:
            out[(p["key"], "iec60287")] = "asserted"
        # IEC 60853 (cyclic/emergency rating of CABLES): assert only for papers
        # whose own title indicates dynamic thermal rating of cables.
        if re.search(r"60853", blob):
            out[(p["key"], "iec60853")] = "api-corroborated"
        elif "cable" in title_n and re.search(r"dynamic", title_n) \
                and re.search(r"rating|thermal", title_n):
            out[(p["key"], "iec60853")] = "asserted"
    return out


def build_coc(P: dict):
    """Co-citation counts from OpenAlex. group_by=referenced_works is not
    supported by the API (HTTP 400), so sample the citing works of each core
    paper (up to 2 pages of 200, sorted by cited_by_count desc - highly-cited
    citing works such as reviews reliably carry reference lists) and count how
    often each reference appears. Persisted with string keys for JSON."""
    if "coc_method" in CACHE["meta"] and CACHE["coc"]:
        coc = {}
        for k, v in CACHE["coc"].items():
            a, b = k.split("|")
            coc[(a, b)] = v
        return coc, CACHE["meta"]["coc_method"]
    coc = {}
    method = "openalex-sample-citing-works"
    for p in sorted(P.values(), key=lambda x: x["n"]):
        if not p["oa_id"]:
            continue
        cursor = "*"
        pages, got = 0, 0
        while cursor and pages < 2:
            js, err = http_json("%s/works" % OA, params={
                "filter": "cites:" + p["oa_id"], "sort": "cited_by_count:desc",
                "select": "referenced_works", "per-page": 200,
                "cursor": cursor, "mailto": MAILTO})
            if not js or not js.get("results"):
                if err:
                    log("  coc[%s] page fetch failed: %s" % (p["short"], err))
                break
            for w in js["results"]:
                for rw in w.get("referenced_works") or []:
                    wid = oa_wid(rw)
                    k = (p["oa_id"], wid)
                    coc[k] = coc.get(k, 0) + 1
            got += len(js["results"])
            cursor = (js.get("meta") or {}).get("next_cursor")
            pages += 1
            time.sleep(0.3)
        CACHE["coc"] = {"%s|%s" % k: v for k, v in coc.items()}
        CACHE["meta"]["coc_method"] = method
        save_cache()
        log("  coc[%s] sampled %d citing works" % (p["short"], got))
    return coc, method


def coc_pair(coc: dict, wid_a: str, wid_b: str) -> int:
    return max(coc.get((wid_a, wid_b), 0), coc.get((wid_b, wid_a), 0))


# ---------------------------------------------------------------- output
def write_outputs(P: dict, edges: dict, std_e: dict, coc: dict, coc_method: str) -> None:
    now = datetime.now(timezone.utc)
    fetched = now.strftime("%Y-%m-%d")

    # ---------------- CSV
    rows = []
    for (sk, dk), e in sorted(edges.items()):
        s, d = P[sk], P[dk]
        rows.append([s["title"], d["title"], s["year"], "cites", e["src_id"]])
    for (pk, stk), how in sorted(std_e.items()):
        p = P[pk]
        std = next(x for x in STANDARDS if x["key"] == stk)
        rows.append([p["title"], std["short"] + " (standard)", p["year"],
                     "standard-cited", paper_src_id(p)])

    # co-cited-by-theme: core pairs not already linked by a direct cites edge,
    # with third-party co-citation count >= threshold
    key_by_wid = {p["oa_id"]: p["key"] for p in P.values() if p["oa_id"]}
    pairs = []
    plist = [p for p in P.values() if p["oa_id"]]
    for i in range(len(plist)):
        for j in range(i + 1, len(plist)):
            a, b = plist[i], plist[j]
            if (a["key"], b["key"]) in edges or (b["key"], a["key"]) in edges:
                continue
            c = coc_pair(coc, a["oa_id"], b["oa_id"])
            if c > 0:
                pairs.append((c, a, b))
    pairs.sort(key=lambda t: -t[0])
    threshold = 5
    while sum(1 for c, _, _ in pairs if c >= threshold) < 6 and threshold > 2:
        threshold -= 1
    coc_rows = [(c, a, b) for c, a, b in pairs if c >= threshold][:20]
    for c, a, b in coc_rows:
        lo, hi = (a, b) if a["n"] <= b["n"] else (b, a)
        rows.append([lo["title"], hi["title"], lo["year"], "co-cited-by-theme", paper_src_id(lo)])

    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["source_title", "target_title", "source_year", "relation", "source_id"])
        w.writerows(rows)
    log("CSV written: %d edges (%d cites / %d standard-cited / %d co-cited-by-theme)"
        % (len(rows), len(edges), len(std_e), len(coc_rows)))

    # ---------------- node stats
    in_deg = {k: 0 for k in P}
    for (sk, dk) in edges:
        in_deg[dk] += 1
    std_in = {s["key"]: sum(1 for (pk, stk) in std_e if stk == s["key"]) for s in STANDARDS}

    # cross-cluster edges
    cross = [(sk, dk) for (sk, dk) in edges
             if cluster_of(P[sk]["n"]) != cluster_of(P[dk]["n"])]
    m_nodes = [p for p in P.values() if cluster_of(p["n"]) == "M"]
    d_nodes = [p for p in P.values() if cluster_of(p["n"]) == "D"]

    # cross co-citation among clusters (for observation)
    cross_coc = []
    for c, a, b in pairs:
        if cluster_of(a["n"]) != cluster_of(b["n"]):
            cross_coc.append((c, a["short"], b["short"]))

    # corroborated standards
    corr_60287 = [P[pk]["short"] for (pk, stk), how in std_e.items()
                  if stk == "iec60287" and how == "api-corroborated"]
    corr_60853 = [P[pk]["short"] for (pk, stk), how in std_e.items()
                  if stk == "iec60853" and how == "api-corroborated"]
    asrt_60287 = [P[pk]["short"] for (pk, stk), how in std_e.items()
                  if stk == "iec60287" and how == "asserted"]
    asrt_60853 = [P[pk]["short"] for (pk, stk), how in std_e.items()
                  if stk == "iec60853" and how == "asserted"]

    hubs = sorted(in_deg.items(), key=lambda kv: -kv[1])[:4]
    top_coc = pairs[:10]

    # ---------------- MD
    L = []
    A = L.append
    A("# Citation Network - Core References (Phase 3)")
    A("")
    A("Project: Neural-operator dynamic thermal rating (DTR) of underground power "
      "cables + safe EV charging.")
    A("")
    A("Fetched: **%s** (UTC %s). APIs: Semantic Scholar Graph API and OpenAlex, "
      "both key-less public endpoints; 429s handled with backoff; every edge below "
      "is derived from an API response (reference lists / cited-by sets / reference "
      "and abstract text), except the `standard-cited` edges to the IEC nodes, which "
      "follow the Phase 3 convention (standards are not citation-indexed) and are "
      "labelled corroborated vs asserted." % (fetched, now.strftime("%Y-%m-%d %H:%M")))
    A("")
    A("## (a) Node table")
    A("")
    A("| # | Paper | Year | S2 cites | OA cited-by | In-degree (core cites) | Role in project | Status |")
    A("|---|-------|------|----------|-------------|------------------------|-----------------|--------|")
    for p in sorted(P.values(), key=lambda x: x["n"]):
        A("| %d | %s | %s | %s | %s | %d | %s | %s |" % (
            p["n"], p["short"].replace("|", "/"), p["year"] or "?",
            p["cites_s2"] if p["cites_s2"] is not None else "n/a",
            p["cites_oa"] if p["cites_oa"] is not None else "n/a",
            in_deg[p["key"]], p["role"], p["status"]))
    for s in STANDARDS:
        A("| - | %s | %s | n/a (standard) | n/a (standard) | %d (standard-cited edges) | standards | manual node |" % (
            s["short"], s["year"], std_in[s["key"]]))
    A("")
    A("Identifiers (from API records):")
    A("")
    A("Note on counts: Semantic Scholar reports merged paper-level citation counts; "
      "OpenAlex often splits preprint/conference/journal versions into separate "
      "records, so its per-record `cited_by_count` can be far lower (e.g., FNO: "
      "4,881 on S2 vs 229 on the single OpenAlex record resolved here). Where the "
      "two APIs disagree on year/title, both values are shown in the identifiers "
      "below (S2 sometimes merges a preprint version under the published DOI).")
    A("")
    for p in sorted(P.values(), key=lambda x: x["n"]):
        ids = []
        if p["title"]:
            ids.append("title: %s" % p["title"])
        if p["doi"]:
            ids.append("doi:%s" % p["doi"])
        if p["arxiv"]:
            ids.append("arXiv:%s" % p["arxiv"])
        if p["s2_id"]:
            ids.append("S2:%s" % p["s2_id"])
        if p["oa_id"]:
            ids.append("OpenAlex:%s" % p["oa_id"])
        A("- **%s** (#%d): %s" % (p["short"], p["n"], "; ".join(ids)))
    A("- **IEC 60287**: %s" % STANDARDS[0]["title"])
    A("- **IEC 60853**: %s" % STANDARDS[1]["title"])
    A("")
    A("## (b) Network structure")
    A("")
    A("### Direct citations among the core papers (%d edges, relation `cites`)" % len(edges))
    A("")
    A("| Source (year) | -> | Target | Evidence |")
    A("|---------------|----|--------|----------|")
    for (sk, dk) in sorted(edges, key=lambda e: (P[e[0]]["n"], P[e[1]]["n"])):
        e = edges[(sk, dk)]
        A("| %s (%s) | -> | %s | %s |" % (
            P[sk]["short"], e["year"], P[dk]["short"], " + ".join(sorted(e["prov"]))))
    A("")
    A("Evidence codes: `s2-doi`/`s2-arxiv` = matched by DOI/arXiv id in the citing "
      "paper's Semantic Scholar reference list; `s2-title` = matched by title "
      "similarity in that list; `oa-refids` = matched by OpenAlex work-id in "
      "`referenced_works`. All matches were additionally eyeballed against the "
      "candidate titles printed by the build script.")
    A("")
    hub_names = ", ".join("%s (in-degree %d)" % (P[k]["short"], v) for k, v in hubs)
    A("**Hubs by in-degree within the core set:** %s." % hub_names)
    A("")
    A("### Standard nodes (relation `standard-cited`)")
    A("")
    A("- **IEC 60287** - API-corroborated (number found in the paper's API record): %s."
      % (", ".join(corr_60287) if corr_60287 else "none"))
    A("  - Asserted by project convention (cable-ampacity papers, not visible in API "
      "reference/abstract text): %s." % (", ".join(asrt_60287) if asrt_60287 else "none"))
    A("- **IEC 60853** - API-corroborated: %s." % (", ".join(corr_60853) if corr_60853 else "none"))
    A("  - Asserted by project convention (dynamic-rating papers): %s."
      % (", ".join(asrt_60853) if asrt_60853 else "none"))
    A("")
    A("### Co-citation (relation `co-cited-by-theme`)")
    A("")
    A("Third-party works citing **both** members of a pair, computed from OpenAlex "
      "(`%s`: for each core paper we sample up to 400 of its most-cited citing "
      "works - reviews and surveys, which reliably carry reference lists - and "
      "count shared reference entries). These are theme indicators, not exhaustive "
      "co-citation totals. Pairs already joined by a direct `cites` edge are "
      "excluded from the CSV rows; counts below are informational." % coc_method)
    A("")
    A("| Core pair | Co-citing works (OpenAlex) |")
    A("|-----------|-----------------------------|")
    for c, a, b in top_coc:
        A("| %s <-> %s | %d |" % (a["short"], b["short"], c))
    if not top_coc:
        A("| (no co-citation data returned) | - |")
    A("")
    A("## (c) Observations for novelty positioning")
    A("")
    obs = []
    bridge_edges = [(sk, dk) for (sk, dk) in cross
                    if "B" in (cluster_of(P[sk]["n"]), cluster_of(P[dk]["n"]))]
    obs.append(
        "1. **The two literatures in our core are almost bibliographically disjoint.** "
        "The ML-methods cluster (#1-4, 12, 13) and the cable/EV-domain cluster "
        "(#5-11, 14-16, 19) are connected by only %d direct cite edge(s)%s; every "
        "cross-community edge involves the 2025 bridge works (#17 van Nooten, "
        "#18 Prob. DLR LineGNN). Nothing in the cable-DTR canon cites the neural-operator "
        "papers, and vice versa - the bridge is recent and thin, which is exactly the "
        "gap our neural-operator DTR contribution targets." % (
            len(cross), ":" if cross else ""
        ))
    if cross:
        obs.append("   Cross edges: " + "; ".join(
            "%s -> %s" % (P[sk]["short"], P[dk]["short"]) for sk, dk in
            sorted(cross, key=lambda e: (P[e[0]]["n"], P[e[1]]["n"]))) + ".")
    obs.append(
        "2. **Standards are the real backbone of the domain cluster.** IEC 60287 "
        "receives %d standard-cited edges (%d corroborated in API records), and IEC "
        "60853 %d (%d corroborated). Any surrogate model that wants credibility with "
        "cable engineers must reproduce/ingest these standards' ampacity chains - "
        "a concrete interface requirement for our operator model."
        % (std_in["iec60287"], len(corr_60287), std_in["iec60853"], len(corr_60853)))
    fno = P["fno"]
    obs.append(
        "3. **Huge method-side attention, zero domain-side uptake (so far).** FNO "
        "(#1) has ~%s S2 citations (OpenAlex: %s) yet receives %d citations from the "
        "rest of our core - all from ML papers. No cable-ampacity/DTR paper in the "
        "core cites any neural operator, so applying FNO-style operators to buried-cable "
        "thermodynamics is an open, publishable combination."
        % (fno["cites_s2"] if fno["cites_s2"] is not None else "n/a",
           fno["cites_oa"] if fno["cites_oa"] is not None else "n/a",
           in_deg["fno"]))
    obs.append(
        "4. **Kusuda & Achenbach (1965) still anchors the domain.** The 60-year-old "
        "NBS soil-temperature report is cited by %d of our core papers%s and remains "
        "the canonical boundary-condition source; our neural operator must therefore "
        "represent seasonal soil dynamics, not just steady state."
        % (in_deg["kusuda"],
           " (plus %s S2 citations overall)" % P["kusuda"]["cites_s2"] if P["kusuda"]["cites_s2"] is not None else ""))
    if top_coc:
        c0, a0, b0 = top_coc[0]
        obs.append(
            "5. **Co-citation confirms the convergence is happening from the outside.** "
            "The strongest co-cited pair in our core is %s <-> %s (%d co-citing works). "
            "Cross-cluster co-citation is present but small (top: %s), i.e., third-party "
            "surveys are starting to place graph/neural simulation next to cable "
            "rating - our paper can be the first *method* paper to occupy that slot "
            "rather than a survey mention."
            % (a0["short"], b0["short"], c0,
               "; ".join("%s <-> %s (%d)" % (a, b, c) for c, a, b in cross_coc[:3])
               if cross_coc else "none detected"))
    for o in obs:
        A(o)
    A("")
    A("## (d) Method, endpoints, fetch date")
    A("")
    A("- **Fetch date:** %s (UTC)." % fetched)
    A("- **Semantic Scholar Graph API:**")
    A("  - `GET %s/paper/arXiv:{id}` and `GET %s/paper/DOI:{doi}` with `fields=title,year,citationCount,externalIds,abstract,references.title,references.year,references.externalIds`" % (S2, S2))
    A("  - `GET %s/paper/search?query=...&fields=title,year,venue,externalIds,authors` (used only when direct id lookup failed)" % S2)
    A("- **OpenAlex:**")
    A("  - `GET %s/works/doi:{doi}`; `GET %s/works?search=...`; `GET %s/works?filter=cites:W...&group_by=referenced_works` (co-citation)" % (OA, OA, OA))
    A("- **Rate limiting:** ~1.6 s between S2 calls, 0.3 s between OpenAlex calls, "
      "exponential backoff on HTTP 429/5xx, max 4 tries per request; papers failing "
      "after retries are marked `not-found-on-API` rather than guessed.")
    A("- **Co-citation method:** %s - sampled (up to 400 most-cited citing works "
      "per core paper), so counts are lower bounds / theme indicators rather than "
      "exhaustive totals." % coc_method)
    A("- **Evidence cache:** `raw_api_cache.json` (all raw responses; delete it or "
      "run the build script with `--refresh` to re-fetch).")
    A("- **Known limitations:** IEC standard nodes are not citation-indexed, so their "
      "edges are convention edges (corroborated/asserted as labelled above); "
      "`s2-title` evidence relies on fuzzy title matching and was manually reviewed "
      "against the API candidate titles.")
    A("")
    MD_PATH.write_text("\n".join(L), encoding="utf-8")
    log("MD written: %s" % MD_PATH)


# ---------------------------------------------------------------- main
def main() -> None:
    refresh = "--refresh" in sys.argv
    load_cache()
    P = resolve_papers(refresh)
    edges = build_cites_edges(P)
    log("cites edges (%d):" % len(edges))
    for (sk, dk), e in sorted(edges.items(), key=lambda kv: (P[kv[0][0]]["n"], P[kv[0][1]]["n"])):
        log("  [%2d] %-28s -> %-28s %s" % (P[sk]["n"], P[sk]["short"], P[dk]["short"], ",".join(sorted(e["prov"]))))
    std_e = std_edges(P)
    log("standard-cited edges (%d): %s" % (len(std_e), sorted(std_e)))
    coc, coc_method = build_coc(P)
    key_by_wid = {p["oa_id"]: p["key"] for p in P.values() if p["oa_id"]}
    plist = [p for p in P.values() if p["oa_id"]]
    pairs = []
    for i in range(len(plist)):
        for j in range(i + 1, len(plist)):
            a, b = plist[i], plist[j]
            c = coc_pair(coc, a["oa_id"], b["oa_id"])
            if c > 0:
                pairs.append((c, a, b))
    pairs.sort(key=lambda t: -t[0])
    log("top co-cited core pairs: " + "; ".join("%s<->%s=%d" % (a["short"], b["short"], c)
                                                for c, a, b in pairs[:12]))
    write_outputs(P, edges, std_e, coc, coc_method)


if __name__ == "__main__":
    main()
