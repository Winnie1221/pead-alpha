"""
Step 1: point-in-time earnings announcement dates from SEC EDGAR.

Source of truth: 8-K filings tagged Item 2.02 ("Results of Operations and
Financial Condition"). Item 2.02 has been required since August 2004 and is
furnished within four business days of the earnings release (in practice
usually the same day). Its EDGAR filing date is used as the announcement date.

Only Item 2.02 8-Ks count as earnings events. Other 8-Ks (M&A, officer
changes, debt issuance, ...) are ignored: treating them as earnings events
would attach a quarter's EPS surprise to dates before that EPS was public.

Fallback: a 10-Q/10-K is used as the event only when no Item 2.02 8-K was
filed between that report's period end and its filing date. The filing
date is then the announcement date (the EPS was public by then at the
latest). These rows are tagged `source = "10Q_fallback"`.

The fiscal quarter an 8-K refers to is not in the filing index (an 8-K's
"reportDate" is the event date, not the quarter end); it is assigned in
step 2 from the XBRL EPS data.

Output: data/edgar/earnings_dates.csv
    ticker, cik, announce_date, fiscal_period_end, form_type, source, accession
"""

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from . import config
from .universe import UNIVERSE

log = logging.getLogger(__name__)

EDGAR_BASE = "https://data.sec.gov"
_last_request = 0.0


# ── HTTP with rate limiting and disk cache ────────────────────────────────────

def sec_get(url: str, cache_key: str | None = None, _retry: bool = True):
    """GET a SEC JSON endpoint with a ≤10 req/s rate limit and a disk cache."""
    global _last_request
    if cache_key:
        path = config.CACHE_DIR / f"{cache_key}.json"
        if path.exists():
            return json.loads(path.read_text())

    wait = config.SEC_MIN_INTERVAL - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    try:
        resp = requests.get(url, headers=config.sec_headers(), timeout=20)
    except requests.RequestException as e:
        log.warning(f"Request failed for {url}: {e}")
        return None
    finally:
        _last_request = time.time()

    if resp.status_code == 429 and _retry:
        log.warning("Rate limited by SEC; sleeping 60s and retrying once")
        time.sleep(60)
        return sec_get(url, cache_key, _retry=False)
    if resp.status_code != 200:
        log.debug(f"HTTP {resp.status_code} for {url}")
        return None

    data = resp.json()
    if cache_key:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (config.CACHE_DIR / f"{cache_key}.json").write_text(json.dumps(data))
    return data


def get_cik_map(tickers: list[str]) -> dict[str, str]:
    """{TICKER: 10-digit CIK} from SEC's company_tickers.json."""
    data = sec_get("https://www.sec.gov/files/company_tickers.json",
                   cache_key="company_tickers")
    if data is None:
        raise RuntimeError("Cannot fetch SEC company_tickers.json")
    lookup = {e["ticker"].upper(): str(e["cik_str"]).zfill(10) for e in data.values()}
    out = {t.upper(): lookup[t.upper()] for t in tickers if t.upper() in lookup}
    missing = [t for t in tickers if t.upper() not in lookup]
    if missing:
        log.warning(f"CIK not found for {len(missing)} tickers "
                    f"(delisted or renamed): {missing}")
    log.info(f"Resolved {len(out)}/{len(tickers)} tickers to CIK")
    return out


# ── Filing index ──────────────────────────────────────────────────────────────

_FIELDS = {
    "form": "form",
    "filingDate": "filing_date",
    "reportDate": "report_date",
    "items": "items",
    "accessionNumber": "accession",
}


def _block_to_frame(block: dict) -> pd.DataFrame:
    n = len(block.get("form", []))
    cols = {}
    for src, dst in _FIELDS.items():
        vals = list(block.get(src, []))
        cols[dst] = vals + [""] * (n - len(vals))
    return pd.DataFrame(cols)


def load_filing_index(cik: str) -> pd.DataFrame:
    """All filings for a CIK: the inline 'recent' block plus older pages."""
    sub = sec_get(f"{EDGAR_BASE}/submissions/CIK{cik}.json",
                  cache_key=f"submissions_{cik}")
    if sub is None:
        return pd.DataFrame(columns=list(_FIELDS.values()))
    frames = [_block_to_frame(sub.get("filings", {}).get("recent", {}))]
    for f in sub.get("filings", {}).get("files", []):
        name = f.get("name", "")
        if not name:
            continue
        chunk = sec_get(f"{EDGAR_BASE}/submissions/{name}",
                        cache_key=f"submissions_{cik}_{name.replace('.json', '')}")
        if chunk:
            frames.append(_block_to_frame(chunk))
    return (pd.concat(frames, ignore_index=True)
              .drop_duplicates(subset="accession"))


# ── Event extraction (pure function; unit-tested) ─────────────────────────────

def extract_earnings_events(filings: pd.DataFrame, ticker: str, cik: str,
                            start_date: str, end_date: str) -> pd.DataFrame:
    """
    Turn a filing index into earnings events.

    Primary events : original 8-Ks whose items include 2.02.
    Fallback events: 10-Q / 10-K with no Item 2.02 8-K filed in
                     (period_end, filing_date].
    """
    cols = ["ticker", "cik", "announce_date", "fiscal_period_end",
            "form_type", "source", "accession"]
    if filings.empty:
        return pd.DataFrame(columns=cols)

    f = filings.copy()
    f["form"] = f["form"].astype(str).str.upper().str.strip()
    f["filing_date"] = pd.to_datetime(f["filing_date"], errors="coerce")
    f["report_date"] = pd.to_datetime(f["report_date"], errors="coerce")
    f = f.dropna(subset=["filing_date"])

    is_202 = (f["form"] == "8-K") & f["items"].astype(str).str.contains("2.02", regex=False)
    k8 = f[is_202].copy()
    k8 = k8.drop_duplicates(subset="filing_date")  # one release per day
    primary = pd.DataFrame({
        "ticker": ticker,
        "cik": cik,
        "announce_date": k8["filing_date"],
        "fiscal_period_end": pd.NaT,  # assigned from XBRL in step 2
        "form_type": k8["form"],
        "source": "8K_item202",
        "accession": k8["accession"],
    })

    periodic = f[f["form"].isin(["10-Q", "10-K"])].dropna(subset=["report_date"])
    k8_dates = k8["filing_date"].sort_values().values
    fallback_rows = []
    for _, r in periodic.iterrows():
        covered = ((k8_dates > r["report_date"].to_datetime64())
                   & (k8_dates <= r["filing_date"].to_datetime64())).any()
        if not covered:
            fallback_rows.append({
                "ticker": ticker, "cik": cik,
                "announce_date": r["filing_date"],
                "fiscal_period_end": r["report_date"],
                "form_type": r["form"], "source": "10Q_fallback",
                "accession": r["accession"],
            })
    fallback = pd.DataFrame(fallback_rows, columns=cols)

    events = pd.concat([primary, fallback], ignore_index=True)
    sd, ed = pd.Timestamp(start_date), pd.Timestamp(end_date)
    events = events[(events["announce_date"] >= sd) & (events["announce_date"] <= ed)]
    return events[cols].sort_values("announce_date").reset_index(drop=True)


# ── Main entry point ──────────────────────────────────────────────────────────

def build_earnings_date_table(tickers: list[str],
                              start_date: str = config.DEFAULT_START,
                              end_date: str = config.DEFAULT_END,
                              output_path: Path = config.EARNINGS_CSV) -> pd.DataFrame:
    config.ensure_dirs()
    log.info(f"Building earnings date table for {len(tickers)} tickers "
             f"({start_date} → {end_date})")
    cik_map = get_cik_map(tickers)

    frames = []
    for i, t in enumerate(tickers, 1):
        cik = cik_map.get(t.upper())
        if not cik:
            continue
        log.info(f"[{i}/{len(tickers)}] {t} (CIK {cik})")
        frames.append(extract_earnings_events(load_filing_index(cik), t, cik,
                                              start_date, end_date))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        log.error("No earnings dates found; check network access to EDGAR")
        return df

    n = len(df)
    n_8k = int((df["source"] == "8K_item202").sum())
    per_year = n / df["ticker"].nunique() / max(
        (df["announce_date"].max() - df["announce_date"].min()).days / 365.25, 1)
    log.info(f"Earnings events: {n} ({n_8k/n:.1%} from 8-K Item 2.02, "
             f"{1 - n_8k/n:.1%} 10-Q/10-K fallback)")
    log.info(f"  Events per ticker-year: {per_year:.2f} (expect ≈ 4)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info(f"Saved to {output_path}")
    return df


def main():
    config.setup_logging()
    p = argparse.ArgumentParser(description="Fetch earnings dates from SEC EDGAR")
    p.add_argument("--tickers", nargs="+", default=None)
    p.add_argument("--start-date", default=config.DEFAULT_START)
    p.add_argument("--end-date", default=config.DEFAULT_END)
    p.add_argument("--output", default=str(config.EARNINGS_CSV))
    a = p.parse_args()
    build_earnings_date_table(a.tickers or UNIVERSE, a.start_date, a.end_date,
                              Path(a.output))


if __name__ == "__main__":
    main()
