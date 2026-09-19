"""
01_fetch_earnings_dates.py
==========================
Module 1 of the PEAD (Post-Earnings Announcement Drift) Alpha Pipeline.

Purpose:
    Download quarterly earnings announcement dates from SEC EDGAR 8-K filings.
    Output a point-in-time (PIT) table of:
        ticker | fiscal_period_end | announce_date | form_type | source

Key design decisions:
    - Use 8-K Item 2.02 ("Results of Operations") as the canonical earnings date.
      This is the SEC-mandated earnings release form, filed within 4 days of announcement.
    - Fall back to 10-Q/10-K filed date only when no 8-K 2.02 exists (pre-2004 filers).
    - NEVER use fiscal year/quarter end date as the announcement date (PIT violation).
    - Rate-limit SEC EDGAR requests to 10 req/sec (SEC fair-access policy).
    - Cache all raw EDGAR responses to data/edgar/cache/ to avoid re-downloading.

Output:
    data/edgar/earnings_dates.csv
    Columns: ticker, cik, fiscal_period_end, announce_date, form_type, source

Author: Tingxuan Wu
"""

import re
import time
import json
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).parent.parent
DATA_DIR   = _ROOT / "data"
EDGAR_DIR  = DATA_DIR / "edgar"
CACHE_DIR  = EDGAR_DIR / "cache"
OUTPUT_CSV = EDGAR_DIR / "earnings_dates.csv"

EDGAR_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── SEC EDGAR configuration ───────────────────────────────────────────────────
EDGAR_BASE   = "https://data.sec.gov"
HEADERS      = {
    "User-Agent": "PEAD Research tw3196@nyu.edu",   # SEC requires contact info
    "Accept-Encoding": "gzip, deflate",
}
_MIN_INTERVAL = 0.11          # 10 req/sec limit (SEC fair-access policy)
_last_request_time = 0.0


def _get(url: str, cache_key: str = None):
    """
    HTTP GET with rate-limiting and optional disk cache.
    Returns parsed JSON dict, raw text, or None on failure.
    """
    global _last_request_time

    # Check disk cache first
    if cache_key:
        cache_path = CACHE_DIR / f"{cache_key}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)

    # Rate limit: maintain >= 0.11s between requests
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        _last_request_time = time.time()

        if resp.status_code == 429:
            log.warning("Rate limited by SEC — sleeping 60s")
            time.sleep(60)
            return _get(url, cache_key)  # retry once

        if resp.status_code != 200:
            log.debug(f"HTTP {resp.status_code} for {url}")
            return None

        try:
            data = resp.json()
        except Exception:
            data = resp.text

        # Cache to disk
        if cache_key and isinstance(data, (dict, list)):
            with open(CACHE_DIR / f"{cache_key}.json", "w") as f:
                json.dump(data, f)

        return data

    except requests.RequestException as e:
        log.warning(f"Request failed: {e}")
        return None


# ── Step 1: Resolve ticker → CIK ─────────────────────────────────────────────

def get_cik_map(tickers: list) -> dict:
    """
    Returns {ticker: CIK_string (zero-padded to 10 digits)} for all tickers.
    Uses SEC EDGAR company_tickers.json (refreshed daily by SEC).
    """
    data = _get(
        "https://www.sec.gov/files/company_tickers.json",
        cache_key="company_tickers",
    )
    if data is None:
        raise RuntimeError("Cannot fetch SEC EDGAR company_tickers.json")

    # Build lookup: {TICKER_UPPER: CIK_10digit}
    lookup = {}
    for entry in data.values():
        t = entry.get("ticker", "").upper()
        cik = str(entry.get("cik_str", "")).zfill(10)
        lookup[t] = cik

    result = {}
    missing = []
    for ticker in tickers:
        cik = lookup.get(ticker.upper())
        if cik:
            result[ticker.upper()] = cik
        else:
            missing.append(ticker)

    if missing:
        log.warning(f"CIK not found for {len(missing)} tickers: {missing[:10]}...")

    log.info(f"Resolved {len(result)}/{len(tickers)} tickers to CIK")
    return result


# ── Step 2: Fetch filing submissions for a CIK ───────────────────────────────

def fetch_submissions(cik: str) -> dict | None:
    """
    Fetch the EDGAR submissions JSON for a CIK.
    Contains metadata for all filings: form type, filed date, accession number.
    """
    url = f"{EDGAR_BASE}/submissions/CIK{cik}.json"
    return _get(url, cache_key=f"submissions_{cik}")


# ── Step 3: Extract 8-K earnings announcement dates ──────────────────────────

def _parse_fiscal_period(period_of_report: str) -> Optional[str]:
    """
    Convert EDGAR period_of_report (YYYYMMDD or YYYY-MM-DD) to YYYY-MM-DD.
    Returns None if unparseable.
    """
    if not period_of_report:
        return None
    s = str(period_of_report).replace("-", "")
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return None


# Regex to detect "Item 2.02" in 8-K exhibit names / descriptions
_ITEM202_RE = re.compile(r"2\.02|results\s+of\s+operations", re.IGNORECASE)


def extract_earnings_dates_from_submissions(
    ticker: str,
    cik: str,
    start_date: str = "2003-01-01",
    end_date: str   = "2025-06-30",
) -> list:
    """
    Extract earnings announcement dates for one ticker from EDGAR submissions JSON.

    Strategy:
        1. Primary: 8-K with Item 2.02 (Results of Operations) — mandatory since Aug 2004.
           The 'filed' date is the announcement date (within 4 calendar days of the call).
        2. Secondary: 8-K without explicit Item 2.02 tag (pre-2004 / some filers).
           Use filed date; flag source = '8K_no_item'.
        3. Fallback: 10-Q or 10-K filed date when no 8-K found for that fiscal quarter.
           Much noisier — filers have 40-75 days; flag source = '10Q_fallback'.

    Returns list of dicts with keys:
        ticker, cik, fiscal_period_end, announce_date, form_type, source
    """
    submissions = fetch_submissions(cik)
    if submissions is None:
        log.warning(f"{ticker}: no submissions data")
        return []

    recent = submissions.get("filings", {}).get("recent", {})
    if not recent:
        return []

    # Unpack parallel arrays into rows
    form_types   = recent.get("form", [])
    filed_dates  = recent.get("filingDate", [])
    periods      = recent.get("reportDate", [])
    accessions   = recent.get("accessionNumber", [])
    items        = recent.get("items", [])      # list of "2.02" strings; may be empty

    n = len(form_types)
    rows = [
        {
            "form":     form_types[i] if i < len(form_types) else "",
            "filed":    filed_dates[i] if i < len(filed_dates) else "",
            "period":   periods[i] if i < len(periods) else "",
            "accession": accessions[i] if i < len(accessions) else "",
            "items":    items[i] if i < len(items) else "",
        }
        for i in range(n)
    ]

    # Filter by date range
    sd = pd.Timestamp(start_date)
    ed = pd.Timestamp(end_date)

    results = []
    seen_periods = set()   # deduplicate on fiscal period

    for row in rows:
        filed_str = row["filed"]
        if not filed_str:
            continue
        try:
            filed_ts = pd.Timestamp(filed_str)
        except Exception:
            continue
        if filed_ts < sd or filed_ts > ed:
            continue

        form = (row["form"] or "").upper().strip()
        period = _parse_fiscal_period(row["period"])

        # ── Primary: 8-K Item 2.02 ──────────────────────────────────
        if form in ("8-K", "8-K/A"):
            item_str = str(row["items"]) if row["items"] else ""
            has_202 = "2.02" in item_str or _ITEM202_RE.search(item_str)

            if has_202:
                source = "8K_item202"
            else:
                source = "8K_no_item"

            if period and period not in seen_periods:
                seen_periods.add(period)
                results.append({
                    "ticker":            ticker,
                    "cik":               cik,
                    "fiscal_period_end": period,
                    "announce_date":     filed_str,
                    "form_type":         row["form"],
                    "source":            source,
                })

        # ── Fallback: 10-Q / 10-K ────────────────────────────────────
        elif form in ("10-Q", "10-K", "10-Q/A", "10-K/A"):
            if period and period not in seen_periods:
                seen_periods.add(period)
                results.append({
                    "ticker":            ticker,
                    "cik":               cik,
                    "fiscal_period_end": period,
                    "announce_date":     filed_str,
                    "form_type":         row["form"],
                    "source":            "10Q_fallback",
                })

    log.debug(f"{ticker}: {len(results)} earnings dates extracted")
    return results


# ── Step 4: Handle paginated older filings ────────────────────────────────────

def fetch_all_earnings_dates(
    ticker: str,
    cik: str,
    start_date: str = "2003-01-01",
    end_date: str   = "2025-06-30",
) -> list:
    """
    Fetch earnings dates, handling EDGAR pagination.

    EDGAR submissions JSON only contains the most recent ~1,000 filings inline.
    Older filings are referenced in submissions["filings"]["files"] — a list of
    additional JSON chunks (filings0001.json, etc.). We fetch those too.
    """
    submissions = fetch_submissions(cik)
    if submissions is None:
        return []

    # Process main (recent) submissions
    results = extract_earnings_dates_from_submissions(ticker, cik, start_date, end_date)

    # Process older filing chunks
    older_files = submissions.get("filings", {}).get("files", [])
    for finfo in older_files:
        chunk_name = finfo.get("name", "")
        if not chunk_name:
            continue
        chunk_url = f"{EDGAR_BASE}/submissions/{chunk_name}"
        chunk_key = f"submissions_{cik}_{chunk_name.replace('.json','')}"
        chunk = _get(chunk_url, cache_key=chunk_key)
        if chunk is None:
            continue

        # Merge chunk data into a fake submissions-like structure and re-parse
        fake_sub = {"filings": {"recent": chunk, "files": []}}

        # Temporarily override fetch_submissions to return chunk
        original = {}
        original.update({"form": chunk.get("form",[]),
                          "filingDate": chunk.get("filingDate",[]),
                          "periodOfReport": chunk.get("reportDate",[]),
                          "accessionNumber": chunk.get("accessionNumber",[]),
                          "items": chunk.get("items",[])})

        # Re-run extraction on this chunk
        n = len(original["form"])
        rows = [
            {
                "form":      original["form"][i] if i < len(original["form"]) else "",
                "filed":     original["filingDate"][i] if i < len(original["filingDate"]) else "",
                "period":    original["periodOfReport"][i] if i < len(original["periodOfReport"]) else "",
                "accession": original["accessionNumber"][i] if i < len(original["accessionNumber"]) else "",
                "items":     original["items"][i] if i < len(original["items"]) else "",
            }
            for i in range(n)
        ]

        sd = pd.Timestamp(start_date)
        ed = pd.Timestamp(end_date)
        seen = {r["fiscal_period_end"] for r in results}

        for row in rows:
            filed_str = row["filed"]
            if not filed_str:
                continue
            try:
                filed_ts = pd.Timestamp(filed_str)
            except Exception:
                continue
            if filed_ts < sd or filed_ts > ed:
                continue

            form   = (row["form"] or "").upper().strip()
            period = _parse_fiscal_period(row["period"])

            if form in ("8-K", "8-K/A"):
                item_str = str(row["items"]) if row["items"] else ""
                source   = "8K_item202" if "2.02" in item_str else "8K_no_item"
                if period and period not in seen:
                    seen.add(period)
                    results.append({
                        "ticker": ticker, "cik": cik,
                        "fiscal_period_end": period,
                        "announce_date": filed_str,
                        "form_type": row["form"], "source": source,
                    })
            elif form in ("10-Q", "10-K", "10-Q/A", "10-K/A"):
                if period and period not in seen:
                    seen.add(period)
                    results.append({
                        "ticker": ticker, "cik": cik,
                        "fiscal_period_end": period,
                        "announce_date": filed_str,
                        "form_type": row["form"], "source": "10Q_fallback",
                    })

    return results


# ── Step 5: Main pipeline function ────────────────────────────────────────────

def build_earnings_date_table(
    tickers: list,
    start_date: str = "2003-01-01",
    end_date: str   = "2025-06-30",
    output_path: Path = OUTPUT_CSV,
) -> pd.DataFrame:
    """
    Main entry point.
    Downloads earnings announcement dates for all tickers and saves to CSV.

    Returns DataFrame sorted by (ticker, announce_date).
    """
    log.info(f"Building earnings date table for {len(tickers)} tickers")
    log.info(f"Date range: {start_date} → {end_date}")

    # Step 1: resolve tickers to CIKs
    cik_map = get_cik_map(tickers)

    all_records = []
    for i, ticker in enumerate(tickers):
        cik = cik_map.get(ticker.upper())
        if not cik:
            log.warning(f"[{i+1}/{len(tickers)}] {ticker}: CIK not found, skipping")
            continue

        log.info(f"[{i+1}/{len(tickers)}] {ticker} (CIK {cik})")
        records = fetch_all_earnings_dates(ticker, cik, start_date, end_date)
        all_records.extend(records)

    if not all_records:
        log.error("No earnings dates found — check network or EDGAR access")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["announce_date"]     = pd.to_datetime(df["announce_date"])
    df["fiscal_period_end"] = pd.to_datetime(df["fiscal_period_end"])
    df = df.sort_values(["ticker", "announce_date"]).reset_index(drop=True)

    # Quality check
    n_total   = len(df)
    n_8k_202  = (df["source"] == "8K_item202").sum()
    n_8k_other= (df["source"] == "8K_no_item").sum()
    n_fallback= (df["source"] == "10Q_fallback").sum()

    log.info(f"\n{'='*50}")
    log.info(f"Earnings dates extracted: {n_total}")
    log.info(f"  8-K Item 2.02 (gold standard): {n_8k_202} ({n_8k_202/n_total:.1%})")
    log.info(f"  8-K (no item tag):              {n_8k_other} ({n_8k_other/n_total:.1%})")
    log.info(f"  10-Q/10-K fallback:             {n_fallback} ({n_fallback/n_total:.1%})")

    # PIT sanity check: announce_date should be AFTER fiscal_period_end
    bad_pit = df[df["announce_date"] < df["fiscal_period_end"]]
    if len(bad_pit) > 0:
        log.warning(f"  PIT violation: {len(bad_pit)} rows where announce < period_end (dropped)")
        df = df[df["announce_date"] >= df["fiscal_period_end"]].copy()

    # Lag check: warn if median lag is suspiciously long
    df["lag_days"] = (df["announce_date"] - df["fiscal_period_end"]).dt.days
    median_lag = df["lag_days"].median()
    log.info(f"  Median announcement lag: {median_lag:.0f} days (expected 15-45 for 8-K, 40-75 for 10-Q)")
    if median_lag > 90:
        log.warning("  High median lag — check whether 10-Q fallback dominates")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info(f"\nSaved to {output_path}")
    log.info(f"{'='*50}")

    return df


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    # Default small-cap universe for PEAD (small caps have stronger drift)
    _DEFAULT_TICKERS = [
        # Large cap anchors (always in S&P 500)
        "AAPL","MSFT","AMZN","GOOGL","META","NVDA","BRK.B","JPM","JNJ","V",
        "UNH","XOM","PG","MA","HD","CVX","MRK","ABBV","PEP","KO",
        # Mid cap tech
        "ANET","ZBRA","CRUS","SWKS","QRVO","MRVL","KEYS","KLAC",
        # Industrials
        "ETN","EMR","ROK","PH","HON","RTX","NOC","LMT","GD","TDG",
        # Healthcare
        "TMO","IQV","WST","CTLT","IDXX","HOLX","TECH","NEOG",
        # Consumer
        "CHD","CL","KMB","GIS","LW","MKC","SJM","HRL",
        # Energy
        "SLB","HAL","BKR","CLB",
        # Financials
        "ICE","CME","CBOE","MKTX","LPLA",
    ]

    parser = argparse.ArgumentParser(description="Fetch earnings announcement dates from SEC EDGAR")
    parser.add_argument("--tickers",    nargs="+", default=None,
                        help="Space-separated tickers (default: built-in list)")
    parser.add_argument("--start-date", default="2003-01-01")
    parser.add_argument("--end-date",   default="2025-06-30")
    parser.add_argument("--output",     default=str(OUTPUT_CSV))
    args = parser.parse_args()

    tickers = args.tickers if args.tickers else _DEFAULT_TICKERS

    df = build_earnings_date_table(
        tickers    = tickers,
        start_date = args.start_date,
        end_date   = args.end_date,
        output_path= Path(args.output),
    )

    print(f"\nSample output:")
    print(df.head(20).to_string(index=False))
