# Transparent SF 311 Blotter — weekly detail + YTD graffiti counts & resolution speed
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
BASE_URL   = "https://data.sfgov.org/resource/vw6y-z8j6.json"
PAGE_LIMIT = 50_000
APP_TOKEN  = "RVn1I62ZHXFcXKgya4u3N1rAn"          # or set SOCRATA_APP_TOKEN

VALID_DISTS = set(range(1, 12))                   # 1-11 inclusive

# -------------------------------------------------------------------
# Resilient session
# -------------------------------------------------------------------
def make_session() -> requests.Session:
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    s = requests.Session()
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

SESSION = make_session()

def token() -> str:
    t = os.getenv("SOCRATA_APP_TOKEN", APP_TOKEN).strip()
    if not t:
        raise ValueError("Missing DataSF token.")
    return t

# -------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------
def pct_change(new: float, old: float) -> float:
    return 0.0 if old == 0 else round((new - old) / old * 100, 1)

def fmt_pct(val: float) -> str:
    if val > 0:  return f"+{val}%"
    if val < 0:  return f"{val}%"
    return "0%"

def first_photo(df: pd.DataFrame) -> Optional[str]:
    for col in ["media_url", "source"]:
        if col in df.columns:
            link = df[col].dropna().astype(str).str.strip()
            for x in link:
                if x.startswith("http"):
                    return x
    return None

# -------------------------------------------------------------------
# Generic fetch (paging)
# -------------------------------------------------------------------
def fetch_rows(where: str, select: str) -> List[dict]:
    rows, offset = [], 0
    while True:
        params = {
            "$$app_token": token(),
            "$select": select,
            "$where":  where,
            "$limit":  PAGE_LIMIT,
            "$offset": offset,
            "$order":  "requested_datetime",
        }
        batch = SESSION.get(BASE_URL, params=params, timeout=180).json()
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT
    return rows

# -------------------------------------------------------------------
# YTD graffiti counts + avg resolution hours
# -------------------------------------------------------------------
def graffiti_ytd_stats(year: int) -> Tuple[pd.DataFrame, int, float]:
    today = datetime.utcnow()
    start = f"{year}-01-01T00:00:00"
    end   = f"{year}-{today.month:02d}-{today.day:02d}T23:59:59"
    where = (
        f"requested_datetime between '{start}' and '{end}' "
        f"and upper(service_name) like '%GRAFFITI%'"
    )
    select = "service_request_id, supervisor_district, requested_datetime, closed_date"
    rows   = fetch_rows(where, select)
    df     = pd.DataFrame(rows)

    if df.empty:
        empty = pd.DataFrame(columns=["supervisor_district", "total", "avg_hours"])
        return empty, 0, 0.0

    df["supervisor_district"] = pd.to_numeric(df["supervisor_district"], errors="coerce")
    df = df[df["supervisor_district"].isin(VALID_DISTS)]
    df["req"]   = pd.to_datetime(df["requested_datetime"], errors="coerce")
    df["close"] = pd.to_datetime(df["closed_date"],        errors="coerce")
    df = df.dropna(subset=["req", "close"])
    df["res_hours"] = (df["close"] - df["req"]).dt.total_seconds() / 3600

    agg = (
        df.groupby("supervisor_district")
          .agg(total=("res_hours", "size"), avg_hours=("res_hours", "mean"))
          .reset_index()
    )
    agg["total"]     = agg["total"].astype(int)
    agg["avg_hours"] = agg["avg_hours"].round(1)

    return agg, int(agg["total"].sum()), round(df["res_hours"].mean(), 1)

# -------------------------------------------------------------------
# Weekly fetch (14-day window then split)
# -------------------------------------------------------------------
def fetch_weekly() -> Tuple[pd.DataFrame, pd.DataFrame]:
    fourteen_days_ago = datetime.utcnow() - timedelta(days=14)
    where = f"requested_datetime >= '{fourteen_days_ago.strftime('%Y-%m-%dT00:00:00')}'"
    rows  = fetch_rows(where, "*")
    if not rows:
        sys.exit("No 311 data returned — check token or API availability.")

    df = pd.DataFrame(rows)
    df["opened_date"]         = pd.to_datetime(df["requested_datetime"], errors="coerce")
    df["service_name"]        = df["service_name"].fillna("Unknown")
    df["supervisor_district"] = pd.to_numeric(df["supervisor_district"], errors="coerce")
    df = df[df["supervisor_district"].isin(VALID_DISTS)]

    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    df_this = df[df["opened_date"] >= seven_days_ago]
    df_prev = df[df["opened_date"] < seven_days_ago]
    if df_this.empty:
        sys.exit("Zero 311 cases in the last 7 days — unlikely. Check API response.")
    return df_this, df_prev

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main() -> None:
    # ---------- weekly data ----------
    df_this, df_prev = fetch_weekly()

    total_this, total_prev = len(df_this), len(df_prev)
    city_pct = pct_change(total_this, total_prev)

    dist_this = df_this.groupby("supervisor_district").size().reset_index(name="this_total")
    dist_prev = df_prev.groupby("supervisor_district").size().reset_index(name="prev_total")
    dist_trend = (
        pd.merge(dist_this, dist_prev, on="supervisor_district", how="outer")
        .fillna(0)
    )
    dist_trend["this_total"] = dist_trend["this_total"].astype(int)
    dist_trend["prev_total"] = dist_trend["prev_total"].astype(int)
    dist_trend["pct_total"]  = dist_trend.apply(
        lambda r: pct_change(r.this_total, r.prev_total), axis=1
    )

    this_counts = (
        df_this.groupby(["supervisor_district", "service_name"])
        .size()
        .reset_index(name="this_week")
    )
    prev_counts = (
        df_prev.groupby(["supervisor_district", "service_name"])
        .size()
        .reset_index(name="last_week")
    )
    trend = pd.merge(
        this_counts, prev_counts, on=["supervisor_district", "service_name"], how="outer"
    ).fillna(0)
    trend[["this_week", "last_week"]] = trend[["this_week", "last_week"]].astype(int)
    trend["pct"] = trend.apply(lambda r: pct_change(r.this_week, r.last_week), axis=1)

    # Weekly graffiti stats
    gra_this_mask = df_this["service_name"].str.contains("graffiti", case=False, na=False)
    gra_prev_mask = df_prev["service_name"].str.contains("graffiti", case=False, na=False)
    gra_this_total = int(gra_this_mask.sum())
    gra_prev_total = int(gra_prev_mask.sum())
    gra_delta      = gra_this_total - gra_prev_total
    gra_pct        = pct_change(gra_this_total, gra_prev_total)

    gra_trend = trend[trend["service_name"].str.contains("graffiti", case=False, na=False)]
    inc_row = gra_trend.loc[gra_trend["pct"].idxmax()] if not gra_trend.empty else None
    dec_row = gra_trend.loc[gra_trend["pct"].idxmin()] if not gra_trend.empty else None

    # ---------- YTD graffiti 2025 vs 2024 ----------
    df25, y25_total, y25_avg = graffiti_ytd_stats(2025)
    df24, y24_total, y24_avg = graffiti_ytd_stats(2024)

    ytd_delta_cnt = y25_total - y24_total
    ytd_pct_cnt   = pct_change(y25_total, y24_total)
    ytd_delta_hr  = round(y25_avg - y24_avg, 1)
    ytd_pct_hr    = pct_change(y25_avg, y24_avg)

    ytd_merge = (
        df25.rename(columns={"total": "total_25", "avg_hours": "avg_25"})
           .merge(df24.rename(columns={"total": "total_24", "avg_hours": "avg_24"}),
                  on="supervisor_district", how="outer")
           .fillna(0)
    )
    for col in ["total_25", "total_24"]:
        ytd_merge[col] = ytd_merge[col].astype(int)
    ytd_merge["delta"]    = ytd_merge["total_25"] - ytd_merge["total_24"]
    ytd_merge["pct"]      = ytd_merge.apply(lambda r: pct_change(r.total_25, r.total_24), axis=1)
    ytd_merge["avg_25"]   = ytd_merge["avg_25"].round(1)
    ytd_merge["avg_24"]   = ytd_merge["avg_24"].round(1)
    ytd_merge["avg_diff"] = (ytd_merge["avg_25"] - ytd_merge["avg_24"]).round(1)

    # ---------- output ----------
    start_date = (datetime.utcnow() - timedelta(days=7)).date()
    end_date   = datetime.utcnow().date()

    print("\nTransparent SF 311 Blotter")
    print(f"Date Range: {start_date} – {end_date}\n")
    print(f"{int(total_this):,} total 311 requests citywide, {fmt_pct(city_pct)} vs. prior week\n")

    for _, d in dist_trend.sort_values("supervisor_district").iterrows():
        dist = int(d.supervisor_district)
        print(f"District {dist}: {int(d.this_total):,} calls, {fmt_pct(d.pct_total)} w/w")
        top3 = (
            trend[trend.supervisor_district == dist]
            .sort_values("this_week", ascending=False)
            .head(3)
        )
        for _, row in top3.iterrows():
            print(f"  {int(row.this_week):,} × {row.service_name} ({fmt_pct(row.pct)})")
        print()

    # Weekly graffiti watch
    print("Graffiti Watch — past week")
    print(
        f"  Citywide: {gra_this_total:,} calls "
        f"({gra_delta:+,} w/w, {fmt_pct(gra_pct)})"
    )
    def gra_line(r, label: str) -> None:
        delta = int(r.this_week - r.last_week)
        dist  = int(r.supervisor_district)
        print(
            f"  {label}: District {dist} — "
            f"{int(r.this_week):,} calls ({delta:+}, {fmt_pct(r.pct)})"
        )
        photo = first_photo(
            df_this[
                (df_this["supervisor_district"] == dist)
                & df_this["service_name"].str.contains("graffiti", case=False, na=False)
            ]
        )
        if photo:
            print(f"    sample photo: {photo}")
    if inc_row is not None and dec_row is not None:
        gra_line(inc_row, "Biggest jump")
        gra_line(dec_row, "Biggest drop")
    print()

    # YTD graffiti scoreboard
    today_str = datetime.utcnow().strftime("%B %d")
    print(f"Graffiti YTD — Jan 1–{today_str}")
    print(f"  2025 citywide: {y25_total:,} calls, avg resolution {y25_avg} h")
    print(f"  2024 citywide: {y24_total:,} calls, avg resolution {y24_avg} h")
    print(
        f"  Change: {ytd_delta_cnt:+,} calls ({fmt_pct(ytd_pct_cnt)}), "
        f"avg time {ytd_delta_hr:+} h ({fmt_pct(ytd_pct_hr)})\n"
    )

    for _, r in ytd_merge.sort_values("supervisor_district").iterrows():
        d = int(r.supervisor_district)
        print(
            f"  District {d}: "
            f"{int(r.total_25):,} vs {int(r.total_24):,} "
            f"({int(r.delta):+,}, {fmt_pct(r.pct)}) | "
            f"avg {r.avg_25} h vs {r.avg_24} h ({r.avg_diff:+} h)"
        )

if __name__ == "__main__":
    main()
