#!/usr/bin/env python3
"""
oi_viewer.py  —  Calendar-based 0DTE OI intensity viewer.

Pick any date from the historical backfill; the app renders a discrete
intensity heatmap of that morning's 0DTE chain, with cells colored by
where their OI falls in the historically-calibrated percentile buckets
(Regular / Weekly / Monthly tiers, ±20 strikes from ATM).

Usage:
    python oi_viewer.py
"""

import calendar as _cal
import io
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pytz
import requests
import yfinance as yf

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg

import tkinter as tk
from tkcalendar import Calendar as TkCalendar

from utils import (
    _ensure_calendar_loaded, next_trading_day, nominal_friday, prior_trading_day,
    target_expirations,
)

# ── paths and remote data ──────────────────────────────────────────────────────

import sys as _sys
ROOT = Path(_sys._MEIPASS) if getattr(_sys, "frozen", False) else Path(__file__).parent
R2_BASE = "https://pub-4d5c916b8cb74ffb8c0abd7dfadb02cf.r2.dev"
_ET     = pytz.timezone("America/New_York")

# ── timing log ─────────────────────────────────────────────────────────────────

_log_path = Path(__file__).parent / "oi_viewer_timing.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_path, mode="w", encoding="utf-8"),
    ],
)
_log = logging.getLogger("timing")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# ── display config ─────────────────────────────────────────────────────────────

DISPLAY_WINDOW   = 11   # ±N strikes shown in heatmap
SCROLL_MAX       = 33 - DISPLAY_WINDOW   # max scroll offset (data covers ±33)
N_SCROLL_STEPS   = 5
SCROLL_POSITIONS = [int(round(v)) for v in np.linspace(-SCROLL_MAX, SCROLL_MAX, N_SCROLL_STEPS)]

# 6-level discrete palette
# Level 0 = zero OI (background), levels 1-5 = increasing intensity
CALL_COLORS = [
    "#0d1117",   # 0  zero OI
    "#0f3d22",   # 1  < p25
    "#1a6635",   # 2  p25–p50
    "#008c38",   # 3  p50–p75
    "#00cc55",   # 4  p75–p90
    "#88ffcc",   # 5  > p90  ("wall")
]
PUT_COLORS = [
    "#0d1117",   # 0  zero OI
    "#400d0d",   # 1  < p25
    "#6b1515",   # 2  p25–p50
    "#a01800",   # 3  p50–p75
    "#ee3300",   # 4  p75–p90
    "#ffaa88",   # 5  > p90  ("wall")
]

TOP5_CALL_EDGE = "#00cc55"
TOP5_CALL_FILL = "#0f3d22"
TOP5_PUT_EDGE  = "#ff7043"
TOP5_PUT_FILL  = "#3d0f08"

TIER_COLORS = {
    "0DTE_Regular": "#4a90d9",
    "0DTE_Weekly":  "#e8c84b",
    "0DTE_Monthly": "#e8604b",
}

BG     = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
FG     = "#c9d1d9"
DIM    = "#9ca6b0"

# ── trading calendar ───────────────────────────────────────────────────────────

_nyse   = mcal.get_calendar("NYSE")
_valid: set[date] = set()


def _bootstrap(start: date, end: date):
    global _valid
    days = _nyse.valid_days(start_date=start.isoformat(), end_date=end.isoformat())
    _valid.update(d.date() for d in days)
    _ensure_calendar_loaded(start, end + timedelta(days=120))


def _monthly_opex(year: int, month: int) -> date | None:
    """3rd Friday, rolled back if it's a NYSE holiday."""
    count = 0
    for day in range(1, _cal.monthrange(year, month)[1] + 1):
        if date(year, month, day).weekday() == 4:
            count += 1
            if count == 3:
                d = date(year, month, day)
                while d not in _valid:
                    d -= timedelta(days=1)
                return d
    return None


def classify_tier(trade_date: date) -> str:
    label_map = dict(target_expirations(trade_date))
    p1d = label_map.get("+1D")
    if p1d is None:
        return "0DTE_Regular"
    eow = prior_trading_day(nominal_friday(trade_date))
    if p1d != eow:
        return "0DTE_Regular"
    opex = _monthly_opex(p1d.year, p1d.month)
    return "0DTE_Monthly" if p1d == opex else "0DTE_Weekly"


# ── data helpers ───────────────────────────────────────────────────────────────

def available_dates() -> set[date]:
    resp = requests.get(f"{R2_BASE}/manifest.json", timeout=10)
    resp.raise_for_status()
    return {date.fromisoformat(d) for d in resp.json()["dates"]}


def load_day(d: date) -> pd.DataFrame | None:
    fname = f"qqq_chain_{d.strftime('%Y%m%d')}.csv"
    for key in [f"raw/{fname}", f"raw/opex/{fname}"]:
        t0 = time.perf_counter()
        resp = requests.get(f"{R2_BASE}/{key}", timeout=15)
        elapsed = time.perf_counter() - t0
        if resp.status_code == 200:
            _log.debug(f"[network] fetched {key} in {elapsed:.2f}s")
            return pd.read_csv(io.StringIO(resp.text))
        _log.debug(f"[network] miss    {key} in {elapsed:.2f}s (HTTP {resp.status_code})")
    return None


def load_ranges() -> pd.DataFrame:
    resp = requests.get(f"{R2_BASE}/derived/OIranges.csv", timeout=15)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def load_intraday(exp_date: date) -> pd.DataFrame | None:
    """Try R2 first, fall back to yfinance for 10-min (or 5-min) QQQ bars on exp_date."""
    fname = f"qqq_intraday_{exp_date.strftime('%Y%m%d')}.csv"
    t0 = time.perf_counter()
    resp = requests.get(f"{R2_BASE}/prices/{fname}", timeout=15)
    _log.debug(f"[network] prices/{fname}: HTTP {resp.status_code} in {time.perf_counter()-t0:.2f}s")
    if resp.status_code == 200:
        return pd.read_csv(io.StringIO(resp.text))

    # yfinance fallback — only for dates that have already traded
    if exp_date > date.today():
        return None
    try:
        for interval in ("5m", "2m"):
            df = yf.download(
                "QQQ",
                start=exp_date.isoformat(),
                end=(exp_date + timedelta(days=1)).isoformat(),
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
            if not df.empty:
                break
        else:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert(_ET)
        df = df.between_time("09:30", "15:59")
        if df.empty:
            return None

        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
        dt_col = next((c for c in df.columns if "datetime" in c), None)
        if dt_col and dt_col != "datetime":
            df = df.rename(columns={dt_col: "datetime"})
        df["datetime"] = df["datetime"].astype(str)
        return df[["datetime", "open", "high", "low", "close", "volume"]]
    except Exception as e:
        _log.debug(f"yfinance intraday fallback failed for {exp_date}: {e}")
        return None


def effective_thresholds(tier: str, ranges: pd.DataFrame, offsets: list[int]) -> dict:
    """
    {offset: {"call": [p25,p50,p75,p90], "put": [p25,p50,p75,p90]}}
    Non-Regular tiers scale the Regular thresholds by their median adj multiplier.
    """
    reg = ranges[ranges["Tier"] == "0DTE_Regular"].set_index("StrikeOffset")
    adj = (ranges[ranges["Tier"] == tier].set_index("StrikeOffset")
           if tier != "0DTE_Regular" else None)

    out = {}
    for o in offsets:
        if o not in reg.index:
            out[o] = {"call": [0]*4, "put": [0]*4}
            continue
        r  = reg.loc[o]
        cp = [r["Call_p25"], r["Call_p50"], r["Call_p75"], r["Call_p90"]]
        pp = [r["Put_p25"],  r["Put_p50"],  r["Put_p75"],  r["Put_p90"]]
        if adj is not None and o in adj.index:
            ca = float(adj.loc[o, "Call_adj"] or 1.0)
            pa = float(adj.loc[o, "Put_adj"]  or 1.0)
            cp = [v * ca for v in cp]
            pp = [v * pa for v in pp]
        out[o] = {"call": cp, "put": pp}
    return out


def oi_bucket(oi: float, thresholds: list) -> int:
    if oi == 0:
        return 0
    p25, p50, p75, p90 = thresholds
    if oi < p25: return 1
    if oi < p50: return 2
    if oi < p75: return 3
    if oi < p90: return 4
    return 5


def fmt_oi(v: int) -> str:
    if v == 0:    return ""
    if v < 1000:  return str(v)
    if v < 10000: return f"{v/1000:.1f}K"
    return f"{v//1000}K"


def load_daily_window(exp_date: date, window: int = 2) -> pd.DataFrame | None:
    """Fetch `window` trading days on each side of exp_date from yfinance."""
    buf = window * 4
    start = exp_date - timedelta(days=buf)
    end   = min(exp_date + timedelta(days=buf), date.today() + timedelta(days=1))
    try:
        df = yf.download("QQQ", start=start.isoformat(), end=end.isoformat(),
                         interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.reset_index()
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
        exp_ts = pd.Timestamp(exp_date)
        mask = df[date_col] == exp_ts
        if not mask.any():
            return None
        ci = int(df.index[mask][0])
        return df.iloc[max(0, ci - window): ci + window + 1].reset_index(drop=True)
    except Exception:
        return None


def render_daily_context(ax: plt.Axes, df: pd.DataFrame | None, exp_date: date | None):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

    if exp_date is None or df is None or df.empty:
        ax.set_xticks([])
        ax.set_yticks([])
        return

    date_col = df.columns[0]
    opens  = pd.to_numeric(df["Open"],  errors="coerce").values
    highs  = pd.to_numeric(df["High"],  errors="coerce").values
    lows   = pd.to_numeric(df["Low"],   errors="coerce").values
    closes = pd.to_numeric(df["Close"], errors="coerce").values

    n   = len(df)
    xs  = np.arange(n)
    up  = closes >= opens
    exp_ts = pd.Timestamp(exp_date)
    center = next((i for i, v in enumerate(df[date_col]) if pd.Timestamp(v) == exp_ts), None)

    ax.vlines(xs[ up], lows[ up], highs[ up], color="#00cc55", linewidth=0.8, zorder=2)
    ax.vlines(xs[~up], lows[~up], highs[~up], color="#ee3300", linewidth=0.8, zorder=2)

    for i in xs:
        o, c = opens[i], closes[i]
        alpha = 1.0 if i == center else 0.4
        ax.add_patch(mpatches.Rectangle(
            (i - 0.35, min(o, c)), 0.7, max(abs(c - o), 0.05),
            color=("#00cc55" if c >= o else "#ee3300"), alpha=alpha, zorder=3,
        ))
    if center is not None:
        o, c = opens[center], closes[center]
        ax.add_patch(mpatches.Rectangle(
            (center - 0.38, min(o, c) - 0.03), 0.76, max(abs(c - o), 0.05) + 0.06,
            fill=False, edgecolor="#cccccc", linewidth=0.7, zorder=5,
        ))

    price_lo, price_hi = np.nanmin(lows), np.nanmax(highs)
    pad = (price_hi - price_lo) * 0.15 or 1.0
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(price_lo - pad, price_hi + pad)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Daily", color=DIM, fontsize=8, pad=2)


def render_intraday(ax: plt.Axes, df: pd.DataFrame | None, exp_date: date | None):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

    if exp_date is None:
        ax.set_xticks([])
        ax.set_yticks([])
        return

    if df is None or df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                color=DIM, transform=ax.transAxes, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(str(exp_date), color=DIM, fontsize=7, pad=3)
        return

    opens  = pd.to_numeric(df["open"],  errors="coerce").values
    highs  = pd.to_numeric(df["high"],  errors="coerce").values
    lows   = pd.to_numeric(df["low"],   errors="coerce").values
    closes = pd.to_numeric(df["close"], errors="coerce").values

    n  = len(df)
    xs = np.arange(n)
    up = closes >= opens

    ax.vlines(xs[ up],  lows[ up],  highs[ up],  color="#00cc55", linewidth=0.6, zorder=2)
    ax.vlines(xs[~up],  lows[~up],  highs[~up],  color="#ee3300", linewidth=0.6, zorder=2)

    for i in xs:
        o, c = opens[i], closes[i]
        ax.add_patch(mpatches.Rectangle(
            (i - 0.35, min(o, c)), 0.7, max(abs(c - o), 0.01),
            color=("#00cc55" if c >= o else "#ee3300"), zorder=3,
        ))

    price_lo, price_hi = np.nanmin(lows), np.nanmax(highs)
    pad = (price_hi - price_lo) * 0.06 or 0.5
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(price_lo - pad, price_hi + pad)

    # x-axis: three evenly-spaced time labels
    if "datetime" in df.columns and n > 0:
        def _hhmm(s):
            try:
                t = pd.Timestamp(str(s))
                return f"{t.hour}:{t.minute:02d}"
            except Exception:
                return ""
        ticks = [0, n // 2, n - 1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([_hhmm(df["datetime"].iloc[i]) for i in ticks],
                           fontsize=5.5, color=DIM)
    else:
        ax.set_xticks([])
    ax.tick_params(axis="x", colors=DIM, length=2, pad=2)

    ax.yaxis.tick_right()
    ax.yaxis.set_tick_params(labelsize=5.5, colors=DIM, pad=1, length=2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3, integer=False))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))

    ax.set_title(str(exp_date), color=DIM, fontsize=7, pad=3)


# ── rendering ──────────────────────────────────────────────────────────────────

def render_top5(ax: plt.Axes, df: pd.DataFrame, exp_str: str, spot: float,
                compact: bool = False):
    ax.set_facecolor(BG)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["bottom"].set_color(BORDER)
    ax.set_yticks([])

    sub = df[df["Expiration"] == exp_str].copy()
    sub["OI"] = pd.to_numeric(sub["OpenInterest"], errors="coerce").fillna(0).astype(int)
    top5 = sub.nlargest(5, "OI")[["Strike", "Type", "OI"]].reset_index(drop=True)

    if top5.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                color=DIM, transform=ax.transAxes, fontsize=9)
        return

    top5["rank"] = range(1, len(top5) + 1)
    top5["pct"]  = (top5["Strike"] - spot) / spot * 100

    if compact:
        rank_fs  = 7;  label_fs = 6;  tick_fs = 7;  spot_fs = 7
        s_min, s_max = 150, 280
        coll_thr = max(top5["pct"].abs().max(), 6.0) * 0.22
        row_y    = 0.65;  lbl_off = -0.32;  lbl_bot = 0.50
    else:
        rank_fs  = 9;  label_fs = 8;  tick_fs = 8;  spot_fs = 8
        s_min, s_max = 300, 500
        coll_thr = max(top5["pct"].abs().max(), 6.0) * 0.16
        row_y    = 0.75;  lbl_off = -0.38;  lbl_bot = 0.60

    sorted_rows = sorted(top5.to_dict("records"), key=lambda r: r["pct"])
    placed = []
    for row in sorted_rows:
        x = row["pct"]
        bump = 0
        while any(abs(p["x"] - x) < coll_thr and p["bump"] == bump
                  for p in placed):
            bump += 1
        placed.append({**row, "x": x, "bump": bump})

    max_bump = max(p["bump"] for p in placed)
    max_y    = max_bump * row_y

    max_abs = max(6.0, np.ceil(top5["pct"].abs().max()))
    ax.set_xlim(-max_abs * 1.05, max_abs * 1.05)
    ax.set_ylim(-lbl_bot, max_y + row_y * 0.55)

    ax.axvline(0, color=DIM, linewidth=0.5, linestyle="--", zorder=1)

    ticks = [-max_abs, -max_abs / 2, 0, max_abs / 2, max_abs]
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [f"{v:+.0f}%" if v != 0 else "0%" for v in ticks],
        fontsize=tick_fs, color=DIM,
    )
    ax.tick_params(axis="x", colors=DIM, length=3, pad=2)

    min_oi = min(p["OI"] for p in placed)
    max_oi = max(p["OI"] for p in placed)

    def mk_size(oi):
        if max_oi == min_oi:
            return (s_min + s_max) / 2
        t = (oi - min_oi) / (max_oi - min_oi)
        return s_min + t * (s_max - s_min)

    for p in placed:
        x   = p["x"]
        y   = p["bump"] * row_y
        is_call = p["Type"] == "call"
        edge = TOP5_CALL_EDGE if is_call else TOP5_PUT_EDGE
        fill = TOP5_CALL_FILL if is_call else TOP5_PUT_FILL

        ax.scatter([x], [y], s=mk_size(p["OI"]),
                   color=fill, edgecolors=edge, linewidths=1.5, zorder=3)
        ax.text(x, y, str(p["rank"]),
                ha="center", va="center",
                fontsize=rank_fs, fontweight="bold", color=edge, zorder=4)
        ax.text(x, y + lbl_off, str(int(round(float(p["Strike"])))),
                ha="center", va="top", fontsize=label_fs, color=FG, zorder=4)


def render(fig: plt.Figure, trade_date: date, df: pd.DataFrame,
           exp_str: str, tier: str, ranges: pd.DataFrame,
           scroll_offset: int = 0):
    fig.clear()
    fig.patch.set_facecolor(BG)

    sub = df[df["Expiration"] == exp_str]
    if sub.empty:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No data for this expiry",
                ha="center", va="center", color=FG, fontsize=12,
                transform=ax.transAxes)
        ax.set_facecolor(BG)
        return

    spot    = sub["UnderlyingPrice"].iloc[0]

    _s = sub.copy()
    _s["_oi"] = pd.to_numeric(_s["OpenInterest"], errors="coerce").fillna(0)
    top5_rank_call: dict[int, int] = {}
    top5_rank_put:  dict[int, int] = {}
    for _rank, (_, _r) in enumerate(_s.nlargest(5, "_oi").iterrows(), 1):
        _sk = int(round(float(_r["Strike"])))
        if str(_r["Type"]).lower() == "call":
            top5_rank_call.setdefault(_sk, _rank)
        else:
            top5_rank_put.setdefault(_sk, _rank)

    strikes = np.sort(sub["Strike"].unique())
    atm     = min(strikes, key=lambda s: abs(s - spot))

    top     = DISPLAY_WINDOW + scroll_offset
    bot     = -DISPLAY_WINDOW + scroll_offset
    offsets = list(range(top, bot - 1, -1))
    atm_row = offsets.index(0) if 0 in offsets else None
    n       = len(offsets)

    thresh     = effective_thresholds(tier, ranges, offsets)
    call_buck  = []
    put_buck   = []
    call_oi    = []
    put_oi     = []
    abs_strikes = []

    for o in offsets:
        sk = atm + o
        abs_strikes.append(f"{sk:.0f}")
        c = sub[(sub["Strike"] == sk) & (sub["Type"] == "call")]
        p = sub[(sub["Strike"] == sk) & (sub["Type"] == "put")]
        coi = int(c["OpenInterest"].sum()) if not c.empty else 0
        poi = int(p["OpenInterest"].sum()) if not p.empty else 0
        call_oi.append(coi)
        put_oi.append(poi)
        t = thresh[o]
        call_buck.append(oi_bucket(coi, t["call"]))
        put_buck.append(oi_bucket(poi, t["put"]))

    c_arr = np.array(call_buck).reshape(-1, 1)
    p_arr = np.array(put_buck).reshape(-1, 1)

    c_cmap = mcolors.ListedColormap(CALL_COLORS)
    p_cmap = mcolors.ListedColormap(PUT_COLORS)

    ext = [-0.5, 0.5, n - 0.5, -0.5]

    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[5, 2, 5],
        left=0.06, right=0.97,
        top=0.91, bottom=0.08,
        wspace=0.03,
    )
    ax_c   = fig.add_subplot(gs[0])
    ax_lbl = fig.add_subplot(gs[1])
    ax_p   = fig.add_subplot(gs[2])

    for ax, arr, cmap, oi_vals, rank_map in (
        (ax_c, c_arr, c_cmap, call_oi, top5_rank_call),
        (ax_p, p_arr, p_cmap, put_oi,  top5_rank_put),
    ):
        ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=5,
                  interpolation="nearest", extent=ext)
        ax.set_facecolor(BG)
        ax.set_xticks([])
        ax.set_ylim(n - 0.5, -0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)

        for i, oi in enumerate(oi_vals):
            sk_int = int(round(float(atm + offsets[i])))
            badge  = rank_map.get(sk_int)
            if oi == 0 and not badge:
                continue
            b = arr[i, 0]
            txt_col = "#000000" if b >= 4 else "#ffffff"
            if oi > 0:
                ax.text(0, i, fmt_oi(oi),
                        ha="center", va="center",
                        fontsize=14, color=txt_col, fontweight="bold")
            if badge:
                ax.text(0.44, i, f"#{badge}",
                        ha="right", va="center",
                        fontsize=10, color=txt_col, fontweight="bold")

        if atm_row is not None:
            ax.axhspan(atm_row - 0.5, atm_row + 0.5,
                       color="#ffffff", alpha=0.06, zorder=0)

    ax_c.set_yticks(range(n))
    ax_c.set_yticklabels(
        ["ATM" if o == 0 else f"{o:+d}" for o in offsets],
        fontsize=9, color=DIM,
    )
    ax_c.yaxis.set_tick_params(length=0, pad=2)
    ax_c.set_title("CALLS", color="#00cc55", fontsize=13, fontweight="bold", pad=5)

    ax_p.set_yticks([])
    ax_p.set_title("PUTS", color="#ee3300", fontsize=13, fontweight="bold", pad=5)

    ax_lbl.set_facecolor(BG)
    ax_lbl.set_xlim(0, 1)
    ax_lbl.set_ylim(n - 0.5, -0.5)
    ax_lbl.set_xticks([])
    ax_lbl.set_yticks([])
    for spine in ax_lbl.spines.values():
        spine.set_edgecolor(BORDER)
    for i, (sk_str, o) in enumerate(zip(abs_strikes, offsets)):
        is_atm = (o == 0)
        ax_lbl.text(
            0.5, i, sk_str,
            ha="center", va="center",
            fontsize=11,
            color="#ffffff" if is_atm else FG,
            fontweight="bold" if is_atm else "normal",
        )
    ax_lbl.set_title("Strike", color=DIM, fontsize=9, pad=5)

    _exp_d  = date.fromisoformat(exp_str)
    _exp_fmt = f"{_exp_d.strftime('%b')} {_exp_d.day}"
    _cap_fmt = f"{trade_date.month}/{trade_date.day}/{trade_date.year}"
    _tier_labels = {
        "0DTE_Regular": "Regular Trading Day",
        "0DTE_Weekly":  "Weekly Expiration",
        "0DTE_Monthly": "Monthly Expiration",
    }
    fig.text(0.5, 0.962,
             f"QQQ Options Chain Expiring {_exp_fmt}, as Captured at 7 PM on {_cap_fmt} (Spot ${spot:.2f})",
             ha="center", color=FG, fontsize=16, fontweight="bold")
    fig.text(0.5, 0.940,
             f"[ {_tier_labels.get(tier, tier)} ]",
             ha="center", color=TIER_COLORS.get(tier, FG), fontsize=12, fontweight="bold")

    level_labels = ["< p25", "p25–p50", "p50–p75", "p75–p90", "> p90"]
    c_patches = [mpatches.Patch(facecolor=CALL_COLORS[i+1], edgecolor=BORDER,
                                label=level_labels[i]) for i in range(5)]
    p_patches = [mpatches.Patch(facecolor=PUT_COLORS[i+1],  edgecolor=BORDER,
                                label=level_labels[i]) for i in range(5)]
    spacer    =  mpatches.Patch(facecolor=BG, edgecolor=BG, label="  ")

    fig.legend(
        handles=c_patches + [spacer] + p_patches,
        loc="lower center",
        ncol=11,
        fontsize=8,
        facecolor=PANEL,
        edgecolor=BORDER,
        labelcolor=FG,
        framealpha=1.0,
        bbox_to_anchor=(0.5, 0.0),
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=0.6,
    )

    fig.canvas.draw_idle()


def _update_render(fig: plt.Figure, trade_date: date, df: pd.DataFrame,
                   exp_str: str, tier: str, ranges: pd.DataFrame,
                   scroll_offset: int = 0):
    """Update an already-rendered figure in place for a new scroll offset.
    Skips fig.clear() and layout — only swaps data that actually changes between frames."""
    sub = df[df["Expiration"] == exp_str]
    if sub.empty:
        return

    spot = sub["UnderlyingPrice"].iloc[0]

    _s = sub.copy()
    _s["_oi"] = pd.to_numeric(_s["OpenInterest"], errors="coerce").fillna(0)
    top5_rank_call: dict[int, int] = {}
    top5_rank_put:  dict[int, int] = {}
    for _rank, (_, _r) in enumerate(_s.nlargest(5, "_oi").iterrows(), 1):
        _sk = int(round(float(_r["Strike"])))
        if str(_r["Type"]).lower() == "call":
            top5_rank_call.setdefault(_sk, _rank)
        else:
            top5_rank_put.setdefault(_sk, _rank)

    strikes = np.sort(sub["Strike"].unique())
    atm     = min(strikes, key=lambda s: abs(s - spot))

    top     = DISPLAY_WINDOW + scroll_offset
    bot     = -DISPLAY_WINDOW + scroll_offset
    offsets = list(range(top, bot - 1, -1))
    atm_row = offsets.index(0) if 0 in offsets else None

    thresh    = effective_thresholds(tier, ranges, offsets)
    call_buck, put_buck = [], []
    call_oi,  put_oi   = [], []
    abs_strikes = []

    for o in offsets:
        sk = atm + o
        abs_strikes.append(f"{sk:.0f}")
        c = sub[(sub["Strike"] == sk) & (sub["Type"] == "call")]
        p = sub[(sub["Strike"] == sk) & (sub["Type"] == "put")]
        coi = int(c["OpenInterest"].sum()) if not c.empty else 0
        poi = int(p["OpenInterest"].sum()) if not p.empty else 0
        call_oi.append(coi)
        put_oi.append(poi)
        t = thresh[o]
        call_buck.append(oi_bucket(coi, t["call"]))
        put_buck.append(oi_bucket(poi, t["put"]))

    c_arr = np.array(call_buck).reshape(-1, 1)
    p_arr = np.array(put_buck).reshape(-1, 1)

    ax_c, ax_lbl, ax_p = fig.axes[:3]

    ax_c.images[0].set_data(c_arr)
    ax_p.images[0].set_data(p_arr)

    ax_c.set_yticklabels(
        ["ATM" if o == 0 else f"{o:+d}" for o in offsets],
        fontsize=9, color=DIM,
    )

    for ax, oi_vals, rank_map, arr in (
        (ax_c, call_oi, top5_rank_call, c_arr),
        (ax_p, put_oi,  top5_rank_put,  p_arr),
    ):
        for txt in ax.texts[:]:
            txt.remove()
        for patch in ax.patches[:]:
            patch.remove()

        for i, oi in enumerate(oi_vals):
            sk_int = int(round(float(atm + offsets[i])))
            badge  = rank_map.get(sk_int)
            if oi == 0 and not badge:
                continue
            b = arr[i, 0]
            txt_col = "#000000" if b >= 4 else "#ffffff"
            if oi > 0:
                ax.text(0, i, fmt_oi(oi),
                        ha="center", va="center",
                        fontsize=14, color=txt_col, fontweight="bold")
            if badge:
                ax.text(0.44, i, f"#{badge}",
                        ha="right", va="center",
                        fontsize=10, color=txt_col, fontweight="bold")

        if atm_row is not None:
            ax.axhspan(atm_row - 0.5, atm_row + 0.5,
                       color="#ffffff", alpha=0.06, zorder=0)

    for txt in ax_lbl.texts[:]:
        txt.remove()
    for i, (sk_str, o) in enumerate(zip(abs_strikes, offsets)):
        is_atm = (o == 0)
        ax_lbl.text(
            0.5, i, sk_str,
            ha="center", va="center",
            fontsize=11,
            color="#ffffff" if is_atm else FG,
            fontweight="bold" if is_atm else "normal",
        )


# ── application ────────────────────────────────────────────────────────────────

class OIViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Options View")
        self.configure(bg=BG)
        self.resizable(True, True)

        # App icon — drop icon.png next to this script to activate
        _icon_path = ROOT / "icon.png"
        if _icon_path.exists():
            try:
                self._icon = tk.PhotoImage(file=str(_icon_path))
                self.iconphoto(True, self._icon)
            except Exception:
                pass

        capture_dates      = available_dates()
        self.ranges        = load_ranges()
        self.scroll_offset = 0
        self._cur: dict    = {}

        sorted_captures = sorted(capture_dates)
        _bootstrap(sorted_captures[0] - timedelta(days=5),
                   sorted_captures[-1] + timedelta(days=60))

        # Key structural choice: the UI is indexed by expiry date, not capture date.
        # Each chain is keyed by the date the session traded, not the date it was captured.
        self._expiry_capture: dict[date, date] = {
            next_trading_day(c): c for c in capture_dates
        }
        self.avail = set(self._expiry_capture.keys())

        sorted_expiries = sorted(self.avail)
        lo, hi = sorted_expiries[0], sorted_expiries[-1]
        self._build(lo, hi)
        self.after(100, lambda: self.show_date(hi))

    # ── UI construction ────────────────────────────────────────────────────────

    def _build(self, lo: date, hi: date):
        left = tk.Frame(self, bg=BG, width=310)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 4), pady=10)
        left.pack_propagate(False)

        self.cal = TkCalendar(
            left,
            selectmode="day",
            year=hi.year, month=hi.month, day=hi.day,
            mindate=datetime(lo.year, lo.month, lo.day),
            maxdate=datetime(hi.year, hi.month, hi.day),
            background=PANEL,
            foreground=FG,
            selectbackground="#2ea043",
            selectforeground="#ffffff",
            headersbackground="#1c2128",
            headersforeground=FG,
            normalbackground=PANEL,
            normalforeground=FG,
            weekendbackground=PANEL,
            weekendforeground=DIM,
            othermonthbackground=BG,
            othermonthforeground="#444444",
            bordercolor=BORDER,
            font=("Segoe UI", 9),
        )
        self.cal.pack(fill=tk.X, pady=(0, 4))

        for d in self.avail:
            try:
                self.cal.calevent_create(
                    datetime(d.year, d.month, d.day), "", "have_data"
                )
            except Exception:
                pass
        self.cal.tag_config("have_data", background="#1a3a28", foreground="#4dff9a")

        tk.Button(
            left,
            text="View Date",
            command=self._on_select,
            bg="#2ea043", fg="#ffffff",
            activebackground="#3fb454", activeforeground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            relief="flat", padx=8, pady=5,
            cursor="hand2",
        ).pack(fill=tk.X, pady=(0, 6))

        self.fig_price = plt.Figure(figsize=(3.0, 2.0), facecolor=BG)
        self.canvas_price = FigureCanvasTkAgg(self.fig_price, master=left)
        self.canvas_price.get_tk_widget().pack(fill=tk.X, pady=(6, 0))

        self.fig_top5 = plt.Figure(figsize=(3.0, 1.8), facecolor=BG)
        self.canvas_top5 = FigureCanvasTkAgg(self.fig_top5, master=left)
        self.canvas_top5.get_tk_widget().pack(fill=tk.X, pady=(4, 0))

        right = tk.Frame(self, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                   pady=10, padx=(4, 10))

        self.scroll_scale = tk.Scale(
            right,
            orient=tk.VERTICAL,
            from_=SCROLL_MAX, to=-SCROLL_MAX,
            resolution=1,
            showvalue=False,
            bg=PANEL, troughcolor=BG,
            activebackground=BORDER,
            highlightthickness=0, bd=0,
            width=10, sliderlength=18,
            command=self._on_scale,
        )
        self.scroll_scale.pack(side=tk.RIGHT, fill=tk.Y, padx=(3, 0))

        self.fig = plt.Figure(figsize=(12, 9.5), facecolor=BG)
        _ax = self.fig.add_axes([0, 0, 1, 1])
        _ax.set_facecolor(BG)
        _ax.set_axis_off()
        self._display_img = _ax.imshow(
            np.zeros((950, 1200, 4), dtype=np.uint8),
            aspect="auto", interpolation="antialiased",
        )
        self._display_ax = _ax
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        self._loading_label = tk.Label(
            self.canvas.get_tk_widget(),
            bg=PANEL, fg=FG,
            font=("Segoe UI", 13),
            padx=24, pady=14,
        )

    # ── event handlers ─────────────────────────────────────────────────────────

    def _on_select(self):
        sel = self.cal.selection_get()
        self.show_date(sel if isinstance(sel, date) else sel.date())

    _SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def _set_loading(self, frame_idx: int):
        n    = len(SCROLL_POSITIONS)
        spin = self._SPINNER[frame_idx % len(self._SPINNER)]
        self._loading_label.config(
            text=f"{spin}  Rendering data...  {frame_idx} / {n}"
        )
        self._loading_label.place(relx=0.5, rely=0.5, anchor="center")
        self.update_idletasks()

    def _prerender_all(self, d: date, df: pd.DataFrame, exp_str: str, tier: str) -> dict:
        offscreen = plt.Figure(figsize=(12, 9.5), facecolor=BG, dpi=72)
        FigureCanvasAgg(offscreen)
        frames = {}
        t_total = time.perf_counter()
        for i, offset in enumerate(SCROLL_POSITIONS):
            t0 = time.perf_counter()
            if i == 0:
                render(offscreen, d, df, exp_str, tier, self.ranges, scroll_offset=offset)
            else:
                _update_render(offscreen, d, df, exp_str, tier, self.ranges, scroll_offset=offset)
            t1 = time.perf_counter()
            offscreen.canvas.draw()
            t2 = time.perf_counter()
            buf = offscreen.canvas.buffer_rgba()
            w, h = offscreen.canvas.get_width_height()
            frames[offset] = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4).copy()
            label = "render" if i == 0 else "update"
            _log.debug(f"[{label:6s}] offset {offset:+3d}: artists={t1-t0:.2f}s  draw={t2-t1:.2f}s")
            self._set_loading(i + 1)
        _log.debug(f"[total ]  {len(frames)} frames: {time.perf_counter() - t_total:.2f}s")
        return frames

    def show_date(self, d: date):
        """d is the expiry date (session date). Load the chain captured the prior day."""
        self._loading_label.config(text="⠋  Fetching data...")
        self._loading_label.place(relx=0.5, rely=0.5, anchor="center")
        self.update_idletasks()

        capture = self._expiry_capture.get(d)
        if capture is None:
            self._loading_label.place_forget()
            return
        df = load_day(capture)
        if df is None:
            self._loading_label.place_forget()
            return
        exp_str = d.isoformat()
        tier = classify_tier(capture)
        self._cur = {"d": capture, "df": df, "exp_str": exp_str, "tier": tier}
        self.scroll_offset = 0
        self.scroll_scale.set(0)
        self._set_loading(0)
        self._prerendered = self._prerender_all(capture, df, exp_str, tier)
        self._loading_label.place_forget()
        self._rerender()

        sub5 = df[df["Expiration"] == exp_str]
        spot5 = sub5["UnderlyingPrice"].iloc[0] if not sub5.empty else None

        # Price panel: intraday 5-min bars (left) + daily context candles (right).
        # Both keyed on the expiry date; blank if the expiry is still in the future.
        exp_date_for_panel = d if d <= date.today() else None
        intraday_df = load_intraday(d) if exp_date_for_panel is not None else None
        daily_df    = load_daily_window(d) if exp_date_for_panel is not None else None

        # Top widget: intraday price only
        self.fig_price.clear()
        ax_id = self.fig_price.add_subplot(111)
        self.fig_price.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.22)
        render_intraday(ax_id, intraday_df, exp_date_for_panel)
        self.canvas_price.draw()

        # Bottom widget: top-5 strikes (left) + daily candles (right)
        self.fig_top5.clear()
        gs5 = self.fig_top5.add_gridspec(
            1, 2, width_ratios=[7, 3],
            left=0.06, right=0.97, top=0.88, bottom=0.24, wspace=0.38,
        )
        ax5     = self.fig_top5.add_subplot(gs5[0])
        ax_daily = self.fig_top5.add_subplot(gs5[1])
        if spot5 is not None:
            render_top5(ax5, df, exp_str, spot5, compact=True)
        render_daily_context(ax_daily, daily_df, exp_date_for_panel)
        self.canvas_top5.draw()

    def _rerender(self):
        if not self._cur or not hasattr(self, "_prerendered"):
            return
        arr = self._prerendered.get(self.scroll_offset)
        if arr is None:
            return
        self._display_img.set_data(arr)
        self.canvas.draw_idle()

    def _on_scroll(self, event):
        if not self._cur:
            return
        delta = 1 if event.button == "up" else -1
        idx = SCROLL_POSITIONS.index(self.scroll_offset)
        new_idx = max(0, min(len(SCROLL_POSITIONS) - 1, idx + delta))
        new_offset = SCROLL_POSITIONS[new_idx]
        if new_offset != self.scroll_offset:
            self.scroll_offset = new_offset
            self.scroll_scale.set(new_offset)
            self._rerender()

    def _on_scale(self, value):
        raw = int(round(float(value)))
        new_offset = min(SCROLL_POSITIONS, key=lambda p: abs(p - raw))
        if new_offset != self.scroll_offset:
            self.scroll_offset = new_offset
            self._rerender()


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = OIViewer()
    app.mainloop()
