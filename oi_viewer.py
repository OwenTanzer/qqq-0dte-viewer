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
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import requests

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkcalendar import Calendar as TkCalendar

from utils import (
    _ensure_calendar_loaded, nominal_friday, prior_trading_day, target_expirations,
)

# ── paths and remote data ──────────────────────────────────────────────────────

import sys as _sys
ROOT = Path(_sys._MEIPASS) if getattr(_sys, "frozen", False) else Path(__file__).parent
R2_BASE = "https://pub-4d5c916b8cb74ffb8c0abd7dfadb02cf.r2.dev"

# ── display config ─────────────────────────────────────────────────────────────

DISPLAY_WINDOW = 20   # ±N strikes shown in heatmap
SCROLL_MAX     = 33 - DISPLAY_WINDOW   # max scroll offset (data covers ±33)

# 6-level discrete palette
# Level 0 = zero OI (background), levels 1-5 = increasing intensity
CALL_COLORS = [
    "#0d1117",   # 0  zero OI
    "#0a1f14",   # 1  < p25
    "#0a3020",   # 2  p25–p50
    "#007730",   # 3  p50–p75
    "#00cc55",   # 4  p75–p90
    "#88ffcc",   # 5  > p90  ("wall")
]
PUT_COLORS = [
    "#0d1117",   # 0  zero OI
    "#1a0d0d",   # 1  < p25
    "#2a0a0a",   # 2  p25–p50
    "#881100",   # 3  p50–p75
    "#ee3300",   # 4  p75–p90
    "#ffaa88",   # 5  > p90  ("wall")
]

TOP5_CALL_EDGE = "#4a9eff"
TOP5_CALL_FILL = "#0a1f3d"
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
DIM    = "#6e7681"

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
        resp = requests.get(f"{R2_BASE}/{key}", timeout=15)
        if resp.status_code == 200:
            return pd.read_csv(io.StringIO(resp.text))
    return None


def load_ranges() -> pd.DataFrame:
    resp = requests.get(f"{R2_BASE}/derived/OIranges.csv", timeout=15)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


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
            txt_col = "#ffffff" if b >= 3 else ("#aaaaaa" if b == 2 else DIM)
            if oi > 0:
                ax.text(0, i, fmt_oi(oi),
                        ha="center", va="center",
                        fontsize=7, color=txt_col, fontweight="bold")
            if badge:
                ax.text(0.44, i, f"#{badge}",
                        ha="right", va="center",
                        fontsize=5.5, color="#000000", fontweight="bold")

        if atm_row is not None:
            ax.axhspan(atm_row - 0.5, atm_row + 0.5,
                       color="#ffffff", alpha=0.06, zorder=0)

    ax_c.set_yticks(range(n))
    ax_c.set_yticklabels(
        ["ATM" if o == 0 else f"{o:+d}" for o in offsets],
        fontsize=6.5, color=DIM,
    )
    ax_c.yaxis.set_tick_params(length=0, pad=2)
    ax_c.set_title("CALLS", color="#00cc55", fontsize=11, fontweight="bold", pad=5)

    ax_p.set_yticks([])
    ax_p.set_title("PUTS", color="#ee3300", fontsize=11, fontweight="bold", pad=5)

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
            fontsize=7,
            color="#ffffff" if is_atm else FG,
            fontweight="bold" if is_atm else "normal",
        )
    ax_lbl.set_title("Strike", color=DIM, fontsize=8, pad=5)

    tier_short = tier.replace("0DTE_", "")
    tier_col   = TIER_COLORS.get(tier, FG)
    fig.text(0.5, 0.957,
             f"QQQ 0DTE  ·  {trade_date}  ·  ${spot:.2f}",
             ha="center", color=FG, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.938,
             f"[ {tier_short} ]",
             ha="center", color=tier_col, fontsize=10, fontweight="bold")

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
        fontsize=7.5,
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

        self.avail         = available_dates()
        self.ranges        = load_ranges()
        self.scroll_offset = 0
        self._cur: dict    = {}

        sorted_dates = sorted(self.avail)
        lo, hi = sorted_dates[0], sorted_dates[-1]
        _bootstrap(lo - timedelta(days=5), hi + timedelta(days=60))

        self._build(lo, hi)
        self.after(100, lambda: self.show_date(hi))

    # ── UI construction ────────────────────────────────────────────────────────

    def _build(self, lo: date, hi: date):
        left = tk.Frame(self, bg=BG, width=210)
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

        self.fig_top5 = plt.Figure(figsize=(2.0, 1.8), facecolor=BG)
        self.fig_top5.subplots_adjust(left=0.06, right=0.96, top=0.94, bottom=0.24)
        self.canvas_top5 = FigureCanvasTkAgg(self.fig_top5, master=left)
        self.canvas_top5.get_tk_widget().pack(fill=tk.X, pady=(6, 0))

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
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    # ── event handlers ─────────────────────────────────────────────────────────

    def _on_select(self):
        sel = self.cal.selection_get()
        self.show_date(sel if isinstance(sel, date) else sel.date())

    def show_date(self, d: date):
        df = load_day(d)
        if df is None:
            return
        label_map = dict(target_expirations(d))
        exp_date  = label_map.get("+1D")
        if exp_date is None:
            return
        exp_str = exp_date.isoformat()
        self._cur = {"d": d, "df": df, "exp_str": exp_str,
                     "tier": classify_tier(d)}
        self.scroll_offset = 0
        self.scroll_scale.set(0)
        self._rerender()

        sub5 = df[df["Expiration"] == exp_str]
        if not sub5.empty:
            spot5 = sub5["UnderlyingPrice"].iloc[0]
            self.fig_top5.clear()
            self.fig_top5.subplots_adjust(left=0.06, right=0.96, top=0.94, bottom=0.24)
            ax5 = self.fig_top5.add_subplot(111)
            render_top5(ax5, df, exp_str, spot5, compact=True)
            self.canvas_top5.draw()

    def _rerender(self):
        if not self._cur:
            return
        c = self._cur
        render(self.fig, c["d"], c["df"], c["exp_str"], c["tier"],
               self.ranges, scroll_offset=self.scroll_offset)
        self.canvas.draw()

    def _on_scroll(self, event):
        if not self._cur:
            return
        delta = 1 if event.button == "up" else -1
        self.scroll_offset = max(-SCROLL_MAX, min(SCROLL_MAX,
                                                   self.scroll_offset + delta))
        self.scroll_scale.set(self.scroll_offset)
        self._rerender()

    def _on_scale(self, value):
        new_offset = int(round(float(value)))
        if new_offset != self.scroll_offset:
            self.scroll_offset = new_offset
            self._rerender()


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = OIViewer()
    app.mainloop()
