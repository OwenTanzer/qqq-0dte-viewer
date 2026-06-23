#!/usr/bin/env python3
"""
collector.py — QQQ 0DTE live chain snapshot service.

Authenticates with tastytrade, subscribes to the QQQ 0DTE option chain via
DXLink websocket, and uploads a snapshot to R2 every 11 minutes.

R2 output:
  intraday/YYYYMMDD/snapshot_HHMM.csv   — archived snapshots
  intraday/latest.json                   — live feed for the web viewer

Environment variables (set in Railway dashboard):
  TASTY_LOGIN            tastytrade username
  TASTY_PASSWORD         tastytrade password
  R2_ACCOUNT_ID          Cloudflare account ID
  R2_ACCESS_KEY_ID       R2 access key
  R2_SECRET_ACCESS_KEY   R2 secret key
  R2_BUCKET_NAME         bucket name (default: pub-4d5c916b8cb74ffb8c0abd7dfadb02cf)
"""

import io
import json
import logging
import os
import sys
import threading
import time
from datetime import date, datetime, timezone
from typing import Optional

import boto3
import pandas as pd
import pytz
import requests
import websocket

# ── logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("collector")

# ── config ─────────────────────────────────────────────────────────────────────

ET             = pytz.timezone("America/New_York")
TASTY_BASE     = "https://api.tastyworks.com"
TICKER         = "QQQ"
STRIKE_WINDOW  = 33
SNAPSHOT_SECS  = 11 * 60
PRICES_SECS    = 30          # how often to push prices.json
PREMARKET_HOUR = 6           # start at 6:00 AM ET
STOP_HOUR      = 16
STOP_MIN       = 15
R2_BUCKET      = os.environ.get("R2_BUCKET_NAME", "pub-4d5c916b8cb74ffb8c0abd7dfadb02cf")

# Display label → DXLink symbol
# VIX index is $VIX.X in dxfeed; BTC/USD via Coinbase on tastytrade
PRICE_TICKERS: dict[str, str] = {
    "QQQ":    "QQQ",
    "USO":    "USO",
    "VIX":    "$VIX.X",
    "SMH":    "SMH",
    "IGV":    "IGV",
    "JPY/USD": "/6J:XCME",   # CME yen futures, quoted USD-per-JPY → invert for ¥/$ display
    "BTC/USD": "BTC/USD:CXERX",
    "META":   "META",
    "GOOGL":  "GOOGL",
    "AMZN":   "AMZN",
    "TSLA":   "TSLA",
}


# ── tastytrade auth ────────────────────────────────────────────────────────────

def tasty_auth(login: str, password: str) -> dict:
    """Authenticate and return {session_token, streamer_token, streamer_url}."""
    resp = requests.post(
        f"{TASTY_BASE}/sessions",
        json={"login": login, "password": password, "remember-me": True},
        headers={"Content-Type": "application/json"},
        timeout=15,
    )
    resp.raise_for_status()
    session_token = resp.json()["data"]["session-token"]
    log.info("tastytrade session established")

    resp2 = requests.get(
        f"{TASTY_BASE}/api-quote-tokens",
        headers={"Authorization": session_token},
        timeout=10,
    )
    resp2.raise_for_status()
    d = resp2.json()["data"]
    streamer_token = d["token"]
    streamer_url   = d.get("dxlink-url") or d.get("websocket-url") or \
                     "wss://tasty-openapi-ws.dxfeed.com/realtime"
    log.info(f"streamer token obtained  url={streamer_url}")
    return {
        "session_token":  session_token,
        "streamer_token": streamer_token,
        "streamer_url":   streamer_url,
    }


# ── option chain structure ─────────────────────────────────────────────────────

def _dxlink_symbol(occ_symbol: str) -> str:
    """Convert OCC symbol to DXLink format: 'QQQ   260623C00480000' → '.QQQ260623C00480000'"""
    return "." + occ_symbol.replace(" ", "")


def _build_symbol(strike: float, exp_date: str, option_type: str) -> str:
    """Build DXLink symbol from components when streamer-symbol is missing."""
    yy, mm, dd = exp_date[2:4], exp_date[5:7], exp_date[8:10]
    side = "C" if option_type.lower() == "call" else "P"
    strike_int = int(round(strike * 1000))
    return f".{TICKER}{yy}{mm}{dd}{side}{strike_int:08d}"


def load_chain(session_token: str, today: date) -> tuple[list[dict], str]:
    """
    Returns (strikes, expiration_date_str).
    Each strike: {"strike": float, "call_sym": str, "put_sym": str,
                  "call_occ": str, "put_occ": str}
    Picks today's expiration, or the nearest upcoming one pre-market.
    """
    resp = requests.get(
        f"{TASTY_BASE}/option-chains/{TICKER}/nested",
        headers={"Authorization": session_token},
        timeout=30,
    )
    resp.raise_for_status()

    items = resp.json().get("data", {}).get("items", [])
    if not items:
        raise RuntimeError("empty option chain response")

    today_str = today.isoformat()
    expirations = items[0].get("expirations", [])

    # Pick today's expiry, or nearest future
    target = None
    for exp in sorted(expirations, key=lambda e: e.get("expiration-date", "")):
        if exp.get("expiration-date", "") >= today_str:
            target = exp
            break
    if target is None:
        raise RuntimeError(f"no upcoming expiration found in chain for {today_str}")

    exp_date = target["expiration-date"]
    log.info(f"chain expiration: {exp_date}  ({len(target.get('strikes', []))} strikes)")

    strikes = []
    for s in target.get("strikes", []):
        strike = float(s.get("strike-price", 0))
        c = s.get("call", {})
        p = s.get("put",  {})

        call_occ = c.get("symbol", "")
        put_occ  = p.get("symbol", "")
        call_sym = c.get("streamer-symbol") or (_dxlink_symbol(call_occ) if call_occ else
                                                 _build_symbol(strike, exp_date, "call"))
        put_sym  = p.get("streamer-symbol") or (_dxlink_symbol(put_occ) if put_occ else
                                                 _build_symbol(strike, exp_date, "put"))
        strikes.append({
            "strike":   strike,
            "call_sym": call_sym,
            "put_sym":  put_sym,
            "call_occ": call_occ,
            "put_occ":  put_occ,
        })

    return strikes, exp_date


# ── DXLink websocket feed ──────────────────────────────────────────────────────

class DXLinkFeed:
    """
    Maintains latest market data for subscribed symbols.
    Runs the websocket in a background thread; thread-safe reads via get_state().
    """

    _DXLINK_VERSION = "0.1-js/1.0.0"

    def __init__(self, url: str, token: str):
        self._url      = url
        self._token    = token
        self._state: dict[str, dict] = {}
        self._lock     = threading.Lock()
        self._ws: Optional[websocket.WebSocketApp] = None
        self._ready    = threading.Event()
        self._subs: list[dict] = []

    # ── public API ─────────────────────────────────────────────────────────────

    def set_subscriptions(self, option_symbols: list[str], price_symbols: list[str]):
        """
        option_symbols: full options chain — Quote, Summary, Trade, Greeks
        price_symbols:  equity/index tickers — Quote, Trade, Summary (for prev close)
        """
        self._subs = []
        for sym in option_symbols:
            for event_type in ("Quote", "Summary", "Trade", "Greeks"):
                self._subs.append({"type": event_type, "symbol": sym})
        for sym in price_symbols:
            for event_type in ("Quote", "Trade", "Summary"):
                self._subs.append({"type": event_type, "symbol": sym})

    def get_state(self) -> dict[str, dict]:
        with self._lock:
            return {k: dict(v) for k, v in self._state.items()}

    def wait_ready(self, timeout: float = 60.0) -> bool:
        return self._ready.wait(timeout=timeout)

    def start(self):
        self._ws = websocket.WebSocketApp(
            self._url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        t = threading.Thread(
            target=self._ws.run_forever,
            kwargs={"reconnect": 5},
            daemon=True,
        )
        t.start()
        log.info("DXLink feed thread started")

    def stop(self):
        if self._ws:
            self._ws.close()

    # ── websocket handlers ─────────────────────────────────────────────────────

    def _send(self, msg: dict):
        if self._ws:
            self._ws.send(json.dumps(msg))

    def _on_open(self, ws):
        log.info("DXLink connected — sending SETUP")
        self._send({
            "type": "SETUP", "channel": 0,
            "version": self._DXLINK_VERSION,
            "keepaliveTimeout": 60,
            "acceptKeepaliveTimeout": 60,
        })

    def _on_message(self, ws, raw: str):
        try:
            msg = json.loads(raw)
        except Exception:
            return

        mtype = msg.get("type")

        if mtype == "SETUP":
            self._send({"type": "AUTH", "channel": 0, "token": self._token})

        elif mtype == "AUTH_STATE":
            state = msg.get("state")
            if state == "AUTHORIZED":
                log.info("DXLink authorized — requesting channel")
                self._send({
                    "type": "CHANNEL_REQUEST", "channel": 1,
                    "service": "FEED",
                    "parameters": {"contract": "AUTO"},
                })
            else:
                log.error(f"DXLink auth failed: {msg}")

        elif mtype == "CHANNEL_OPENED":
            log.info("DXLink channel 1 open — sending FEED_SETUP")
            self._send({
                "type": "FEED_SETUP", "channel": 1,
                "acceptDataFormat": "FULL",
                "acceptEventFields": {
                    "Quote":   ["eventSymbol", "bidPrice", "askPrice"],
                    "Summary": ["eventSymbol", "openInterest", "prevDayClosePrice", "dayOpenPrice"],
                    "Trade":   ["eventSymbol", "dayVolume", "price"],
                    "Greeks":  ["eventSymbol", "volatility", "delta", "gamma", "theta", "vega"],
                },
            })
            if self._subs:
                self._send({
                    "type": "FEED_SUBSCRIPTION", "channel": 1,
                    "reset": True, "add": self._subs,
                })
                log.info(f"subscribed to {len(self._subs)} event/symbol pairs")
            self._ready.set()

        elif mtype == "FEED_DATA":
            self._ingest(msg.get("data", []))

        elif mtype == "KEEPALIVE":
            self._send({"type": "KEEPALIVE", "channel": 0})

        elif mtype == "ERROR":
            log.error(f"DXLink server error: {msg}")

    def _ingest(self, data):
        if not isinstance(data, list):
            return
        for event in data:
            if not isinstance(event, dict):
                continue
            et  = event.get("eventType")
            sym = event.get("eventSymbol")
            if not sym:
                continue
            with self._lock:
                s = self._state.setdefault(sym, {})
                if et == "Quote":
                    if event.get("bidPrice") is not None:
                        s["bid"] = event["bidPrice"]
                    if event.get("askPrice") is not None:
                        s["ask"] = event["askPrice"]
                elif et == "Summary":
                    if event.get("openInterest") is not None:
                        s["oi"] = int(event["openInterest"])
                    if event.get("prevDayClosePrice") is not None:
                        s["prev_close"] = event["prevDayClosePrice"]
                    if event.get("dayOpenPrice") is not None:
                        s["day_open"] = event["dayOpenPrice"]
                elif et == "Trade":
                    if event.get("dayVolume") is not None:
                        s["volume"] = int(event["dayVolume"])
                    if event.get("price") is not None:
                        s["last"] = event["price"]
                elif et == "Greeks":
                    for field in ("volatility", "delta", "gamma", "theta", "vega"):
                        if event.get(field) is not None:
                            s[field] = event[field]

    def _on_error(self, ws, error):
        log.error(f"DXLink error: {error}")

    def _on_close(self, ws, code, msg):
        log.warning(f"DXLink closed: code={code}")
        self._ready.clear()


# ── tier classification (mirrors oi_viewer.py) ─────────────────────────────────

def _load_calendar():
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        from datetime import timedelta
        start = date.today()
        end   = start + timedelta(days=90)
        return {d.date() for d in nyse.valid_days(start_date=start.isoformat(),
                                                    end_date=end.isoformat())}
    except Exception:
        return set()


def classify_tier(today: date) -> str:
    from datetime import timedelta
    import calendar as _cal

    valid = _load_calendar()

    def prior_td(d):
        while d not in valid:
            d -= timedelta(days=1)
        return d

    def next_td(d):
        d += timedelta(days=1)
        while d not in valid:
            d += timedelta(days=1)
        return d

    def nominal_fri(d):
        return d + timedelta(days=(4 - d.weekday()) % 7)

    eow = prior_td(nominal_fri(today))
    plus1d = next_td(today)
    if plus1d != eow:
        return "0DTE_Regular"

    # It's a Thursday (or Wed-before-holiday Friday) — check if monthly OpEx
    count, opex = 0, None
    for day in range(1, _cal.monthrange(plus1d.year, plus1d.month)[1] + 1):
        if date(plus1d.year, plus1d.month, day).weekday() == 4:
            count += 1
            if count == 3:
                opex = prior_td(date(plus1d.year, plus1d.month, day))
                break
    return "0DTE_Monthly" if plus1d == opex else "0DTE_Weekly"


# ── prices.json upload (every 30s) ────────────────────────────────────────────

def push_prices(s3, feed: DXLinkFeed):
    state = feed.get_state()
    ts_et  = datetime.now(ET)
    ts_utc = datetime.now(timezone.utc)

    prices = {}
    for label, dxlink_sym in PRICE_TICKERS.items():
        d = state.get(dxlink_sym, {})
        bid  = d.get("bid")
        ask  = d.get("ask")
        last = d.get("last")
        mid  = round((bid + ask) / 2, 4) if bid is not None and ask is not None else None
        price = last or mid
        prev  = d.get("prev_close")
        chg_pct = None
        if price and prev and prev != 0:
            chg_pct = round((price - prev) / prev * 100, 2)
        prices[label] = {
            "price":    price,
            "bid":      bid,
            "ask":      ask,
            "prev_close": prev,
            "chg_pct":  chg_pct,
            "volume":   d.get("volume"),
        }

    payload = json.dumps({
        "timestamp":     ts_utc.isoformat(),
        "snapshot_time": ts_et.strftime("%H:%M ET"),
        "prices":        prices,
    }, default=str)

    s3.put_object(
        Bucket=R2_BUCKET, Key="intraday/prices.json",
        Body=payload.encode(),
        ContentType="application/json",
        CacheControl="no-cache, max-age=0",
    )


def prices_loop(s3, feed: DXLinkFeed):
    while not past_stop():
        try:
            push_prices(s3, feed)
        except Exception as e:
            log.error(f"prices.json error: {e}")
        time.sleep(PRICES_SECS)
    log.info("prices loop stopped")


# ── snapshot upload ────────────────────────────────────────────────────────────

def make_s3():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def _fmt_oi(v: int) -> str:
    if v == 0:    return ""
    if v < 1000:  return str(v)
    if v < 10000: return f"{v/1000:.1f}K"
    return f"{v//1000}K"


def take_snapshot(s3, feed: DXLinkFeed, strikes: list[dict],
                  exp_date: str, tier: str, today: date):
    state = feed.get_state()
    ts_et = datetime.now(ET)
    ts_utc = datetime.now(timezone.utc)

    # Underlying price from QQQ quote
    qqq = state.get(TICKER, {})
    bid, ask = qqq.get("bid"), qqq.get("ask")
    underlying = round((bid + ask) / 2, 2) if bid and ask else (qqq.get("last") or None)

    # ATM from underlying price
    atm = round(underlying) if underlying else None

    rows = []
    for s in strikes:
        strike = s["strike"]
        if atm is not None and abs(strike - atm) > STRIKE_WINDOW:
            continue
        for option_type, sym_key, occ_key in (
            ("call", "call_sym", "call_occ"),
            ("put",  "put_sym",  "put_occ"),
        ):
            sym  = s[sym_key]
            data = state.get(sym, {})
            b    = data.get("bid")
            a    = data.get("ask")
            mid  = round((b + a) / 2, 4) if b is not None and a is not None else None
            rows.append({
                "TradeDate":       today.isoformat(),
                "Expiration":      exp_date,
                "Strike":          strike,
                "Type":            option_type,
                "OptionSymbol":    s[occ_key],
                "DTE":             0,
                "OpenInterest":    data.get("oi", 0) or 0,
                "Volume":          data.get("volume", 0) or 0,
                "Bid":             b,
                "Mid":             mid,
                "Ask":             a,
                "Last":            data.get("last"),
                "IV":              data.get("volatility"),
                "Delta":           data.get("delta"),
                "Gamma":           data.get("gamma"),
                "Theta":           data.get("theta"),
                "Vega":            data.get("vega"),
                "UnderlyingPrice": underlying,
            })

    if not rows:
        log.warning("snapshot empty — state not populated yet")
        return

    # Upload CSV
    df = pd.DataFrame(rows)
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)

    date_str = today.strftime("%Y%m%d")
    time_str = ts_et.strftime("%H%M")
    csv_key  = f"intraday/{date_str}/snapshot_{time_str}.csv"

    s3.put_object(
        Bucket=R2_BUCKET, Key=csv_key,
        Body=csv_buf.getvalue().encode(),
        ContentType="text/csv",
    )
    log.info(f"→ {csv_key}  ({len(rows)} rows,  underlying={underlying})")

    # Upload latest.json
    payload = {
        "timestamp":        ts_utc.isoformat(),
        "snapshot_time":    ts_et.strftime("%H:%M ET"),
        "date":             today.isoformat(),
        "expiration":       exp_date,
        "tier":             tier,
        "underlying_price": underlying,
        "snapshot_key":     csv_key,
        "rows":             rows,
    }
    s3.put_object(
        Bucket=R2_BUCKET, Key="intraday/latest.json",
        Body=json.dumps(payload, default=str).encode(),
        ContentType="application/json",
        CacheControl="no-cache, max-age=0",
    )
    log.info("→ intraday/latest.json updated")


# ── main ───────────────────────────────────────────────────────────────────────

def past_stop() -> bool:
    et = datetime.now(ET)
    return (et.hour, et.minute) >= (STOP_HOUR, STOP_MIN)


def wait_for_premarket():
    while True:
        et = datetime.now(ET)
        if et.hour >= PREMARKET_HOUR:
            return
        log.info(f"waiting for {PREMARKET_HOUR:02d}:00 ET pre-market window")
        time.sleep(60)


def main():
    login    = os.environ["TASTY_LOGIN"]
    password = os.environ["TASTY_PASSWORD"]

    wait_for_premarket()

    auth = tasty_auth(login, password)
    today = date.today()
    tier  = classify_tier(today)
    log.info(f"session date={today}  tier={tier}")

    strikes, exp_date = load_chain(auth["session_token"], today)

    option_syms = []
    for s in strikes:
        option_syms.append(s["call_sym"])
        option_syms.append(s["put_sym"])

    price_syms = list(PRICE_TICKERS.values())
    log.info(f"subscribing to {len(option_syms)} option symbols + {len(price_syms)} price tickers")

    feed = DXLinkFeed(auth["streamer_url"], auth["streamer_token"])
    feed.set_subscriptions(option_syms, price_syms)
    feed.start()

    if not feed.wait_ready(timeout=30):
        log.warning("DXLink channel not open after 30s — proceeding anyway")

    log.info("waiting 20s for initial data flush...")
    time.sleep(20)

    s3 = make_s3()

    # Start prices.json refresh thread (every 30s)
    prices_thread = threading.Thread(target=prices_loop, args=(s3, feed), daemon=True)
    prices_thread.start()
    log.info(f"prices thread started (every {PRICES_SECS}s)")

    log.info(f"snapshot loop started (every {SNAPSHOT_SECS // 60}m, stop {STOP_HOUR:02d}:{STOP_MIN:02d} ET)")

    while not past_stop():
        try:
            take_snapshot(s3, feed, strikes, exp_date, tier, today)
        except Exception as e:
            log.error(f"snapshot error: {e}")
        time.sleep(SNAPSHOT_SECS)

    log.info("past stop time — shutting down")
    feed.stop()


if __name__ == "__main__":
    main()
