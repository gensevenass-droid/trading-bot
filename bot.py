import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd
import requests

SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "BNB": "BNB/USDT",
    "EMAS (PAXG)": "PAXG/USDT",
}
PRIMARY = os.environ.get("PRIMARY_EXCHANGE", "okx").strip() or "okx"
EXCHANGES = [PRIMARY] + [
    e for e in ("okx", "kucoin", "bitget", "gate", "mexc") if e != PRIMARY
]
MARKET = os.environ.get("MARKET", "spot").strip() or "spot"
MIN_SCORE = 4
COOLDOWN_CANDLES = 3
SWING_K = 5
MAX_RISK_PCT = 8.0
STATE_FILE = "state.json"
ZONES_FILE = "zones.json"
WIB = timezone(timedelta(hours=7))
H4_MS = 4 * 60 * 60 * 1000


_EX_CACHE = {}


def get_exchange_for(symbol):
    for name in EXCHANGES:
        if name in _EX_CACHE and _EX_CACHE[name] is None:
            continue
        try:
            if name not in _EX_CACHE:
                opts = {"enableRateLimit": True}
                if MARKET == "swap":
                    opts["options"] = {"defaultType": "swap"}
                ex = getattr(ccxt, name)(opts)
                ex.load_markets()
                _EX_CACHE[name] = ex
            ex = _EX_CACHE[name]
            if symbol in ex.markets:
                return ex
        except Exception as e:
            print(f"{name} gagal: {e}")
            _EX_CACHE[name] = None
    return None


def fetch_closed(ex, symbol, tf, limit=300):
    raw = ex.fetch_ohlcv(symbol, tf, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    return df.iloc[:-1].reset_index(drop=True)


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def atr(df, period=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def find_zones(df, k=SWING_K, width=0.75, min_touches=2):
    a = atr(df).iloc[-1]
    if pd.isna(a) or a <= 0:
        return []
    highs = df["high"].values
    lows = df["low"].values
    pts = []
    for i in range(k, len(df) - k):
        if highs[i] == highs[i - k : i + k + 1].max():
            pts.append(float(highs[i]))
        if lows[i] == lows[i - k : i + k + 1].min():
            pts.append(float(lows[i]))
    pts.sort()
    clusters = []
    for p in pts:
        if clusters and p - clusters[-1][0] <= width * a:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    zones = []
    for c in clusters:
        if len(c) >= min_touches:
            zones.append(
                {
                    "low": min(c) - 0.1 * a,
                    "high": max(c) + 0.1 * a,
                    "touches": len(c),
                    "src": "auto",
                    "weak": len(c) < 2,
                }
            )
    last = float(df["close"].iloc[-1])
    return [z for z in zones if abs((z["low"] + z["high"]) / 2 - last) / last <= 0.30]


def merge_zones(zones):
    zones = sorted(zones, key=lambda z: z["low"])
    out = []
    for z in zones:
        if out and z["low"] <= out[-1]["high"]:
            m = out[-1]
            m["high"] = max(m["high"], z["high"])
            m["touches"] = max(m["touches"], z["touches"])
            m["weak"] = m["weak"] and z["weak"]
            if z["src"] == "manual":
                m["src"] = "manual"
        else:
            out.append(dict(z))
    return out


def load_manual_zones(key):
    if not os.path.exists(ZONES_FILE):
        return []
    try:
        with open(ZONES_FILE) as f:
            data = json.load(f)
        return [
            {"low": float(lo), "high": float(hi), "touches": 3, "src": "manual", "weak": False}
            for lo, hi in data.get(key, [])
        ]
    except Exception as e:
        print(f"zones.json bermasalah: {e}")
        return []


def pin_bar(c):
    rng = c["high"] - c["low"]
    if rng <= 0:
        return None
    body = abs(c["close"] - c["open"])
    upper = c["high"] - max(c["open"], c["close"])
    lower = min(c["open"], c["close"]) - c["low"]
    if lower >= 2 * body and lower >= 0.6 * rng:
        return "bull"
    if upper >= 2 * body and upper >= 0.6 * rng:
        return "bear"
    return None


def engulfing(prev, c):
    if (
        prev["close"] < prev["open"]
        and c["close"] > c["open"]
        and c["close"] >= prev["open"]
        and c["open"] <= prev["close"]
    ):
        return "bull"
    if (
        prev["close"] > prev["open"]
        and c["close"] < c["open"]
        and c["close"] <= prev["open"]
        and c["open"] >= prev["close"]
    ):
        return "bear"
    return None


def fmt(x):
    return f"{x:,.2f}"


def zone_text(z):
    tag = "lemah 1x" if z.get("weak") else f"{z['touches']}x"
    return f"{fmt(z['low'])} - {fmt(z['high'])} ({tag}, {z['src']})"


def nearest_zones(zones, last):
    step = 10 ** math.floor(math.log10(last)) / 2
    above = [z for z in zones if z["low"] > last]
    below = [z for z in zones if z["high"] < last]
    inside = [z for z in zones if z["low"] <= last <= z["high"]]
    parts = []
    if above:
        parts.append("  Resisten: " + zone_text(min(above, key=lambda z: z["low"])))
    else:
        lvl = math.floor(last / step) * step + step
        parts.append(f"  Resisten: tidak ada zona di data, level bulat {fmt(lvl)}")
    if inside:
        parts.append("  Di dalam zona: " + zone_text(inside[0]))
    if below:
        parts.append("  Support: " + zone_text(max(below, key=lambda z: z["high"])))
    else:
        lvl = math.floor(last / step) * step
        if lvl >= last:
            lvl -= step
        parts.append(f"  Support: tidak ada zona di data, level bulat {fmt(lvl)}")
    return parts


def evaluate(name, h4, zones):
    close = h4["close"]
    last = float(close.iloc[-1])
    e50 = float(ema(close, 50).iloc[-1])
    e200 = float(ema(close, 200).iloc[-1])
    rsi_v = float(rsi(close).iloc[-1])
    atr_v = float(atr(h4).iloc[-1])
    vol_avg = float(h4["volume"].iloc[-21:-1].mean())
    c = h4.iloc[-1]
    prev = h4.iloc[-2]
    ts = int(c["ts"])

    summary = (
        f"{name}: close H4 {fmt(last)} | RSI {rsi_v:.0f} | "
        f"{'di bawah' if last < e200 else 'di atas'} EMA200"
    )

    pin = pin_bar(c)
    eng = engulfing(prev, c)
    direction = None
    pola = ""
    if pin == "bull" or eng == "bull":
        direction = "long"
        pola = " + ".join(p for p, ok in (("Pin bar", pin == "bull"), ("Engulfing", eng == "bull")) if ok)
    elif pin == "bear" or eng == "bear":
        direction = "short"
        pola = " + ".join(p for p, ok in (("Pin bar", pin == "bear"), ("Engulfing", eng == "bear")) if ok)
    if direction is None or pd.isna(atr_v):
        return summary, None

    cands = []
    for z in zones:
        if direction == "long" and c["low"] <= z["high"] and c["close"] >= z["low"]:
            cands.append(z)
        if direction == "short" and c["high"] >= z["low"] and c["close"] <= z["high"]:
            cands.append(z)
    if not cands:
        return summary, None
    z = max(cands, key=lambda zz: zz["touches"])

    entry = last
    if direction == "long":
        sl = float(c["low"]) - 0.5 * atr_v
        risk = entry - sl
    else:
        sl = float(c["high"]) + 0.5 * atr_v
        risk = sl - entry
    if risk <= 0 or risk / entry * 100 > MAX_RISK_PCT:
        return summary, None
    sign = 1 if direction == "long" else -1
    tp1 = entry + sign * 1.5 * risk
    tp2 = entry + sign * 3.0 * risk

    if direction == "long":
        beyond = [zz["low"] - entry for zz in zones if zz["low"] > entry]
    else:
        beyond = [entry - zz["high"] for zz in zones if zz["high"] < entry]
    room = bool(beyond) and min(beyond) >= 2 * risk

    checks = [
        (
            "Searah tren (EMA 200)",
            (last > e200) if direction == "long" else (last < e200),
        ),
        (
            "Posisi terhadap EMA 50 mendukung",
            (last > e50) if direction == "long" else (last < e50),
        ),
        (
            "RSI mendukung (sisi 50)",
            (rsi_v > 50) if direction == "long" else (rsi_v < 50),
        ),
        ("Zona kuat (3+ sentuhan atau manual)", z["touches"] >= 3),
        ("Ruang gerak ke zona lawan minimal 2R", room),
        ("Volume di atas rata-rata 20 candle", float(c["volume"]) > vol_avg),
    ]
    score = sum(1 for _, ok in checks if ok)
    if score < MIN_SCORE:
        return summary, None

    close_time = datetime.fromtimestamp((ts + H4_MS) / 1000, WIB).strftime("%d %b %H:%M")
    role = "support" if direction == "long" else "resistance"
    lines = [
        f"SETUP {direction.upper()} - {name} (H4)",
        f"Candle close: {close_time} WIB",
        f"Pola: {pola} di zona {role} {fmt(z['low'])} - {fmt(z['high'])}",
        f"Sumber zona: {z['src']} ({z['touches']} sentuhan)",
        f"Skor: {score}/6",
        "",
    ]
    for text, ok in checks:
        lines.append(("OK  " if ok else "NO  ") + text)
    lines += [
        "",
        "Rencana contoh (bukan perintah):",
        f"Entry: {fmt(entry)}",
        f"SL: {fmt(sl)} (jarak {fmt(risk)} = {risk / entry * 100:.2f}%)",
        f"TP1: {fmt(tp1)} (1,5R)",
        f"TP2: {fmt(tp2)} (3R)",
        f"RSI: {rsi_v:.0f} | EMA50: {fmt(e50)} | EMA200: {fmt(e200)}",
        "{LIVE}",
        "",
        "Cek chart sendiri dulu. Alat bantu belajar, bukan sinyal.",
    ]
    return summary, {"direction": direction, "ts": ts, "text": "\n".join(lines)}


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"alerts": {}, "heartbeat": ""}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def send_telegram(text):
    token = os.environ["TELEGRAM_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=30)
    resp.raise_for_status()


def main():
    state = load_state()
    state.setdefault("alerts", {})
    summaries = []
    sent = 0
    for name, symbol in SYMBOLS.items():
        try:
            if MARKET == "swap" and ":" not in symbol:
                symbol = symbol + ":USDT"
            ex = get_exchange_for(symbol)
            if ex is None:
                summaries.append(f"{name}: pair {symbol} tidak ada di exchange yang bisa diakses")
                continue
            live = float(ex.fetch_ticker(symbol)["last"])
            daily = fetch_closed(ex, symbol, "1d")
            h4 = fetch_closed(ex, symbol, "4h")
            weekly_zones = []
            try:
                weekly = fetch_closed(ex, symbol, "1w", limit=300)
                weekly_zones = find_zones(weekly, k=3, width=0.5, min_touches=1)
            except Exception as e:
                print(f"{name} weekly gagal: {e}")
            zones = merge_zones(
                find_zones(daily) + weekly_zones + load_manual_zones(name.split()[0])
            )
            detect_zones = [z for z in zones if not z.get("weak")]
            summary, alert = evaluate(name, h4, detect_zones)
            summaries.append(summary + f" | live {fmt(live)} [{ex.id}]")
            summaries.extend(nearest_zones(zones, live))
            if alert:
                entry_txt = alert["text"]
                alert["text"] = entry_txt.replace(
                    "{LIVE}",
                    f"Harga live: {fmt(live)} | sumber: {ex.id} ({MARKET})",
                )
                key = f"{name}:{alert['direction']}"
                last_ts = state["alerts"].get(key, 0)
                if alert["ts"] - last_ts >= COOLDOWN_CANDLES * H4_MS:
                    send_telegram(alert["text"])
                    state["alerts"][key] = alert["ts"]
                    sent += 1
        except Exception as e:
            summaries.append(f"{name}: gagal ({e})")
            print(f"{name} error: {e}")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state.get("heartbeat") != today or os.environ.get("SEND_SUMMARY") == "1":
        msg = [
            "BOT AKTIF - ringkasan H4",
            f"Sumber utama: {PRIMARY} ({MARKET}). Tanda [ ] = exchange yang dipakai per aset",
            "close H4 = harga penutupan candle terakhir, bukan harga live",
            f"Setup terkirim run ini: {sent}",
            "",
        ] + summaries
        send_telegram("\n".join(msg))
        state["heartbeat"] = today
    save_state(state)
    print(f"Selesai. Alert terkirim: {sent}")


if __name__ == "__main__":
    main()
