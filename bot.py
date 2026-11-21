import html
import json
import math
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone

import ccxt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
from matplotlib.patches import Rectangle

SYMBOLS = {
    "BTC": "BTC/USDT",
    "ETH": "ETH/USDT",
    "SOL": "SOL/USDT",
    "BNB": "BNB/USDT",
    "EMAS (PAXG)": "PAXG/USDT",
}
ALIASES = {
    "btc": "BTC",
    "eth": "ETH",
    "sol": "SOL",
    "bnb": "BNB",
    "emas": "EMAS (PAXG)",
    "paxg": "EMAS (PAXG)",
    "gold": "EMAS (PAXG)",
    "xau": "EMAS (PAXG)",
    "xauusd": "EMAS (PAXG)",
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
DEFAULT_KURS = 16500.0
DEFAULT_RISK = 1.0
MAX_RISK_SETTING = 2.0
SETTINGS_FILE = "settings.json"
MARGIN_TARGET = 0.4
MAX_LEV = 3
FEE_ROUNDTRIP = 0.001
STATE_FILE = "state.json"
JOURNAL_FILE = "journal.json"
EXPIRE_CANDLES = 30
RETEST_MAX_AGE = 12
MONTHS = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
ZONES_FILE = "zones.json"
WIB = timezone(timedelta(hours=7))
H4_MS = 4 * 60 * 60 * 1000

_EX_CACHE = {}
_KURS_CACHE = {}


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


def _idn(s):
    return s.replace(",", "\u00a7").replace(".", ",").replace("\u00a7", ".")


def fmt(x):
    ax = abs(x)
    d = 0 if ax >= 10000 else 1 if ax >= 1000 else 2 if ax >= 1 else 4
    return _idn(f"{x:,.{d}f}")


def num(x, d=2):
    return _idn(f"{x:.{d}f}")


def pct(x, d=1, sign=False):
    return _idn(f"{x:+.{d}f}" if sign else f"{x:.{d}f}") + "%"


def esc(x):
    return html.escape(str(x), quote=False)


def rupiah(x):
    return "Rp" + f"{round(x):,}".replace(",", ".")


def rp_short(x):
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        v = f"{x / 1e6:.2f}".rstrip("0").rstrip(".")
        return f"{sign}Rp{v.replace('.', ',')}jt"
    if x >= 1e5:
        return f"{sign}Rp{x / 1e3:.0f}rb"
    if x >= 1e3:
        v = f"{x / 1e3:.1f}".rstrip("0").rstrip(".")
        return f"{sign}Rp{v.replace('.', ',')}rb"
    return f"{sign}Rp{x:.0f}"


def fmt_qty(q):
    if q < 0.01:
        return _idn(f"{q:.5f}")
    if q < 1:
        return _idn(f"{q:.4f}")
    return _idn(f"{q:.3f}")


def id_date(dt):
    return f"{dt.day} {MONTHS[dt.month - 1]} {dt:%H:%M}"


def bar(n, total):
    filled = max(0, min(total, round(n)))
    return "\u25b0" * filled + "\u25b1" * (total - filled)


def zone_range(z):
    return f"{fmt(z['low'])}\u2013{fmt(z['high'])}"


def zone_short(z):
    tag = "lemah" if z.get("weak") else f"{z['touches']}x"
    return f"{zone_range(z)} ({tag})"


def nearest_zones(zones, last):
    step = 10 ** math.floor(math.log10(last)) / 2
    above = [z for z in zones if z["low"] > last]
    below = [z for z in zones if z["high"] < last]
    inside = [z for z in zones if z["low"] <= last <= z["high"]]
    out = []
    if above:
        out.append("   \u2b06 " + zone_short(min(above, key=lambda z: z["low"])))
    else:
        out.append(f"   \u2b06 {fmt(math.floor(last / step) * step + step)} (level bulat)")
    if inside:
        out.append(f"   \u23fa di dalam zona {zone_range(inside[0])}")
    if below:
        out.append("   \u2b07 " + zone_short(max(below, key=lambda z: z["high"])))
    else:
        lvl = math.floor(last / step) * step
        if lvl >= last:
            lvl -= step
        out.append(f"   \u2b07 {fmt(lvl)} (level bulat)")
    return out


def make_plan(direction, entry, extreme, atr_v):
    if direction == "long":
        sl = extreme - 0.5 * atr_v
        risk = entry - sl
    else:
        sl = extreme + 0.5 * atr_v
        risk = sl - entry
    if risk <= 0 or risk / entry * 100 > MAX_RISK_PCT:
        return None
    sign = 1 if direction == "long" else -1
    return {
        "entry": entry,
        "sl": sl,
        "tp1": entry + sign * 1.5 * risk,
        "tp2": entry + sign * 3.0 * risk,
        "risk": risk,
    }


def room_ok(direction, entry, risk, zones):
    if direction == "long":
        gaps = [z["low"] - entry for z in zones if z["low"] > entry]
    else:
        gaps = [entry - z["high"] for z in zones if z["high"] < entry]
    return bool(gaps) and min(gaps) >= 2 * risk


def build_alert(kind, direction, ts, title, pola, zone, plan, checks):
    score = sum(1 for ok, _, _ in checks if ok)
    if score < MIN_SCORE:
        return None
    return {
        "kind": kind,
        "direction": direction,
        "ts": ts,
        "title": title,
        "pola": pola,
        "zone": zone,
        "checks": checks,
        "score": score,
        "entry": plan["entry"],
        "sl": plan["sl"],
        "tp1": plan["tp1"],
        "tp2": plan["tp2"],
    }


def find_bounce(h4, zones, ind):
    c, prev = h4.iloc[-1], h4.iloc[-2]
    last, atr_v = ind["last"], ind["atr"]
    pin, eng = pin_bar(c), engulfing(prev, c)
    if pin == "bull" or eng == "bull":
        direction = "long"
        flags = (("Pin bar", pin == "bull"), ("Engulfing", eng == "bull"))
    elif pin == "bear" or eng == "bear":
        direction = "short"
        flags = (("Pin bar", pin == "bear"), ("Engulfing", eng == "bear"))
    else:
        return None
    pola = " + ".join(p for p, ok in flags if ok)
    cands = []
    for z in zones:
        if direction == "long" and c["low"] <= z["high"] and c["close"] >= z["low"]:
            cands.append(z)
        if direction == "short" and c["high"] >= z["low"] and c["close"] <= z["high"]:
            cands.append(z)
    if not cands:
        return None
    z = max(cands, key=lambda zz: zz["touches"])
    long = direction == "long"
    plan = make_plan(direction, last, float(c["low"] if long else c["high"]), atr_v)
    if not plan:
        return None
    vol_avg = float(h4["volume"].iloc[-21:-1].mean())
    checks = [
        ((last > ind["e200"]) if long else (last < ind["e200"]), "Searah tren besar", "Melawan tren besar"),
        ((last > ind["e50"]) if long else (last < ind["e50"]), "Posisi terhadap EMA 50 mendukung", "Melawan EMA 50"),
        ((ind["rsi"] > 50) if long else (ind["rsi"] < 50), "RSI mendukung", "RSI belum mendukung"),
        (z["touches"] >= 3, "Zona kuat", "Zona biasa"),
        (room_ok(direction, last, plan["risk"], zones), "Ruang gerak lega", "Ruang gerak sempit"),
        (float(c["volume"]) > vol_avg, "Volume di atas rata-rata", "Volume biasa"),
    ]
    title = "Pantulan dari support" if long else "Penolakan di resisten"
    return build_alert("pantulan", direction, int(c["ts"]), title, pola, z, plan, checks)


def find_retest(h4, zones, ind):
    n = len(h4)
    c, prev = h4.iloc[-1], h4.iloc[-2]
    last, atr_v = ind["last"], ind["atr"]
    pin, eng = pin_bar(c), engulfing(prev, c)
    closes = h4["close"]
    best = None
    for z in zones:
        for direction in ("short", "long"):
            short = direction == "short"
            b = None
            for i in range(max(1, n - 1 - RETEST_MAX_AGE), n - 1):
                ci, cp = closes.iloc[i], closes.iloc[i - 1]
                if short and ci < z["low"] and cp >= z["low"]:
                    b = i
                    break
                if (not short) and ci > z["high"] and cp <= z["high"]:
                    b = i
                    break
            if b is None:
                continue
            after = closes.iloc[b + 1 : n - 1]
            if short and (after > z["high"]).any():
                continue
            if (not short) and (after < z["low"]).any():
                continue
            if short:
                touched = c["high"] >= z["low"] and c["close"] < z["low"]
                rejected = c["close"] < c["open"] or pin == "bear" or eng == "bear"
                pat = "Pin bar" if pin == "bear" else "Engulfing" if eng == "bear" else "Candle penolakan"
            else:
                touched = c["low"] <= z["high"] and c["close"] > z["high"]
                rejected = c["close"] > c["open"] or pin == "bull" or eng == "bull"
                pat = "Pin bar" if pin == "bull" else "Engulfing" if eng == "bull" else "Candle penolakan"
            if not (touched and rejected):
                continue
            extreme = max(float(c["high"]), z["high"]) if short else min(float(c["low"]), z["low"])
            plan = make_plan(direction, last, extreme, atr_v)
            if not plan:
                continue
            bc = h4.iloc[b]
            strong = abs(float(bc["close"]) - float(bc["open"])) >= 0.7 * atr_v
            vol_ref = h4["volume"].iloc[max(0, b - 20) : b].mean()
            vol_ok = bool(vol_ref == vol_ref) and float(bc["volume"]) > float(vol_ref)
            checks = [
                ((last < ind["e200"]) if short else (last > ind["e200"]), "Searah tren besar", "Melawan tren besar"),
                (strong, "Breakout kuat", "Breakout lemah"),
                (vol_ok, "Volume breakout naik", "Volume breakout biasa"),
                (z["touches"] >= 3, "Zona kuat", "Zona biasa"),
                ((ind["rsi"] < 50) if short else (ind["rsi"] > 50), "RSI mendukung", "RSI belum mendukung"),
                (room_ok(direction, last, plan["risk"], zones), "Ruang gerak lega", "Ruang gerak sempit"),
            ]
            hours = (n - 1 - b) * 4
            title = "Retest setelah breakdown" if short else "Retest setelah breakout"
            alert = build_alert(
                "retest", direction, int(c["ts"]), title,
                f"{pat}, breakout {hours} jam lalu", z, plan, checks,
            )
            if alert and (best is None or (alert["score"], z["touches"]) > (best["score"], best["zone"]["touches"])):
                best = alert
    return best


def evaluate(name, h4, zones):
    close = h4["close"]
    ind = {
        "last": float(close.iloc[-1]),
        "e50": float(ema(close, 50).iloc[-1]),
        "e200": float(ema(close, 200).iloc[-1]),
        "rsi": float(rsi(close).iloc[-1]),
        "atr": float(atr(h4).iloc[-1]),
    }
    summary = {"rsi": ind["rsi"], "below200": ind["last"] < ind["e200"]}
    if pd.isna(ind["atr"]) or len(h4) < 60:
        return summary, None
    alert = find_retest(h4, zones, ind) or find_bounce(h4, zones, ind)
    return summary, alert


def get_kurs(st):
    manual = st.get("kurs_manual")
    if manual:
        return float(manual), "manual"
    if "v" in _KURS_CACHE:
        return _KURS_CACHE["v"]
    try:
        r = requests.get("https://indodax.com/api/ticker/usdtidr", timeout=8)
        r.raise_for_status()
        val = float(r.json()["ticker"]["last"])
        if 5000 < val < 50000:
            _KURS_CACHE["v"] = (val, "Indodax")
            return _KURS_CACHE["v"]
    except Exception as e:
        print(f"kurs otomatis gagal: {e}")
    _KURS_CACHE["v"] = (DEFAULT_KURS, "asumsi, atur dengan /kurs")
    return _KURS_CACHE["v"]


def load_settings():
    st = {"modal_idr": [1000000], "risk_pct": DEFAULT_RISK, "kurs_manual": None}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                data = json.load(f)
            st.update({k: v for k, v in data.items() if k in st})
        except Exception as e:
            print(f"settings.json bermasalah: {e}")
    modals = st["modal_idr"]
    if isinstance(modals, (int, float)):
        modals = [modals]
    st["modal_idr"] = [float(m) for m in modals if isinstance(m, (int, float)) and m >= 10000][:5]
    try:
        st["risk_pct"] = min(max(float(st["risk_pct"]), 0.1), MAX_RISK_SETTING)
    except (TypeError, ValueError):
        st["risk_pct"] = DEFAULT_RISK
    return st


def sizing_lines(name, direction, entry, sl, tp1, tp2, st):
    modals = st.get("modal_idr") or []
    if not modals:
        return ["<b>UKURAN POSISI</b>", "Isi daftar modal di settings.json"]
    kurs, kurs_note = get_kurs(st)
    risk_pct = float(st.get("risk_pct", DEFAULT_RISK))
    dist = abs(entry - sl)
    dist_pct = dist / entry * 100
    gold = name.startswith("EMAS")
    base = "PAXG" if gold else name.split()[0]
    out = [f"<b>UKURAN POSISI</b> \u00b7 risiko {num(risk_pct, 1).rstrip('0').rstrip(',')}% \u00b7 kurs {fmt(kurs)}"]
    if kurs_note.startswith("asumsi"):
        out.append("\u2139\ufe0f kurs masih asumsi, isi kurs_manual di settings.json")
    for modal in modals:
        modal_usd = modal / kurs
        risk_usd = modal_usd * risk_pct / 100
        qty = risk_usd / dist
        notional = qty * entry
        lev = max(1, math.ceil(notional / (MARGIN_TARGET * modal_usd)))
        if lev > MAX_LEV:
            out.append(f"<b>{rp_short(modal)}</b> \u00b7 terlalu besar untuk modal ini, lewati")
            continue
        margin = notional / lev
        liq_pct = max(100 / lev - 0.5, 0)
        fee = notional * FEE_ROUNDTRIP
        loss = risk_usd + fee
        gain2 = qty * abs(tp2 - entry) - fee
        out.append(
            f"<b>{rp_short(modal)}</b> \u00b7 {fmt_qty(qty)} {base} \u00b7 {lev}x \u00b7 margin {rp_short(margin * kurs)}"
        )
        out.append(f"   rugi SL -{rp_short(loss * kurs)} \u00b7 untung TP2 +{rp_short(gain2 * kurs)}")
        if liq_pct < 2 * dist_pct:
            out.append("   \u26a0\ufe0f likuidasi dekat SL, kecilkan posisi")
        if gold:
            lot = qty / 100
            if lot < 0.01:
                out.append("   di bawah 0,01 lot XAUUSD, pakai PAXG di exchange")
            else:
                out.append(f"   \u2248 {num(lot, 2)} lot XAUUSD")
        if notional < 10:
            out.append("   \u26a0\ufe0f posisi sangat kecil, cek minimum order exchange")
    return out


QUALITY = {6: "Sangat kuat", 5: "Kuat", 4: "Cukup"}


def alert_message(name, a, live, src, ts_close, st):
    long = a["direction"] == "long"
    z = a["zone"]
    risk_pct = abs(a["entry"] - a["sl"]) / a["entry"] * 100
    head = ("\U0001f7e2 <b>BELI (LONG)" if long else "\U0001f534 <b>JUAL (SHORT)") + f" \u00b7 {esc(name)}</b>"
    lines = [
        head,
        f"<i>{esc(a['title'])} \u00b7 H4</i>",
        "",
        f"Skor {bar(a['score'], 6)} <b>{a['score']}/6</b> \u00b7 {QUALITY.get(a['score'], '')}",
        "",
        "<b>RENCANA</b>",
        "<pre>"
        f"Entry  {fmt(a['entry'])}\n"
        f"SL     {fmt(a['sl'])}  (risiko {pct(risk_pct)})\n"
        f"TP1    {fmt(a['tp1'])}  (1,5R)\n"
        f"TP2    {fmt(a['tp2'])}  (3R)"
        "</pre>",
        f"\U0001f4cd {esc(a['pola'])} \u00b7 zona {zone_range(z)} ({z['touches']}x, {esc(z['src'])})",
        "",
        "<b>ALASAN</b>",
    ]
    for ok, t_ok, t_no in a["checks"]:
        lines.append(("\u2705 " + t_ok) if ok else ("\u274c " + t_no))
    lines.append("")
    lines += sizing_lines(name, a["direction"], a["entry"], a["sl"], a["tp1"], a["tp2"], st)
    drift = (live - a["entry"]) / a["entry"] * 100
    lines.append("")
    if abs(drift) > 0.5:
        lines.append(f"\u26a0\ufe0f Harga live {fmt(live)} ({pct(drift, 1, True)} dari entry). Sesuaikan level, jangan kejar harga.")
    else:
        lines.append(f"Harga live {fmt(live)}")
    lines.append(f"\U0001f550 Close {id_date(ts_close)} WIB \u00b7 {esc(src.upper())}")
    if not long:
        lines.append("\u2139\ufe0f SHORT hanya bisa di futures atau margin.")
    lines += ["", "<i>Alat bantu belajar, bukan saran keuangan. Cek chart sendiri dulu.</i>"]
    return "\n".join(lines)


def make_chart(path, title, h4, zones, plan=None, n=100):
    df = h4.iloc[-n:].reset_index(drop=True)
    m = len(df)
    close = h4["close"]
    e50 = ema(close, 50).iloc[-m:].reset_index(drop=True)
    e200 = ema(close, 200).iloc[-m:].reset_index(drop=True)
    rs = rsi(close).iloc[-m:].reset_index(drop=True)
    bg, fg, grid = "#131722", "#d1d4dc", "#2a2e39"

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.05},
    )
    fig.patch.set_facecolor(bg)
    for a_ in (ax, axr):
        a_.set_facecolor(bg)
        a_.tick_params(colors=fg, labelsize=8)
        for sp in a_.spines.values():
            sp.set_color(grid)
        a_.grid(color=grid, linewidth=0.5, alpha=0.6)

    lo, hi = float(df["low"].min()), float(df["high"].max())
    if plan:
        vals = [plan["entry"], plan["sl"], plan["tp1"]]
        lo, hi = min(lo, *vals), max(hi, *vals)
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad
    tiny = (hi - lo) * 0.001

    for i, row in df.iterrows():
        col = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
        ax.plot([i, i], [row["low"], row["high"]], color=col, linewidth=1)
        body_lo = min(row["open"], row["close"])
        body_h = max(abs(row["close"] - row["open"]), tiny)
        ax.add_patch(Rectangle((i - 0.3, body_lo), 0.6, body_h, color=col))

    for z in zones:
        if z["high"] < lo or z["low"] > hi:
            continue
        if z.get("weak"):
            color, alpha = "#9e9e9e", 0.12
        elif z["src"] == "manual":
            color, alpha = "#f5a623", 0.2
        else:
            color, alpha = "#2962ff", 0.2
        ax.axhspan(max(z["low"], lo), min(z["high"], hi), color=color, alpha=alpha, linewidth=0)
        ax.text(0.5, min(z["high"], hi), f"zona {z['touches']}x", color=fg, fontsize=7, va="top", alpha=0.8)

    ax.plot(e50, color="#f5a623", linewidth=1, label="EMA 50")
    ax.plot(e200, color="#ffeb3b", linewidth=1.4, label="EMA 200")

    if plan:
        levels = [
            (plan["entry"], "#64b5f6", "Entry"),
            (plan["sl"], "#ef5350", "SL"),
            (plan["tp1"], "#66bb6a", "TP1"),
        ]
        if lo <= plan["tp2"] <= hi:
            levels.append((plan["tp2"], "#2e7d32", "TP2"))
        for y, col, lab in levels:
            ax.axhline(y, color=col, linestyle="--", linewidth=1)
            ax.text(m + 9.5, y, f" {lab} {fmt(y)}", color=col, fontsize=8, va="bottom", ha="right")
        last = df.iloc[-1]
        if plan["direction"] == "long":
            ax.scatter([m - 1], [last["low"] - pad * 0.6], marker="^", color="#64b5f6", s=60, zorder=5)
        else:
            ax.scatter([m - 1], [last["high"] + pad * 0.6], marker="v", color="#64b5f6", s=60, zorder=5)

    ax.set_ylim(lo, hi)
    ax.set_xlim(-1, m + 10)
    ax.set_title(title, color=fg, fontsize=11, loc="left")
    leg = ax.legend(loc="upper left", fontsize=7, facecolor=bg, edgecolor=grid)
    for t in leg.get_texts():
        t.set_color(fg)

    axr.plot(rs, color="#b388ff", linewidth=1.2)
    for lvl, ls in ((70, "--"), (50, ":"), (30, "--")):
        axr.axhline(lvl, color=fg, linestyle=ls, linewidth=0.6, alpha=0.5)
    axr.set_ylim(0, 100)
    axr.set_ylabel("RSI 14", color=fg, fontsize=8)

    ticks = list(range(0, m, 12))
    axr.set_xticks(ticks)
    axr.set_xticklabels(
        [datetime.fromtimestamp(df["ts"].iloc[i] / 1000, WIB).strftime("%d/%m %H:%M") for i in ticks],
        rotation=0, fontsize=7,
    )
    fig.savefig(path, dpi=110, facecolor=bg, bbox_inches="tight")
    plt.close(fig)


def load_journal():
    if os.path.exists(JOURNAL_FILE):
        try:
            with open(JOURNAL_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_journal(journal):
    with open(JOURNAL_FILE, "w") as f:
        json.dump(journal, f, indent=1)


def resolve_trade(t, df):
    """Model hasil: entry di close candle sinyal, 50% ditutup di TP1 dan 50% di TP2,
    SL tidak digeser. Jika SL dan TP kena di candle yang sama, SL dihitung lebih dulu."""
    long = t["direction"] == "long"
    risk = abs(t["entry"] - t["sl"])
    tp1_hit = False
    n = 0
    last_close = t["entry"]
    for _, c in df.iterrows():
        n += 1
        hi, lo, ts = float(c["high"]), float(c["low"]), int(c["ts"])
        last_close = float(c["close"])
        sl_hit = lo <= t["sl"] if long else hi >= t["sl"]
        tp1_c = hi >= t["tp1"] if long else lo <= t["tp1"]
        tp2_c = hi >= t["tp2"] if long else lo <= t["tp2"]
        if not tp1_hit:
            if sl_hit:
                return "sl", -1.0, ts, False
            if tp1_c:
                tp1_hit = True
                if tp2_c:
                    return "tp2", 2.25, ts, True
        else:
            if sl_hit:
                return "tp1_sl", 0.25, ts, True
            if tp2_c:
                return "tp2", 2.25, ts, True
        if n >= EXPIRE_CANDLES:
            mtm = ((last_close - t["entry"]) if long else (t["entry"] - last_close)) / risk
            r = (0.75 + 0.5 * mtm) if tp1_hit else mtm
            return "expired", round(r, 2), ts, tp1_hit
    return "open", None, None, tp1_hit


def rfmt(r):
    return _idn(f"{r:+.2f}") + "R"


RESULT_HEAD = {
    "sl": ("\u274c", "Kena SL"),
    "tp1_sl": ("\U0001f7e1", "TP1 tercapai, lalu kena SL"),
    "tp2": ("\u2705", "TP1 dan TP2 tercapai"),
    "expired": ("\u23f3", "Kedaluwarsa (5 hari tanpa hasil)"),
}


def update_journal(journal, assets):
    msgs = []
    for t in journal:
        if t["status"] != "open":
            continue
        data = assets.get(t["name"])
        if not data:
            continue
        h4 = data["h4"]
        status, r, rts, tp1 = resolve_trade(t, h4[h4["ts"] > t["ts"]])
        arah = "LONG" if t["direction"] == "long" else "SHORT"
        if status != "open":
            t.update(status=status, r=r, resolved=rts, tp1_hit=tp1)
            icon, text = RESULT_HEAD[status]
            msgs.append(
                f"{icon} <b>HASIL \u00b7 {esc(t['name'])} {arah}</b>\n"
                f"{text} \u00b7 <b>{rfmt(r)}</b>\n"
                f"Entry {fmt(t['entry'])} \u2192 SL {fmt(t['sl'])} / TP2 {fmt(t['tp2'])}"
            )
        elif tp1 and not t.get("tp1_notified"):
            t["tp1_hit"] = True
            t["tp1_notified"] = True
            msgs.append(
                f"\U0001f3af <b>UPDATE \u00b7 {esc(t['name'])} {arah}</b>\n"
                f"TP1 tercapai ({fmt(t['tp1'])}). Sisa posisi menuju TP2 {fmt(t['tp2'])}."
            )
    return msgs


def trade_stats(trades):
    closed = [t for t in trades if t["status"] != "open" and t.get("r") is not None]
    rs = [t["r"] for t in closed]
    n = len(rs)
    wins = [r for r in rs if r > 0]
    gp = sum(wins)
    gl = abs(sum(r for r in rs if r < 0))
    pf = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)
    cum = peak = dd = 0.0
    for t in sorted(closed, key=lambda x: x.get("resolved") or 0):
        cum += t["r"]
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return {
        "n": n,
        "wins": len(wins),
        "wr": (len(wins) / n * 100) if n else 0.0,
        "total": sum(rs),
        "avg": (sum(rs) / n) if n else 0.0,
        "pf": pf,
        "dd": dd,
    }


def pf_text(pf, n=1):
    if n == 0:
        return "-"
    return "belum ada rugi" if pf == float("inf") else num(pf, 2)


def build_report(journal, state, st, now):
    start = datetime.strptime(state.get("start_date", now.strftime("%Y-%m-%d")), "%Y-%m-%d")
    week_no = (now.replace(tzinfo=None) - start).days // 7 + 1
    cutoff = (now - timedelta(days=7)).timestamp() * 1000
    week_alerts = [t for t in journal if datetime.fromisoformat(t["created"]).timestamp() * 1000 >= cutoff]
    week_done = [t for t in journal if t["status"] != "open" and (t.get("resolved") or 0) >= cutoff]
    open_n = sum(1 for t in journal if t["status"] == "open")
    sw, sa = trade_stats(week_done), trade_stats(journal)
    closed = [t for t in journal if t["status"] != "open" and t.get("r") is not None]

    lines = [
        f"\U0001f4ca <b>LAPORAN MINGGUAN \u00b7 Minggu {week_no}</b>",
        f"<i>{id_date(now)} WIB</i>",
        "",
        "<b>7 HARI TERAKHIR</b>",
        f"Alert {len(week_alerts)} \u00b7 selesai {sw['n']} \u00b7 berjalan {open_n}",
    ]
    if sw["n"]:
        lines.append(f"Menang {sw['wins']} \u00b7 Kalah {sw['n'] - sw['wins']} \u00b7 <b>{rfmt(sw['total'])}</b>")
    else:
        lines.append("Belum ada trade yang selesai.")

    lines += ["", f"<b>SEJAK AWAL</b> ({sa['n']} trade selesai)"]
    if sa["n"]:
        lines.append(f"Win rate {sa['wr']:.0f}% \u00b7 Total <b>{rfmt(sa['total'])}</b> \u00b7 rata-rata {rfmt(sa['avg'])}")
        lines.append(f"Profit factor {pf_text(sa['pf'], sa['n'])} \u00b7 Drawdown maks {num(sa['dd'], 1)}R")
        modals = st.get("modal_idr") or []
        if modals:
            rp = float(st.get("risk_pct", DEFAULT_RISK))
            eq = sa["total"] * modals[0] * rp / 100
            lines.append(
                f"\u2248 {'+' if eq >= 0 else '-'}{rp_short(abs(eq))} "
                f"(modal {rp_short(modals[0])}, risiko {num(rp, 1).rstrip('0').rstrip(',')}%)"
            )

        def row(label, sub):
            ss = trade_stats(sub)
            return f"{label}  {ss['n']} trade \u00b7 menang {ss['wr']:.0f}% \u00b7 {rfmt(ss['total'])}"

        lines += ["", "<b>PER ASET</b>"]
        for nm in sorted({t["name"] for t in closed}):
            lines.append(row(esc(nm), [t for t in closed if t["name"] == nm]))
        lines += ["", "<b>PER SETUP</b>"]
        for kind, label in (("pantulan", "Pantulan"), ("retest", "Retest")):
            sub = [t for t in closed if t.get("kind", "pantulan") == kind]
            if sub:
                lines.append(row(label, sub))
        lines += ["", "<b>PER SKOR</b>"]
        for label, cond in (("Skor 5-6", lambda t: t["score"] >= 5), ("Skor 4", lambda t: t["score"] == 4)):
            sub = [t for t in closed if cond(t)]
            if sub:
                lines.append(row(label, sub))
    else:
        lines.append("Belum ada trade yang selesai.")

    ok_n = "\u2705" if sa["n"] >= 30 else "\u274c"
    ok_pf = "\u2705" if sa["pf"] > 1.3 else "\u274c"
    lines += [
        "",
        "\U0001f393 <b>TARGET LULUS</b>",
        f"Trade selesai {bar(min(sa['n'], 30) / 30 * 10, 10)} {sa['n']}/30 {ok_n}",
        f"Profit factor di atas 1,3: {pf_text(sa['pf'], sa['n'])} {ok_pf}",
    ]
    if week_no in (4, 8, 12):
        lines.append(f"\U0001f4cc Minggu {week_no} = titik evaluasi. Ubah aturan hanya sekarang, dan hanya jika sampel cukup.")
    if sa["n"] < 30:
        lines.append("\u26a0\ufe0f Sampel masih kecil, jangan simpulkan apa pun dari angka ini.")
    lines += [
        "",
        "<i>Model hasil: entry di close candle sinyal, 50% keluar di TP1 dan 50% di TP2, "
        "SL tidak digeser, SL dihitung dulu jika SL dan TP kena satu candle. "
        "Hasil nyata bisa berbeda (fee, slippage). Alat bantu belajar.</i>",
    ]
    return "\n".join(lines)


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


def tg_url(method):
    return f"https://api.telegram.org/bot{os.environ['TELEGRAM_TOKEN']}/{method}"


def send_telegram(text):
    chat = os.environ["TELEGRAM_CHAT_ID"]
    resp = requests.post(
        tg_url("sendMessage"),
        data={"chat_id": chat, "text": text[:4000], "parse_mode": "HTML", "disable_web_page_preview": True},
        timeout=30,
    )
    if resp.status_code == 400:
        plain = html.unescape(re.sub(r"<[^>]+>", "", text))[:4000]
        resp = requests.post(tg_url("sendMessage"), data={"chat_id": chat, "text": plain}, timeout=30)
    resp.raise_for_status()


def send_photo(path, caption=""):
    with open(path, "rb") as f:
        resp = requests.post(
            tg_url("sendPhoto"),
            data={"chat_id": os.environ["TELEGRAM_CHAT_ID"], "caption": caption[:1000]},
            files={"photo": f},
            timeout=60,
        )
    resp.raise_for_status()


def safe_reply(text):
    try:
        send_telegram(text)
    except Exception as e:
        print(f"balasan gagal: {e}")


def main():
    state = load_state()
    state.setdefault("alerts", {})
    st = load_settings()
    journal = load_journal()
    now_wib = datetime.now(WIB)
    state.setdefault("start_date", now_wib.strftime("%Y-%m-%d"))
    req = ALIASES.get(os.environ.get("CHART_REQUEST", "").strip().lower())
    chart_reqs = [req] if req else []

    blocks = []
    assets = {}
    sent = 0
    used = set()
    for name, symbol in SYMBOLS.items():
        try:
            if MARKET == "swap" and ":" not in symbol:
                symbol = symbol + ":USDT"
            ex = get_exchange_for(symbol)
            if ex is None:
                blocks += [f"\u26a0\ufe0f <b>{esc(name)}</b> \u00b7 data tidak tersedia", ""]
                continue
            used.add(ex.id.upper())
            live = float(ex.fetch_ticker(symbol)["last"])
            daily = fetch_closed(ex, symbol, "1d")
            h4 = fetch_closed(ex, symbol, "4h")
            weekly_zones = []
            try:
                weekly = fetch_closed(ex, symbol, "1w", limit=300)
                weekly_zones = find_zones(weekly, k=3, width=0.5, min_touches=1)
            except Exception as e:
                print(f"{name} weekly gagal: {e}")
            zones = merge_zones(find_zones(daily) + weekly_zones + load_manual_zones(name.split()[0]))
            detect_zones = [z for z in zones if not z.get("weak")]
            assets[name] = {"h4": h4, "zones": zones, "live": live, "ex": ex.id}
            summary, alert = evaluate(name, h4, detect_zones)
            icon = "\U0001f4c9" if summary["below200"] else "\U0001f4c8"
            blocks.append(f"{icon} <b>{esc(name)}</b>  {fmt(live)} \u00b7 RSI {summary['rsi']:.0f}")
            blocks += nearest_zones(zones, live)
            blocks.append("")
            if alert:
                key = f"{name}:{alert['direction']}:{alert['kind']}"
                last_ts = state["alerts"].get(key, 0)
                if alert["ts"] - last_ts >= COOLDOWN_CANDLES * H4_MS:
                    close_dt = datetime.fromtimestamp((alert["ts"] + H4_MS) / 1000, WIB)
                    text = alert_message(name, alert, live, ex.id, close_dt, st)
                    try:
                        path = os.path.join(tempfile.gettempdir(), f"chart_{name.split()[0]}.png")
                        arah = "BELI" if alert["direction"] == "long" else "JUAL"
                        make_chart(
                            path,
                            f"{name} \u00b7 {arah} \u00b7 {alert['title']} \u00b7 skor {alert['score']}/6 \u00b7 {ex.id}",
                            h4, zones, plan=alert,
                        )
                        send_photo(path, f"{name} H4 \u00b7 {arah}")
                    except Exception as e:
                        print(f"chart gagal: {e}")
                    send_telegram(text)
                    state["alerts"][key] = alert["ts"]
                    sent += 1
                    journal.append(
                        {
                            "id": f"{key}:{alert['ts']}",
                            "name": name,
                            "direction": alert["direction"],
                            "kind": alert["kind"],
                            "ts": alert["ts"],
                            "entry": alert["entry"],
                            "sl": alert["sl"],
                            "tp1": alert["tp1"],
                            "tp2": alert["tp2"],
                            "score": alert["score"],
                            "exchange": ex.id,
                            "created": datetime.now(timezone.utc).isoformat(),
                            "status": "open",
                            "r": None,
                            "resolved": None,
                            "tp1_hit": False,
                            "tp1_notified": False,
                        }
                    )
        except Exception as e:
            blocks += [f"\u26a0\ufe0f <b>{esc(name)}</b> \u00b7 gagal ambil data", ""]
            print(f"{name} error: {e}")

    for m in update_journal(journal, assets):
        safe_reply(m)

    for req in chart_reqs:
        data = assets.get(req)
        if not data:
            safe_reply(f"Data {req} belum tersedia saat ini.")
            continue
        try:
            path = os.path.join(tempfile.gettempdir(), f"chart_{req.split()[0]}.png")
            make_chart(path, f"{req} H4 \u00b7 {data['ex']}", data["h4"], data["zones"])
            send_photo(path, f"{req} H4 \u00b7 live {fmt(data['live'])}")
        except Exception as e:
            safe_reply(f"Gagal membuat chart {req}: {e}")

    week_key = now_wib.strftime("%G-W%V")
    force_report = os.environ.get("REPORT_NOW") == "1"
    if force_report or (
        now_wib.weekday() == 6 and now_wib.hour >= 20 and state.get("last_report") != week_key
    ):
        try:
            send_telegram(build_report(journal, state, st, now_wib))
            if not force_report:
                state["last_report"] = week_key
        except Exception as e:
            print(f"laporan gagal: {e}")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state.get("heartbeat") != today or os.environ.get("SEND_SUMMARY") == "1":
        head = [
            f"\U0001f7e2 <b>BOT AKTIF</b> \u00b7 {id_date(now_wib)} WIB",
            f"Data: {esc(', '.join(sorted(used)) or '-')} \u00b7 setup baru: {sent}",
            "",
        ]
        foot = ["\U0001f4c9 / \U0001f4c8 = harga di bawah / atas EMA 200 (H4)", "\u2b06 resisten \u00b7 \u2b07 support terdekat"]
        send_telegram("\n".join(head + blocks + foot))
        state["heartbeat"] = today
    save_state(state)
    save_journal(journal)
    print(f"Selesai. Alert terkirim: {sent}")


if __name__ == "__main__":
    main()
