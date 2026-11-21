import json
import math
import os
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


def fmt(x):
    return f"{x:,.2f}"


def rupiah(x):
    return "Rp" + f"{round(x):,}".replace(",", ".")


def fmt_qty(q):
    if q < 0.01:
        return f"{q:.5f}"
    if q < 1:
        return f"{q:.4f}"
    return f"{q:.3f}"


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
        ("Searah tren (EMA 200)", (last > e200) if direction == "long" else (last < e200)),
        ("Posisi terhadap EMA 50 mendukung", (last > e50) if direction == "long" else (last < e50)),
        ("RSI mendukung (sisi 50)", (rsi_v > 50) if direction == "long" else (rsi_v < 50)),
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
        f"Entry: {fmt(entry)} (pakai order Limit di sekitar level ini)",
        f"SL: {fmt(sl)} (jarak {fmt(risk)} = {risk / entry * 100:.2f}%)",
        f"TP1: {fmt(tp1)} (1,5R)",
        f"TP2: {fmt(tp2)} (3R)",
        f"RSI: {rsi_v:.0f} | EMA50: {fmt(e50)} | EMA200: {fmt(e200)}",
        "{LIVE}",
        "",
        "{PLAN}",
        "",
        "Cek chart sendiri dulu. Alat bantu belajar, bukan sinyal.",
    ]
    return summary, {
        "direction": direction,
        "ts": ts,
        "text": "\n".join(lines),
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "score": score,
    }


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


def position_guide(name, direction, entry, sl, tp1, tp2, st):
    modals = st.get("modal_idr") or []
    if not modals:
        return "PANDUAN ENTRY: isi daftar modal di settings.json."
    kurs, kurs_note = get_kurs(st)
    risk_pct = float(st.get("risk_pct", DEFAULT_RISK))
    dist = abs(entry - sl)
    dist_pct = dist / entry * 100
    gold = name.startswith("EMAS")
    base = "PAXG" if gold else name.split()[0]
    lines = [
        f"PANDUAN ENTRY (risiko {risk_pct:g}% per transaksi)",
        f"Kurs USDT: {kurs:,.0f} ({kurs_note}) | Jarak SL: {dist_pct:.2f}%",
    ]
    for modal in modals:
        modal_usd = modal / kurs
        risk_usd = modal_usd * risk_pct / 100
        qty = risk_usd / dist
        notional = qty * entry
        lev = max(1, math.ceil(notional / (MARGIN_TARGET * modal_usd)))
        lines += ["", f"Modal {rupiah(modal)} (risiko {rupiah(modal * risk_pct / 100)})"]
        if lev > MAX_LEV:
            lines.append(f"  Terlalu besar untuk modal ini (butuh leverage {lev}x). Lewati.")
            continue
        margin = notional / lev
        liq_pct = max(100 / lev - 0.5, 0)
        safe = liq_pct >= 2 * dist_pct
        fee = notional * FEE_ROUNDTRIP
        loss = risk_usd + fee
        gain1 = qty * abs(tp1 - entry) - fee
        gain2 = qty * abs(tp2 - entry) - fee
        lines.append(f"  Qty {fmt_qty(qty)} {base} | posisi {rupiah(notional * kurs)}")
        lines.append(
            f"  Leverage {lev}x | margin {rupiah(margin * kurs)} | likuidasi sekitar {liq_pct:.0f}%"
            + ("" if safe else " (DEKAT SL!)")
        )
        lines.append(
            f"  SL -{rupiah(loss * kurs)} | TP1 +{rupiah(gain1 * kurs)} | TP2 +{rupiah(gain2 * kurs)}"
        )
        if gold:
            lot = qty / 100
            lines.append(
                f"  Setara {lot:.4f} lot XAUUSD"
                + (" (di bawah 0,01 lot, pakai PAXG)" if lot < 0.01 else "")
            )
        if notional < 10:
            lines.append("  Catatan: posisi di bawah sekitar 10 USDT, mungkin di bawah minimum order.")
    lines += [
        "",
        "Sudah termasuk estimasi fee. Likuidasi hanya perkiraan kasar.",
        "Checklist: order Limit, SL dan TP terpasang, margin Isolated, cek berita besar.",
    ]
    if direction == "short":
        lines.append("SHORT hanya bisa di futures atau margin, bukan spot.")
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
    resp = requests.post(
        tg_url("sendMessage"),
        data={"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": text[:4000]},
        timeout=30,
    )
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
    req = ALIASES.get(os.environ.get("CHART_REQUEST", "").strip().lower())
    chart_reqs = [req] if req else []

    summaries = []
    assets = {}
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
            zones = merge_zones(find_zones(daily) + weekly_zones + load_manual_zones(name.split()[0]))
            detect_zones = [z for z in zones if not z.get("weak")]
            assets[name] = {"h4": h4, "zones": zones, "live": live, "ex": ex.id}
            summary, alert = evaluate(name, h4, detect_zones)
            summaries.append(summary + f" | live {fmt(live)} [{ex.id}]")
            summaries.extend(nearest_zones(zones, live))
            if alert:
                key = f"{name}:{alert['direction']}"
                last_ts = state["alerts"].get(key, 0)
                if alert["ts"] - last_ts >= COOLDOWN_CANDLES * H4_MS:
                    plan_text = position_guide(
                        name, alert["direction"], alert["entry"], alert["sl"],
                        alert["tp1"], alert["tp2"], st,
                    )
                    live_line = f"Harga live: {fmt(live)} | sumber: {ex.id} ({MARKET})"
                    drift = (live - alert["entry"]) / alert["entry"] * 100
                    if abs(drift) > 0.5:
                        live_line += (
                            f"\nPERHATIAN: harga live sudah {drift:+.2f}% dari entry contoh. "
                            "Sesuaikan level, jangan kejar harga."
                        )
                    text = alert["text"].replace("{LIVE}", live_line).replace("{PLAN}", plan_text)
                    try:
                        path = os.path.join(tempfile.gettempdir(), f"chart_{name.split()[0]}.png")
                        make_chart(
                            path, f"{name} H4 | {alert['direction'].upper()} | skor {alert['score']}/6 | {ex.id}",
                            h4, zones, plan=alert,
                        )
                        send_photo(path, f"{name} H4 - {alert['direction'].upper()}")
                    except Exception as e:
                        print(f"chart gagal: {e}")
                    send_telegram(text)
                    state["alerts"][key] = alert["ts"]
                    sent += 1
        except Exception as e:
            summaries.append(f"{name}: gagal ({e})")
            print(f"{name} error: {e}")

    for req in chart_reqs:
        data = assets.get(req)
        if not data:
            safe_reply(f"Data {req} belum tersedia saat ini.")
            continue
        try:
            path = os.path.join(tempfile.gettempdir(), f"chart_{req.split()[0]}.png")
            make_chart(path, f"{req} H4 | {data['ex']}", data["h4"], data["zones"])
            send_photo(path, f"{req} H4 | live {fmt(data['live'])} | zona dari bot")
        except Exception as e:
            safe_reply(f"Gagal membuat chart {req}: {e}")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state.get("heartbeat") != today or os.environ.get("SEND_SUMMARY") == "1":
        msg = [
            "BOT AKTIF - ringkasan H4",
            f"Sumber utama: {PRIMARY} ({MARKET}). Tanda [ ] = exchange yang dipakai per aset",
            "close H4 = harga penutupan candle terakhir, bukan harga live",
            f"Setup terkirim run ini: {sent}",
            "Modal dan risiko diatur di settings.json",
            "",
        ] + summaries
        send_telegram("\n".join(msg))
        state["heartbeat"] = today
    save_state(state)
    print(f"Selesai. Alert terkirim: {sent}")


if __name__ == "__main__":
    main()
