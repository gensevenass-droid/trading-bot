import os
import sys

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
TIMEFRAME = "4h"
EXCHANGES = ["kucoin", "okx", "bybit"]


def get_exchange():
    for name in EXCHANGES:
        try:
            ex = getattr(ccxt, name)({"enableRateLimit": True})
            ex.load_markets()
            print(f"Exchange dipakai: {name}")
            return ex
        except Exception as e:
            print(f"{name} gagal: {e}")
    sys.exit("Semua exchange gagal.")


def fetch_closed(ex, symbol):
    raw = ex.fetch_ohlcv(symbol, TIMEFRAME, limit=300)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.iloc[:-1]
    return df.reset_index(drop=True)


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def analyze(df):
    close = df["close"]
    last = close.iloc[-1]
    e50 = ema(close, 50).iloc[-1]
    e200 = ema(close, 200).iloc[-1]
    r = rsi(close).iloc[-1]
    if last < e200:
        bias = "BEARISH (di bawah EMA 200)"
    else:
        bias = "BULLISH (di atas EMA 200)"
    pos50 = "di atas EMA 50" if last > e50 else "di bawah EMA 50"
    return last, e50, e200, r, bias, pos50


def fmt(x):
    return f"{x:,.2f}"


def send_telegram(text):
    token = os.environ["TELEGRAM_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=30)
    resp.raise_for_status()


def main():
    ex = get_exchange()
    lines = ["RINGKASAN H4 (candle close, tes tahap 1)", ""]
    for name, symbol in SYMBOLS.items():
        try:
            df = fetch_closed(ex, symbol)
            last, e50, e200, r, bias, pos50 = analyze(df)
            lines.append(f"{name}: {fmt(last)}")
            lines.append(f"  Bias: {bias}")
            lines.append(f"  EMA50 {fmt(e50)} | EMA200 {fmt(e200)} ({pos50})")
            lines.append(f"  RSI14: {r:.1f}")
            lines.append("")
        except Exception as e:
            lines.append(f"{name}: gagal ambil data ({e})")
            lines.append("")
    lines.append("Ini alat bantu belajar, bukan sinyal beli/jual.")
    send_telegram("\n".join(lines))
    print("Terkirim.")


if __name__ == "__main__":
    main()
