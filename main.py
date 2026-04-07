"""
Range Scalper — Level Scanner
Avvisa via Telegram quando il prezzo si avvicina a livelli S/R chiave.
La decisione finale è dell'utente (come fa Eliz).
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go

# ============================================================
# CONFIG
# ============================================================
COINGECKO_BASE  = "https://api.coingecko.com/api/v3"
STABLECOINS     = {"USDT","USDC","BUSD","DAI","TUSD","FDUSD","USDS","USDP","USDE","PYUSD","USD1","USDG"}
PROXIMITY_PCT   = 0.015   # segnala se prezzo è entro 1.5% da un livello
SR_BOX_PCT      = 0.005   # ±0.5% ampiezza box S/R
SR_MIN_TOUCHES  = 2       # tocchi minimi per validare livello

YAHOO_SYMBOL_MAP = {
    "WBTUSDT":   "BTC-USD",
    "WBTCUSDT":  "BTC-USD",
    "SUIUSDT":   "SUI20947-USD",
    "TAOUSDT":   "TAO22974-USD",
    "USD1USDT":  None,
    "USDGUSDT":  None,
    "XAUTUSDT":  "XAU-USD",
    "PAXGUSDT":  "PAXG-USD",
    "PYUSDUSDT": None,
    "ASTERUSDT": None,
}

# ============================================================
# TELEGRAM
# ============================================================
def send_telegram(message):
    try:
        token   = st.secrets["TELEGRAM_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10
        )
        return r.status_code == 200
    except:
        return False

# ============================================================
# DATA
# ============================================================
@st.cache_data(ttl=900)
def get_top30_symbols():
    try:
        r = requests.get(
            f"{COINGECKO_BASE}/coins/markets",
            params={"vs_currency":"usd","order":"market_cap_desc","per_page":80,"page":1},
            timeout=10
        )
        symbols = []
        for coin in r.json():
            sym = coin["symbol"].upper()
            if sym not in STABLECOINS and coin.get("total_volume",0) > 50_000_000:
                symbols.append({
                    "symbol":     sym+"USDT",
                    "name":       coin["name"],
                    "volume_24h": coin.get("total_volume",0),
                    "price":      coin.get("current_price",0),
                    "change_24h": coin.get("price_change_percentage_24h",0),
                })
            if len(symbols) >= 30:
                break
        return symbols
    except:
        base = ["BTC","ETH","BNB","SOL","XRP","ADA","AVAX","DOGE","DOT","LINK",
                "TRX","LTC","SHIB","UNI","ATOM","XLM","NEAR","APT","INJ","SUI",
                "ARB","OP","FIL","HBAR","VET","ALGO","ICP","AAVE","ZEC","TON"]
        return [{"symbol":s+"USDT","name":s,"volume_24h":100_000_000,"price":0,"change_24h":0} for s in base]


def get_ohlcv(symbol, interval="1h", limit=100):
    if YAHOO_SYMBOL_MAP.get(symbol) is None and symbol in YAHOO_SYMBOL_MAP:
        return None
    ticker = YAHOO_SYMBOL_MAP.get(symbol, symbol.replace("USDT","")+"-USD")
    yf_map = {"1h":("1h","59d"), "1d":("1d","730d")}
    yf_interval, period = yf_map.get(interval, ("1h","59d"))
    try:
        df = yf.download(ticker, period=period, interval=yf_interval, progress=False, auto_adjust=True)
        if df is None or len(df) < 10:
            return None
        df = df.reset_index()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        time_col = "Datetime" if "Datetime" in df.columns else "Date"
        df = df.rename(columns={time_col:"open_time","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
        df["open_time"] = pd.to_datetime(df["open_time"])
        for col in ["open","high","low","close","volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[df["close"]>0].dropna().reset_index(drop=True)
        if len(df) > 1:
            df = df.iloc[:-1]  # escludi candela in formazione
        return df.tail(limit).reset_index(drop=True) if len(df)>=10 else None
    except:
        return None


def detect_sr_levels(candles_d1):
    """Rileva livelli S/R chiave da daily"""
    if candles_d1 is None or len(candles_d1) < 20:
        return [], []
    highs = candles_d1["high"].values
    lows  = candles_d1["low"].values
    support_lvls    = []
    resistance_lvls = []

    for i in range(2, len(highs)-2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            level   = highs[i]
            touches = sum(1 for h in highs if abs(h-level)/level < SR_BOX_PCT)
            if touches >= SR_MIN_TOUCHES:
                resistance_lvls.append(level)

    for i in range(2, len(lows)-2):
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            level   = lows[i]
            touches = sum(1 for l in lows if abs(l-level)/level < SR_BOX_PCT)
            if touches >= SR_MIN_TOUCHES:
                support_lvls.append(level)

    def cluster(levels):
        if not levels:
            return []
        levels = sorted(set(levels))
        out = [levels[0]]
        for l in levels[1:]:
            if abs(l-out[-1])/out[-1] > 0.012:
                out.append(l)
        return out

    return cluster(support_lvls), cluster(resistance_lvls)


def get_monday_range(candles_h1):
    """Monday Range high/low"""
    if candles_h1 is None:
        return None, None
    try:
        candles_h1["weekday"] = candles_h1["open_time"].dt.weekday
        mon = candles_h1[candles_h1["weekday"] == 0]
        if mon.empty:
            return None, None
        last_mon = mon.iloc[-1]
        return float(last_mon["high"]), float(last_mon["low"])
    except:
        return None, None


# ============================================================
# SCANNER CORE
# ============================================================
def scan_asset(asset_info):
    """
    Scan semplificato: avvisa quando il prezzo è vicino a un livello S/R.
    Nessuna condizione entry complessa — l'utente decide.
    """
    symbol = asset_info["symbol"]
    price  = asset_info.get("price", 0)
    if price == 0:
        return None, f"SKIP {symbol}: prezzo non disponibile"

    candles_d1 = get_ohlcv(symbol, "1d", 200)
    candles_h1 = get_ohlcv(symbol, "1h", 60)

    if candles_d1 is None:
        return None, f"SKIP {symbol}: dati daily non disponibili"

    support_lvls, resistance_lvls = detect_sr_levels(candles_d1)
    mr_high, mr_low = get_monday_range(candles_h1)

    alerts = []

    # Controlla prossimità a livelli Support
    for sup in support_lvls:
        dist = (price - sup) / sup
        if 0 <= dist <= PROXIMITY_PCT:
            alerts.append({
                "type":    "🟢 NEAR SUPPORT",
                "level":   round(sup, 6),
                "price":   round(price, 6),
                "dist_pct": round(dist*100, 2),
                "note":    "Prezzo sopra supporto — potenziale rimbalzo"
            })
        elif -PROXIMITY_PCT/2 <= dist < 0:
            alerts.append({
                "type":    "⚠️ BELOW SUPPORT",
                "level":   round(sup, 6),
                "price":   round(price, 6),
                "dist_pct": round(abs(dist)*100, 2),
                "note":    "Prezzo ha rotto il supporto — attenzione"
            })

    # Controlla prossimità a livelli Resistance
    for res in resistance_lvls:
        dist = (res - price) / price
        if 0 <= dist <= PROXIMITY_PCT:
            alerts.append({
                "type":    "🔴 NEAR RESISTANCE",
                "level":   round(res, 6),
                "price":   round(price, 6),
                "dist_pct": round(dist*100, 2),
                "note":    "Prezzo sotto resistenza — possibile rejection"
            })

    # Monday Range
    if mr_high and abs(price - mr_high)/price <= PROXIMITY_PCT:
        alerts.append({
            "type":    "📅 NEAR MR HIGH",
            "level":   round(mr_high, 6),
            "price":   round(price, 6),
            "dist_pct": round(abs(price-mr_high)/price*100, 2),
            "note":    "Vicino al Monday Range High"
        })
    if mr_low and abs(price - mr_low)/price <= PROXIMITY_PCT:
        alerts.append({
            "type":    "📅 NEAR MR LOW",
            "level":   round(mr_low, 6),
            "price":   round(price, 6),
            "dist_pct": round(abs(price-mr_low)/price*100, 2),
            "note":    "Vicino al Monday Range Low"
        })

    if not alerts:
        return None, f"OK {symbol}: prezzo @ {price:.4g} — lontano da tutti i livelli"

    # Aggiungi info asset agli alert
    for a in alerts:
        a["symbol"]     = symbol.replace("USDT","")
        a["support_lvls"] = [round(s,6) for s in support_lvls[-3:]]
        a["resistance_lvls"] = [round(r,6) for r in resistance_lvls[:3]]

    log = f"🎯 {symbol}: {len(alerts)} alert — " + " | ".join(a["type"] for a in alerts)
    return alerts, log


def is_weekend():
    return datetime.utcnow().weekday() >= 5

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Range Scalper Scanner", page_icon="🎯", layout="wide")

st.title("🎯 Range Scalper — Level Scanner")
st.caption("Avvisa quando il prezzo si avvicina a livelli S/R chiave · Tu decidi se entrare")

for key, default in [("alerts",[]),("log",[]),("last_scan",None),("alert_history",[])]:
    if key not in st.session_state:
        st.session_state[key] = default

tab1, tab2 = st.tabs(["📡 Scanner", "📜 Log"])

with tab1:
    col_info, col_btn = st.columns([4,1])

    with col_btn:
        if is_weekend():
            st.warning("⛔ Weekend\nVolume basso")
        else:
            st.success("✅ Market active")
        scan_btn = st.button("🔍 Scan Now", type="primary", use_container_width=True)
        if st.session_state.last_scan:
            st.caption(f"Ultimo: {st.session_state.last_scan}")

    with col_info:
        run_scan = scan_btn or st.session_state.last_scan is None

        if run_scan:
            with st.spinner("Scansione livelli su 30 asset..."):
                symbols    = get_top30_symbols()
                all_alerts = []
                new_log    = []

                for asset in symbols:
                    alerts, log = scan_asset(asset)
                    new_log.append(log)
                    if alerts:
                        all_alerts.extend(alerts)
                        # Telegram per ogni alert
                        for a in alerts:
                            msg = (
                                f"{a['type']} — <b>{a['symbol']}</b>\n"
                                f"💰 Prezzo: {a['price']}\n"
                                f"📍 Livello: {a['level']} ({a['dist_pct']}% di distanza)\n"
                                f"💡 {a['note']}\n"
                                f"🛡 Supports: {a['support_lvls']}\n"
                                f"🔒 Resistances: {a['resistance_lvls']}\n"
                                f"⏰ {datetime.utcnow().strftime('%H:%M UTC')}"
                            )
                            send_telegram(msg)

                st.session_state.alerts      = all_alerts
                st.session_state.log         = new_log
                st.session_state.last_scan   = datetime.utcnow().strftime("%H:%M UTC")
                st.session_state.alert_history.extend(all_alerts)

        alerts = st.session_state.alerts

        # Metriche
        m1,m2,m3,m4 = st.columns(4)
        near_sup = [a for a in alerts if "SUPPORT" in a["type"] and "BELOW" not in a["type"]]
        near_res = [a for a in alerts if "RESISTANCE" in a["type"]]
        broken   = [a for a in alerts if "BELOW" in a["type"]]
        mr_alerts= [a for a in alerts if "MR" in a["type"]]
        m1.metric("📍 Near Support",    len(near_sup))
        m2.metric("📍 Near Resistance", len(near_res))
        m3.metric("📅 Near MR Level",   len(mr_alerts))
        m4.metric("⚠️ Broken Support",  len(broken))

        if not alerts:
            st.info("🧘 Nessun asset vicino a livelli chiave — aspetta.")
        else:
            # Raggruppa per asset
            by_asset = {}
            for a in alerts:
                by_asset.setdefault(a["symbol"], []).append(a)

            for sym, sym_alerts in by_asset.items():
                with st.expander(f"**{sym}** — {len(sym_alerts)} alert", expanded=True):
                    for a in sym_alerts:
                        color = "green" if "SUPPORT" in a["type"] and "BELOW" not in a["type"] else \
                                "red"   if "RESISTANCE" in a["type"] else \
                                "orange"if "BELOW" in a["type"] else "blue"
                        st.markdown(f":{color}[**{a['type']}**]")
                        col1,col2,col3 = st.columns(3)
                        col1.metric("Prezzo",  a["price"])
                        col2.metric("Livello", a["level"])
                        col3.metric("Distanza",f"{a['dist_pct']}%")
                        st.caption(f"💡 {a['note']}")
                        st.caption(f"Supports: {a['support_lvls']} | Resistances: {a['resistance_lvls']}")

            if alerts:
                df_export = pd.DataFrame(alerts)[["symbol","type","price","level","dist_pct","note"]]
                st.download_button("📥 Export CSV", df_export.to_csv(index=False), "scanner.csv","text/csv")

with tab2:
    st.subheader("📜 Session Log")
    if st.session_state.log:
        st.text_area("", "\n".join(st.session_state.log), height=400)
        if st.button("🗑️ Clear"):
            st.session_state.log = []
            st.rerun()
    else:
        st.info("Esegui uno scan per popolare il log.")

    if st.session_state.alert_history:
        st.subheader("Storico alert sessione")
        df_hist = pd.DataFrame(st.session_state.alert_history)[["symbol","type","price","level","dist_pct","note"]]
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
