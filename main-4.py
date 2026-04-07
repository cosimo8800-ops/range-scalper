"""
Range Scalper Signal Dashboard
Ispirato alla metodologia di Eliz (@eliz883)
Replit-ready | Streamlit + Binance API + CoinGecko
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import plotly.graph_objects as go

# ============================================================
# CONFIG
# ============================================================
BINANCE_BASE  = "https://api.binance.com/api/v3"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

STABLECOINS      = {"USDT","USDC","BUSD","DAI","TUSD","FDUSD","USDS","USDP","USDE"}
MIN_VOLUME_USD   = 50_000_000   # $50M filtro volume 24h
MAX_TP_PCT       = 0.04         # TP max 4% da entry (scalp only)
MAX_SL_PCT       = 0.015        # SL max 1.5% da entry
VOL_MULTIPLIER   = 1.5          # volume candle >= 1.5x media
VOL_LOOKBACK     = 20           # periodi media volume
SR_BOX_PCT       = 0.005        # ±0.5% ampiezza box S/R
SR_MIN_TOUCHES   = 2            # tocchi minimi per validare livello

# ============================================================
# DATA FETCHING
# ============================================================

@st.cache_data(ttl=900)
def get_top30_symbols():
    """Top 30 crypto per market cap da CoinGecko (escludi stablecoin)"""
    try:
        r = requests.get(
            f"{COINGECKO_BASE}/coins/markets",
            params={"vs_currency":"usd","order":"market_cap_desc","per_page":80,"page":1},
            timeout=10
        )
        symbols = []
        for coin in r.json():
            sym = coin["symbol"].upper()
            if sym not in STABLECOINS and coin.get("total_volume", 0) > MIN_VOLUME_USD:
                symbols.append({
                    "symbol":     sym + "USDT",
                    "name":       coin["name"],
                    "volume_24h": coin.get("total_volume", 0),
                    "market_cap": coin.get("market_cap", 0)
                })
            if len(symbols) >= 30:
                break
        return symbols
    except:
        # Fallback hardcoded
        base = ["BTC","ETH","BNB","SOL","XRP","ADA","AVAX","DOGE","DOT","LINK",
                "TRX","LTC","SHIB","UNI","ATOM","XLM","NEAR","APT","INJ","SUI",
                "ARB","OP","FIL","HBAR","VET","ALGO","ICP","AAVE","ZEC","TON"]
        return [{"symbol":s+"USDT","name":s,"volume_24h":100_000_000,"market_cap":0} for s in base]


# Mappa intervalli -> Bybit
INTERVAL_MAP = {
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "1d": "D",
    "1w": "W",
}

def get_candles(symbol, interval, limit=100):
    """Fetch OHLCV da Bybit public API (stabile da cloud)"""
    bybit_interval = INTERVAL_MAP.get(interval, "60")
    try:
        r = requests.get(
            "https://api.bybit.com/v5/market/kline",
            params={
                "category": "spot",
                "symbol":   symbol,
                "interval": bybit_interval,
                "limit":    min(limit, 200),
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15
        )
        if r.status_code != 200:
            return None
        js = r.json()
        if js.get("retCode") != 0:
            return None
        rows = js["result"]["list"]
        if not rows:
            return None
        # Bybit restituisce [timestamp, open, high, low, close, volume, turnover]
        # ordinati dal più recente al più vecchio -> inverti
        rows = list(reversed(rows))
        df = pd.DataFrame(rows, columns=["open_time","open","high","low","close","volume","turnover"])
        df["open_time"] = pd.to_datetime(df["open_time"].astype(float), unit="ms")
        for col in ["open","high","low","close","volume"]:
            df[col] = df[col].astype(float)
        df = df[df["close"] > 0].reset_index(drop=True)
        return df if len(df) > 5 else None
    except:
        return None

# ============================================================
# SIGNAL LOGIC
# ============================================================

def detect_sr_boxes(candles_d1):
    """Rileva livelli S/R da D1 (swing highs/lows con >= 2 tocchi)"""
    if candles_d1 is None or len(candles_d1) < 20:
        return [], []

    highs   = candles_d1["high"].values
    lows    = candles_d1["low"].values
    support_levels    = []
    resistance_levels = []

    # Swing highs → resistance
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] \
           and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            level   = highs[i]
            touches = sum(1 for h in highs if abs(h - level) / level < SR_BOX_PCT)
            if touches >= SR_MIN_TOUCHES:
                resistance_levels.append(level)

    # Swing lows → support
    for i in range(2, len(lows) - 2):
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] \
           and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            level   = lows[i]
            touches = sum(1 for l in lows if abs(l - level) / level < SR_BOX_PCT)
            if touches >= SR_MIN_TOUCHES:
                support_levels.append(level)

    def cluster(levels):
        if not levels:
            return []
        levels = sorted(set(levels))
        out = [levels[0]]
        for l in levels[1:]:
            if abs(l - out[-1]) / out[-1] > 0.01:
                out.append(l)
        return out

    return cluster(support_levels), cluster(resistance_levels)


def get_monday_range(symbol):
    """High/Low del primo H4 candle del lunedì corrente"""
    try:
        candles = get_candles(symbol, "4h", 50)
        if candles is None:
            return None, None
        candles["weekday"] = candles["open_time"].dt.weekday
        mon = candles[candles["weekday"] == 0]
        if mon.empty:
            return None, None
        last_mon = mon.iloc[-1]
        return float(last_mon["high"]), float(last_mon["low"])
    except:
        return None, None


def is_weekend():
    return datetime.utcnow().weekday() >= 5


def analyze_asset(asset_info, btc_breakdown):
    """
    Analisi completa di un asset.
    Ritorna (signal_dict | None, [log_entries])
    TUTTE E 3 le condizioni devono essere TRUE → segnale
    """
    symbol = asset_info["symbol"]

    # --- FILTRI PRE-ENTRY ---
    if is_weekend():
        return None, [f"SKIP {symbol}: weekend — low volume suppressed"]

    if asset_info["volume_24h"] < MIN_VOLUME_USD:
        return None, [f"SKIP {symbol}: volume 24h ${asset_info['volume_24h']:,.0f} < min ${MIN_VOLUME_USD:,.0f}"]

    if symbol != "BTCUSDT" and btc_breakdown:
        return None, [f"SKIP {symbol}: BTC breakdown — altcoin signals suppressed"]

    # --- FETCH DATI ---
    candles_h1 = get_candles(symbol, "1h", 60)
    candles_d1 = get_candles(symbol, "1d", 100)

    if candles_h1 is None or len(candles_h1) < VOL_LOOKBACK + 5:
        return None, [f"SKIP {symbol}: insufficient H1 data"]

    support_levels, resistance_levels = detect_sr_boxes(candles_d1)
    mr_high, mr_low = get_monday_range(symbol)

    last  = candles_h1.iloc[-1]
    prev  = candles_h1.iloc[-2]
    price = float(last["close"])

    # --- CONDIZIONE 3: VOLUME ---
    vol_avg = float(np.mean(candles_h1["volume"].values[-VOL_LOOKBACK-1:-1]))
    vol_ok  = float(last["volume"]) >= vol_avg * VOL_MULTIPLIER

    if not vol_ok:
        return None, [f"NO TRADE {symbol}: volume {last['volume']:.0f} < {vol_avg * VOL_MULTIPLIER:.0f} required"]

    # --- CONDIZIONE 1 LONG: Liquidity Sweep + Rebound ---
    long_sweep = False
    long_sup   = None
    for sup in support_levels:
        box_bottom = sup * (1 - SR_BOX_PCT)
        if float(last["low"]) < box_bottom and float(last["close"]) > box_bottom:
            long_sweep = True
            long_sup   = sup
            break

    # --- CONDIZIONE 2 LONG: Flip + Acceptance ---
    long_flip = False
    long_res_flip = None
    for res in resistance_levels:
        if float(prev["close"]) < res and float(last["close"]) > res:
            long_flip    = True
            long_res_flip = res
            break

    # --- CONDIZIONE 1 SHORT: Liquidity Sweep + Rejection ---
    short_sweep = False
    short_res   = None
    for res in resistance_levels:
        box_top = res * (1 + SR_BOX_PCT)
        if float(last["high"]) > box_top and float(last["close"]) < box_top:
            short_sweep = True
            short_res   = res
            break

    # --- CONDIZIONE 2 SHORT: Flip + Acceptance ---
    short_flip = False
    short_sup_flip = None
    for sup in support_levels:
        if float(prev["close"]) > sup and float(last["close"]) < sup:
            short_flip     = True
            short_sup_flip = sup
            break

    # --- EMISSIONE SEGNALE (AND STRICT) ---

    # LONG: tutte e 3 vere
    if long_sweep and long_flip and vol_ok:
        sl     = round(float(last["low"]) * (1 - 0.003), 8)
        sl_pct = abs(price - sl) / price
        if sl_pct > MAX_SL_PCT:
            return None, [f"NO TRADE {symbol} LONG: SL {sl_pct:.1%} troppo largo (max {MAX_SL_PCT:.1%})"]

        res_above = sorted([r for r in resistance_levels if r > price])
        tp1 = res_above[0] if res_above else price * 1.008
        tp2 = (mr_high if mr_high and mr_high > tp1 else tp1 * 1.005)

        if abs(tp2 - price) / price > MAX_TP_PCT:
            tp2 = price * (1 + MAX_TP_PCT)

        context = f"Support box @ {long_sup:.6g}"
        if mr_low and price > mr_low:
            context += " | Above MR Low"

        signal = _build_signal(symbol, "LONG", price, sl, tp1, tp2,
                               "Both", True, context)
        return signal, [f"✅ LONG signal → {symbol} | Entry {price:.6g} SL {sl:.6g} TP1 {tp1:.6g} TP2 {tp2:.6g}"]

    # SHORT: tutte e 3 vere
    elif short_sweep and short_flip and vol_ok:
        sl     = round(float(last["high"]) * (1 + 0.003), 8)
        sl_pct = abs(sl - price) / price
        if sl_pct > MAX_SL_PCT:
            return None, [f"NO TRADE {symbol} SHORT: SL {sl_pct:.1%} troppo largo (max {MAX_SL_PCT:.1%})"]

        sup_below = sorted([s for s in support_levels if s < price], reverse=True)
        tp1 = sup_below[0] if sup_below else price * 0.992
        tp2 = (mr_low if mr_low and mr_low < tp1 else tp1 * 0.995)

        if abs(price - tp2) / price > MAX_TP_PCT:
            tp2 = price * (1 - MAX_TP_PCT)

        context = f"Resistance box @ {short_res:.6g}"
        if mr_high and price < mr_high:
            context += " | Below MR High"

        signal = _build_signal(symbol, "SHORT", price, sl, tp1, tp2,
                               "Both", True, context)
        return signal, [f"✅ SHORT signal → {symbol} | Entry {price:.6g} SL {sl:.6g} TP1 {tp1:.6g} TP2 {tp2:.6g}"]

    else:
        reasons = []
        if not long_sweep and not short_sweep:
            reasons.append("no sweep rilevato")
        if not long_flip and not short_flip:
            reasons.append("no flip+acceptance")
        return None, [f"NO TRADE {symbol}: {', '.join(reasons)}"]


def _build_signal(symbol, direction, price, sl, tp1, tp2, trigger, vol_ok, context):
    entry_low  = round(price * 0.9990, 8)
    entry_high = round(price * 1.0010, 8)
    return {
        "asset":            symbol.replace("USDT",""),
        "direction":        direction,
        "timeframe":        "H1",
        "entry_low":        entry_low,
        "entry_high":       entry_high,
        "sl":               round(sl, 8),
        "tp1":              round(tp1, 8),
        "tp2":              round(tp2, 8),
        "trigger":          trigger,
        "volume_confirmed": vol_ok,
        "macro_context":    context,
        "status":           "ACTIVE",
        "timestamp":        datetime.utcnow().strftime("%H:%M:%S")
    }


def check_btc_breakdown():
    candles_d1      = get_candles("BTCUSDT", "1d", 100)
    candles_h1      = get_candles("BTCUSDT", "1h", 5)
    if candles_d1 is None or candles_h1 is None:
        return False
    support, _      = detect_sr_boxes(candles_d1)
    if not support:
        return False
    current_price   = float(candles_h1["close"].iloc[-1])
    lowest_support  = min(support)
    return current_price < lowest_support * (1 - SR_BOX_PCT)

# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(symbols, months=6, starting_capital=10_000, risk_pct=0.01):
    """Backtest storico su dati Binance H1"""
    equity       = float(starting_capital)
    trade_num    = 0
    results      = []
    equity_curve = [{"trade": 0, "equity": equity}]

    limit = min(months * 30 * 24, 1000)

    progress    = st.progress(0)
    status_text = st.empty()

    for idx, sym_info in enumerate(symbols):
        symbol = sym_info["symbol"]
        status_text.text(f"Backtesting {symbol} ({idx+1}/{len(symbols)})...")
        progress.progress((idx + 1) / len(symbols))

        candles_h1 = get_candles(symbol, "1h", limit)
        candles_d1 = get_candles(symbol, "1d", 200)
        if candles_h1 is None or len(candles_h1) < 50:
            continue

        support_levels, resistance_levels = detect_sr_boxes(candles_d1)
        if not support_levels or not resistance_levels:
            continue

        for i in range(VOL_LOOKBACK + 5, len(candles_h1) - 13):
            candle      = candles_h1.iloc[i]
            prev_candle = candles_h1.iloc[i - 1]

            if candle["open_time"].weekday() >= 5:
                continue

            vol_window = candles_h1["volume"].values[i - VOL_LOOKBACK:i]
            vol_avg    = float(np.mean(vol_window))
            if float(candle["volume"]) < vol_avg * VOL_MULTIPLIER:
                continue

            price = float(candle["close"])

            # Condizioni LONG
            long_sweep = any(
                float(candle["low"]) < s*(1-SR_BOX_PCT) and float(candle["close"]) > s*(1-SR_BOX_PCT)
                for s in support_levels
            )
            long_flip = any(
                float(candle["close"]) > r * (1 - SR_BOX_PCT)
                for r in resistance_levels
                if r > float(candle["close"]) * 0.98
            )

            # Condizioni SHORT
            short_sweep = any(
                float(candle["high"]) > r*(1+SR_BOX_PCT) and float(candle["close"]) < r*(1+SR_BOX_PCT)
                for r in resistance_levels
            )
            short_flip = any(
                float(candle["close"]) < s * (1 + SR_BOX_PCT)
                for s in support_levels
                if s < float(candle["close"]) * 1.02
            )

            direction = sl = tp1 = tp2 = None

            if long_sweep and long_flip:
                sl     = float(candle["low"]) * (1 - 0.003)
                sl_pct = abs(price - sl) / price
                if sl_pct > MAX_SL_PCT:
                    continue
                res_above = sorted([r for r in resistance_levels if r > price])
                if not res_above:
                    continue
                tp1 = res_above[0]
                tp2 = tp1 * 1.005
                if abs(tp2 - price) / price > MAX_TP_PCT:
                    continue
                direction = "LONG"

            elif short_sweep and short_flip:
                sl     = float(candle["high"]) * (1 + 0.003)
                sl_pct = abs(sl - price) / price
                if sl_pct > MAX_SL_PCT:
                    continue
                sup_below = sorted([s for s in support_levels if s < price], reverse=True)
                if not sup_below:
                    continue
                tp1 = sup_below[0]
                tp2 = tp1 * 0.995
                if abs(price - tp2) / price > MAX_TP_PCT:
                    continue
                direction = "SHORT"

            if direction is None:
                continue

            # Risk sizing — 1% equity, max 5% hard cap
            risk_amount = min(equity * risk_pct, equity * 0.05)
            sl_dist     = abs(price - sl)
            if sl_dist == 0:
                continue

            # Simula le candele successive (max 12h)
            future  = candles_h1.iloc[i+1:i+13]
            outcome = "TIMEOUT"
            profit  = 0.0
            tp1_hit = False

            for _, fc in future.iterrows():
                if direction == "LONG":
                    if float(fc["low"]) <= sl:
                        outcome = "PARTIAL_WIN" if tp1_hit else "LOSS"
                        profit  = risk_amount * abs(tp1 - price) / sl_dist * 0.5 if tp1_hit else -risk_amount
                        break
                    if not tp1_hit and float(fc["high"]) >= tp1:
                        tp1_hit = True
                    if tp1_hit and float(fc["high"]) >= tp2:
                        r1      = min(abs(tp1 - price) / sl_dist, 3.0)
                        r2      = min(abs(tp2 - price) / sl_dist, 3.0)
                        profit  = risk_amount * (r1 * 0.5 + r2 * 0.5)
                        outcome = "WIN"
                        break
                else:  # SHORT
                    if float(fc["high"]) >= sl:
                        outcome = "PARTIAL_WIN" if tp1_hit else "LOSS"
                        profit  = risk_amount * abs(price - tp1) / sl_dist * 0.5 if tp1_hit else -risk_amount
                        break
                    if not tp1_hit and float(fc["low"]) <= tp1:
                        tp1_hit = True
                    if tp1_hit and float(fc["low"]) <= tp2:
                        r1      = min(abs(price - tp1) / sl_dist, 3.0)
                        r2      = min(abs(price - tp2) / sl_dist, 3.0)
                        profit  = risk_amount * (r1 * 0.5 + r2 * 0.5)
                        outcome = "WIN"
                        break

            equity    = max(equity + profit, 1.0)
            trade_num += 1

            results.append({
                "trade":      trade_num,
                "date":       candle["open_time"].strftime("%Y-%m-%d %H:%M"),
                "asset":      symbol.replace("USDT",""),
                "direction":  direction,
                "entry":      round(price, 6),
                "sl":         round(sl, 6),
                "tp1":        round(tp1, 6),
                "tp2":        round(tp2, 6),
                "outcome":    outcome,
                "profit_usd": round(profit, 2),
                "equity":     round(equity, 2)
            })
            equity_curve.append({"trade": trade_num, "equity": round(equity, 2)})

    progress.empty()
    status_text.empty()
    return results, equity_curve


def compute_metrics(results, starting_capital):
    if not results:
        return {}
    df          = pd.DataFrame(results)
    total       = len(df)
    wins        = len(df[df["outcome"] == "WIN"])
    losses      = len(df[df["outcome"] == "LOSS"])
    partials    = len(df[df["outcome"] == "PARTIAL_WIN"])
    win_rate    = wins / total * 100 if total else 0

    gross_profit = df[df["profit_usd"] > 0]["profit_usd"].sum()
    gross_loss   = abs(df[df["profit_usd"] < 0]["profit_usd"].sum())
    pf           = gross_profit / gross_loss if gross_loss > 0 else 9.99

    eq = df["equity"].values
    peak   = float(starting_capital)
    max_dd = 0.0
    for e in eq:
        if e > peak:
            peak = e
        dd = (peak - e) / peak
        if dd > max_dd:
            max_dd = dd

    avg_win  = df[df["profit_usd"] > 0]["profit_usd"].mean() if wins else 0
    avg_loss = abs(df[df["profit_usd"] < 0]["profit_usd"].mean()) if losses else 1
    avg_rr   = avg_win / avg_loss if avg_loss else 0

    final_eq      = df["equity"].iloc[-1]
    total_return  = (final_eq - starting_capital) / starting_capital * 100

    return {
        "total_trades":  total,
        "wins":          wins,
        "losses":        losses,
        "partials":      partials,
        "win_rate":      round(win_rate, 1),
        "profit_factor": round(min(pf, 99.0), 2),
        "max_drawdown":  round(max_dd * 100, 1),
        "avg_rr":        round(avg_rr, 2),
        "total_return":  round(total_return, 1),
        "final_equity":  round(final_eq, 2)
    }

# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Range Scalper Signals", page_icon="📈", layout="wide")

st.markdown("""
<style>
body { background: #0e1117; }
.block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Range Scalper Signal Dashboard")
st.caption("H1/H2 signal engine · ICT/SMC methodology · Top 30 crypto by market cap")

# Session state
for key, default in [
    ("signals", []),
    ("log", []),
    ("last_scan", None),
    ("signal_history", [])
]:
    if key not in st.session_state:
        st.session_state[key] = default

tab1, tab2, tab3 = st.tabs(["🎯 Dashboard", "📊 Backtest", "📜 Log"])

# ---- TAB 1: DASHBOARD ----
with tab1:
    header_col, btn_col = st.columns([4, 1])

    with btn_col:
        if is_weekend():
            st.error("⛔ Weekend\nSegnali soppressi")
        else:
            st.success("✅ Market active")
        scan_btn = st.button("🔍 Scan Now", type="primary", use_container_width=True)
        if st.session_state.last_scan:
            st.caption(f"Ultimo scan: {st.session_state.last_scan}")

    with header_col:
        run_scan = scan_btn or st.session_state.last_scan is None

        if run_scan:
            with st.spinner("Scansione 30 asset in corso..."):
                symbols        = get_top30_symbols()
                new_signals    = []
                new_log        = []
                btc_breakdown  = check_btc_breakdown()

                if btc_breakdown:
                    new_log.append("⚠️ BTC BREAKDOWN — altcoin signals soppressi")

                for asset in symbols:
                    signal, logs = analyze_asset(asset, btc_breakdown)
                    new_log.extend(logs)
                    if signal:
                        new_signals.append(signal)
                        st.session_state.signal_history.append({
                            **signal,
                            "scan_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                        })

                st.session_state.signals   = new_signals
                st.session_state.log       = new_log
                st.session_state.last_scan = datetime.utcnow().strftime("%H:%M:%S UTC")

        signals = st.session_state.signals
        longs   = [s for s in signals if s["direction"] == "LONG"]
        shorts  = [s for s in signals if s["direction"] == "SHORT"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Signals", len(signals))
        m2.metric("📗 Long",       len(longs))
        m3.metric("📕 Short",      len(shorts))
        m4.metric("Active",        len(signals))

        if len(signals) > 6:
            st.warning(f"⚠️ {len(signals)} segnali — troppi. Qualcosa non va, controlla i threshold.")

        if not signals:
            st.info("🧘 Nessun segnale — in attesa di un setup valido. **Patience is a position.**")
        else:
            rows = [{
                "Asset":        s["asset"],
                "Dir":          s["direction"],
                "TF":           s["timeframe"],
                "Entry Zone":   f"{s['entry_low']} – {s['entry_high']}",
                "SL":           s["sl"],
                "TP1 (50%)":    s["tp1"],
                "TP2 (50%)":    s["tp2"],
                "Trigger":      s["trigger"],
                "Vol ✓":        "✅" if s["volume_confirmed"] else "❌",
                "Macro Context":s["macro_context"],
                "Time":         s["timestamp"]
            } for s in signals]

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            csv = pd.DataFrame(rows).to_csv(index=False)
            st.download_button("📥 Export CSV", csv, "signals.csv", "text/csv")

# ---- TAB 2: BACKTEST ----
with tab2:
    st.subheader("📊 Backtest Engine")
    st.caption("Stessa logica segnali applicata su dati storici Binance H1")

    c1, c2, c3 = st.columns(3)
    bt_capital = c1.number_input("Capital ($)",       value=10000, step=1000, min_value=1000)
    bt_risk    = c2.number_input("Risk/trade (%)",    value=1.0, min_value=0.1, max_value=5.0, step=0.1)
    bt_months  = c3.selectbox("Periodo",              [3, 6, 12], index=1)
    bt_scope   = st.selectbox("Asset da testare",     ["BTC only", "BTC + ETH", "Top 10", "All 30"])

    if st.button("▶️ Run Backtest", type="primary"):
        symbols = get_top30_symbols()
        if bt_scope == "BTC only":
            symbols = [s for s in symbols if s["symbol"] == "BTCUSDT"]
        elif bt_scope == "BTC + ETH":
            symbols = [s for s in symbols if s["symbol"] in ["BTCUSDT","ETHUSDT"]]
        elif bt_scope == "Top 10":
            symbols = symbols[:10]

        results, equity_curve = run_backtest(
            symbols,
            months=bt_months,
            starting_capital=bt_capital,
            risk_pct=bt_risk / 100
        )

        if not results:
            st.warning("Nessun trade generato. Prova a estendere il periodo o usare più asset.")
        else:
            m = compute_metrics(results, bt_capital)

            # Sanity check
            if m["final_equity"] > bt_capital * 20:
                st.error("🚨 CALCULATION ERROR: equity > 20x — risultati inaffidabili")
            else:
                cols = st.columns(6)
                cols[0].metric("Win Rate",      f"{m['win_rate']}%",       f"{m['wins']}W / {m['losses']}L")
                cols[1].metric("Partial Wins",  m['partials'])
                cols[2].metric("Avg R:R",        m['avg_rr'])
                cols[3].metric("Total Trades",   m['total_trades'])
                cols[4].metric("Max Drawdown",  f"{m['max_drawdown']}%")
                cols[5].metric("Profit Factor",  m['profit_factor'])

                st.metric("Total Return", f"{m['total_return']}%",
                          f"${m['final_equity']:,.0f} equity finale")

                # Equity curve
                eq_df = pd.DataFrame(equity_curve)
                fig   = go.Figure()
                fig.add_trace(go.Scatter(
                    x=eq_df["trade"], y=eq_df["equity"],
                    mode="lines", name="Equity",
                    line=dict(color="#00ff88", width=2)
                ))
                fig.add_hline(y=bt_capital, line_dash="dash", line_color="#888",
                              annotation_text="Starting Capital")
                fig.update_layout(
                    title="Equity Curve (trade per trade)",
                    xaxis_title="Trade #",
                    yaxis_title="Equity ($)",
                    template="plotly_dark",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

                df_res = pd.DataFrame(results)[
                    ["trade","date","asset","direction","entry","sl","tp1","tp2","outcome","profit_usd","equity"]
                ]
                st.dataframe(df_res, use_container_width=True, hide_index=True)
                st.download_button("📥 Export Backtest CSV",
                                   df_res.to_csv(index=False),
                                   "backtest.csv", "text/csv")

# ---- TAB 3: LOG ----
with tab3:
    st.subheader("📜 Session Log")
    if st.session_state.log:
        st.text_area("", "\n".join(st.session_state.log), height=400)
        if st.button("🗑️ Clear Log"):
            st.session_state.log = []
            st.rerun()
    else:
        st.info("Esegui uno scan per popolare il log.")

    if st.session_state.signal_history:
        st.subheader("Storico segnali (sessione corrente)")
        st.dataframe(pd.DataFrame(st.session_state.signal_history),
                     use_container_width=True, hide_index=True)
