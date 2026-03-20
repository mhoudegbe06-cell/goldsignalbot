#!/usr/bin/env python3
import os
import asyncio
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
VOTRE_CHAT_ID  = int(os.environ.get("VOTRE_CHAT_ID", "0"))
SCAN_MINUTES   = 1
SCAN_TF        = ["1m", "5m"]   # timeframes surveillés automatiquement

ACTIFS = {
    "gold": {"symbol": "GC=F", "nom": "Gold", "emoji": "🥇"},
}

TIMEFRAMES = {
    "1m": {"period": "1d",  "interval": "1m"},
    "5m": {"period": "1d",  "interval": "5m"},
    "1h": {"period": "5d",  "interval": "1h"},
    "1d": {"period": "90d", "interval": "1d"},
}

derniers_signaux = {}


# ════════════════════════════════════════════
#  DONNÉES
# ════════════════════════════════════════════

def get_data(symbol, period, interval):
    url     = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    headers = {"User-Agent": "Mozilla/5.0"}
    period_map   = {"1d": "1d", "5d": "5d", "90d": "3mo"}
    interval_map = {"1m": "1m", "5m": "5m", "1h": "1h", "1d": "1d"}
    params = {
        "range":    period_map.get(period, "1d"),
        "interval": interval_map.get(interval, interval),
    }
    try:
        r   = requests.get(url, headers=headers, params=params, timeout=15)
        d   = r.json()
        res = d["chart"]["result"][0]
        q   = res["indicators"]["quote"][0]
        ts  = res["timestamp"]
        df  = pd.DataFrame({
            "Open":   q["open"],
            "High":   q["high"],
            "Low":    q["low"],
            "Close":  q["close"],
            "Volume": q.get("volume", [0] * len(ts)),
        }, index=pd.to_datetime(ts, unit="s"))
        df.dropna(inplace=True)
        if len(df) < 10:
            return pd.DataFrame()
        return df
    except Exception as e:
        print(f"Erreur get_data {symbol} {interval}: {e}")
        return pd.DataFrame()


# ════════════════════════════════════════════
#  SUPERTREND + ATR
# ════════════════════════════════════════════

def calc_atr(df, period=14):
    pc = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - pc).abs(),
        (df["Low"]  - pc).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def calc_supertrend(df, period=10, mult=3.0):
    df  = df.copy()
    hl2 = (df["High"] + df["Low"]) / 2
    atr = calc_atr(df, period)
    ub  = hl2 + mult * atr
    lb  = hl2 - mult * atr
    st  = pd.Series(np.nan, index=df.index)
    di  = pd.Series(1, index=df.index, dtype=int)
    for i in range(1, len(df)):
        pu = ub.iloc[i-1]
        pl = lb.iloc[i-1]
        c  = df["Close"].iloc[i-1]
        lb.iloc[i] = lb.iloc[i] if lb.iloc[i] < pl or c < pl else pl
        ub.iloc[i] = ub.iloc[i] if ub.iloc[i] > pu or c > pu else pu
        ps = st.iloc[i-1] if not np.isnan(st.iloc[i-1]) else ub.iloc[i]
        if ps == pu:
            st.iloc[i], di.iloc[i] = (lb.iloc[i], 1) if df["Close"].iloc[i] > ub.iloc[i] else (ub.iloc[i], -1)
        else:
            st.iloc[i], di.iloc[i] = (ub.iloc[i], -1) if df["Close"].iloc[i] < lb.iloc[i] else (lb.iloc[i], 1)
    df["st"]  = st
    df["dir"] = di
    df["atr"] = atr
    return df


def get_signaux(df):
    sigs = []
    for i in range(1, len(df)):
        pd_, cd = df["dir"].iloc[i-1], df["dir"].iloc[i]
        if pd_ == -1 and cd == 1:
            sigs.append({"date": df.index[i], "type": "BUY",  "prix": df["Close"].iloc[i], "atr": df["atr"].iloc[i]})
        elif pd_ == 1 and cd == -1:
            sigs.append({"date": df.index[i], "type": "SELL", "prix": df["Close"].iloc[i], "atr": df["atr"].iloc[i]})
    return sigs


def calc_sl_tp(sig):
    """
    SL = 1.5x ATR depuis le prix d'entrée
    TP = 3.0x ATR (ratio Risk/Reward 1:2)
    """
    prix = sig["prix"]
    atr  = sig["atr"]
    sl_dist = round(atr * 1.5, 3)
    tp_dist = round(atr * 3.0, 3)

    if sig["type"] == "BUY":
        sl = round(prix - sl_dist, 3)
        tp = round(prix + tp_dist, 3)
    else:
        sl = round(prix + sl_dist, 3)
        tp = round(prix - tp_dist, 3)

    return sl, tp


# ════════════════════════════════════════════
#  GRAPHIQUE (matplotlib pur — stable)
# ════════════════════════════════════════════

def make_chart(data, nom, tf, sl=None, tp=None):
    df   = calc_supertrend(data).tail(80).copy()
    sigs = get_signaux(df)
    n    = len(df)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        facecolor="#131722"
    )
    ax1.set_facecolor("#131722")
    ax2.set_facecolor("#131722")
    fig.subplots_adjust(hspace=0.05)

    # ── Bougies ──
    for i in range(n):
        o = df["Open"].iloc[i]
        h = df["High"].iloc[i]
        l = df["Low"].iloc[i]
        c = df["Close"].iloc[i]
        color = "#26a69a" if c >= o else "#ef5350"
        ax1.plot([i, i], [l, h], color=color, linewidth=0.8, zorder=1)
        rect = plt.Rectangle((i - 0.35, min(o, c)), 0.7, max(abs(c - o), 0.001),
            color=color, zorder=2)
        ax1.add_patch(rect)

    # ── Supertrend ──
    for i in range(1, n):
        sv = df["st"].iloc[i]
        sp = df["st"].iloc[i-1]
        if not np.isnan(sv) and not np.isnan(sp):
            col = "#00bcd4" if df["dir"].iloc[i] == 1 else "#ff5252"
            ax1.plot([i-1, i], [sp, sv], color=col, linewidth=1.8, zorder=3)

    # ── Flèches blanches ──
    for sig in sigs:
        if sig["date"] not in df.index:
            continue
        idx  = df.index.get_loc(sig["date"])
        prix = sig["prix"]
        rng  = df["High"].max() - df["Low"].min()
        off  = rng * 0.012
        if sig["type"] == "BUY":
            ax1.annotate("",
                xy=(idx, prix - off * 0.6),
                xytext=(idx, prix - off * 1.8),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2, mutation_scale=14),
                zorder=5)
            ax1.text(idx, prix - off * 2.2, f"{prix:.3f}",
                color="white", fontsize=7.5, ha="center", va="top", fontweight="bold", zorder=5)
        else:
            ax1.annotate("",
                xy=(idx, prix + off * 0.6),
                xytext=(idx, prix + off * 1.8),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2, mutation_scale=14),
                zorder=5)
            ax1.text(idx, prix + off * 2.2, f"{prix:.3f}",
                color="white", fontsize=7.5, ha="center", va="bottom", fontweight="bold", zorder=5)

    # ── SL / TP sur le graphique ──
    if sl is not None:
        ax1.axhline(sl, color="#ff4444", lw=1.2, linestyle="--", alpha=0.9, zorder=4)
        ax1.text(n - 1, sl, f"  SL {sl:.3f}", color="#ff4444",
            fontsize=8.5, va="center", fontweight="bold")
    if tp is not None:
        ax1.axhline(tp, color="#00e676", lw=1.2, linestyle="--", alpha=0.9, zorder=4)
        ax1.text(n - 1, tp, f"  TP {tp:.3f}", color="#00e676",
            fontsize=8.5, va="center", fontweight="bold")

    # ── Prix actuel ──
    lp = df["Close"].iloc[-1]
    ax1.axhline(lp, color="#ff4081", lw=0.9, linestyle=":", alpha=0.7, zorder=4)
    ax1.text(n + 0.3, lp, f"  {lp:.3f}", color="white", fontsize=9,
        fontweight="bold", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ef5350", edgecolor="none"),
        zorder=5)

    # ── Volume ──
    for i in range(n):
        o = df["Open"].iloc[i]
        c = df["Close"].iloc[i]
        v = df["Volume"].iloc[i] if df["Volume"].iloc[i] else 0
        color = "#26a69a55" if c >= o else "#ef535055"
        ax2.bar(i, v, color=color, width=0.7)

    # ── Labels X ──
    step = max(1, n // 8)
    ticks = list(range(0, n, step))
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([""] * len(ticks))
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(
        [df.index[i].strftime("%H:%M") for i in ticks],
        color="#888888", fontsize=8
    )

    # ── Style axes ──
    for ax in [ax1, ax2]:
        ax.set_facecolor("#131722")
        ax.tick_params(colors="#888888")
        for spine in ax.spines.values():
            spine.set_color("#2a2a3a")
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", colors="#888888", labelsize=8)
        ax.grid(axis="y", color="#2a2a3a", linestyle=":", linewidth=0.5)
        ax.set_xlim(-1, n + 3)

    ax1.set_title(f"  {nom} - {tf.upper()} - Yahoo Finance",
        color="#cccccc", fontsize=11, loc="left", pad=8)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#131722")
    plt.close(fig)
    buf.seek(0)
    return buf


# ════════════════════════════════════════════
#  SCAN AUTOMATIQUE
# ════════════════════════════════════════════

async def scan(app):
    print(f"Scan {datetime.now().strftime('%H:%M:%S')}")
    info = ACTIFS["gold"]
    for tf in SCAN_TF:
        try:
            cfg = TIMEFRAMES[tf]
            df  = get_data(info["symbol"], cfg["period"], cfg["interval"])
            if df.empty or len(df) < 20:
                continue

            df_st = calc_supertrend(df)
            sigs  = get_signaux(df_st)
            if not sigs:
                continue

            last     = sigs[-1]
            cle      = f"gold_{tf}"
            date_str = str(last["date"])
            if derniers_signaux.get(cle) == date_str:
                continue
            derniers_signaux[cle] = date_str

            sl, tp = calc_sl_tp(last)
            buf    = make_chart(df, info["nom"], tf, sl=sl, tp=tp)

            emoji  = "🟢" if last["type"] == "BUY" else "🔴"
            action = "ACHAT ▲" if last["type"] == "BUY" else "VENTE ▼"
            rr     = "1 : 2"

            caption = (
                f"{emoji} *Signal {action}*\n"
                f"{info['emoji']} *{info['nom']} — {tf.upper()}*\n\n"
                f"💰 *Entrée :* `{last['prix']:.3f}`\n"
                f"🛑 *Stop Loss :* `{sl:.3f}`\n"
                f"🎯 *Take Profit :* `{tp:.3f}`\n"
                f"📊 *Risk/Reward :* `{rr}`\n\n"
                f"🕐 {last['date'].strftime('%d/%m %H:%M')}\n\n"
                f"_⚠️ Fais ta propre analyse avant de trader._"
            )

            await app.bot.send_photo(
                chat_id=VOTRE_CHAT_ID,
                photo=buf,
                caption=caption,
                parse_mode="Markdown"
            )
            print(f"Signal envoye : Gold {tf} {last['type']} | SL:{sl} TP:{tp}")

        except Exception as e:
            print(f"Erreur scan gold {tf}: {e}")
        await asyncio.sleep(1)


# ════════════════════════════════════════════
#  COMMANDES TELEGRAM
# ════════════════════════════════════════════

async def on_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        if not update.message or not update.message.text:
            return
        texte = update.message.text.strip().lower()

        if texte == "/start":
            chat_id = update.effective_chat.id
            await update.message.reply_text(
                f"Bonjour ! Je suis GoldSignalBot.\n\n"
                f"Ton Chat ID : {chat_id}\n\n"
                f"Je surveille automatiquement Gold en 1m et 5m.\n"
                f"Des qu une fleche apparait je t envoie :\n"
                f"le graphique + Entree + SL + TP\n\n"
                f"Commandes :\n"
                f"fc gold 1m\n"
                f"fc gold 5m\n"
                f"fc gold 1h\n"
                f"fc gold 1d"
            )
            return

        if texte.startswith("fc "):
            parts = texte.split()
            actif = parts[1] if len(parts) > 1 else "gold"
            tf    = parts[2] if len(parts) > 2 else "5m"

            if actif not in ACTIFS:
                await update.message.reply_text("Actif inconnu. Seul 'gold' est disponible.")
                return
            if tf not in TIMEFRAMES:
                await update.message.reply_text(f"Timeframe inconnu. Disponibles : {', '.join(TIMEFRAMES.keys())}")
                return

            info = ACTIFS[actif]
            cfg  = TIMEFRAMES[tf]
            msg  = await update.message.reply_text(f"Generation du graphique Gold {tf.upper()}...")

            df = get_data(info["symbol"], cfg["period"], cfg["interval"])
            if df.empty:
                await msg.edit_text("Donnees indisponibles. Reessaie dans quelques secondes.")
                return

            df_st    = calc_supertrend(df)
            sigs     = get_signaux(df_st)
            last     = sigs[-1] if sigs else None
            lp       = df_st["Close"].iloc[-1]
            tendance = "Haussier" if df_st["dir"].iloc[-1] == 1 else "Baissier"

            sl, tp = (None, None)
            if last:
                sl, tp = calc_sl_tp(last)

            buf = make_chart(df, info["nom"], tf, sl=sl, tp=tp)

            sig_txt = ""
            if last:
                e = "ACHAT" if last["type"] == "BUY" else "VENTE"
                sig_txt = (
                    f"\nDernier signal : {e} @ {last['prix']:.3f}"
                    f"\nSL : {sl:.3f}  |  TP : {tp:.3f}"
                )

            caption = (
                f"Gold - {tf.upper()}\n"
                f"Prix actuel : {lp:.3f}\n"
                f"Tendance : {tendance}"
                f"{sig_txt}"
            )
            await msg.delete()
            await update.message.reply_photo(photo=buf, caption=caption)

    except Exception as e:
        print(f"Erreur message : {e}")
        try:
            await update.message.reply_text(f"Erreur : {e}")
        except:
            pass


# ════════════════════════════════════════════
#  LANCEMENT
# ════════════════════════════════════════════

def main():
    print("GoldSignalBot demarre...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT, on_message))

    scheduler = AsyncIOScheduler(timezone="Europe/Paris")
    scheduler.add_job(scan, "interval", minutes=SCAN_MINUTES, args=[app], id="scan")

    async def post_init(application):
        scheduler.start()
        print(f"Pret ! Surveillance Gold 1m et 5m active.")

    app.post_init = post_init
    app.run_polling()


if __name__ == "__main__":
    main()
