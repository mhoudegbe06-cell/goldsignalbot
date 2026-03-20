#!/usr/bin/env python3
import os
import asyncio
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
from io import BytesIO
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
VOTRE_CHAT_ID  = int(os.environ.get("VOTRE_CHAT_ID", "0"))
SCAN_MINUTES   = 1

ACTIFS = {
    "gold":   {"symbol": "GC=F",    "nom": "Gold",    "emoji": "🥇"},
    "xauusd": {"symbol": "GC=F",    "nom": "Gold",    "emoji": "🥇"},
    "btc":    {"symbol": "BTC-USD", "nom": "Bitcoin", "emoji": "₿"},
    "silver": {"symbol": "SI=F",    "nom": "Silver",  "emoji": "🥈"},
}

TIMEFRAMES = {
    "1m":  {"period": "1d",  "interval": "1m"},
    "5m":  {"period": "1d",  "interval": "5m"},
    "15m": {"period": "5d",  "interval": "15m"},
    "1h":  {"period": "5d",  "interval": "1h"},
    "1d":  {"period": "90d", "interval": "1d"},
}

derniers_signaux = {}


def get_data(symbol, period, interval):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    headers = {"User-Agent": "Mozilla/5.0"}
    period_yf = {"1d": "1d", "5d": "5d", "90d": "3mo"}
    inter_yf  = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "1d": "1d"}
    params = {
        "range":    period_yf.get(period, "5d"),
        "interval": inter_yf.get(interval, interval),
    }
    try:
        r   = requests.get(url, headers=headers, params=params, timeout=15)
        res = r.json()["chart"]["result"][0]
        q   = res["indicators"]["quote"][0]
        df  = pd.DataFrame({
            "Open":   q["open"],
            "High":   q["high"],
            "Low":    q["low"],
            "Close":  q["close"],
            "Volume": q.get("volume", [0]*len(res["timestamp"])),
        }, index=pd.to_datetime(res["timestamp"], unit="s"))
        df.dropna(inplace=True)
        if interval == "15m":
            df = df.resample("15min").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
        return df
    except Exception as e:
        print(f"Erreur donnees {symbol}: {e}")
        return pd.DataFrame()


def calc_supertrend(df, period=10, mult=3.0):
    df  = df.copy()
    hl2 = (df["High"] + df["Low"]) / 2
    pc  = df["Close"].shift(1)
    tr  = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - pc).abs(),
        (df["Low"]  - pc).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    ub  = hl2 + mult * atr
    lb  = hl2 - mult * atr
    st  = pd.Series(np.nan, index=df.index)
    di  = pd.Series(1, index=df.index, dtype=int)
    for i in range(1, len(df)):
        pu, pl, c = ub.iloc[i-1], lb.iloc[i-1], df["Close"].iloc[i-1]
        lb.iloc[i] = lb.iloc[i] if lb.iloc[i] < pl or c < pl else pl
        ub.iloc[i] = ub.iloc[i] if ub.iloc[i] > pu or c > pu else pu
        ps = st.iloc[i-1] if not np.isnan(st.iloc[i-1]) else ub.iloc[i]
        if ps == pu:
            st.iloc[i], di.iloc[i] = (lb.iloc[i], 1) if df["Close"].iloc[i] > ub.iloc[i] else (ub.iloc[i], -1)
        else:
            st.iloc[i], di.iloc[i] = (ub.iloc[i], -1) if df["Close"].iloc[i] < lb.iloc[i] else (lb.iloc[i], 1)
    df["st"]  = st
    df["dir"] = di
    return df


def get_signaux(df):
    sigs = []
    for i in range(1, len(df)):
        pd_, cd = df["dir"].iloc[i-1], df["dir"].iloc[i]
        if pd_ == -1 and cd == 1:
            sigs.append({"date": df.index[i], "type": "BUY",  "prix": df["Close"].iloc[i]})
        elif pd_ == 1 and cd == -1:
            sigs.append({"date": df.index[i], "type": "SELL", "prix": df["Close"].iloc[i]})
    return sigs


def make_chart(data, nom, tf):
    df   = calc_supertrend(data).tail(80).copy()
    sigs = get_signaux(df)
    n    = len(df)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8),
        gridspec_kw={"height_ratios": [3, 1]}, facecolor="#131722")
    ax1.set_facecolor("#131722")
    ax2.set_facecolor("#131722")
    fig.subplots_adjust(hspace=0.05)

    # ── Bougies ──
    for i in range(n):
        o, h, l, c = df["Open"].iloc[i], df["High"].iloc[i], df["Low"].iloc[i], df["Close"].iloc[i]
        color = "#26a69a" if c >= o else "#ef5350"
        ax1.plot([i, i], [l, h], color=color, linewidth=0.8)
        ax1.add_patch(plt.Rectangle((i - 0.35, min(o, c)), 0.7, abs(c - o),
            color=color, zorder=2))

    # ── Supertrend ──
    for i in range(1, n):
        if not np.isnan(df["st"].iloc[i]) and not np.isnan(df["st"].iloc[i-1]):
            col = "#00bcd4" if df["dir"].iloc[i] == 1 else "#ff5252"
            ax1.plot([i-1, i], [df["st"].iloc[i-1], df["st"].iloc[i]], color=col, linewidth=1.8, zorder=3)

    # ── Fleches blanches ──
    for sig in sigs:
        if sig["date"] not in df.index:
            continue
        idx  = df.index.get_loc(sig["date"])
        prix = sig["prix"]
        if sig["type"] == "BUY":
            ax1.annotate("", xy=(idx, prix * 0.9984), xytext=(idx, prix * 0.9968),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2, mutation_scale=14))
            ax1.text(idx, prix * 0.9955, f"{prix:.3f}", color="white",
                fontsize=7.5, ha="center", va="top", fontweight="bold")
        else:
            ax1.annotate("", xy=(idx, prix * 1.0016), xytext=(idx, prix * 1.0032),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2, mutation_scale=14))
            ax1.text(idx, prix * 1.0045, f"{prix:.3f}", color="white",
                fontsize=7.5, ha="center", va="bottom", fontweight="bold")

    # ── Prix actuel ──
    lp = df["Close"].iloc[-1]
    ax1.axhline(lp, color="#ff4081", lw=0.9, linestyle="--", alpha=0.85)
    ax1.text(n + 0.5, lp, f"  {lp:.3f}", color="white", fontsize=9,
        fontweight="bold", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ef5350", edgecolor="none"))

    # ── Volume ──
    for i in range(n):
        o, c = df["Open"].iloc[i], df["Close"].iloc[i]
        color = "#26a69a55" if c >= o else "#ef535055"
        vol = df["Volume"].iloc[i] if df["Volume"].iloc[i] else 0
        ax2.bar(i, vol, color=color, width=0.7)

    # ── Labels X (heures) ──
    step = max(1, n // 8)
    ticks = range(0, n, step)
    ax1.set_xticks(list(ticks))
    ax1.set_xticklabels([""] * len(list(ticks)))
    ax2.set_xticks(list(ticks))
    ax2.set_xticklabels(
        [df.index[i].strftime("%H:%M") for i in ticks],
        color="#888888", fontsize=8
    )

    # ── Style ──
    for ax in [ax1, ax2]:
        ax.tick_params(colors="#888888")
        ax.spines["bottom"].set_color("#2a2a3a")
        ax.spines["top"].set_color("#2a2a3a")
        ax.spines["left"].set_color("#2a2a3a")
        ax.spines["right"].set_color("#2a2a3a")
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", colors="#888888", labelsize=8)
        ax.grid(axis="y", color="#2a2a3a", linestyle=":", linewidth=0.5)
        ax.set_xlim(-1, n + 2)

    ax1.set_title(f"  {nom} - {tf.upper()} - Yahoo Finance",
        color="#cccccc", fontsize=11, loc="left", pad=8)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#131722")
    plt.close(fig)
    buf.seek(0)
    return buf


async def scan(app):
    print(f"Scan {datetime.now().strftime('%H:%M:%S')}")
    for key, info in ACTIFS.items():
        if key == "xauusd":
            continue
        for tf, cfg in [("1m", TIMEFRAMES["1m"]), ("1h", TIMEFRAMES["1h"])]:
            try:
                df = get_data(info["symbol"], cfg["period"], cfg["interval"])
                if df.empty or len(df) < 20:
                    continue
                df_st = calc_supertrend(df)
                sigs  = get_signaux(df_st)
                if not sigs:
                    continue
                last     = sigs[-1]
                cle      = f"{key}_{tf}"
                date_str = str(last["date"])
                if derniers_signaux.get(cle) == date_str:
                    continue
                derniers_signaux[cle] = date_str
                buf    = make_chart(df, info["nom"], tf)
                emoji  = "🟢" if last["type"] == "BUY" else "🔴"
                action = "ACHAT" if last["type"] == "BUY" else "VENTE"
                caption = (
                    f"{emoji} Signal {action} - {info['emoji']} {info['nom']} ({tf.upper()})\n\n"
                    f"Prix : {last['prix']:.3f}\n"
                    f"Heure : {last['date'].strftime('%d/%m %H:%M')}\n\n"
                    f"Fais ta propre analyse avant de trader."
                )
                await app.bot.send_photo(chat_id=VOTRE_CHAT_ID, photo=buf, caption=caption)
                print(f"Signal envoye : {info['nom']} {tf} {last['type']}")
            except Exception as e:
                print(f"Erreur scan {key} {tf}: {e}")
            await asyncio.sleep(1)


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
                f"Alertes automatiques actives !\n\n"
                f"Commandes :\n"
                f"fc gold 1m\n"
                f"fc gold 5m\n"
                f"fc gold 1h\n"
                f"fc btc 1m\n"
                f"fc silver 1d"
            )
            return

        if texte.startswith("fc "):
            parts = texte.split()
            actif = parts[1] if len(parts) > 1 else "gold"
            tf    = parts[2] if len(parts) > 2 else "1m"

            if actif not in ACTIFS:
                await update.message.reply_text(f"Actif inconnu. Disponibles : {', '.join(ACTIFS.keys())}")
                return
            if tf not in TIMEFRAMES:
                await update.message.reply_text(f"Timeframe inconnu. Disponibles : {', '.join(TIMEFRAMES.keys())}")
                return

            info = ACTIFS[actif]
            cfg  = TIMEFRAMES[tf]
            msg  = await update.message.reply_text(f"Generation du graphique {info['nom']} {tf.upper()}...")

            df = get_data(info["symbol"], cfg["period"], cfg["interval"])
            if df.empty:
                await msg.edit_text("Donnees indisponibles. Reessaie.")
                return

            df_st    = calc_supertrend(df)
            sigs     = get_signaux(df_st)
            last     = sigs[-1] if sigs else None
            buf      = make_chart(df, info["nom"], tf)
            lp       = df_st["Close"].iloc[-1]
            tendance = "Haussier" if df_st["dir"].iloc[-1] == 1 else "Baissier"
            sig_txt  = ""
            if last:
                e = "ACHAT" if last["type"] == "BUY" else "VENTE"
                sig_txt = f"\nDernier signal : {e} @ {last['prix']:.3f} - {last['date'].strftime('%d/%m %H:%M')}"
            caption = (
                f"{info['nom']} - {tf.upper()}\n"
                f"Prix : {lp:.3f}\n"
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


def main():
    print("GoldSignalBot demarre...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT, on_message))
    scheduler = AsyncIOScheduler(timezone="Europe/Paris")
    scheduler.add_job(scan, "interval", minutes=SCAN_MINUTES, args=[app], id="scan")

    async def post_init(application):
        scheduler.start()
        print(f"Pret ! Scan toutes les {SCAN_MINUTES} minute(s)")

    app.post_init = post_init
    app.run_polling()


if __name__ == "__main__":
    main()
