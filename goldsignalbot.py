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
from datetime import datetime, timezone
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, MessageHandler, CallbackQueryHandler,
    CommandHandler, filters, ContextTypes
)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
VOTRE_CHAT_ID  = int(os.environ.get("VOTRE_CHAT_ID", "0"))
SCAN_MINUTES   = 1
FUSEAU         = "Africa/Porto-Novo"  # heure du Bénin (UTC+1)

TIMEFRAMES = {
    "1m": {"range": "1d",  "interval": "1m"},
    "5m": {"range": "5d",  "interval": "5m"},
    "1h": {"range": "1mo", "interval": "1h"},
    "1d": {"range": "1y",  "interval": "1d"},
}

# Mémorise la direction précédente pour détecter le CHANGEMENT en temps réel
direction_precedente = {}


def get_data(interval, range_):
    url     = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    params  = {"range": range_, "interval": interval, "includePrePost": "false"}
    try:
        r   = requests.get(url, headers=headers, params=params, timeout=15)
        res = r.json()["chart"]["result"][0]
        q   = res["indicators"]["quote"][0]
        ts  = res["timestamp"]
        df  = pd.DataFrame({
            "Open":   q["open"],
            "High":   q["high"],
            "Low":    q["low"],
            "Close":  q["close"],
            "Volume": q.get("volume", [0]*len(ts)),
        }, index=pd.to_datetime(ts, unit="s", utc=True))
        df.dropna(inplace=True)
        return df if len(df) >= 15 else pd.DataFrame()
    except Exception as e:
        print(f"Erreur get_data {interval}: {e}")
        return pd.DataFrame()


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


def calc_sl_tp(prix, atr, type_signal):
    if type_signal == "BUY":
        return round(prix - atr * 1.5, 3), round(prix + atr * 3.0, 3)
    else:
        return round(prix + atr * 1.5, 3), round(prix - atr * 3.0, 3)


def heure_locale():
    """Retourne l'heure actuelle au format local Bénin"""
    from datetime import timezone as tz
    import zoneinfo
    try:
        zone = zoneinfo.ZoneInfo(FUSEAU)
        return datetime.now(zone).strftime("%d/%m/%Y %H:%M")
    except:
        return datetime.utcnow().strftime("%d/%m/%Y %H:%M") + " UTC"


def make_chart(df, tf, sl=None, tp=None):
    df   = calc_supertrend(df).tail(80).copy()
    n    = len(df)

    # Flèches uniquement sur les changements de direction
    sigs = []
    for i in range(1, n):
        pd_, cd = df["dir"].iloc[i-1], df["dir"].iloc[i]
        if pd_ == -1 and cd == 1:
            sigs.append({"i": i, "type": "BUY",  "prix": float(df["Close"].iloc[i])})
        elif pd_ == 1 and cd == -1:
            sigs.append({"i": i, "type": "SELL", "prix": float(df["Close"].iloc[i])})

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8),
        gridspec_kw={"height_ratios": [3, 1]}, facecolor="#131722")
    ax1.set_facecolor("#131722")
    ax2.set_facecolor("#131722")
    fig.subplots_adjust(hspace=0.05)

    # Bougies
    for i in range(n):
        o = float(df["Open"].iloc[i])
        h = float(df["High"].iloc[i])
        l = float(df["Low"].iloc[i])
        c = float(df["Close"].iloc[i])
        color = "#26a69a" if c >= o else "#ef5350"
        ax1.plot([i, i], [l, h], color=color, linewidth=0.8, zorder=1)
        ax1.add_patch(plt.Rectangle((i-0.35, min(o,c)), 0.7,
            max(abs(c-o), 0.001), color=color, zorder=2))

    # Supertrend
    for i in range(1, n):
        sv = df["st"].iloc[i]
        sp = df["st"].iloc[i-1]
        if not np.isnan(sv) and not np.isnan(sp):
            col = "#00bcd4" if df["dir"].iloc[i] == 1 else "#ff5252"
            ax1.plot([i-1, i], [float(sp), float(sv)], color=col, linewidth=1.8, zorder=3)

    # Flèches
    rng = float(df["High"].max() - df["Low"].min())
    off = max(rng * 0.012, 0.1)
    for sig in sigs:
        idx, prix = sig["i"], sig["prix"]
        if sig["type"] == "BUY":
            ax1.annotate("", xy=(idx, prix-off*0.6), xytext=(idx, prix-off*1.8),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2, mutation_scale=14), zorder=5)
            ax1.text(idx, prix-off*2.2, f"{prix:.3f}", color="white",
                fontsize=7.5, ha="center", va="top", fontweight="bold", zorder=5)
        else:
            ax1.annotate("", xy=(idx, prix+off*0.6), xytext=(idx, prix+off*1.8),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2, mutation_scale=14), zorder=5)
            ax1.text(idx, prix+off*2.2, f"{prix:.3f}", color="white",
                fontsize=7.5, ha="center", va="bottom", fontweight="bold", zorder=5)

    # SL / TP
    if sl:
        ax1.axhline(sl, color="#ff4444", lw=1.2, linestyle="--", alpha=0.9, zorder=4)
        ax1.text(n-2, sl, f" SL {sl:.3f}", color="#ff4444", fontsize=8.5, va="bottom", fontweight="bold")
    if tp:
        ax1.axhline(tp, color="#00e676", lw=1.2, linestyle="--", alpha=0.9, zorder=4)
        ax1.text(n-2, tp, f" TP {tp:.3f}", color="#00e676", fontsize=8.5, va="bottom", fontweight="bold")

    # Prix actuel
    lp = float(df["Close"].iloc[-1])
    ax1.axhline(lp, color="#ff4081", lw=0.9, linestyle=":", alpha=0.7, zorder=4)
    ax1.text(n+0.3, lp, f"  {lp:.3f}", color="white", fontsize=9, fontweight="bold", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ef5350", edgecolor="none"), zorder=5)

    # Volume
    for i in range(n):
        v = float(df["Volume"].iloc[i]) if df["Volume"].iloc[i] else 0
        c = float(df["Close"].iloc[i])
        o = float(df["Open"].iloc[i])
        ax2.bar(i, v, color="#26a69a55" if c >= o else "#ef535055", width=0.7)

    # Labels X
    step  = max(1, n // 8)
    ticks = list(range(0, n, step))
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([""] * len(ticks))
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(
        [df.index[i].strftime("%H:%M") for i in ticks],
        color="#888", fontsize=8
    )

    for ax in [ax1, ax2]:
        ax.set_facecolor("#131722")
        ax.tick_params(colors="#888888")
        for spine in ax.spines.values():
            spine.set_color("#2a2a3a")
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", colors="#888888", labelsize=8)
        ax.grid(axis="y", color="#2a2a3a", linestyle=":", linewidth=0.5)
        ax.set_xlim(-1, n+3)

    ax1.set_title(f"  Gold - {tf.upper()} - {heure_locale()}",
        color="#cccccc", fontsize=11, loc="left", pad=8)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#131722")
    plt.close(fig)
    buf.seek(0)
    return buf


def menu():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🥇 Gold 1m", callback_data="chart_1m"),
            InlineKeyboardButton("🥇 Gold 5m", callback_data="chart_5m"),
        ],
        [
            InlineKeyboardButton("🥇 Gold 1h", callback_data="chart_1h"),
            InlineKeyboardButton("🥇 Gold 1j", callback_data="chart_1d"),
        ],
    ])


async def scan(app):
    """
    Vérifie UNIQUEMENT si la direction du Supertrend
    vient de changer sur la DERNIÈRE bougie fermée.
    Si oui → signal en temps réel. Sinon → rien.
    """
    print(f"Scan {heure_locale()}")
    for tf in ["1m", "5m"]:
        try:
            cfg = TIMEFRAMES[tf]
            df  = get_data(cfg["interval"], cfg["range"])
            if df.empty or len(df) < 15:
                continue

            df_st = calc_supertrend(df)

            # Direction actuelle et précédente sur les 2 dernières bougies FERMÉES
            # On ignore la dernière bougie (en cours de formation)
            dir_actuelle  = int(df_st["dir"].iloc[-2])
            dir_avant     = int(df_st["dir"].iloc[-3])
            prix_entree   = float(df_st["Close"].iloc[-2])
            atr_actuel    = float(df_st["atr"].iloc[-2])
            heure_bougie  = df_st.index[-2]

            cle = f"gold_{tf}"

            # Pas de changement → on ne fait rien
            if dir_actuelle == dir_avant:
                direction_precedente[cle] = dir_actuelle
                continue

            # Changement détecté mais déjà envoyé
            if direction_precedente.get(cle) == dir_actuelle:
                continue

            direction_precedente[cle] = dir_actuelle

            type_signal = "BUY" if dir_actuelle == 1 else "SELL"
            sl, tp      = calc_sl_tp(prix_entree, atr_actuel, type_signal)
            buf         = make_chart(df, tf, sl=sl, tp=tp)
            emoji       = "🟢" if type_signal == "BUY" else "🔴"
            action      = "ACHAT ▲" if type_signal == "BUY" else "VENTE ▼"
            maintenant  = heure_locale()

            caption = (
                f"{emoji} *Signal {action}*\n"
                f"🥇 *Gold — {tf.upper()}*\n\n"
                f"💰 *Entree :* `{prix_entree:.3f}`\n"
                f"🛑 *Stop Loss :* `{sl:.3f}`\n"
                f"🎯 *Take Profit :* `{tp:.3f}`\n"
                f"📊 *Risk/Reward :* `1 : 2`\n\n"
                f"🕐 *Heure :* `{maintenant}`\n\n"
                f"_Fais ta propre analyse avant de trader._"
            )
            await app.bot.send_photo(
                chat_id=VOTRE_CHAT_ID, photo=buf,
                caption=caption, parse_mode="Markdown",
                reply_markup=menu()
            )
            print(f"✅ Signal REEL Gold {tf} {type_signal} @ {prix_entree} | SL:{sl} TP:{tp}")

        except Exception as e:
            print(f"Erreur scan {tf}: {e}")
        await asyncio.sleep(1)


async def afficher_graphique(message, tf):
    cfg = TIMEFRAMES.get(tf, TIMEFRAMES["5m"])
    df  = get_data(cfg["interval"], cfg["range"])
    if df.empty:
        await message.reply_text(
            "Marche ferme ou donnees indisponibles.\n"
            "Gold trade du lundi 01h au vendredi 23h.",
            reply_markup=menu()
        )
        return
    df_st    = calc_supertrend(df)
    dir_     = int(df_st["dir"].iloc[-1])
    lp       = float(df_st["Close"].iloc[-1])
    atr      = float(df_st["atr"].iloc[-1])
    tendance = "🟢 Haussier" if dir_ == 1 else "🔴 Baissier"

    # Dernier changement de direction
    sl, tp, sig_txt = None, None, ""
    for i in range(len(df_st)-1, 0, -1):
        if df_st["dir"].iloc[i] != df_st["dir"].iloc[i-1]:
            type_sig = "BUY" if df_st["dir"].iloc[i] == 1 else "SELL"
            prix_sig = float(df_st["Close"].iloc[i])
            atr_sig  = float(df_st["atr"].iloc[i])
            sl, tp   = calc_sl_tp(prix_sig, atr_sig, type_sig)
            e        = "ACHAT" if type_sig == "BUY" else "VENTE"
            sig_txt  = f"\nDernier signal : {e} @ {prix_sig:.3f}\nSL : {sl:.3f}  |  TP : {tp:.3f}"
            break

    buf     = make_chart(df, tf, sl=sl, tp=tp)
    caption = (
        f"🥇 Gold - {tf.upper()}\n"
        f"Prix actuel : {lp:.3f}\n"
        f"Tendance : {tendance}\n"
        f"Heure : {heure_locale()}"
        f"{sig_txt}"
    )
    await message.reply_photo(photo=buf, caption=caption, reply_markup=menu())


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Bonjour ! Je suis GoldSignalBot.\n\n"
        "Je surveille Gold en 1m et 5m.\n"
        "Des qu une fleche apparait je t envoie :\n"
        "graphique + Entree + SL + TP en temps reel\n\n"
        "Appuie sur un bouton :",
        reply_markup=menu()
    )


async def on_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    tf  = query.data.replace("chart_", "")
    msg = await query.message.reply_text(f"Generation graphique Gold {tf.upper()}...")
    await afficher_graphique(query.message, tf)
    try:
        await msg.delete()
    except:
        pass


async def on_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    await update.message.reply_text(
        "Appuie sur un bouton pour voir un graphique :",
        reply_markup=menu()
    )


def main():
    print("GoldSignalBot demarre...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(scan, "interval", minutes=SCAN_MINUTES, args=[app], id="scan")

    async def post_init(application):
        scheduler.start()
        print("Pret ! Surveillance Gold 1m et 5m en temps reel.")

    app.post_init = post_init
    app.run_polling()


if __name__ == "__main__":
    main()
