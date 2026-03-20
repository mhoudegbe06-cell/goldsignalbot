#!/usr/bin/env python3
import os
import asyncio
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf
from io import BytesIO
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, MessageHandler, CallbackQueryHandler,
    CommandHandler, filters, ContextTypes
)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
VOTRE_CHAT_ID  = int(os.environ.get("VOTRE_CHAT_ID", "0"))
SCAN_MINUTES   = 1

TIMEFRAMES = {
    "1m": {"period": "1d",  "interval": "1m"},
    "5m": {"period": "1d",  "interval": "5m"},
    "1h": {"period": "5d",  "interval": "1h"},
    "1d": {"period": "60d", "interval": "1d"},
}

derniers_signaux = {}


# ════════════════════════════════════════════
#  DONNÉES
# ════════════════════════════════════════════

def get_data(interval, period):
    try:
        df = yf.download("GC=F", period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 15:
            return pd.DataFrame()
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        df.dropna(inplace=True)
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        return df
    except Exception as e:
        print(f"Erreur get_data {interval}: {e}")
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
            sigs.append({"date": df.index[i], "type": "BUY",
                         "prix": float(df["Close"].iloc[i]), "atr": float(df["atr"].iloc[i])})
        elif pd_ == 1 and cd == -1:
            sigs.append({"date": df.index[i], "type": "SELL",
                         "prix": float(df["Close"].iloc[i]), "atr": float(df["atr"].iloc[i])})
    return sigs


def calc_sl_tp(sig):
    prix = sig["prix"]
    atr  = sig["atr"]
    if sig["type"] == "BUY":
        return round(prix - atr * 1.5, 3), round(prix + atr * 3.0, 3)
    else:
        return round(prix + atr * 1.5, 3), round(prix - atr * 3.0, 3)


# ════════════════════════════════════════════
#  GRAPHIQUE
# ════════════════════════════════════════════

def make_chart(df, tf, sl=None, tp=None):
    df   = calc_supertrend(df).tail(80).copy()
    sigs = get_signaux(df)
    n    = len(df)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8),
        gridspec_kw={"height_ratios": [3, 1]}, facecolor="#131722")
    ax1.set_facecolor("#131722")
    ax2.set_facecolor("#131722")
    fig.subplots_adjust(hspace=0.05)

    for i in range(n):
        o = float(df["Open"].iloc[i])
        h = float(df["High"].iloc[i])
        l = float(df["Low"].iloc[i])
        c = float(df["Close"].iloc[i])
        color = "#26a69a" if c >= o else "#ef5350"
        ax1.plot([i, i], [l, h], color=color, linewidth=0.8, zorder=1)
        ax1.add_patch(plt.Rectangle((i-0.35, min(o,c)), 0.7,
            max(abs(c-o), 0.001), color=color, zorder=2))

    for i in range(1, n):
        sv = df["st"].iloc[i]
        sp = df["st"].iloc[i-1]
        if not np.isnan(sv) and not np.isnan(sp):
            col = "#00bcd4" if df["dir"].iloc[i] == 1 else "#ff5252"
            ax1.plot([i-1, i], [float(sp), float(sv)], color=col, linewidth=1.8, zorder=3)

    rng = float(df["High"].max() - df["Low"].min())
    off = rng * 0.012
    for sig in sigs:
        if sig["date"] not in df.index:
            continue
        idx  = df.index.get_loc(sig["date"])
        prix = sig["prix"]
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

    if sl:
        ax1.axhline(sl, color="#ff4444", lw=1.2, linestyle="--", alpha=0.9, zorder=4)
        ax1.text(n-2, sl, f" SL {sl:.3f}", color="#ff4444", fontsize=8.5, va="bottom", fontweight="bold")
    if tp:
        ax1.axhline(tp, color="#00e676", lw=1.2, linestyle="--", alpha=0.9, zorder=4)
        ax1.text(n-2, tp, f" TP {tp:.3f}", color="#00e676", fontsize=8.5, va="bottom", fontweight="bold")

    lp = float(df["Close"].iloc[-1])
    ax1.axhline(lp, color="#ff4081", lw=0.9, linestyle=":", alpha=0.7, zorder=4)
    ax1.text(n+0.3, lp, f"  {lp:.3f}", color="white", fontsize=9, fontweight="bold", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ef5350", edgecolor="none"), zorder=5)

    for i in range(n):
        o = float(df["Open"].iloc[i])
        c = float(df["Close"].iloc[i])
        v = float(df["Volume"].iloc[i]) if df["Volume"].iloc[i] else 0
        ax2.bar(i, v, color="#26a69a55" if c >= o else "#ef535055", width=0.7)

    step  = max(1, n // 8)
    ticks = list(range(0, n, step))
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([""] * len(ticks))
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([df.index[i].strftime("%H:%M") for i in ticks], color="#888", fontsize=8)

    for ax in [ax1, ax2]:
        ax.set_facecolor("#131722")
        ax.tick_params(colors="#888888")
        for spine in ax.spines.values():
            spine.set_color("#2a2a3a")
        ax.yaxis.tick_right()
        ax.tick_params(axis="y", colors="#888888", labelsize=8)
        ax.grid(axis="y", color="#2a2a3a", linestyle=":", linewidth=0.5)
        ax.set_xlim(-1, n+3)

    ax1.set_title(f"  Gold - {tf.upper()} - Temps reel",
        color="#cccccc", fontsize=11, loc="left", pad=8)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#131722")
    plt.close(fig)
    buf.seek(0)
    return buf


# ════════════════════════════════════════════
#  CLAVIER BOUTONS
# ════════════════════════════════════════════

def menu_principal():
    keyboard = [
        [
            InlineKeyboardButton("🥇 Gold 1m", callback_data="chart_1m"),
            InlineKeyboardButton("🥇 Gold 5m", callback_data="chart_5m"),
        ],
        [
            InlineKeyboardButton("🥇 Gold 1h", callback_data="chart_1h"),
            InlineKeyboardButton("🥇 Gold 1j", callback_data="chart_1d"),
        ],
        [
            InlineKeyboardButton("📊 Signal actuel", callback_data="signal_now"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


# ════════════════════════════════════════════
#  SCAN AUTOMATIQUE
# ════════════════════════════════════════════

async def scan(app):
    print(f"Scan {datetime.now().strftime('%H:%M:%S')}")
    for tf in ["1m", "5m"]:
        try:
            cfg = TIMEFRAMES[tf]
            df  = get_data(cfg["interval"], cfg["period"])
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
            buf    = make_chart(df, tf, sl=sl, tp=tp)
            emoji  = "🟢" if last["type"] == "BUY" else "🔴"
            action = "ACHAT ▲" if last["type"] == "BUY" else "VENTE ▼"
            caption = (
                f"{emoji} *Signal {action}*\n"
                f"🥇 *Gold — {tf.upper()}*\n\n"
                f"💰 *Entree :* `{last['prix']:.3f}`\n"
                f"🛑 *Stop Loss :* `{sl:.3f}`\n"
                f"🎯 *Take Profit :* `{tp:.3f}`\n"
                f"📊 *Risk/Reward :* `1 : 2`\n\n"
                f"🕐 {last['date'].strftime('%d/%m %H:%M')}\n\n"
                f"_Fais ta propre analyse avant de trader._"
            )
            await app.bot.send_photo(
                chat_id=VOTRE_CHAT_ID,
                photo=buf,
                caption=caption,
                parse_mode="Markdown",
                reply_markup=menu_principal()
            )
            print(f"Signal Gold {tf} {last['type']} SL:{sl} TP:{tp}")
        except Exception as e:
            print(f"Erreur scan {tf}: {e}")
        await asyncio.sleep(1)


# ════════════════════════════════════════════
#  HANDLERS
# ════════════════════════════════════════════

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Bonjour ! Je suis GoldSignalBot.\n\n"
        f"Je surveille Gold en 1m et 5m.\n"
        f"Des qu une fleche apparait tu recois :\n"
        f"graphique + Entree + SL + TP\n\n"
        f"Appuie sur un bouton pour voir un graphique :",
        reply_markup=menu_principal()
    )


async def on_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data

    if data == "signal_now":
        # Montre le signal le plus recent sur 5m
        tf  = "5m"
        cfg = TIMEFRAMES[tf]
        msg = await query.message.reply_text("Analyse en cours...")
        df  = get_data(cfg["interval"], cfg["period"])
        if df.empty:
            await msg.edit_text("Donnees indisponibles. Reessaie.")
            return
        df_st    = calc_supertrend(df)
        sigs     = get_signaux(df_st)
        last     = sigs[-1] if sigs else None
        sl, tp   = calc_sl_tp(last) if last else (None, None)
        buf      = make_chart(df, tf, sl=sl, tp=tp)
        lp       = float(df_st["Close"].iloc[-1])
        tendance = "Haussier" if df_st["dir"].iloc[-1] == 1 else "Baissier"
        sig_txt  = ""
        if last:
            e = "ACHAT" if last["type"] == "BUY" else "VENTE"
            sig_txt = f"\nSignal : {e} @ {last['prix']:.3f}\nSL : {sl:.3f}  |  TP : {tp:.3f}"
        caption = f"Gold - {tf.upper()}\nPrix : {lp:.3f}\nTendance : {tendance}{sig_txt}"
        await msg.delete()
        await query.message.reply_photo(photo=buf, caption=caption, reply_markup=menu_principal())
        return

    if data.startswith("chart_"):
        tf  = data.replace("chart_", "")
        cfg = TIMEFRAMES.get(tf)
        if not cfg:
            return
        msg = await query.message.reply_text(f"Generation graphique Gold {tf.upper()}...")
        df  = get_data(cfg["interval"], cfg["period"])
        if df.empty:
            await msg.edit_text("Donnees indisponibles. Reessaie.")
            return
        df_st    = calc_supertrend(df)
        sigs     = get_signaux(df_st)
        last     = sigs[-1] if sigs else None
        sl, tp   = calc_sl_tp(last) if last else (None, None)
        buf      = make_chart(df, tf, sl=sl, tp=tp)
        lp       = float(df_st["Close"].iloc[-1])
        tendance = "Haussier" if df_st["dir"].iloc[-1] == 1 else "Baissier"
        sig_txt  = ""
        if last:
            e = "ACHAT" if last["type"] == "BUY" else "VENTE"
            sig_txt = f"\nSignal : {e} @ {last['prix']:.3f}\nSL : {sl:.3f}  |  TP : {tp:.3f}"
        caption = f"Gold - {tf.upper()}\nPrix : {lp:.3f}\nTendance : {tendance}{sig_txt}"
        await msg.delete()
        await query.message.reply_photo(photo=buf, caption=caption, reply_markup=menu_principal())


async def on_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    await update.message.reply_text(
        "Appuie sur un bouton pour voir un graphique :",
        reply_markup=menu_principal()
    )


# ════════════════════════════════════════════
#  LANCEMENT
# ════════════════════════════════════════════

def main():
    print("GoldSignalBot demarre...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    scheduler = AsyncIOScheduler(timezone="Europe/Paris")
    scheduler.add_job(scan, "interval", minutes=SCAN_MINUTES, args=[app], id="scan")

    async def post_init(application):
        scheduler.start()
        print("Pret ! Surveillance Gold 1m et 5m active.")

    app.post_init = post_init
    app.run_polling()


if __name__ == "__main__":
    main()
