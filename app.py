"""
Rut-Trigger Predictor (Streamlit app)

Predicts when a flexible pavement section will first reach a 10 mm rut depth, from the
XGBoost-Cox model of the manuscript "Within-region skill, cross-region loss:
censoring-aware machine learning for time-to-trigger rutting prognosis from LTPP data".

Run:  streamlit run app.py      (rut_trigger_model.json must sit next to this file)
"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rut_engine import RutTriggerModel

MODEL_FILE = Path(__file__).with_name("rut_trigger_model.json")
GROUPS = [("Condition", "🛣️"), ("Traffic", "🚚"), ("Climate", "🌦️"), ("Structure", "🧱"), ("Materials", "⚗️")]

# palette
NAVY, TEAL, SKY, INK = "#0B2545", "#13807A", "#2E86DE", "#1B1F2A"
TIER = {"Low": ("#1E8E5A", "#E7F6EE"), "Moderate": ("#C98A0B", "#FFF5DC"), "High": ("#C0392B", "#FDECEA")}
COV = {"Typical": ("#1E8E5A", "#E7F6EE"), "Near": ("#C98A0B", "#FFF5DC"), "Outside": ("#C0392B", "#FDECEA")}
HORIZON = {5: "#2E86DE", 10: "#8E44AD", 15: "#D35400"}

st.set_page_config(page_title="Rut-Trigger Predictor", page_icon="🛣️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(f"""
<style>
html, body, [class*="css"] {{ font-family: "Source Sans Pro", "Segoe UI", sans-serif; }}
.block-container {{ padding-top: 2.6rem; max-width: 1350px; }}
section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, {NAVY} 0%, #12355B 100%); }}
section[data-testid="stSidebar"] * {{ color: #F2F6FA; }}
section[data-testid="stSidebar"] input {{ color: {INK} !important; background: #FFFFFF !important; }}
section[data-testid="stSidebar"] [data-testid="stExpander"] {{ background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.18); border-radius: 12px; }}
section[data-testid="stSidebar"] [data-testid="stExpander"] details,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary {{ background: transparent !important; }}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {{ background: rgba(255,255,255,0.08) !important; }}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary p,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary span {{ color: #FFFFFF !important; font-weight: 700;
    font-size: 1.02rem; }}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary svg {{ fill: #FFFFFF !important; color: #FFFFFF !important; }}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] button {{ background: #DCE6F2 !important; }}
section[data-testid="stSidebar"] [data-testid="stNumberInput"] button svg {{ fill: #0B2545 !important; color: #0B2545 !important; }}
section[data-testid="stSidebar"] [data-testid="stNumberInputContainer"] {{ background: #FFFFFF; border-radius: 8px; }}
section[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg {{ stroke: #BFD7EE !important; }}
section[data-testid="stSidebar"] button[kind="secondary"] {{ background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.35); }}
section[data-testid="stSidebar"] button[kind="secondary"] p {{ color: #FFFFFF; }}
div[data-testid="stFormSubmitButton"] button {{ width: 100%; height: 3.1rem; border: 0; border-radius: 12px;
    background: linear-gradient(90deg, #F39C12, #E74C3C); box-shadow: 0 6px 18px rgba(231,76,60,0.35); }}
div[data-testid="stFormSubmitButton"] button p {{ color: #FFFFFF !important; font-size: 1.12rem; font-weight: 800;
    letter-spacing: 0.3px; }}
div[data-testid="stFormSubmitButton"] button:hover {{ filter: brightness(1.08); transform: translateY(-1px); }}
.hero {{ background: linear-gradient(115deg, {NAVY} 0%, #134074 45%, {TEAL} 100%); color: #FFF;
    border-radius: 18px; padding: 26px 32px; margin-bottom: 18px; box-shadow: 0 10px 30px rgba(11,37,69,0.25); }}
.hero h1 {{ color: #FFF; font-size: 2.3rem; margin: 0 0 4px 0; line-height: 1.1; }}
.hero p {{ color: #D6E6F5; margin: 0; font-size: 1.02rem; }}
.chips span {{ display: inline-block; margin: 12px 8px 0 0; padding: 4px 12px; border-radius: 20px;
    background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.3); font-size: 0.86rem; }}
.card {{ background: #FFFFFF; border-radius: 16px; padding: 18px 20px; box-shadow: 0 4px 16px rgba(20,40,80,0.10);
    border: 1px solid #E6ECF3; height: 100%; }}
.big {{ background: linear-gradient(135deg, #0B2545, #1D4E89); color: #FFF; border-radius: 16px; padding: 20px 24px;
    box-shadow: 0 8px 22px rgba(11,37,69,0.28); }}
.big .lab {{ color: #BFD7EE; font-size: 0.95rem; font-weight: 600; }}
.big .val {{ font-size: 3.1rem; font-weight: 800; line-height: 1.05; }}
.big .val small {{ font-size: 1.2rem; font-weight: 600; color: #D6E6F5; }}
.big .sub {{ color: #D6E6F5; font-size: 0.92rem; margin-top: 6px; }}
.pcard {{ border-radius: 16px; padding: 16px 18px; color: #FFF; box-shadow: 0 6px 16px rgba(0,0,0,0.12); }}
.pcard .lab {{ font-size: 0.92rem; font-weight: 600; opacity: 0.92; }}
.pcard .val {{ font-size: 2.2rem; font-weight: 800; line-height: 1.1; }}
.pcard .ci {{ font-size: 0.86rem; opacity: 0.92; }}
.pill {{ display: inline-block; padding: 7px 14px; border-radius: 22px; font-weight: 700; margin: 4px 8px 4px 0;
    font-size: 0.95rem; }}
.note {{ background: #F4F8FC; border-left: 0; border-radius: 12px; padding: 12px 16px; color: #33475B;
    font-size: 0.93rem; }}
.tile {{ border-radius: 16px; padding: 18px; color: #FFF; min-height: 168px; }}
.tile h4 {{ color: #FFF; margin: 0 0 6px 0; }}
.tile p {{ margin: 0; font-size: 0.93rem; color: #F1F5F9; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{ background: #EEF3F8; border-radius: 10px 10px 0 0; padding: 8px 18px; font-weight: 600; }}
.stTabs [aria-selected="true"] {{ background: {NAVY}; color: #FFF !important; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str) -> RutTriggerModel:
    return RutTriggerModel(path)


def fmt_num(v: float) -> str:
    a = abs(v)
    if a >= 10000:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:.1f}"
    return f"{v:.2f}".rstrip("0").rstrip(".")


def pct(p: float) -> str:
    return f"{p * 100:.1f}%" if p < 0.1 else f"{p * 100:.0f}%"


def yrs(t, t_max) -> str:
    return f"> {t_max:.0f}" if t is None else f"{t:.1f}"


def pill(text, key, table):
    fg, bg = table[key]
    return f"<span class='pill' style='color:{fg};background:{bg};border:1.5px solid {fg}'>{text}</span>"


if not MODEL_FILE.exists():
    st.error("Model file **rut_trigger_model.json** not found. Run 10_rut_trigger_tool.py and copy the file "
             "from its output folder (10_tool) next to app.py.")
    st.stop()

M = load_model(str(MODEL_FILE))
FEAT = {f["input"]: f for f in M.features}
ss = st.session_state
ss.setdefault("result", None)
ss.setdefault("history", [])


@st.cache_data(show_spinner=False)
def typical_curve(path: str):
    m = load_model(path).predict(load_model(path).typical).margin
    return load_model(path).curve(m)[:2]


def tier_position(p: float) -> float:
    """Map a 10-year probability onto a 0-3 scale with equal width for each tier."""
    q1, q2 = M.tertiles
    if p < q1:
        return p / q1
    if p < q2:
        return 1 + (p - q1) / (q2 - q1)
    return 2 + min((p - q2) / (1 - q2), 1.0)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🛣️ Section inputs")
    st.caption("Leave a field empty if unknown: the training median is used and flagged.")
    b1, b2 = st.columns(2)
    if b1.button("Load typical", width="stretch"):
        for f in M.features:
            ss[f"in_{f['input']}"] = float(f["typical"])
    if b2.button("Clear", width="stretch"):
        for f in M.features:
            ss[f"in_{f['input']}"] = None
        ss.result = None
    with st.form("inputs", border=False):
        ss.setdefault("sec_name", "Section 1")
        st.text_input("Section name", key="sec_name")
        raw = {}
        for g, icon in GROUPS:
            with st.expander(f"{icon}  {g}", expanded=g in ("Condition", "Traffic")):
                for f in [f for f in M.features if f["group"] == g]:
                    raw[f["input"]] = st.number_input(
                        f"{f['label']} ({f['unit']})", value=None, key=f"in_{f['input']}",
                        placeholder=f"typical {fmt_num(f['typical'])}", format="%g",
                        step=1.0 if f["input"] == "n_layers" else None,
                        help=f"LTPP range {fmt_num(f['min'])} – {fmt_num(f['max'])} {f['unit']}")
        go_btn = st.form_submit_button("🔮  PREDICT", width="stretch")
    st.caption(f"Model: XGBoost-Cox · {M.meta['n_cycles']:,} LTPP cycles · 200 bootstrap refits")

if go_btn:
    rut = raw.get("rut0_mm")
    if rut is None:
        ss.result = {"error": "Enter at least the current rut depth, or click **Load typical**."}
    elif rut >= M.threshold:
        ss.result = {"error": f"The entered rut depth is already {M.threshold:g} mm or more: the trigger has been reached."}
    else:
        rawn = {k: (float("nan") if v is None else float(v)) for k, v in raw.items()}
        with st.spinner("Running the model and its 200 bootstrap refits…"):
            r = M.predict(rawn)
        ss.result = {"r": r, "raw": rawn, "name": ss.sec_name}
        ss.history.insert(0, {"Time": datetime.now().strftime("%H:%M:%S"), "Section": ss.sec_name,
                              "Rut depth (mm)": rut, "Years to 10 mm": yrs(r.t50, M.t_max),
                              "P(5 yr)": round(r.p5, 3), "P(10 yr)": round(r.p10, 3),
                              "Band (10 yr)": f"{r.ci10[0]:.2f}–{r.ci10[1]:.2f}",
                              "P(15 yr)": round(r.p15, 3), "Risk tier": r.tier, "Coverage": r.coverage})

# ------------------------------------------------------------------ hero
st.markdown(f"""
<div class="hero"><h1>Rut-Trigger Predictor</h1>
<p>When will this pavement section reach a {M.threshold:g} mm rut depth? Survival-model prediction with
uncertainty bands, risk tier and data-coverage check.</p>
<div class="chips"><span>XGBoost-Cox survival model</span><span>{M.meta['n_cycles']:,} LTPP cycles · {M.meta['n_events']} events</span>
<span>Cross-validated Harrell's C 0.85</span><span>Research prototype</span></div></div>
""", unsafe_allow_html=True)

tab_pred, tab_what, tab_hist, tab_batch, tab_about = st.tabs(
    ["📈 Prediction", "🎛️ What-if explorer", "🗂️ History", "📑 Batch (CSV)", "ℹ️ Model and guidance"])

# ------------------------------------------------------------------ prediction tab
with tab_pred:
    res = ss.result
    if res is None:
        st.markdown("#### Enter the section data in the left panel and press **PREDICT**")
        t1, t2, t3 = st.columns(3)
        for col, bg, h, p in [
            (t1, f"linear-gradient(135deg,{SKY},#1B4F72)", "⏱️ Time to trigger",
             "Predicted years until the rut depth reaches 10 mm, with 10% and 25% planning horizons."),
            (t2, "linear-gradient(135deg,#8E44AD,#5B2C6F)", "📊 Probability and uncertainty",
             "Chance of reaching 10 mm within 5, 10 and 15 years, with 90% bands from 200 model refits."),
            (t3, f"linear-gradient(135deg,{TEAL},#0E5E59)", "🧭 Risk tier and coverage",
             "Where the section sits among LTPP sections, and how similar it is to the training data.")]:
            col.markdown(f"<div class='tile' style='background:{bg}'><h4>{h}</h4><p>{p}</p></div>",
                         unsafe_allow_html=True)
        st.markdown("<div class='note' style='margin-top:16px'>Tip: click <b>Load typical</b> for a median LTPP "
                    "section, change a few values, then press <b>PREDICT</b>.</div>", unsafe_allow_html=True)
    elif "error" in res:
        st.warning(res["error"])
    else:
        r = res["r"]
        st.markdown(f"#### Results for **{res['name']}**")
        c1, c2, c3, c4 = st.columns([1.35, 1, 1, 1], gap="medium")
        sub = ("Time at which the probability of reaching 10 mm passes 50%" if r.t50 is not None
               else f"50% probability not reached within {M.t_max:.0f} years")
        plan = f"10% chance by {yrs(r.t10, M.t_max)} yr" + (
            f" · 25% by {yrs(r.t25, M.t_max)} yr" if r.t10 is not None else "")
        c1.markdown(f"<div class='big'><div class='lab'>Predicted time to 10 mm</div>"
                    f"<div class='val'>{yrs(r.t50, M.t_max)} <small>years</small></div>"
                    f"<div class='sub'>{sub}</div><div class='sub'><b>{plan}</b></div></div>",
                    unsafe_allow_html=True)
        for col, h, p, ci in [(c2, 5, r.p5, r.ci5), (c3, 10, r.p10, r.ci10), (c4, 15, r.p15, r.ci15)]:
            col.markdown(f"<div class='pcard' style='background:linear-gradient(135deg,{HORIZON[h]},{HORIZON[h]}CC)'>"
                         f"<div class='lab'>Within {h} years</div><div class='val'>{pct(p)}</div>"
                         f"<div class='ci'>90% band {pct(ci[0])} – {pct(ci[1])}</div></div>",
                         unsafe_allow_html=True)

        st.write("")
        tier_txt = f"Risk tier: {r.tier}" + (f" (band spans {r.tier_band})" if r.tier_band else "")
        st.markdown(pill(tier_txt, r.tier, TIER) + pill(f"Data coverage: {r.coverage}", r.coverage.split()[0], COV),
                    unsafe_allow_html=True)
        if r.imputed:
            st.caption("🔸 Unknown, training median used: " + ", ".join(r.imputed) + ".")
        if r.out_of_range:
            st.warning("Outside the LTPP training range: " + ", ".join(r.out_of_range) + ".")

        g1, g2 = st.columns([1.65, 1], gap="large")
        with g1:
            ts, p, lo, hi = M.curve(r.margin, r.boot_margins)
            tt, pt = typical_curve(str(MODEL_FILE))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.r_[ts, ts[::-1]], y=np.r_[hi, lo[::-1]], fill="toself",
                                     fillcolor="rgba(46,134,222,0.18)", line=dict(width=0),
                                     hoverinfo="skip", name="90% band"))
            fig.add_trace(go.Scatter(x=tt, y=pt, mode="lines", name="Typical LTPP section",
                                     line=dict(color="#95A5A6", width=2.2, dash="dash"),
                                     hovertemplate="%{x:.1f} yr: %{y:.0%}<extra>typical</extra>"))
            fig.add_trace(go.Scatter(x=ts, y=p, mode="lines", name=res["name"],
                                     line=dict(color=NAVY, width=4, shape="hv"),
                                     hovertemplate="%{x:.1f} yr: %{y:.1%}<extra></extra>"))
            for h, ph in [(5, r.p5), (10, r.p10), (15, r.p15)]:
                fig.add_trace(go.Scatter(x=[h], y=[ph], mode="markers+text", showlegend=False,
                                         marker=dict(size=13, color=HORIZON[h], line=dict(color="white", width=2)),
                                         text=[pct(ph)], textposition="top left",
                                         textfont=dict(color=HORIZON[h], size=13), hoverinfo="skip"))
            fig.add_hline(y=0.5, line=dict(color="#7F8C8D", width=1, dash="dot"),
                          annotation_text="50%", annotation_position="top left")
            if r.t50 is not None:
                fig.add_vline(x=r.t50, line=dict(color="#E74C3C", width=1.6, dash="dot"),
                              annotation_text=f"{r.t50:.1f} yr", annotation_position="top")
            fig.update_layout(title=dict(text="Probability of reaching 10 mm over time", font=dict(size=17)),
                              height=440, margin=dict(l=10, r=10, t=60, b=10), plot_bgcolor="#FBFCFE",
                              xaxis=dict(title="Years from survey", range=[0, M.t_max], dtick=5,
                                         gridcolor="#E5EAF0"),
                              yaxis=dict(title="Probability", range=[0, 1.02], tickformat=".0%",
                                         gridcolor="#E5EAF0"),
                              legend=dict(orientation="h", y=-0.18, x=0), font=dict(size=13))
            st.plotly_chart(fig, width="stretch")
        with g2:
            # risk-tier scale
            pos, lo_p, hi_p = tier_position(r.p10), tier_position(r.ci10[0]), tier_position(r.ci10[1])
            fr = go.Figure()
            for i, (name, (fg, bg)) in enumerate(TIER.items()):
                fr.add_shape(type="rect", x0=i, x1=i + 1, y0=0, y1=1, fillcolor=fg, opacity=0.85, line_width=0)
                fr.add_annotation(x=i + 0.5, y=0.5, text=f"<b>{name}</b>", showarrow=False,
                                  font=dict(color="white", size=14))
            fr.add_shape(type="line", x0=lo_p, x1=hi_p, y0=1.35, y1=1.35, line=dict(color=INK, width=4))
            fr.add_trace(go.Scatter(x=[pos], y=[1.35], mode="markers", marker=dict(size=18, color="#FFFFFF",
                                    line=dict(color=INK, width=3)), hoverinfo="skip", showlegend=False))
            fr.add_annotation(x=pos, y=1.8, text=f"P(10 yr) = {pct(r.p10)}", showarrow=False,
                              font=dict(size=13, color=INK))
            q1, q2 = M.tertiles
            fr.update_layout(title=dict(text="Position among LTPP sections", font=dict(size=16)), height=210,
                             margin=dict(l=10, r=10, t=50, b=30), plot_bgcolor="white",
                             xaxis=dict(range=[-0.05, 3.05], tickvals=[0, 1, 2, 3],
                                        ticktext=["0", pct(q1), pct(q2), "100%"], showgrid=False),
                             yaxis=dict(range=[-0.1, 2.1], visible=False))
            st.plotly_chart(fr, width="stretch")
            st.markdown(f"<div class='note'>The white marker is the section's 10-year probability and the bar "
                        f"its 90% band. Tier limits are the thirds of the LTPP training sections "
                        f"({pct(q1)} and {pct(q2)}). A band that crosses a limit means the tier is uncertain.</div>",
                        unsafe_allow_html=True)
            out = pd.DataFrame([{"section": res["name"], **{k: v for k, v in res["raw"].items()},
                                 "years_to_10mm": yrs(r.t50, M.t_max), "P_5yr": round(r.p5, 3),
                                 "P_10yr": round(r.p10, 3), "P_10yr_band_low": round(r.ci10[0], 3),
                                 "P_10yr_band_high": round(r.ci10[1], 3), "P_15yr": round(r.p15, 3),
                                 "risk_tier": r.tier, "data_coverage": r.coverage}])
            st.download_button("⬇️ Download this prediction", out.to_csv(index=False).encode(),
                               f"{res['name']}_prediction.csv", "text/csv", width="stretch")

# ------------------------------------------------------------------ what-if tab
with tab_what:
    res = ss.result
    if not res or "r" not in res:
        st.info("Make a prediction first; the explorer then varies one input of that section at a time.")
    else:
        st.markdown(f"#### How would the 10-year probability change for **{res['name']}**?")
        labels = {f["label"]: f["input"] for f in M.features}
        w1, w2 = st.columns([1, 2.2], gap="large")
        with w1:
            lab = st.selectbox("Input to vary", list(labels), index=0)
            code = labels[lab]
            f = FEAT[code]
            st.markdown(f"<div class='note'>All other inputs stay as entered. Range: LTPP minimum to maximum "
                        f"({fmt_num(f['min'])} – {fmt_num(f['max'])} {f['unit']}).</div>", unsafe_allow_html=True)
        lo_v, hi_v = f["min"], f["max"]
        if code == "rut0_mm":
            hi_v = min(hi_v, M.threshold - 0.1)
        grid = np.linspace(lo_v, hi_v, 40)
        base = dict(res["raw"])
        vals = []
        for v in grid:
            base[code] = v
            x, _ = M.to_vector(base)
            vals.append(M.prob(M.margin(x), 10.0))
        cur = res["raw"][code]
        fw = go.Figure()
        fw.add_trace(go.Scatter(x=grid, y=vals, mode="lines", line=dict(color="#8E44AD", width=4, shape="hv"),
                                fill="tozeroy", fillcolor="rgba(142,68,173,0.12)",
                                hovertemplate="%{x:.3g}: %{y:.1%}<extra></extra>", name="P(10 yr)"))
        if np.isfinite(cur):
            fw.add_vline(x=cur, line=dict(color="#E74C3C", width=2, dash="dash"),
                         annotation_text="entered value", annotation_position="top")
        for q, nm in zip(M.tertiles, ["Low | Moderate", "Moderate | High"]):
            fw.add_hline(y=q, line=dict(color="#7F8C8D", width=1, dash="dot"), annotation_text=nm,
                         annotation_position="top left", annotation_font=dict(size=11, color="#5D6D7E"))
        fw.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="#FBFCFE",
                         xaxis=dict(title=f"{f['label']} ({f['unit']})", gridcolor="#E5EAF0"),
                         yaxis=dict(title="Probability of reaching 10 mm within 10 years", tickformat=".0%",
                                    gridcolor="#E5EAF0", rangemode="tozero"), showlegend=False, font=dict(size=13))
        w2.plotly_chart(fw, width="stretch")
        st.caption("The explorer shows the model's response, not a causal effect. Steps reflect the tree structure "
                   "of the model; for correlated inputs (for example freeze index and temperature) a change in one "
                   "alone may describe an unrealistic section.")

# ------------------------------------------------------------------ history tab
with tab_hist:
    if not ss.history:
        st.info("Predictions made in this session appear here, so sections can be compared.")
    else:
        hd = pd.DataFrame(ss.history)

        def colour_tier(v):
            fg, bg = TIER.get(v, ("#333", "#FFF"))
            return f"background-color:{bg};color:{fg};font-weight:700"
        st.dataframe(hd.style.map(colour_tier, subset=["Risk tier"]).format(precision=3), hide_index=True, width="stretch")
        h1, h2 = st.columns(2)
        h1.download_button("⬇️ Download history", hd.to_csv(index=False).encode(), "prediction_history.csv",
                           "text/csv", width="stretch")
        if h2.button("🗑️ Clear history", width="stretch"):
            ss.history = []
            st.rerun()

# ------------------------------------------------------------------ batch tab
with tab_batch:
    st.markdown("#### Predict many sections from a CSV file")
    st.markdown("One row per section: a `section` column plus the field codes below. Empty cells count as unknown.")
    with st.expander("Field codes and LTPP ranges"):
        st.dataframe(pd.DataFrame([{"field code": f["input"], "description": f["label"], "unit": f["unit"],
                                    "LTPP range": f"{fmt_num(f['min'])} – {fmt_num(f['max'])}"}
                                   for f in M.features]), hide_index=True, width="stretch")
    tmpl = pd.DataFrame([{"section": "example", **{f["input"]: f["typical"] for f in M.features}}])
    st.download_button("⬇️ Download CSV template", tmpl.to_csv(index=False).encode(),
                       "rut_trigger_template.csv", "text/csv")
    up = st.file_uploader("Upload sections (CSV)", type="csv")
    if up is not None and st.button("🚀 Run batch prediction", type="primary"):
        df = pd.read_csv(up)
        df.columns = [c.strip() for c in df.columns]
        missing = [f["input"] for f in M.features if f["input"] not in df.columns]
        if missing:
            st.caption("Columns not found (treated as unknown): " + ", ".join(missing))
        rows, bar = [], st.progress(0.0)
        for i, row in df.iterrows():
            rw = {f["input"]: pd.to_numeric(row.get(f["input"], np.nan), errors="coerce") for f in M.features}
            o = {"section": row["section"] if "section" in df.columns else str(i + 1)}
            if not np.isfinite(rw["rut0_mm"]):
                o["years_to_10mm"] = "rut0_mm missing"
            elif rw["rut0_mm"] >= M.threshold:
                o["years_to_10mm"] = "already reached"
            else:
                p = M.predict(rw)
                o.update({"years_to_10mm": yrs(p.t50, M.t_max), "years_to_10pct": yrs(p.t10, M.t_max),
                          "P_5yr": round(p.p5, 3), "P_10yr": round(p.p10, 3),
                          "P_10yr_band_low": round(p.ci10[0], 3), "P_10yr_band_high": round(p.ci10[1], 3),
                          "P_15yr": round(p.p15, 3), "risk_tier": p.tier,
                          "tier_band": p.tier_band or "within one tier", "data_coverage": p.coverage,
                          "n_imputed": len(p.imputed)})
            rows.append(o)
            bar.progress((i + 1) / len(df), text=f"Predicting… {i + 1}/{len(df)}")
        bar.empty()
        ss.batch = pd.DataFrame(rows)
    if "batch" in ss and ss.batch is not None:
        bd = ss.batch
        if "risk_tier" in bd:
            counts = bd["risk_tier"].value_counts().reindex(list(TIER)).fillna(0)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Sections", len(bd))
            for col, t in zip([k2, k3, k4], TIER):
                col.markdown(f"<div class='pcard' style='background:{TIER[t][0]}'><div class='lab'>{t} risk</div>"
                             f"<div class='val'>{int(counts[t])}</div></div>", unsafe_allow_html=True)
            st.write("")

            def colour_tier(v):
                fg, bg = TIER.get(v, ("#333", "#FFF"))
                return f"background-color:{bg};color:{fg};font-weight:700"
            st.dataframe(bd.style.map(colour_tier, subset=["risk_tier"]).format(precision=3), hide_index=True, width="stretch")
        else:
            st.dataframe(bd, hide_index=True, width="stretch")
        st.download_button("⬇️ Download predictions", bd.to_csv(index=False).encode(),
                           "rut_trigger_predictions.csv", "text/csv")

# ------------------------------------------------------------------ about tab
with tab_about:
    a1, a2 = st.columns([1.4, 1], gap="large")
    with a1:
        st.markdown("#### How to read the results")
        st.markdown(f"""
- **Predicted time to 10 mm**: the time at which the probability of reaching 10 mm passes 50%.
  The 10% and 25% horizons suit agencies that plan treatment before the trigger.
- **Risk tier**: the 10-year probability compared with the LTPP training sections
  (Low below {pct(M.tertiles[0])}, Moderate below {pct(M.tertiles[1])}, High above).
- **90% band**: the spread of the probability over 200 refits of the model on resampled LTPP sections.
  It reflects uncertainty in the fitted model, not the full spread of outcomes for one section.
- **Data coverage**: similarity of the section to the training sections (mean distance to the {M.k} most
  similar ones). "Outside training data" means extrapolation. A "typical" label does not guarantee accuracy:
  sections of an unseen region lost accuracy in the study even when their inputs looked typical.
  Compare predictions with local outcomes before using the tool in a new region.
- Times are counted from the survey whose rut depth is entered. Rehabilitation is not modelled as a
  competing event.
""")
    with a2:
        st.markdown("#### Model")
        st.markdown(f"<div class='card'><b>{M.meta['model']}</b><br><br>Trained on {M.meta['n_cycles']:,} "
                    f"section-construction cycles ({M.meta['n_events']} reached {M.threshold:g} mm).<br>"
                    f"Time axis supported to {M.t_max:.0f} years.<br>Model file generated {M.meta['generated']}."
                    f"<br><br>{M.cv_perf}</div>", unsafe_allow_html=True)
    st.caption("Research prototype accompanying the manuscript “Within-region skill, cross-region loss: "
               "censoring-aware machine learning for time-to-trigger rutting prognosis from LTPP data”. "
               "Not a substitute for engineering judgement or local performance data.")
