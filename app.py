"""
Rut-Trigger Predictor (Streamlit app)

Predicts when a flexible pavement section will first reach a 10 mm rut depth,
from the XGBoost-Cox model of the manuscript
"Within-region skill, cross-region loss: censoring-aware machine learning for
time-to-trigger rutting prognosis from LTPP data".

Run locally:   streamlit run app.py
Needs rut_trigger_model.json (written by 10_rut_trigger_tool.py) in the same folder.
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rut_engine import RutTriggerModel

MODEL_FILE = Path(__file__).with_name("rut_trigger_model.json")
GROUPS = ["Condition", "Traffic", "Climate", "Structure", "Materials"]
CURVE, BAND, REF = "#1F4E79", "rgba(46,117,182,0.18)", "#8A8A8A"
TIER_COLOUR = {"Low": "#2E7D32", "Moderate": "#B8860B", "High": "#B23A1B"}
COV_COLOUR = {"Typical": "#2E7D32", "Near": "#B8860B", "Outside": "#B23A1B"}

st.set_page_config(page_title="Rut-Trigger Predictor", page_icon="🛣️", layout="wide")


@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str) -> RutTriggerModel:
    return RutTriggerModel(path)


@st.cache_data(show_spinner=False)
def typical_curve(path: str):
    M = load_model(path)
    m = M.predict(M.typical).margin
    return M.curve(m)[:2]


def fmt_num(v: float) -> str:
    a = abs(v)
    if a >= 10000:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:.1f}"
    return f"{v:.2f}".rstrip("0").rstrip(".")


def pct(p: float) -> str:
    return f"{p * 100:.1f}%" if p < 0.1 else f"{p * 100:.0f}%"


def yrs(t: float | None, t_max: float) -> str:
    return f"> {t_max:.0f}" if t is None else f"{t:.1f}"


def badge(text: str, colour: str) -> str:
    return (f"<span style='display:inline-block;padding:4px 12px;margin:2px 6px 2px 0;border-radius:14px;"
            f"background:{colour}1A;color:{colour};border:1.5px solid {colour};font-weight:600;"
            f"font-size:0.92rem'>{text}</span>")


if not MODEL_FILE.exists():
    st.error(f"Model file not found: {MODEL_FILE.name}. Run 10_rut_trigger_tool.py and copy "
             "rut_trigger_model.json from its output folder (10_tool) next to app.py.")
    st.stop()

M = load_model(str(MODEL_FILE))
FEAT = {f["input"]: f for f in M.features}

# ---------------------------------------------------------------- header
st.title("Rut-Trigger Predictor")
st.caption(f"Time until a flexible pavement section first reaches a {M.threshold:g} mm rut depth · "
           f"XGBoost-Cox survival model trained on {M.meta['n_cycles']:,} LTPP section-construction "
           f"cycles ({M.meta['n_events']} reached {M.threshold:g} mm) · research prototype")

tab_one, tab_batch, tab_about = st.tabs(["Single section", "Batch (CSV)", "Model and guidance"])

# ---------------------------------------------------------------- inputs (sidebar)
with st.sidebar:
    st.header("Section inputs")
    st.caption("Leave a field empty if unknown; the training median is then used and flagged.")
    c1, c2 = st.columns(2)
    if c1.button("Typical", width="stretch"):
        for f in M.features:
            st.session_state[f"in_{f['input']}"] = float(f["typical"])
    if c2.button("Clear all", width="stretch"):
        for f in M.features:
            st.session_state[f"in_{f['input']}"] = None
    raw = {}
    for g in GROUPS:
        with st.expander(g, expanded=(g in ("Condition", "Traffic", "Climate"))):
            for f in [f for f in M.features if f["group"] == g]:
                raw[f["input"]] = st.number_input(
                    f"{f['label']} ({f['unit']})", value=None, key=f"in_{f['input']}",
                    placeholder=f"typical {fmt_num(f['typical'])}",
                    step=1.0 if f["input"] == "n_layers" else None, format="%g",
                    help=f"LTPP training range {fmt_num(f['min'])} – {fmt_num(f['max'])} {f['unit']}")

# ---------------------------------------------------------------- single section
with tab_one:
    rut = raw.get("rut0_mm")
    if rut is None:
        st.info("Enter at least the current rut depth in the sidebar (or click **Typical section**).")
    elif rut >= M.threshold:
        st.warning(f"The entered rut depth is already {M.threshold:g} mm or more; the trigger has been reached.")
    else:
        rawn = {k: (float("nan") if v is None else v) for k, v in raw.items()}
        with st.spinner("Predicting with the model and its 200 bootstrap refits…"):
            r = M.predict(rawn)

        left, right = st.columns([1.05, 1.6], gap="large")
        with left:
            st.subheader("Predicted time to 10 mm")
            st.markdown(f"<div style='font-size:2.6rem;font-weight:700;color:{CURVE};line-height:1.1'>"
                        f"{yrs(r.t50, M.t_max)} <span style='font-size:1.2rem'>years</span></div>",
                        unsafe_allow_html=True)
            st.caption("Time at which the probability of reaching 10 mm passes 50%"
                       if r.t50 is not None else
                       f"50% probability is not reached within the {M.t_max:.0f}-year period the model supports")
            plan = [f"10% chance by **{yrs(r.t10, M.t_max)}** years"]
            if r.t10 is not None:
                plan.append(f"25% chance by **{yrs(r.t25, M.t_max)}** years")
            st.markdown("Planning horizons: " + " · ".join(plan))

            st.markdown("**Probability of reaching 10 mm**")
            m1, m2, m3 = st.columns(3)
            for col, lab, p, ci in [(m1, "Within 5 years", r.p5, r.ci5),
                                    (m2, "Within 10 years", r.p10, r.ci10),
                                    (m3, "Within 15 years", r.p15, r.ci15)]:
                col.metric(lab, pct(p))
                col.caption(f"90% band {pct(ci[0])}–{pct(ci[1])}")

            tier_txt = f"Risk tier: {r.tier}" + (f" (band spans {r.tier_band})" if r.tier_band else "")
            cov_key = r.coverage.split()[0]
            st.markdown(badge(tier_txt, TIER_COLOUR[r.tier]) + badge(f"Data coverage: {r.coverage}",
                        COV_COLOUR[cov_key]), unsafe_allow_html=True)
            if r.imputed:
                st.caption("Unknown, training median used: " + ", ".join(r.imputed) + ".")
            if r.out_of_range:
                st.warning("Outside the LTPP training range: " + ", ".join(r.out_of_range) + ".")

        with right:
            ts, p, lo, hi = M.curve(r.margin, r.boot_margins)
            tt, pt = typical_curve(str(MODEL_FILE))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.r_[ts, ts[::-1]], y=np.r_[hi, lo[::-1]], fill="toself",
                                     fillcolor=BAND, line=dict(width=0), hoverinfo="skip",
                                     name="90% band (200 refits)"))
            fig.add_trace(go.Scatter(x=tt, y=pt, mode="lines", line=dict(color=REF, width=2, dash="dash"),
                                     name="Typical LTPP section",
                                     hovertemplate="%{x:.1f} yr: %{y:.0%}<extra>typical</extra>"))
            fig.add_trace(go.Scatter(x=ts, y=p, mode="lines", line=dict(color=CURVE, width=3.5, shape="hv"),
                                     name="This section",
                                     hovertemplate="%{x:.1f} yr: %{y:.1%}<extra>this section</extra>"))
            fig.add_hline(y=0.5, line=dict(color="#555", width=1, dash="dot"))
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10),
                              xaxis=dict(title="Years from survey", range=[0, M.t_max], dtick=5,
                                         gridcolor="#E5E5E5"),
                              yaxis=dict(title="Probability of reaching 10 mm", range=[0, 1],
                                         tickformat=".0%", gridcolor="#E5E5E5"),
                              legend=dict(orientation="h", y=1.08, x=0), plot_bgcolor="white",
                              font=dict(size=14))
            st.plotly_chart(fig, width="stretch")

        res = pd.DataFrame([{**{FEAT[k]["label"]: v for k, v in raw.items()},
                             "years_to_10mm": yrs(r.t50, M.t_max), "years_to_10pct": yrs(r.t10, M.t_max),
                             "P_5yr": round(r.p5, 3), "P_10yr": round(r.p10, 3),
                             "P_10yr_band_low": round(r.ci10[0], 3), "P_10yr_band_high": round(r.ci10[1], 3),
                             "P_15yr": round(r.p15, 3), "risk_tier": r.tier,
                             "tier_band": r.tier_band or "within one tier", "data_coverage": r.coverage}])
        st.download_button("Download this prediction (CSV)", res.to_csv(index=False).encode(),
                           "rut_trigger_prediction.csv", "text/csv")

# ---------------------------------------------------------------- batch
with tab_batch:
    st.subheader("Predict many sections from a CSV file")
    st.markdown("One row per section, with a `section` label column and the field codes below. "
                "Empty cells are treated as unknown.")
    st.dataframe(pd.DataFrame([{"field code": f["input"], "description": f["label"], "unit": f["unit"],
                                "LTPP range": f"{fmt_num(f['min'])} – {fmt_num(f['max'])}"}
                               for f in M.features]), hide_index=True, width="stretch")
    tmpl = pd.DataFrame([{"section": "example", **{f["input"]: f["typical"] for f in M.features}}])
    st.download_button("Download CSV template", tmpl.to_csv(index=False).encode(),
                       "rut_trigger_template.csv", "text/csv")
    up = st.file_uploader("Upload sections (CSV)", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        df.columns = [c.strip() for c in df.columns]
        missing = [f["input"] for f in M.features if f["input"] not in df.columns]
        if missing:
            st.caption("Columns not found (treated as unknown): " + ", ".join(missing))
        rows = []
        bar = st.progress(0.0, text="Predicting…")
        for i, row in df.iterrows():
            lab = row["section"] if "section" in df.columns else str(i + 1)
            rw = {f["input"]: pd.to_numeric(row.get(f["input"], np.nan), errors="coerce")
                  for f in M.features}
            out = {"section": lab}
            if not np.isfinite(rw["rut0_mm"]):
                out["years_to_10mm"] = "rut0_mm missing"
            elif rw["rut0_mm"] >= M.threshold:
                out["years_to_10mm"] = "already reached"
            else:
                p = M.predict(rw)
                out.update({"years_to_10mm": yrs(p.t50, M.t_max), "years_to_10pct": yrs(p.t10, M.t_max),
                            "P_5yr": round(p.p5, 3), "P_10yr": round(p.p10, 3),
                            "P_10yr_band_low": round(p.ci10[0], 3), "P_10yr_band_high": round(p.ci10[1], 3),
                            "P_15yr": round(p.p15, 3), "risk_tier": p.tier,
                            "tier_band": p.tier_band or "within one tier",
                            "data_coverage": p.coverage, "n_imputed": len(p.imputed)})
            rows.append(out)
            bar.progress((i + 1) / len(df), text=f"Predicting… {i + 1}/{len(df)}")
        bar.empty()
        out_df = pd.DataFrame(rows)
        st.dataframe(out_df, hide_index=True, width="stretch")
        st.download_button("Download predictions (CSV)", out_df.to_csv(index=False).encode(),
                           "rut_trigger_predictions.csv", "text/csv")

# ---------------------------------------------------------------- about
with tab_about:
    st.subheader("How to read the results")
    st.markdown(f"""
- **Predicted time to 10 mm** is the time at which the probability of reaching 10 mm passes 50%.
  The 10% and 25% horizons suit agencies that plan treatment before the trigger.
- The **risk tier** places the 10-year probability among the LTPP training sections
  (Low below {M.tertiles[0]:.3f}, Moderate below {M.tertiles[1]:.3f}, High above).
- The **90% band** is the spread of the probability over 200 refits of the model on resampled LTPP
  sections. It reflects uncertainty in the fitted model, not the full spread of outcomes for one section.
  When the band spans more than one tier, treat the section as lying between those tiers.
- **Data coverage** measures how similar the section is to the training sections (mean distance to the
  {M.k} most similar ones). "Outside training data" means the prediction is an extrapolation. A "typical"
  label does not guarantee accuracy: in the study, sections of an unseen region lost accuracy even when
  their inputs looked typical, because the link between inputs and rutting differed. Before relying on the
  tool in a new region, compare its predictions with local outcomes.
- Times are counted from the survey whose rut depth is entered. Rehabilitation, which ends a cycle
  before the trigger, is not modelled as a competing event.
""")
    st.subheader("Model")
    st.markdown(f"- {M.meta['model']}\n- Trained on {M.meta['n_cycles']:,} cycles, "
                f"{M.meta['n_events']} reached {M.threshold:g} mm; model file generated {M.meta['generated']}\n"
                f"- Time axis supported to {M.t_max:.0f} years (at least 20 sections at risk)\n- {M.cv_perf}")
    st.caption("Research prototype accompanying the manuscript “Within-region skill, cross-region loss: "
               "censoring-aware machine learning for time-to-trigger rutting prognosis from LTPP data”. "
               "Not a substitute for engineering judgement or local performance data.")
