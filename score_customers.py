"""
RetentionIQ - score_customers.py

Turns the committed model artifacts in data/models/ plus the real
feature_table.parquet into a scored customer table the prototype reads.
Run this on the machine that has the Drive data.

    python score_customers.py \
        --features data/features/feature_table.parquet \
        --models   data/models \
        --out      data/scored

Outputs:
    data/scored/customer_scores.parquet   full scored table
    data/scored/customer_scores_sample.json   top at-risk rows for the UI

Deps:  pip install pandas pyarrow xgboost lifetimes shap
(SHAP and lifetimes degrade gracefully if missing.)

Honest notes baked in:
 - CLV (BG/NBD + Gamma-Gamma) only applies to repeat buyers. One-time buyers
   (frequency==0) get predicted_clv = 0. That's 88% of the base, by design.
 - The recommender routes on the features that actually exist in the table
   (refund_rate, recency_per_gap, p_alive, days_since_last_order, frequency).
   It does NOT claim root cause - drivers are behavioural.
"""
from __future__ import annotations
import argparse, json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

CHURN_WINDOW = 299
CLV_HORIZON_DAYS = 365  # 12-month CLV

# ---- recommender (adapted to the real feature_table columns) -------------
STRATEGY = {
    "service_recovery": "Fix the cause - protect high-value relationships",
    "timing_nudge":     "Re-engage early and cheap, before the window closes",
    "winback":          "Last recoverable moment - margin-aware reactivation",
    "loyalty":          "Reward and deepen - grow LTV of healthy customers",
    "first_repeat":     "Convert one-time buyers to repeat - the 88% problem",
    "hold":             "Protect budget - do not spend on noisy or lost customers",
}
REASON = {
    "refund_rate": "returning a high share of orders",
    "recency_per_gap": "quiet well past their usual reorder rhythm",
    "days_since_last_order": "long silence since last order",
    "monetary_value": "lower repeat-purchase value",
    "avg_order_value": "smaller baskets than peers",
    "frequency": "few or no repeat orders",
    "bgnbd_p_alive": "low modelled probability of still being active",
    "recency": "short active lifespan", "t": "short observation window",
    "total_orders": "low order count", "total_revenue": "low total spend",
}

def intensity(clv):
    return "personal" if clv >= 120 else "semi" if clv >= 30 else "automated"

def why_for_row(r):
    """Behavioural drivers split into risk (what pushes the customer toward churn)
    and protective (what is keeping them). Quantified and version-proof, unlike
    SHAP TreeExplainer on some xgboost builds. Returns {"risk": [...], "protective": [...]}."""
    churn = float(r.get("churn_prob", 0.0) or 0.0)
    clv   = float(r.get("predicted_clv", 0.0) or 0.0)
    freq  = float(r.get("frequency", 0) or 0)
    pa    = float(r.get("bgnbd_p_alive", 1) or 1)
    ds    = float(r.get("days_since_last_order", 0) or 0)
    rr    = float(r.get("refund_rate", 0) or 0)
    rpg   = r.get("recency_per_gap", np.nan)
    rpg   = float(rpg) if rpg is not None and not (isinstance(rpg, float) and np.isnan(rpg)) else np.nan
    # dsl_per_gap = days since last order / typical gap = how OVERDUE they are.
    # (recency_per_gap is lifetime span in gap-units, not overdue-ness, so it is
    # not the right signal for "past their reorder pace".)
    dsl   = r.get("dsl_per_gap", np.nan)
    dsl   = float(dsl) if dsl is not None and not (isinstance(dsl, float) and np.isnan(dsl)) else np.nan

    # --- risk drivers as (severity, quantified phrase); strongest leads ---
    risk = []
    if rr >= 0.15:
        risk.append((rr, f"returns {rr*100:.0f}% of orders"))
    if ds > CHURN_WINDOW:
        risk.append((1.0, f"no order in {int(ds)} days, past the {CHURN_WINDOW}-day window"))
    elif not np.isnan(dsl) and dsl >= 1.5:
        phrase = (f"{dsl:.1f}x past their usual reorder gap" if dsl < 5
                  else "well past their usual reorder pace")
        risk.append((min(dsl / 3, 0.95), phrase))
    if pa <= 0.20:
        risk.append((1 - pa, f"{pa*100:.0f}% modelled chance still active"))
    if freq == 0:
        risk.append((0.5, "one purchase so far, no repeat yet"))
    risk.sort(key=lambda x: -x[0])
    risk_phrases = [t for _, t in risk][:3]

    # --- protective drivers as (strength, quantified phrase); what's keeping them ---
    prot = []
    if freq >= 1:
        prot.append((freq, f"{int(freq) + 1} orders to date"))
    if pa >= 0.60:
        prot.append((pa, f"{pa*100:.0f}% modelled chance still active"))
    if not np.isnan(dsl) and dsl < 1.0 and freq >= 1:
        prot.append((1 - dsl, "ordering on or close to their usual pace"))
    if rr == 0 and freq >= 1:
        prot.append((0.4, "no returns on record"))
    if clv >= 200:
        prot.append((min(clv / 1000, 1.0), f"predicted lifetime value ${clv:,.0f}"))
    prot.sort(key=lambda x: -x[0])
    prot_phrases = [t for _, t in prot][:3]

    if not risk_phrases and not prot_phrases:
        prot_phrases = ["stable, low-risk profile"]

    return {"risk": risk_phrases, "protective": prot_phrases}

def recommend_row(r):
    clv = float(r.get("predicted_clv", 0.0))
    churn = float(r.get("churn_prob", 0.0))
    rr = float(r.get("refund_rate", 0.0) or 0.0)
    rpg = float(r.get("recency_per_gap", np.nan))
    dsl = float(r.get("dsl_per_gap", np.nan))   # overdue = days since last / typical gap
    pa = float(r.get("bgnbd_p_alive", 1.0))
    ds = float(r.get("days_since_last_order", 0.0) or 0.0)
    freq = float(r.get("frequency", 0.0) or 0.0)
    it = intensity(clv)
    if freq == 0:
        cat, act, it = "first_repeat", "First-to-second purchase nudge, timed before the window.", "automated"
    elif rr >= 0.15:
        cat, act = "service_recovery", "Reach out about the returns - exchange / fit help. Do NOT discount."
    elif pa <= 0.20 or ds > CHURN_WINDOW:
        if clv >= 120: cat, act = "winback", "Last-chance personal win-back, justified by value at stake."
        else: cat, act, it = "hold", "Likely gone and low value - do not spend.", "none"
    elif not np.isnan(dsl) and dsl >= 1.6:
        cat, act = "winback", "Timed win-back now - overdue, approaching the window."
    elif not np.isnan(dsl) and dsl >= 1.0:
        cat, act = "timing_nudge", "Light 'time to restock?' nudge - early and cheap."
    elif clv >= 120:
        cat, act = "loyalty", "Healthy and high value - reward to deepen the relationship."
    else:
        cat, act, it = "hold", "No strong actionable signal - monitor.", "none"
    stake = round(clv * churn, 2)
    return cat, act, it, STRATEGY[cat], stake

# ---- main ----------------------------------------------------------------
def main(a):
    mdir = Path(a.models)
    import xgboost as xgb

    # Prefer Tanzeel's TEMPORAL (non-leaky) churn model if it is present.
    # It is trained with a future-window label, so days_since/recency are
    # legitimate features instead of the label in disguise.
    temporal_model = mdir / "churn_xgboost_temporal.json"
    temporal_meta  = mdir / "churn_xgboost_temporal_metrics.json"
    use_temporal = temporal_model.exists() and temporal_meta.exists()

    # The temporal model needs the richer feature table exported by
    # 04_churn_xgboost_temporal (feature_table_temporal.parquet). Auto-pick it.
    feat_path = Path(a.features)
    if use_temporal:
        cand = feat_path.parent / "feature_table_temporal.parquet"
        if feat_path.name != "feature_table_temporal.parquet" and cand.exists():
            feat_path = cand

    print(f"loading feature table: {feat_path}")
    df = pd.read_parquet(feat_path)
    df.columns = [c.strip() for c in df.columns]          # trim only; preserve case
    if not use_temporal:
        df.columns = [c.lower() for c in df.columns]      # snapshot model expects lower-snake columns
    df["customer_id"] = df["customer_id"].astype(str)

    # ---- churn probability (XGBoost) ----
    if use_temporal:
        meta = json.load(open(temporal_meta))
        feats = [c.strip() for c in meta["feature_cols"]]
        # resolve each model feature to an actual table column, ignoring case and
        # surrounding whitespace, so a casing mismatch between the metrics file and
        # the parquet (e.g. "ptype_Combined Collection" vs "ptype_combined collection")
        # does not break scoring.
        lut = {c.strip().lower(): c for c in df.columns}
        resolved, missing = [], []
        for f in feats:
            actual = lut.get(f.strip().lower())
            (resolved.append(actual) if actual is not None else missing.append(f))
        if missing:
            raise SystemExit(
                "The temporal churn model needs features this table does not have:\n"
                f"  {missing}\n"
                "Point --features at the temporal feature table exported by "
                "04_churn_xgboost_temporal (feature_table_temporal.parquet), "
                "not the basic feature_table.parquet.")
        bst = xgb.Booster(); bst.load_model(str(temporal_model))
        print(f"scoring churn (TEMPORAL model, {len(feats)} features, leakage fixed)...")
        # select by the table's actual column names, label with the model's names
        dm = xgb.DMatrix(df[resolved].astype(float).values, feature_names=feats)
        df["churn_prob"] = bst.predict(dm)
    else:
        meta = json.load(open(mdir / "churn_metrics.json"))
        orig_feats = list(meta["feature_cols"])       # names the model was trained with (e.g. "T")
        feats = [c.lower() for c in orig_feats]         # how they appear in the snake_cased table
        for c in feats:
            if c not in df.columns:
                raise SystemExit(f"missing feature column: {c}. Columns present: {list(df.columns)}")
        bst = xgb.Booster(); bst.load_model(str(mdir / "churn_xgboost.json"))
        print("scoring churn (snapshot model - note: known recency leakage)...")
        dm = xgb.DMatrix(df[feats].astype(float).values, feature_names=orig_feats)
        df["churn_prob"] = bst.predict(dm)

    # ---- BG/NBD p_alive + expected purchases, Gamma-Gamma CLV ----
    had_p_alive = "bgnbd_p_alive" in df.columns   # temporal table already carries it
    df["predicted_clv"] = 0.0
    if not had_p_alive:
        df["bgnbd_p_alive"] = 1.0
    try:
        from lifetimes import BetaGeoFitter, GammaGammaFitter
        bg = json.load(open(mdir / "clv_bgnbd_params.json"))
        gg = json.load(open(mdir / "clv_gamma_gamma_params.json"))
        bgf = BetaGeoFitter(); bgf.params_ = pd.Series(bg)
        ggf = GammaGammaFitter(); ggf.params_ = pd.Series(gg)
        f, r_, t_ = df["frequency"].astype(float), df["recency"].astype(float), df["t"].astype(float)
        print("computing p_alive + expected purchases...")
        if not had_p_alive:
            df["bgnbd_p_alive"] = bgf.conditional_probability_alive(f, r_, t_)
        exp_purch = bgf.conditional_expected_number_of_purchases_up_to_time(CLV_HORIZON_DAYS, f, r_, t_)
        repeat = (df["frequency"] > 0) & (df["monetary_value"] > 0)
        profit = pd.Series(0.0, index=df.index)
        profit[repeat] = ggf.conditional_expected_average_profit(
            df.loc[repeat, "frequency"], df.loc[repeat, "monetary_value"])
        df["predicted_clv"] = (exp_purch * profit).clip(lower=0).fillna(0.0)
    except Exception as e:
        print(f"[warn] lifetimes step skipped ({e}); CLV left at 0, p_alive at 1.0")

    # ---- derived signal the recommender needs (skip if the table already has it) ----
    if "recency_per_gap" not in df.columns:
        avg_gap = (df["recency"] / df["frequency"].replace(0, np.nan))
        df["recency_per_gap"] = df["days_since_last_order"] / avg_gap
    if "dsl_per_gap" not in df.columns:
        if "avg_days_between_orders" in df.columns:
            df["dsl_per_gap"] = df["days_since_last_order"] / (df["avg_days_between_orders"] + 1.0)
        else:
            _ag = (df["recency"] / df["frequency"].replace(0, np.nan))
            df["dsl_per_gap"] = df["days_since_last_order"] / _ag
    # reorder gap is undefined with a single order; null it so one-timers don't
    # distort it (they are 88% of the base) and don't read as "overdue".
    df.loc[df["frequency"] == 0, "dsl_per_gap"] = np.nan

    # ---- recommendation per customer ----
    print("recommending...")
    rec = df.apply(lambda r: recommend_row(r), axis=1, result_type="expand")
    rec.columns = ["category", "action", "intensity", "strategy", "value_at_stake"]
    df = pd.concat([df, rec], axis=1)

    # ---- optional: customer segment from 05_segmentation.ipynb (numpy-only) ----
    seg_path = mdir / "segmentation_kmeans.json"
    if seg_path.exists():
        try:
            sp = json.load(open(seg_path))
            logset = set(sp["log_features"])
            Xs = df[[c.lower() for c in sp["features"]]].astype(float).copy()
            Xs.columns = sp["features"]
            for c in logset:
                Xs[c] = np.log1p(Xs[c].clip(lower=0))
            Z = (Xs.values - np.array(sp["scaler_mean"])) / np.array(sp["scaler_scale"])
            cents = np.array(sp["centroids"])
            dists = np.stack([((Z - c) ** 2).sum(1) for c in cents], axis=1)  # (N, k)
            sid = dists.argmin(1)
            names = sp["segment_names"]
            df["segment"] = [names.get(str(i), f"Segment {i}") for i in sid]
            print(f"assigned segments (k={sp['k']}, silhouette={sp.get('silhouette')})")
        except Exception as e:
            print(f"[warn] segmentation skipped ({e})")

    # ---- at-risk flag (the behavioural "why" is computed on the sample below) ----
    atrisk = df["churn_prob"] >= a.atrisk_threshold
    print(f"{atrisk.sum():,} at-risk customers (churn >= {a.atrisk_threshold})")
    df["why"] = [[] for _ in range(len(df))]

    # ---- write outputs ----
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cols = ["customer_id", "churn_prob", "predicted_clv", "bgnbd_p_alive",
            "recency_per_gap", "dsl_per_gap", "days_since_last_order", "frequency", "refund_rate",
            "category", "action", "intensity", "strategy", "value_at_stake", "why"]
    if "segment" in df.columns:
        cols.insert(cols.index("category"), "segment")
    scored = df[cols].copy()
    scored.to_parquet(out / "customer_scores.parquet", index=False)
    # Demo sample: rank by VALUE AT STAKE (clv x churn), actionable plays first.
    # Important with the temporal model: raw churn is highest for one-time buyers
    # (CLV ~0), so ranking by churn would surface people not worth saving. We rank by
    # value at stake so the UI shows the customers whose value is actually exposed.
    ACTIONABLE = {"winback", "service_recovery", "loyalty", "timing_nudge"}
    is_act = scored["category"].isin(ACTIONABLE)
    actionable = scored[is_act].sort_values("value_at_stake", ascending=False)
    rest = scored[~is_act].sort_values("value_at_stake", ascending=False)
    sample = pd.concat([actionable, rest]).head(a.sample_n).copy()
    _wf = list(sample.apply(why_for_row, axis=1))           # behavioural drivers for the UI
    sample["why_risk"] = [w["risk"] for w in _wf]
    sample["why_protective"] = [w["protective"] for w in _wf]
    sample["why"] = [w["risk"] if w["risk"] else w["protective"] for w in _wf]  # flat, backward compat

    # ---- population summary: value-based totals the header shows (not churn-gated) ----
    has_value = scored["value_at_stake"] > 0
    ACTIONABLE_CATS = {"winback", "loyalty", "timing_nudge", "service_recovery"}
    # group-level driver profile. Honest stand-in for per-group SHAP: standardized
    # deviation of each group's mean from the whole-population mean, signed so that
    # positive = pushes churn risk up (red), negative = pulls it down (green).
    DRIVERS = [
        ("Days since last order", "days_since_last_order", +1),
        ("Reorder gap (overdue)", "dsl_per_gap", +1),
        ("Refund rate", "refund_rate", +1),
        ("Order frequency", "frequency", -1),
        ("Avg order value", "avg_order_value", -1),
    ]
    drv = [(lab, col, d) for lab, col, d in DRIVERS if col in df.columns]
    # baseline = repeat buyers (behavioural drivers are meaningful once there is
    # more than one order); winsorize tails so a few very-frequent buyers do not
    # dominate the scale.
    base = df[df["frequency"] >= 1]
    bounds, popstat = {}, {}
    for _, col, _ in drv:
        s = base[col].dropna()
        lo, hi = (float(s.quantile(0.02)), float(s.quantile(0.98))) if len(s) else (0.0, 1.0)
        bounds[col] = (lo, hi)
        cs = s.clip(lo, hi)
        popstat[col] = (float(cs.mean()), (float(cs.std()) or 1.0))

    category_breakdown = {}
    for cat, g in df.groupby("category"):
        gg = g if cat == "first_repeat" else g[g["frequency"] >= 1]
        raw = []
        for lab, col, direction in drv:
            lo, hi = bounds[col]
            mu, sd = popstat[col]
            gm = float(gg[col].clip(lo, hi).mean())
            z = 0.0 if (sd == 0 or gm != gm) else (gm - mu) / sd
            z = max(-2.5, min(2.5, direction * z))            # signed, clipped contribution
            raw.append([lab, z])
        mx = max((abs(z) for _, z in raw), default=1.0) or 1.0
        drivers = sorted(
            [{"label": lab, "z": round(z, 2), "val": round(z / mx, 3)} for lab, z in raw],
            key=lambda d: -abs(d["z"]),
        )
        category_breakdown[str(cat)] = {
            "count": int(len(g)),
            "value_at_stake": round(float(g["value_at_stake"].sum()), 2),
            "avg_churn": round(float(g["churn_prob"].mean()), 4),
            "avg_clv": round(float(g["predicted_clv"].mean()), 2),
            "actionable": bool(cat in ACTIONABLE_CATS),
            "drivers": drivers,
        }
    summary = {
        "total_customers": int(len(scored)),
        "total_at_risk": int(has_value.sum()),                         # customers with CLV value exposed to churn
        "total_value_at_stake": round(float(scored["value_at_stake"].sum()), 2),
        "actionable_at_risk": int(is_act.sum()),
        "actionable_value_at_stake": round(float(scored.loc[is_act, "value_at_stake"].sum()), 2),
        "first_repeat_customers": int((scored["category"] == "first_repeat").sum()),  # one-timers, volume play (CLV ~0)
        "churn_flag_count": int(atrisk.sum()),                         # secondary: churn >= threshold
        "sample_size": int(len(sample)),
        "atrisk_threshold": float(a.atrisk_threshold),
        "category_counts": {k: int(v) for k, v in scored["category"].value_counts().items()},
        "category_breakdown": category_breakdown,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    # JSON the UI reads: real population summary + the top-N sample to render
    payload = {"summary": summary, "customers": json.loads(sample.to_json(orient="records"))}
    (out / "customer_scores_sample.json").write_text(json.dumps(payload))

    print(f"\nwrote {out/'customer_scores.parquet'} ({len(scored):,} rows)")
    print(f"wrote {out/'customer_scores_sample.json'} (top {len(sample)} sample + population summary)")
    print(f"\npopulation: ${summary['total_value_at_stake']:,.0f} total value at stake"
          f" | {summary['actionable_at_risk']:,} actionable customers (${summary['actionable_value_at_stake']:,.0f})"
          f" | {summary['first_repeat_customers']:,} one-timers to nudge")
    print("\ntop 5 in the demo sample:")
    print(sample[["customer_id","churn_prob","predicted_clv","category","value_at_stake"]].head().to_string(index=False))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/features/feature_table.parquet")
    p.add_argument("--models", default="data/models")
    p.add_argument("--out", default="data/scored")
    p.add_argument("--atrisk-threshold", type=float, default=0.5)
    p.add_argument("--shap-cap", type=int, default=5000, help="max rows to run SHAP on")
    p.add_argument("--sample-n", type=int, default=200, help="rows in the UI JSON sample")
    main(p.parse_args())
