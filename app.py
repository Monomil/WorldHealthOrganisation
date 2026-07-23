"""
Streamlit dashboard for the Life Expectancy Predictor.

Run with:  streamlit run app.py

Requires, in the same folder:
  life_expectancy_predictor.py
  models_bound.pkl
  models_unbound.pkl
  preprocessing.pkl
"""

import pickle

import streamlit as st

from life_expectancy_predictor import predict_life_expectancy, _load_artifacts, FEATURE_DESCRIPTIONS

st.set_page_config(page_title="Life Expectancy Predictor", page_icon="\U0001F30D", layout="centered")

_, ARTIFACTS = _load_artifacts()
BOUND_STATS = ARTIFACTS["bound"]["feature_stats"]
UNBOUND_STATS = ARTIFACTS["unbound"]["feature_stats"]
REGIONS = ARTIFACTS["bound"]["region_categories"]
PERF = ARTIFACTS["performance"]

st.title("\U0001F30D Life Expectancy Predictor")
st.caption(
    "Predicts a population's average life expectancy from country-level "
    "statistics. Two model families are available, trained with different "
    "levels of data sensitivity."
)

# --------------------------------------------------------------------------
# Consent (ethical gate)
# --------------------------------------------------------------------------
st.subheader("Data consent")
st.write(
    "Do you consent to using **advanced population data, which may include "
    "protected information**, for better accuracy?"
)
consent_choice = st.radio(
    label="Consent",
    options=["No — use only non-sensitive data", "Yes — use all available data"],
    index=0,
    label_visibility="collapsed",
)
consent = "Y" if consent_choice.startswith("Yes") else "N"

with st.expander("What does this mean?"):
    st.markdown(
        "- **No** uses the *bound* model: only year, region, economic "
        "status, GDP per capita, population, and schooling. No health "
        "data is required or used.\n"
        "- **Yes** unlocks the *unbound* model, which also uses sensitive "
        "public-health statistics (mortality rates, immunization "
        "coverage, HIV incidence, BMI, etc.) for a substantially more "
        "accurate prediction.\n\n"
        f"On held-out test data, the bound model reaches "
        f"R\u00b2={PERF['bound']['Random Forest']['test_r2']:.2f} "
        f"(\u00b1{PERF['bound']['Random Forest']['test_rmse']:.1f} years typical error), "
        f"while the unbound model reaches "
        f"R\u00b2={PERF['unbound']['Random Forest']['test_r2']:.2f} "
        f"(\u00b1{PERF['unbound']['Random Forest']['test_rmse']:.1f} years typical error)."
    )

with st.expander("What do these fields mean? (units, verified against the data)"):
    st.caption(
        "This dataset is the *updated* 'Life Expectancy (WHO) Fixed' release (2,864 rows), "
        "which redefines a few fields compared to the older/classic WHO Kaggle dataset — "
        "most notably **Measles**, which here is immunization coverage (%), not a raw case "
        "count. Ranges below were checked directly against the CSV."
    )
    for field, desc in FEATURE_DESCRIPTIONS.items():
        st.markdown(f"- **{field}**: {desc}")

# --------------------------------------------------------------------------
# Model — chosen automatically, not by the user
# --------------------------------------------------------------------------
st.subheader("Model")
model_choice = "best"  # Random Forest, in both tiers — see rationale below

def _fit_gap(perf, name):
    return round(perf[name]["train_r2"] - perf[name]["test_r2"], 4)

st.markdown(f"**Random Forest** is used automatically to make predictions.")
with st.expander("Why Random Forest, and not the linear models?", expanded=False):
    for tier_label, tier_key in [("Bound (no sensitive data)", "bound"), ("Unbound (all data)", "unbound")]:
        perf = PERF[tier_key]
        rf_gap = _fit_gap(perf, "Random Forest")
        lr_gap = _fit_gap(perf, "Linear Regression")
        st.markdown(f"**{tier_label}**")
        st.markdown(
            f"- Random Forest: Test RMSE = **{perf['Random Forest']['test_rmse']:.2f} yrs** "
            f"(train R\u00b2={perf['Random Forest']['train_r2']:.4f}, test R\u00b2={perf['Random Forest']['test_r2']:.4f}, "
            f"gap={rf_gap:.4f})\n"
            f"- Linear Regression: Test RMSE = **{perf['Linear Regression']['test_rmse']:.2f} yrs** "
            f"(train R\u00b2={perf['Linear Regression']['train_r2']:.4f}, test R\u00b2={perf['Linear Regression']['test_r2']:.4f}, "
            f"gap={lr_gap:.4f})"
        )
        if rf_gap > lr_gap:
            st.caption(
                f"Random Forest shows a larger train-test gap here ({rf_gap:.4f} vs {lr_gap:.4f}) — "
                "i.e. more overfitting — but its test RMSE is still clearly the lowest, so it "
                "generalises better in absolute terms despite fitting the training data more closely. "
                "The linear models underfit by comparison: their smaller gap reflects high bias "
                "(they can't capture the non-linear patterns) rather than better generalisation."
            )
        else:
            st.caption(
                f"Random Forest has both the smaller train-test gap ({rf_gap:.4f} vs {lr_gap:.4f}) "
                "and the lowest test RMSE — it generalises better with no meaningful overfitting cost."
            )
    st.markdown(
        "**Bottom line:** across both tiers, Random Forest has the lowest test RMSE by a wide "
        "margin, so it's used automatically rather than asking you to pick a model."
    )

st.divider()

# --------------------------------------------------------------------------
# Core (always required) inputs
# --------------------------------------------------------------------------
st.subheader("Core population statistics")
col1, col2 = st.columns(2)
with col1:
    year = st.number_input("Year", min_value=2000, max_value=2030,
                             value=2015, step=1)
    region = st.selectbox("Region", REGIONS)
    economic_status_label = st.selectbox("Economic status", ["Developing", "Developed"])
    economic_status = 1 if economic_status_label == "Developed" else 0
with col2:
    gdp_per_capita = st.number_input(
        "GDP per capita (USD)",
        min_value=0.0, value=float(round(BOUND_STATS["gdp_per_capita"]["median"])), step=100.0,
    )
    population_mln = st.number_input(
        "Population (millions)",
        min_value=0.0, value=float(round(BOUND_STATS["population_mln"]["median"], 2)), step=1.0,
    )
    schooling = st.number_input(
        "Average years of schooling",
        min_value=0.0, max_value=25.0,
        value=float(BOUND_STATS["schooling"]["median"]), step=0.1,
    )

# --------------------------------------------------------------------------
# Sensitive inputs — only shown/used if consent given
# --------------------------------------------------------------------------
sensitive_kwargs = {}
if consent == "Y":
    st.subheader("Additional health statistics (sensitive)")
    st.caption("These are only used because you consented above.")
    c1, c2 = st.columns(2)
    with c1:
        sensitive_kwargs["infant_deaths"] = st.number_input(
            "Infant deaths (per 1,000 live births)", min_value=0.0,
            value=float(UNBOUND_STATS["infant_deaths"]["median"]))
        sensitive_kwargs["under_five_deaths"] = st.number_input(
            "Under-five deaths (per 1,000 live births)", min_value=0.0,
            value=float(UNBOUND_STATS["under_five_deaths"]["median"]))
        sensitive_kwargs["adult_mortality"] = st.number_input(
            "Adult mortality rate", min_value=0.0,
            value=float(UNBOUND_STATS["adult_mortality"]["median"]))
        sensitive_kwargs["alcohol_consumption"] = st.number_input(
            "Alcohol consumption (litres per capita)", min_value=0.0,
            value=float(UNBOUND_STATS["alcohol_consumption"]["median"]))
        sensitive_kwargs["hepatitis_b"] = st.number_input(
            "Hepatitis B immunization (%)", min_value=0.0, max_value=100.0,
            value=float(UNBOUND_STATS["hepatitis_b"]["median"]))
        sensitive_kwargs["measles"] = st.number_input(
            "Measles immunization (%)", min_value=0.0, max_value=100.0,
            value=float(UNBOUND_STATS["measles"]["median"]))
    with c2:
        sensitive_kwargs["bmi"] = st.number_input(
            "Average BMI", min_value=0.0,
            value=float(UNBOUND_STATS["bmi"]["median"]))
        sensitive_kwargs["polio"] = st.number_input(
            "Polio immunization (%)", min_value=0.0, max_value=100.0,
            value=float(UNBOUND_STATS["polio"]["median"]))
        sensitive_kwargs["diphtheria"] = st.number_input(
            "Diphtheria immunization (%)", min_value=0.0, max_value=100.0,
            value=float(UNBOUND_STATS["diphtheria"]["median"]))
        sensitive_kwargs["incidents_hiv"] = st.number_input(
            "HIV incidence rate", min_value=0.0,
            value=float(UNBOUND_STATS["incidents_hiv"]["median"]))
        sensitive_kwargs["thinness_five_nineteen_years"] = st.number_input(
            "Thinness prevalence, ages 5-19 (%)", min_value=0.0, max_value=100.0,
            value=float(UNBOUND_STATS["thinness_five_nineteen_years"]["median"]))

st.divider()

# --------------------------------------------------------------------------
# Predict
# --------------------------------------------------------------------------
if st.button("Predict life expectancy", type="primary", use_container_width=True):
    result = predict_life_expectancy(
        consent=consent,
        model_choice=model_choice,
        year=year, region=region, economic_status=economic_status,
        gdp_per_capita=gdp_per_capita, population_mln=population_mln,
        schooling=schooling,
        **sensitive_kwargs,
    )

    st.metric("Predicted life expectancy", f"{result['prediction']} years")

    tier_label = "Unbound (all data)" if result["tier_used"] == "unbound" else "Bound (no sensitive data)"
    st.write(f"**Model used:** {result['model_used']} — *{tier_label}*")
    st.write(
        f"**Held-out accuracy:** R\u00b2 = {result['model_r2']:.3f}, "
        f"typical error \u00b1{result['model_rmse']:.2f} years"
    )
    st.info(result["note"])

st.divider()
with st.expander("Model performance summary (train vs. test)"):
    def _perf_table(perf):
        return {
            name: {
                "Train R\u00b2": v["train_r2"], "Test R\u00b2": v["test_r2"],
                "Train MSE": v["train_mse"], "Test MSE": v["test_mse"],
                "Train RMSE": v["train_rmse"], "Test RMSE": v["test_rmse"],
            }
            for name, v in perf.items()
        }

    st.write("**Bound models** (no sensitive data)")
    st.table(_perf_table(PERF["bound"]))
    st.write("**Unbound models** (all data)")
    st.table(_perf_table(PERF["unbound"]))
