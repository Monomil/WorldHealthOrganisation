"""
life_expectancy_predictor.py

A single, self-contained function for predicting a population's average
life expectancy from population statistics.

ETHICAL DESIGN
--------------
Two families of models were trained (see EDA.ipynb):

  * "bound"   -> models_bound.pkl    trained WITHOUT sensitive/protected
                 population data (only year, region, economic status,
                 GDP per capita, population, and schooling).
  * "unbound" -> models_unbound.pkl  trained WITH additional sensitive
                 health data (mortality rates, disease incidence,
                 immunization coverage, BMI, HIV incidence, etc.)

The unbound models are meaningfully more accurate (R^2 ~0.99 vs ~0.75 on
held-out data) but require detailed, potentially sensitive population
health statistics. Per the brief, the function must ask for explicit,
informed consent before using that sensitive data. If consent is not
given (or the sensitive fields aren't available), the function
transparently falls back to the bound model instead of silently
degrading or fabricating missing values.

Within whichever consent tier is selected, the user can also choose
between:
  * "best"    -> Random Forest (lowest test RMSE in both tiers)
  * "minimal" -> Linear Regression (simplest, most interpretable model)

WHY THIS FILE ALSO SHIPS PREPROCESSING ARTIFACTS
-------------------------------------------------
The saved models expect engineered features (one-hot encoded region,
degree-2 polynomial/interaction terms, standardised, then a fixed
stepwise-selected subset). `preprocessing_artifacts.pkl` stores the
exact fitted encoder-column-order, PolynomialFeatures object, and
StandardScaler for each tier (rebuilt deterministically from the same
train/test split, random_state=42, as the training notebook) so raw,
human-entered values are transformed identically to how the models
were trained.

Required files alongside this script:
  - models_bound.pkl
  - models_unbound.pkl
  - preprocessing_artifacts.pkl
"""

from __future__ import annotations

import os
import pickle
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))


# Artifact loading (cached at module level so repeated calls are cheap)


_MODELS: Optional[dict] = None
_ARTIFACTS: Optional[dict] = None


def _load_artifacts():
    """Lazily load model + preprocessing artifacts from disk (once)."""
    global _MODELS, _ARTIFACTS
    if _MODELS is None or _ARTIFACTS is None:
        with open(os.path.join(_HERE, "models_bound.pkl"), "rb") as f:
            models_bound = pickle.load(f)
        with open(os.path.join(_HERE, "models_unbound.pkl"), "rb") as f:
            models_unbound = pickle.load(f)
        with open(os.path.join(_HERE, "preprocessing_artifacts.pkl"), "rb") as f:
            artifacts = pickle.load(f)

        _MODELS = {"bound": models_bound, "unbound": models_unbound}
        _ARTIFACTS = artifacts
    return _MODELS, _ARTIFACTS


# Human-facing display names for the raw inputs each tier needs.
BOUND_FIELDS = ["year", "region", "economic_status", "gdp_per_capita",
                 "population_mln", "schooling"]

UNBOUND_EXTRA_FIELDS = ["infant_deaths", "under_five_deaths", "adult_mortality",
                          "alcohol_consumption", "hepatitis_b", "measles", "bmi",
                          "polio", "diphtheria", "incidents_hiv",
                          "thinness_five_nineteen_years"]

CONSENT_PROMPT = ("Do you consent to using advanced population data, which may "
                   "include protected information, for better accuracy? (Y/N)")


def _transform(raw_row: dict, tier: str) -> pd.DataFrame:
    """Turn a dict of raw feature values into the exact model-ready
    (encoded -> polynomial -> scaled -> selected) feature row for a tier."""
    _, artifacts = _load_artifacts()
    art = artifacts[tier]

    raw_order = art["raw_feature_order"]
    row = pd.DataFrame([{col: raw_row[col] for col in raw_order}])

    # one-hot encode region to match training columns, then align
    row_encoded = pd.get_dummies(row, columns=["region"], drop_first=True, dtype=int)
    row_encoded = row_encoded.reindex(columns=art["encoded_columns"], fill_value=0)

    # polynomial expansion (fitted transformer, so this matches training exactly)
    poly = art["poly"]
    row_poly = poly.transform(row_encoded)
    row_poly_df = pd.DataFrame(row_poly, columns=poly.get_feature_names_out(row_encoded.columns))

    # scale (fitted scaler from training)
    scaler = art["scaler"]
    row_scaled = pd.DataFrame(scaler.transform(row_poly_df), columns=row_poly_df.columns)

    # keep only the stepwise-selected features the model was trained on
    return row_scaled[art["selected_features"]]


def predict_life_expectancy(
    consent: str,
    model_choice: str = "best",
    *,
    year: int,
    region: str,
    economic_status: int,
    gdp_per_capita: float,
    population_mln: float,
    schooling: float,
    infant_deaths: Optional[float] = None,
    under_five_deaths: Optional[float] = None,
    adult_mortality: Optional[float] = None,
    alcohol_consumption: Optional[float] = None,
    hepatitis_b: Optional[float] = None,
    measles: Optional[float] = None,
    bmi: Optional[float] = None,
    polio: Optional[float] = None,
    diphtheria: Optional[float] = None,
    incidents_hiv: Optional[float] = None,
    thinness_five_nineteen_years: Optional[float] = None,
) -> dict:
    """
    Predict a population's average life expectancy.

    Parameters
    ----------
    consent : str
        "Y"/"yes" to use the unbound model (all data, incl. sensitive
        health statistics) if those fields are supplied; "N"/"no" to
        always use the bound model (no sensitive data required).
    model_choice : str
        "best"    -> Random Forest (highest accuracy)
        "minimal" -> Linear Regression (simplest, most interpretable)
    year, region, economic_status, gdp_per_capita, population_mln, schooling
        Always-required, non-sensitive population statistics.
    infant_deaths ... thinness_five_nineteen_years
        Optional sensitive health statistics. Only used, and only
        requested of the caller, when consent is given.

    Returns
    -------
    dict with keys:
        prediction      : float, predicted life expectancy (years)
        tier_used       : "bound" or "unbound"
        model_used      : model name actually used
        model_r2        : held-out R^2 for that model/tier
        model_rmse      : held-out RMSE (years) for that model/tier
        note            : human-readable explanation of what happened
    """
    models, artifacts = _load_artifacts()

    consent_given = str(consent).strip().lower() in ("y", "yes", "true", "1")
    model_choice = model_choice.strip().lower()
    if model_choice not in ("best", "minimal"):
        raise ValueError('model_choice must be "best" or "minimal"')

    sensitive_values = {
        "infant_deaths": infant_deaths,
        "under_five_deaths": under_five_deaths,
        "adult_mortality": adult_mortality,
        "alcohol_consumption": alcohol_consumption,
        "hepatitis_b": hepatitis_b,
        "measles": measles,
        "bmi": bmi,
        "polio": polio,
        "diphtheria": diphtheria,
        "incidents_hiv": incidents_hiv,
        "thinness_five_nineteen_years": thinness_five_nineteen_years,
    }
    sensitive_complete = all(v is not None for v in sensitive_values.values())

    if consent_given and sensitive_complete:
        tier = "unbound"
        note = "Consent given and full sensitive data supplied: using the unbound model."
    elif consent_given and not sensitive_complete:
        tier = "bound"
        note = ("Consent given, but not all sensitive fields were supplied, so the "
                "privacy-preserving bound model was used instead.")
    else:
        tier = "bound"
        note = "Consent not given: using the bound model (no protected data used)."

    model_name = artifacts["best_model"] if model_choice == "best" else artifacts["minimal_model"]
    model = models[tier][model_name]

    raw_row = {
        "year": year, "region": region, "economic_status": economic_status,
        "gdp_per_capita": gdp_per_capita, "population_mln": population_mln,
        "schooling": schooling,
    }
    if tier == "unbound":
        raw_row.update(sensitive_values)

    X = _transform(raw_row, tier)
    prediction = float(model.predict(X)[0])

    perf = artifacts["performance"][tier][model_name]

    return {
        "prediction": round(prediction, 2),
        "tier_used": tier,
        "model_used": model_name,
        "model_r2": perf["r2"],
        "model_rmse": perf["rmse"],
        "note": note,
    }


def run_interactive():
    """
    Command-line entry point that literally asks the consent question,
    collects the required inputs, and prints a prediction. Run this file
    directly (`python life_expectancy_predictor.py`) for a CLI demo.
    """
    print("=== Life Expectancy Predictor ===\n")

    consent = input(f"{CONSENT_PROMPT} ").strip()
    consent_given = consent.lower() in ("y", "yes")

    print("\nEnter the population's core statistics:")
    year = int(input("  Year (e.g. 2015): "))
    region = input("  Region (e.g. Africa, Asia, European Union, ...): ").strip()
    economic_status = int(input("  Economic status (1 = Developed, 0 = Developing): "))
    gdp_per_capita = float(input("  GDP per capita (USD): "))
    population_mln = float(input("  Population (millions): "))
    schooling = float(input("  Average years of schooling: "))

    kwargs = dict(
        year=year, region=region, economic_status=economic_status,
        gdp_per_capita=gdp_per_capita, population_mln=population_mln,
        schooling=schooling,
    )

    if consent_given:
        print("\nThank you for consenting. Please also enter the following "
              "health statistics for a more accurate prediction:")
        kwargs.update(dict(
            infant_deaths=float(input("  Infant deaths (per 1,000 live births): ")),
            under_five_deaths=float(input("  Under-five deaths (per 1,000 live births): ")),
            adult_mortality=float(input("  Adult mortality rate: ")),
            alcohol_consumption=float(input("  Alcohol consumption (litres per capita): ")),
            hepatitis_b=float(input("  Hepatitis B immunization (%): ")),
            measles=float(input("  Measles immunization (%): ")),
            bmi=float(input("  Average BMI: ")),
            polio=float(input("  Polio immunization (%): ")),
            diphtheria=float(input("  Diphtheria immunization (%): ")),
            incidents_hiv=float(input("  HIV incidence rate: ")),
            thinness_five_nineteen_years=float(input("  Thinness prevalence, ages 5-19 (%): ")),
        ))

    result = predict_life_expectancy(consent="Y" if consent_given else "N",
                                       model_choice="best", **kwargs)

    print(f"\nPredicted life expectancy: {result['prediction']} years")
    print(f"Model used: {result['model_used']} ({result['tier_used']} tier, "
          f"R^2={result['model_r2']}, RMSE={result['model_rmse']} years)")
    print(result["note"])


if __name__ == "__main__":
    run_interactive()
