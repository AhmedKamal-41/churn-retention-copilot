"""The customer-entry form for the dashboard.

Fields are grouped into sections laid out in columns. Each field is stored in
session state under an "f_" key, so the High/Low/Average preset buttons can fill
the whole form by setting those keys and rerunning.
"""

import pandas as pd
import streamlit as st

from app.components.theme import section_label
from src.data.split import FEATURE_COLUMNS

# Valid options for each categorical field (matches the Telco dataset).
_YES_NO = ["Yes", "No"]
_SERVICE = ["No", "Yes", "No internet service"]
CATEGORY_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": _YES_NO,
    "Dependents": _YES_NO,
    "PhoneService": _YES_NO,
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": _SERVICE,
    "OnlineBackup": _SERVICE,
    "DeviceProtection": _SERVICE,
    "TechSupport": _SERVICE,
    "StreamingTV": _SERVICE,
    "StreamingMovies": _SERVICE,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": _YES_NO,
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

# Starting values for the form.
DEFAULTS = {
    "f_gender": "Female", "f_SeniorCitizen": 0, "f_Partner": "No", "f_Dependents": "No",
    "f_tenure": 12, "f_PhoneService": "Yes", "f_MultipleLines": "No",
    "f_InternetService": "Fiber optic", "f_OnlineSecurity": "No", "f_OnlineBackup": "No",
    "f_DeviceProtection": "No", "f_TechSupport": "No", "f_StreamingTV": "No",
    "f_StreamingMovies": "No", "f_Contract": "Month-to-month", "f_PaperlessBilling": "Yes",
    "f_PaymentMethod": "Electronic check", "f_MonthlyCharges": 70.0, "f_TotalCharges": 840.0,
}

# One-click presets that fill the whole form.
PRESETS = {
    "High risk": {
        "f_gender": "Female", "f_SeniorCitizen": 0, "f_Partner": "No", "f_Dependents": "No",
        "f_tenure": 2, "f_PhoneService": "Yes", "f_MultipleLines": "No",
        "f_InternetService": "Fiber optic", "f_OnlineSecurity": "No", "f_OnlineBackup": "No",
        "f_DeviceProtection": "No", "f_TechSupport": "No", "f_StreamingTV": "Yes",
        "f_StreamingMovies": "Yes", "f_Contract": "Month-to-month", "f_PaperlessBilling": "Yes",
        "f_PaymentMethod": "Electronic check", "f_MonthlyCharges": 95.0, "f_TotalCharges": 190.0,
    },
    "Low risk": {
        "f_gender": "Male", "f_SeniorCitizen": 0, "f_Partner": "Yes", "f_Dependents": "Yes",
        "f_tenure": 64, "f_PhoneService": "Yes", "f_MultipleLines": "Yes",
        "f_InternetService": "DSL", "f_OnlineSecurity": "Yes", "f_OnlineBackup": "Yes",
        "f_DeviceProtection": "Yes", "f_TechSupport": "Yes", "f_StreamingTV": "No",
        "f_StreamingMovies": "No", "f_Contract": "Two year", "f_PaperlessBilling": "No",
        "f_PaymentMethod": "Bank transfer (automatic)", "f_MonthlyCharges": 60.0, "f_TotalCharges": 3840.0,
    },
    "Average": {
        "f_gender": "Male", "f_SeniorCitizen": 0, "f_Partner": "No", "f_Dependents": "No",
        "f_tenure": 29, "f_PhoneService": "Yes", "f_MultipleLines": "No",
        "f_InternetService": "DSL", "f_OnlineSecurity": "No", "f_OnlineBackup": "No",
        "f_DeviceProtection": "No", "f_TechSupport": "No", "f_StreamingTV": "No",
        "f_StreamingMovies": "No", "f_Contract": "Month-to-month", "f_PaperlessBilling": "Yes",
        "f_PaymentMethod": "Mailed check", "f_MonthlyCharges": 65.0, "f_TotalCharges": 1400.0,
    },
}


def _ensure_defaults():
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)


def _apply_preset(name: str):
    for key, value in PRESETS[name].items():
        st.session_state[key] = value


def _category(label: str, field: str):
    st.selectbox(label, CATEGORY_OPTIONS[field], key=f"f_{field}")


def _collect_customer() -> pd.DataFrame:
    """Read the current form values into a one-row DataFrame of feature columns."""
    values = {column: st.session_state[f"f_{column}"] for column in FEATURE_COLUMNS}
    return pd.DataFrame([values])[FEATURE_COLUMNS]


def render_input_form():
    """Render the form and return (new_customer DataFrame, analyze_clicked)."""
    _ensure_defaults()

    section_label("Reference profiles")
    st.caption("Load a representative account to explore model behavior before entering live data.")
    preset_columns = st.columns(len(PRESETS))
    for column, name in zip(preset_columns, PRESETS):
        if column.button(name, use_container_width=True):
            _apply_preset(name)
            st.rerun()

    st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)

    section_label("Contract & billing")
    account = st.columns(4)
    with account[0]:
        st.number_input("Tenure (months)", min_value=0, max_value=72, step=1, key="f_tenure")
    with account[1]:
        _category("Contract", "Contract")
    with account[2]:
        _category("Paperless billing", "PaperlessBilling")
    with account[3]:
        _category("Payment method", "PaymentMethod")

    charges = st.columns(2)
    with charges[0]:
        st.number_input(
            "Monthly charges ($)", min_value=0.0, max_value=200.0, step=1.0, key="f_MonthlyCharges"
        )
    with charges[1]:
        st.number_input("Total charges ($)", min_value=0.0, step=1.0, key="f_TotalCharges")

    with st.expander("Service bundle", expanded=False):
        services = st.columns(3)
        with services[0]:
            _category("Phone service", "PhoneService")
            _category("Multiple lines", "MultipleLines")
            _category("Internet service", "InternetService")
        with services[1]:
            _category("Online security", "OnlineSecurity")
            _category("Online backup", "OnlineBackup")
            _category("Device protection", "DeviceProtection")
        with services[2]:
            _category("Tech support", "TechSupport")
            _category("Streaming TV", "StreamingTV")
            _category("Streaming movies", "StreamingMovies")

    with st.expander("Demographics", expanded=False):
        demographics = st.columns(4)
        with demographics[0]:
            _category("Gender", "gender")
        with demographics[1]:
            st.selectbox(
                "Senior citizen", [0, 1],
                format_func=lambda value: "Yes" if value == 1 else "No",
                key="f_SeniorCitizen",
            )
        with demographics[2]:
            _category("Partner", "Partner")
        with demographics[3]:
            _category("Dependents", "Dependents")

    st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
    analyze_clicked = st.button(
        "Analyze Customer", type="primary", key="analyze_button", use_container_width=True
    )
    return _collect_customer(), analyze_clicked
