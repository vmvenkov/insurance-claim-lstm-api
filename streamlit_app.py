import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Insurance Claim Analyzer",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Insurance Claim Analyzer")

st.write(
    "Анализ на описание на застрахователна щета чрез Bidirectional LSTM."
)

description = st.text_area(
    "Описание на щетата",
    height=180,
    placeholder="Например: След силна градушка автомобилът има счупено предно стъкло..."
)

if st.button("🔍 Анализирай"):

    if description.strip() == "":
        st.warning("Моля въведете описание.")
        st.stop()

    with st.spinner("Анализиране..."):

        response = requests.post(
            f"{API_URL}/predict",
            json={
                "description": description
            }
        )

    if response.status_code != 200:
        st.error(response.text)
        st.stop()

    result = response.json()

    st.success("Анализът приключи успешно.")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Insurance Type",
            result["insurance_type"],
            f'{result["insurance_confidence"]*100:.2f}%'
        )

        st.metric(
            "Claim Type",
            result["claim_type"],
            f'{result["claim_confidence"]*100:.2f}%'
        )

    with col2:

        st.metric(
            "Severity",
            result["severity"],
            f'{result["severity_confidence"]*100:.2f}%'
        )

        st.metric(
            "Department",
            result["department"],
            f'{result["department_confidence"]*100:.2f}%'
        )

    st.divider()

    st.subheader("📋 Recommendation")

    st.info(result["recommendation"])
