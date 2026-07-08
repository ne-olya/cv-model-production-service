import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="VisionServe monitoring", layout="wide")
st.title("VisionServe monitoring")
api = os.getenv("VISION_API_URL", "http://localhost:8000")
try:
    data = requests.get(f"{api}/monitoring-summary", timeout=3).json()
    columns = st.columns(4)
    columns[0].metric("Запросы", data["requests"])
    columns[1].metric("Ошибки", data["errors"])
    columns[2].metric("Средняя latency", f"{data['average_latency_ms']:.1f} ms")
    columns[3].metric("Средняя confidence", f"{data['average_confidence']:.1%}")
    st.metric("p95 latency", f"{data['p95_latency_ms']:.1f} ms")
    if data["predicted_classes"]:
        st.bar_chart(pd.Series(data["predicted_classes"], name="predictions"))
    st.bar_chart(pd.Series(data["confidence_histogram"], name="confidence"))
except requests.RequestException as exc:
    st.error(f"API недоступен: {exc}")
