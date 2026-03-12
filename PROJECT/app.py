import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.components.json_loader import IntentJSONLoader
from src.components.json_to_dataframe import flatten_intents_json
from src.components.gemini_nlu import GeminiNLU
from src.components.evaluator import NLUEvaluator
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

# ----------------- SETUP -----------------
logger = setup_logger("STREAMLIT_APP")

st.set_page_config(
    page_title="BotTrainer – Gemini NLU Platform",
    layout="wide",
    page_icon="🤖"
)

# ----------------- LOAD CONFIG & DATA -----------------
config = load_config()
loader = IntentJSONLoader(config["paths"]["intents_path"])
intents_data = loader.load()

df = flatten_intents_json(intents_data)
nlu = GeminiNLU(
    model_name=config["llm"]["model_name"],
    api_key=config["llm"]["api_key"]
)
evaluator = NLUEvaluator()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("## 🤖 BotTrainer")
    st.caption("LLM-Based NLU Platform")
    st.divider()

    page = st.radio(
        "📌 Navigation",
        ["NLU Tester", "Evaluation", "Dataset Overview"]
    )

    st.divider()
    st.markdown("### ⚙️ Model Info")
    st.write("**Model:**", config["llm"]["model_name"])
    st.write("**Inference:** Google Gemini API")
    st.write("**Total Intents:**", df["true_intent"].nunique())
    st.write("**Total Samples:**", len(df))
    st.divider()
    st.caption("BotTrainer • Gemini NLU Version")

# ----------------- NLU TESTER -----------------
if page == "NLU Tester":
    st.title("✨ NLU Tester")
    st.caption("Real-time intent classification and entity extraction powered by Gemini")

    user_text = st.text_area(
        "Enter a user message",
        placeholder="e.g., Book a flight to Delhi tomorrow",
        height=120
    )

    if st.button("🚀 Analyze Message", use_container_width=True):
        if not user_text.strip():
            st.warning("Please enter some text to analyze")
        else:
            with st.spinner("Analyzing with Gemini..."):
                logger.info(f"User input: {user_text}")
                result = nlu.predict(user_text, intents_data)

            st.subheader("🔍 Prediction Result")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("### 🏷 Predicted Intent")
                st.success(f"**{result['intent']}**")

            with col2:
                st.markdown("### 🎯 Confidence Score")
                confidence = float(result.get("confidence", 0))
                st.progress(confidence)
                st.write(f"{confidence * 100:.1f}%")

            st.markdown("### 🧩 Extracted Entities")
            entities = result.get("entities", {})
            if entities:
                entity_df = pd.DataFrame(
                    entities.items(), columns=["Entity", "Value"]
                )
                st.table(entity_df)
            else:
                st.info("No entities detected for this input")

            st.markdown("### 📋 Raw JSON Output")
            st.json(result)

# ----------------- EVALUATION -----------------
elif page == "Evaluation":
    st.title("📊 Model Evaluation")
    st.caption("Evaluate NLU model performance across all intents")

    st.info("ℹ️ Evaluation runs 1 sample per intent — 10 samples total")

    if st.button("▶ Run Evaluation", use_container_width=True):
        with st.spinner("Running evaluation... this may take a moment"):
            logger.info("Evaluation started")

            df_eval = (
                df.groupby("true_intent", group_keys=False)
                  .head(1)
                  .reset_index(drop=True)
            )

            predictions = []
            for text in df_eval["text"]:
                result = nlu.predict(text, intents_data)
                predictions.append(result["intent"])

            df_eval["predicted_intent"] = predictions

            metrics, confusion_df, per_intent_df = evaluator.evaluate_with_results(
                df_eval.rename(columns={"true_intent": "intent"})
            )

        # Overall metrics
        st.subheader("📈 Overall Performance Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("✅ Accuracy",   f"{metrics['accuracy']  * 100:.2f}%")
        c2.metric("🎯 Precision",  f"{metrics['precision'] * 100:.2f}%")
        c3.metric("🔁 Recall",     f"{metrics['recall']    * 100:.2f}%")
        c4.metric("⚖️ F1 Score",   f"{metrics['f1']        * 100:.2f}%")

        st.divider()

        # Per-intent breakdown (extra feature)
        st.subheader("🔍 Per-Intent Breakdown")
        st.dataframe(per_intent_df, use_container_width=True)

        st.divider()

        # Confusion matrix
        st.subheader("🔀 Confusion Matrix")
        st.dataframe(confusion_df, use_container_width=True)

        st.divider()

        # Prediction details
        st.subheader("📋 Prediction Details")
        st.dataframe(df_eval, use_container_width=True)

# ----------------- DATASET OVERVIEW -----------------
elif page == "Dataset Overview":
    st.title("📁 Dataset Overview")
    st.caption("Explore intent distribution and dataset structure")

    # Summary stats
    st.subheader("📊 Dataset Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Samples", len(df))
    c2.metric("Total Intents", df["true_intent"].nunique())
    c3.metric("Avg Samples per Intent", round(len(df) / df["true_intent"].nunique(), 2))

    st.divider()

    # Bar chart
    st.subheader("📌 Intent Distribution — Bar Chart")
    st.bar_chart(df["true_intent"].value_counts())

    st.divider()

    # Pie chart (extra feature)
    st.subheader("🥧 Intent Distribution — Pie Chart")
    intent_counts = df["true_intent"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        intent_counts.values,
        labels=intent_counts.index,
        autopct="%1.1f%%",
        startangle=140
    )
    ax.set_title("Intent Distribution")
    st.pyplot(fig)

    st.divider()

    # Sample data
    st.subheader("🔍 Sample Data Preview")
    st.dataframe(df.sample(10), use_container_width=True)

    st.divider()

    # Intent filter
    st.subheader("🔎 Filter by Intent")
    selected_intent = st.selectbox(
        "Select an intent to view its examples",
        options=sorted(df["true_intent"].unique())
    )
    filtered_df = df[df["true_intent"] == selected_intent]
    st.dataframe(filtered_df, use_container_width=True)