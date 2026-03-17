
import streamlit as st
import pandas as pd
import json
import os

from src.pipeline.predict import predict
from src.pipeline.trainer import train
from src.utils.data_loader import load_dataset

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="NLU Bot Trainer",
    page_icon="🤖",
    layout="wide"
)

# -------------------------
# CUSTOM UI STYLE
# -------------------------

st.markdown("""
<style>

/* Main background */

.main {
    background-color: #f5f7fb;
}

/* Sidebar styling */

[data-testid="stSidebar"] {
    background-color: #111827;
    padding-top: 20px;
}

/* Sidebar text */

[data-testid="stSidebar"] * {
    color: white;
}

/* Sidebar title */

.sidebar-title {
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 20px;
}

/* Buttons */

.stButton>button {
    border-radius:10px;
    height:3em;
    font-size:16px;
    background-color:#4CAF50;
    color:white;
}

/* Inputs */

.stTextInput>div>div>input {
    border-radius:10px;
}

.stTextArea textarea {
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)
# -------------------------
# HEADER
# -------------------------

st.title("🤖 NLU Bot Trainer Platform")
st.write("Train, evaluate and test Machine Learning based NLU models")

# -------------------------
# SIDEBAR
# -------------------------

st.sidebar.markdown("## 🤖 BotTrainer")
st.sidebar.write("ML-based NLU Platform")
st.sidebar.divider()

page = st.sidebar.radio(
    "Go to",
    ["NLU Tester", "Dataset Overview", "Evaluation"]
)

# -------------------------
# NLU TESTER PAGE
# -------------------------

if page == "NLU Tester":

    st.header("✨ Test Your NLU Model")

    user_input = st.text_area("Enter a user message")

    if st.button("🚀 Analyze Message"):

        if user_input.strip() != "":

            intent, confidence, entities = predict(user_input)

            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Detected Intent", intent)

            with col2:
                st.metric("Confidence Score", round(confidence,2))

            st.subheader("🧩 Extracted Entities")

            if entities:
                for key, value in entities.items():
                    st.info(f"{key} : {value}")
            else:
                st.write("No entities detected")

            # Download Result
            results = {
                "message": user_input,
                "intent": intent,
                "confidence": confidence,
                "entities": str(entities)
            }

            df = pd.DataFrame([results])

            st.download_button(
                label="Download Result as CSV",
                data=df.to_csv(index=False),
                file_name="prediction_result.csv",
                mime="text/csv"
            )

        else:
            st.warning("Please enter a message")

    st.divider()

    if st.button("🔄 Retrain Model"):
        train()
        st.success("Model retrained successfully!")

# -------------------------
# DATASET OVERVIEW
# -------------------------

elif page == "Dataset Overview":

    st.header("📊 Dataset Dashboard")

    dataset = load_dataset()

    st.write("Dataset Size:", dataset.shape)

    st.subheader("Dataset Preview")
    st.dataframe(dataset.head())

    st.subheader("Intent Distribution")

    intent_counts = dataset["intent"].value_counts()
    st.bar_chart(intent_counts, height=400)

    # Upload dataset
    st.divider()

    st.subheader("Upload New Dataset")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:

        new_data = pd.read_csv(uploaded_file)

        st.write("Preview of uploaded dataset")
        st.dataframe(new_data.head())

        if st.button("Train Model with Uploaded Dataset"):

            new_data.to_csv("data/raw_data/full_nlu_dataset_200.csv", index=False)

            train()

            st.success("Model retrained with uploaded dataset!")

    # Add training example
    st.divider()

    st.subheader("Add New Training Example")

    new_intent = st.text_input("Intent Name")

    new_sentence = st.text_input("Example Sentence")

    if st.button("Add to Dataset"):

        if new_intent and new_sentence:

            dataset = load_dataset()

            new_row = {
                "intent": new_intent,
                "utterance": new_sentence,
                "entities": ""
            }

            dataset = dataset._append(new_row, ignore_index=True)

            dataset.to_csv("data/raw_data/full_nlu_dataset_200.csv", index=False)

            st.success("Training example added!")

        else:
            st.warning("Please fill both fields")

    if st.button("Retrain Model with New Data"):

        train()

        st.success("Model retrained successfully!")

# -------------------------
# MODEL EVALUATION
# -------------------------

elif page == "Evaluation":

    st.header("📈 Model Evaluation Dashboard")

    dataset = load_dataset()

    col1, col2 = st.columns(2)

    col1.metric("Total Samples", len(dataset))
    col2.metric("Unique Intents", dataset["intent"].nunique())

    st.subheader("Intent Distribution")
    st.bar_chart(dataset["intent"].value_counts(), height=400)

    # Model Metrics

    st.subheader("Model Metrics")

    if os.path.exists("models/metrics.json"):

        with open("models/metrics.json") as f:
            metrics = json.load(f)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", round(metrics["accuracy"],2))
        col2.metric("Precision", round(metrics["precision"],2))
        col3.metric("Recall", round(metrics["recall"],2))
        col4.metric("F1 Score", round(metrics["f1_score"],2))

    else:
        st.info("Train the model to generate evaluation metrics.")

    # Confusion Matrix

    st.subheader("Confusion Matrix")

    if os.path.exists("models/confusion_matrix.png"):
        st.image("models/confusion_matrix.png")
    else:
        st.info("Train the model to generate confusion matrix.")

    # Model Info

    st.subheader("Model Information")

    st.info("""
Algorithm: Logistic Regression  
Vectorizer: TF-IDF  
Entity Recognition: spaCy NLP  
Framework: Streamlit
""")

# -------------------------
# FOOTER
# -------------------------

st.divider()

st.markdown("""
<style>
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
text-align: center;
padding: 10px;
font-size: 14px;
color: gray;
}
</style>

<div class="footer">
Built with ❤️ using <b>Python, Scikit-learn, spaCy and Streamlit</b> 
</div>
""", unsafe_allow_html=True)

