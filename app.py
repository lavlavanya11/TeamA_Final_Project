import streamlit as st

from evaluation.evaluator import evaluate_intents
from src.nlu_pipeline import run_pipeline

import matplotlib.pyplot as plt
import pandas as pd
import plotly.figure_factory as ff


st.set_page_config(
    page_title="BotTrainer – LLM NLU Trainer",
    page_icon="🤖",
    layout="wide",
)

st.title("BotTrainer – LLM-Based NLU Trainer & Evaluator")
st.markdown(
    "Test intents, entities, and evaluation metrics for your chatbot NLU, "
    "powered by Groq + Llama."
)


tab_test, tab_eval, tab_dataset = st.tabs(["🔍 Test NLU", "📊 Evaluation", "📂 Dataset Overview"])


with tab_test:
    user_text = st.text_input("Type a user message", "")

    if st.button("Analyze") and user_text.strip():
        with st.spinner("Running NLU pipeline..."):
            result = run_pipeline(user_text.strip())

        if result.get("error"):
            st.error(result["error"])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Predicted Intent")
            st.write(result.get("intent"))
            confidence = float(result.get("confidence", 0.0))
            st.write(f"Confidence: {confidence:.3f}")
            st.progress(min(max(confidence, 0.0), 1.0))

        with col2:
            st.subheader("Extracted Entities")
            entities = result.get("entities", {}) or {}
            if entities:
                ent_rows = [
                    {"Entity": k, "Value": v} for k, v in entities.items()
                ]
                st.table(ent_rows)
            else:
                st.write("No entities extracted.")

        st.subheader("Raw Model Output")
        st.json(result.get("raw", {}))


with tab_eval:
    st.markdown(
        "Run the full evaluation on the test dataset defined in "
        "`data/test_data.json`."
    )

    use_cache = st.checkbox("Use cached evaluation results", value=True)

    @st.cache_data(show_spinner=False)
    def _cached_eval():
        return evaluate_intents()

    if st.button("Run Evaluation"):
        progress = st.progress(0)

        def _progress_cb(done: int, total: int):
            if total <= 0:
                return
            progress.progress(min(1.0, done / total))

        with st.spinner("Evaluating on test set..."):
            if use_cache:
                metrics, cm, labels, report, examples = _cached_eval()
            else:
                metrics, cm, labels, report, examples = evaluate_intents(
                    progress_cb=_progress_cb
                )

        st.subheader("Classification Metrics")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        col2.metric("Precision", f"{metrics['precision']:.2%}")
        col3.metric("F1 Score", f"{metrics['f1']:.2%}")
        col4.metric("Recall", f"{metrics['recall']:.2%}")
        # This is a stricter metric: the fraction of examples where
        # both the intent AND the full set of entities exactly match
        # the ground truth from the test data.
        if "exact_match" in metrics:
            col5.metric(
                "Exact Match (intent+entities)",
                f"{metrics['exact_match']:.2%}",
            )

        st.subheader("Per-example predictions")
        # Show a compact table so you can visually confirm whether
        # the model is really perfect or not.
        if examples:
            ex_df = pd.DataFrame(examples)
            st.dataframe(ex_df)

        st.subheader("Confusion Matrix")

        cm_df = pd.DataFrame(cm, index=labels, columns=labels)

        fig = ff.create_annotated_heatmap(
            z=cm_df.values,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=True,
        )
        fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Per-intent Breakdown (classification report)")
        st.dataframe(pd.DataFrame(report).T)


with tab_dataset:
    st.markdown("Overview of the NLU dataset defined in `data/intents.json` and `data/test_data.json`.")

    from utils.data_loader import load_intents_dataset
    import json
    from pathlib import Path

    dataset = load_intents_dataset()
    intents = dataset.get("intents", [])
    entities_def = dataset.get("entities", {})

    st.subheader("Intent Distribution (training-style dataset)")
    if intents:
        rows = []
        for intent in intents:
            name = intent.get("name", "")
            examples = intent.get("examples", []) or []
            rows.append(
                {
                    "intent": name,
                    "num_examples": len(examples),
                }
            )
        intent_names = [r["intent"] for r in rows]
        intent_counts = [r["num_examples"] for r in rows]

        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(dict(zip(intent_names, intent_counts)))
        with col2:
            fig, ax = plt.subplots()
            ax.pie(intent_counts, labels=intent_names, autopct="%1.0f%%")
            st.pyplot(fig)

        selected = st.selectbox("Filter by intent", [i["name"] for i in intents])
        selected_intent = next(i for i in intents if i["name"] == selected)
        st.dataframe(pd.DataFrame({"examples": selected_intent.get("examples", [])}))
    else:
        st.write("No intents found in `intents.json`.")

    st.subheader("Entity Definitions")
    if entities_def:
        ent_rows = []
        for name, values in entities_def.items():
            ent_rows.append(
                {
                    "entity": name,
                    "num_values": len(values) if isinstance(values, list) else 0,
                    "example_values": ", ".join(values[:5]) if isinstance(values, list) else "",
                }
            )
        st.dataframe(pd.DataFrame(ent_rows))
    else:
        st.write("No entities defined in `intents.json`.")

    st.subheader("Test Set Preview (`data/test_data.json`)")
    test_path = Path("data") / "test_data.json"
    if test_path.exists():
        test_data = json.loads(test_path.read_text(encoding="utf-8"))
        examples = test_data.get("test_examples", [])
        if examples:
            test_rows = []
            for ex in examples:
                test_rows.append(
                    {
                        "text": ex.get("text", ""),
                        "intent": ex.get("intent", ""),
                        "entities": ex.get("entities", {}),
                    }
                )
            st.dataframe(pd.DataFrame(test_rows))
        else:
            st.write("`test_data.json` has no `test_examples` entries.")
    else:
        st.write("`data/test_data.json` not found.")




