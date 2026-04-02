import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import sys
import pickle
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(
    page_title="FedTriage — AI Federated Medical Triage",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
        color: white; text-align: center;
    }
    .triage-red { background: #ff4444; color: white; padding: 1rem;
                   border-radius: 8px; text-align: center; font-size: 1.5rem; font-weight: bold; }
    .triage-yellow { background: #ffaa00; color: #1a1a1a; padding: 1rem;
                      border-radius: 8px; text-align: center; font-size: 1.5rem; font-weight: bold; }
    .triage-green { background: #00aa44; color: white; padding: 1rem;
                     border-radius: 8px; text-align: center; font-size: 1.5rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    try:
        from models.clinic_model import ClinicTrainer
        from federated.aggregator import EnsemblePredictor

        clinic_ids = ["clinic_1", "clinic_2", "clinic_3"]
        trainers = []
        scalers = {}

        for cid in clinic_ids:
            trainer = ClinicTrainer(cid)
            model_path = f"models/saved/{cid}_model.pkl"
            scaler_path = f"models/saved/{cid}_scaler.pkl"
            if os.path.exists(model_path):
                trainer.load(model_path)
            trainers.append(trainer)
            if os.path.exists(scaler_path):
                with open(scaler_path, "rb") as f:
                    scalers[cid] = pickle.load(f)

        ensemble = EnsemblePredictor(trainers)
        meta_path = "models/saved/meta_learner.pkl"
        if os.path.exists(meta_path):
            ensemble.load_meta_learner(meta_path)

        return trainers, ensemble, scalers, True
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None, None, False


@st.cache_data
def load_training_summary():
    path = "federated/training_summary.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_round_history():
    path = "federated/round_history.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def preprocess_input(inputs_dict, scaler):
    travel_map = {"None": 0, "Domestic": 1, "Europe": 2,
                  "Southeast Asia": 3, "Middle East": 4, "Africa": 5}
    gender_val = 1 if inputs_dict["gender"] == "Male" else 0
    travel_val = travel_map.get(inputs_dict["travel_history"], 0)

    feature_vector = np.array([[
        inputs_dict["age"], gender_val,
        inputs_dict["diabetes"], inputs_dict["hypertension"], inputs_dict["immunocompromised"],
        inputs_dict["fever"], inputs_dict["cough"], inputs_dict["shortness_of_breath"],
        inputs_dict["chest_pain"], inputs_dict["headache"], inputs_dict["fatigue"],
        inputs_dict["nausea"], inputs_dict["vomiting"], inputs_dict["diarrhea"],
        inputs_dict["rash"], inputs_dict["joint_pain"], inputs_dict["loss_of_smell"],
        travel_val,
        inputs_dict["temperature"], inputs_dict["heart_rate"],
        inputs_dict["oxygen_saturation"], inputs_dict["systolic_bp"],
        inputs_dict["symptom_duration"],
    ]], dtype=np.float32)

    if scaler is not None:
        continuous_idx = [0, 18, 19, 20, 21, 22]
        feature_vector[:, continuous_idx] = scaler.transform(
            feature_vector[:, continuous_idx])
    return feature_vector


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 FedTriage")
    st.caption("Federated Learning Medical Triage")
    st.divider()
    page = st.radio("Navigation", [
        "🩺 Patient Triage",
        "📊 Clinic Dashboard",
        "🔬 Federated Learning",
        "ℹ️ About"
    ])

trainers, ensemble, scalers, models_loaded = load_models()
summary = load_training_summary()
round_history = load_round_history()


# ══════════════════════════════════════════════════════════════════════════════
if page == "🩺 Patient Triage":
    st.markdown("""
    <div class="main-header">
        <h1>🏥 AI-Powered Patient Triage</h1>
        <p>Federated Learning model — trained across 3 clinics, privacy preserved</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patient Demographics")
        age = st.slider("Age", 1, 95, 35)
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        col1a, col1b = st.columns(2)
        with col1a:
            diabetes = st.checkbox("Diabetes")
            hypertension = st.checkbox("Hypertension")
        with col1b:
            immunocompromised = st.checkbox("Immunocompromised")
        travel_history = st.selectbox("Travel History (past 30 days)",
            ["None", "Domestic", "Europe", "Southeast Asia", "Middle East", "Africa"])

        st.subheader("Vitals")
        c1, c2 = st.columns(2)
        with c1:
            temperature = st.number_input("Temperature (°C)", 35.0, 42.0, 37.2, 0.1)
            heart_rate = st.number_input("Heart Rate (bpm)", 40, 200, 78)
        with c2:
            oxygen_sat = st.number_input("SpO₂ (%)", 60, 100, 98)
            systolic_bp = st.number_input("Systolic BP (mmHg)", 60, 220, 120)
        symptom_duration = st.slider("Symptom Duration (days)", 0, 30, 2)

    with col2:
        st.subheader("Presenting Symptoms")
        sc1, sc2 = st.columns(2)
        with sc1:
            fever = st.checkbox("🌡️ Fever")
            cough = st.checkbox("😮‍💨 Cough")
            shortness_of_breath = st.checkbox("💨 Shortness of Breath")
            chest_pain = st.checkbox("💔 Chest Pain")
            headache = st.checkbox("🤕 Headache")
            fatigue = st.checkbox("😴 Fatigue")
        with sc2:
            nausea = st.checkbox("🤢 Nausea")
            vomiting = st.checkbox("🤮 Vomiting")
            diarrhea = st.checkbox("🏃 Diarrhea")
            rash = st.checkbox("🔴 Rash")
            joint_pain = st.checkbox("🦴 Joint Pain")
            loss_of_smell = st.checkbox("👃 Loss of Smell")

        st.divider()
        if st.button("🔍 Run Triage Assessment", type="primary", use_container_width=True):
            if not models_loaded:
                st.error("Models not loaded. Run python train.py first.")
            else:
                inputs = {
                    "age": age, "gender": gender,
                    "diabetes": int(diabetes), "hypertension": int(hypertension),
                    "immunocompromised": int(immunocompromised),
                    "fever": int(fever), "cough": int(cough),
                    "shortness_of_breath": int(shortness_of_breath),
                    "chest_pain": int(chest_pain), "headache": int(headache),
                    "fatigue": int(fatigue), "nausea": int(nausea),
                    "vomiting": int(vomiting), "diarrhea": int(diarrhea),
                    "rash": int(rash), "joint_pain": int(joint_pain),
                    "loss_of_smell": int(loss_of_smell),
                    "travel_history": travel_history,
                    "temperature": temperature, "heart_rate": heart_rate,
                    "oxygen_saturation": oxygen_sat, "systolic_bp": systolic_bp,
                    "symptom_duration": symptom_duration,
                }

                scaler = list(scalers.values())[0] if scalers else None
                X = preprocess_input(inputs, scaler)

                label_names = ["Green", "Yellow", "Red"]
                label_classes = ["triage-green", "triage-yellow", "triage-red"]
                label_emojis = ["🟢", "🟡", "🔴"]
                recommendations = [
                    "Non-urgent. Monitor at home. Follow up in 48–72 hours if symptoms worsen.",
                    "Semi-urgent. Schedule appointment within 24 hours. Monitor vitals.",
                    "URGENT. Immediate medical attention required. Do not delay."
                ]

                clinic_probs = [t.predict_proba(X)[0] for t in trainers]
                stacked = np.concatenate(clinic_probs).reshape(1, -1)
                final_probs = ensemble.meta_learner.predict_proba(stacked)[0]
                final_label = np.argmax(final_probs)

                st.subheader("🏥 Triage Result")
                st.markdown(f"""
                <div class="{label_classes[final_label]}">
                    {label_emojis[final_label]} TRIAGE: {label_names[final_label].upper()}
                </div>
                """, unsafe_allow_html=True)
                st.info(f"**Recommendation:** {recommendations[final_label]}")

                st.subheader("Clinic Model Opinions")
                cols = st.columns(3)
                for i, (col, probs) in enumerate(zip(cols, clinic_probs)):
                    with col:
                        st.markdown(f"**Clinic {i+1}**")
                        for j, (name, emoji) in enumerate(zip(label_names, label_emojis)):
                            st.write(f"{emoji} {name}: {probs[j]*100:.1f}%")

                fig = go.Figure(go.Bar(
                    x=label_names, y=final_probs * 100,
                    marker_color=["#00aa44", "#ffaa00", "#ff4444"],
                    text=[f"{p*100:.1f}%" for p in final_probs],
                    textposition="outside"
                ))
                fig.update_layout(title="Meta-Learner Confidence", yaxis_title="Confidence (%)",
                                   height=300, paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Clinic Dashboard":
    st.title("📊 Clinic Performance Dashboard")

    if summary:
        clinics = list(summary["baseline"].keys())
        avg_base = np.mean([summary["baseline"][c]["accuracy"] for c in clinics])
        avg_fed  = np.mean([summary["federated"][c]["accuracy"] for c in clinics])
        meta_acc = summary["ensemble"]["meta_learner_accuracy"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Baseline", f"{avg_base:.1%}")
        c2.metric("Avg Federated", f"{avg_fed:.1%}", delta=f"+{avg_fed-avg_base:.1%}")
        c3.metric("Meta-Learner", f"{meta_acc:.1%}")
        c4.metric("Clinics", "3", delta="Federated")

        st.subheader("Baseline vs Federated Accuracy")
        clinic_labels = [c.replace("_", " ").title() for c in clinics]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Baseline", x=clinic_labels,
                              y=[summary["baseline"][c]["accuracy"] for c in clinics],
                              marker_color="#4a9eff"))
        fig.add_trace(go.Bar(name="Federated", x=clinic_labels,
                              y=[summary["federated"][c]["accuracy"] for c in clinics],
                              marker_color="#00d4a4"))
        fig.update_layout(barmode="group", yaxis=dict(range=[0, 1]), height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Improvement per Clinic")
        improvements = summary.get("improvement", {})
        fig2 = go.Figure(go.Bar(
            x=list(improvements.keys()),
            y=list(improvements.values()),
            marker_color=["#00d4a4" if v >= 0 else "#ff4444" for v in improvements.values()],
            text=[f"+{v:.1%}" if v >= 0 else f"{v:.1%}" for v in improvements.values()],
            textposition="outside"
        ))
        fig2.update_layout(title="Accuracy Improvement After Federation", height=350)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No training summary found. Run python train.py first.")

    st.subheader("Dataset Distribution")
    for clinic_id in ["clinic_1", "clinic_2", "clinic_3"]:
        path = f"data/raw/{clinic_id}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            with st.expander(f"📁 {clinic_id.replace('_',' ').title()} — {len(df)} patients"):
                c1, c2 = st.columns(2)
                with c1:
                    vc = df["triage_name"].value_counts()
                    fig = px.pie(values=vc.values, names=vc.index,
                                 color=vc.index,
                                 color_discrete_map={"Green":"#00aa44","Yellow":"#ffaa00","Red":"#ff4444"},
                                 title="Triage Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig = px.histogram(df, x="age", color="triage_name",
                                       color_discrete_map={"Green":"#00aa44","Yellow":"#ffaa00","Red":"#ff4444"},
                                       title="Age Distribution")
                    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Federated Learning":
    st.title("🔬 Federated Learning Process")
    st.info("Raw patient data never leaves each clinic. Only model weights are shared.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Algorithm", "FedProx")
    c2.metric("Aggregation", "Weighted FedAvg")
    c3.metric("Meta-Learner", "Logistic Regression")

    if round_history:
        st.subheader("Accuracy Over Federated Rounds")
        rounds = [r["round"] for r in round_history]
        colors = {"clinic_1": "#4a9eff", "clinic_2": "#ff9a4a", "clinic_3": "#4aff9a"}
        fig = go.Figure()
        for cid, color in colors.items():
            accs = [r["clinic_metrics"].get(cid, {}).get("accuracy", 0) for r in round_history]
            fig.add_trace(go.Scatter(x=rounds, y=accs, mode="lines+markers",
                                      name=cid.replace("_"," ").title(),
                                      line=dict(color=color, width=2)))
        fig.update_layout(xaxis_title="Round", yaxis_title="Accuracy", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("System Architecture")
    st.dataframe(pd.DataFrame({
        "Component": ["Clinic 1", "Clinic 2", "Clinic 3", "FedProx Aggregator", "Meta-Learner"],
        "Model": ["Gradient Boosting + RL", "Gradient Boosting + RL",
                   "Gradient Boosting + RL", "FedAvg + Proximal Term",
                   "Logistic Regression (Stacked)"],
        "Data Access": ["Local only","Local only","Local only","Weights only","Clinic outputs only"],
        "Status": ["✅","✅","✅","✅","✅"]
    }), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About FedTriage")
    st.markdown("""
    ## Federated Learning Medical Triage System

    | Component | Technology | Purpose |
    |-----------|-----------|---------|
    | Local Models | Gradient Boosting (sklearn) | Per-clinic triage classification |
    | Cost-Sensitive Weights | RL-inspired penalty | Penalizes under-triage of Red patients |
    | RL Agent | Policy Gradient | Adaptive triage thresholds |
    | Aggregation | FedProx | Handles non-IID data across clinics |
    | Meta-Learner | Logistic Regression | Consolidates 3 clinic outputs |
    | Dashboard | Streamlit + Plotly | Interactive visualization |

    ### Triage Categories
    - 🟢 **Green** — Non-urgent
    - 🟡 **Yellow** — Semi-urgent (24hr)
    - 🔴 **Red** — Immediate care required

    ### Dataset
    KTAS Emergency Triage dataset — 1,267 patients split across 3 clinics by age group.
    """)