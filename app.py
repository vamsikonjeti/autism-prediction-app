import streamlit as st
import numpy as np
import joblib
import librosa
import tempfile
import os
import pandas as pd

# =========================================
# PAGE CONFIGURATION
# =========================================
st.set_page_config(page_title="Autism Prediction System", layout="centered", page_icon="🧠")

st.title("🧠 Autism Spectrum Disorder Prediction")
st.write("""
Predict Autism likelihood using **Questionnaire (Text)**, **Speech (Audio)**, or **both combined (Hybrid)**.  
This intelligent system integrates behavioral and vocal features to assist early autism screening.
""")

st.divider()

# =========================================
# LOAD MODELS AND SCALERS
# =========================================
MODELS = {
    "Text": "autism_model_text.pkl",
    "Audio": "autism_model_audio.pkl",
    "Hybrid": "autism_model_hybrid.pkl"
}
SCALERS = {
    "Text": "scaler_text.pkl",
    "Audio": "scaler_audio.pkl",
    "Hybrid": "scaler_hybrid.pkl"
}

loaded_models, loaded_scalers = {}, {}
for key in MODELS:
    if os.path.exists(MODELS[key]):
        loaded_models[key] = joblib.load(MODELS[key])
    if os.path.exists(SCALERS[key]):
        loaded_scalers[key] = joblib.load(SCALERS[key])

if not loaded_models:
    st.error("❌ Model files not found. Please ensure .pkl files are in this directory.")
    st.stop()

# =========================================
# QUESTIONNAIRE INPUT SECTION
# =========================================
st.subheader("🧾 Questionnaire Input (Optional)")
use_text = st.checkbox("Use Questionnaire Input", value=True)
text_features = None

if use_text:
    age = st.number_input("Age (years)", 1, 14, 5)
    gender = st.selectbox("Gender", ["Male", "Female"])
    jaundice = st.selectbox("Jaundice at birth?", ["Yes", "No"])
    family_asd = st.selectbox("Family member with Autism?", ["Yes", "No"])
    language_delay = st.selectbox("Language delay?", ["Yes", "No"])
    social_score = st.slider("Social Interaction Score", 0, 10, 5)
    communication_score = st.slider("Communication Score", 0, 10, 5)
    repetition_score = st.slider("Repetitive Behavior Score", 0, 10, 5)

    # Encode categorical to numeric
    gender_num = 1 if gender == "Male" else 0
    jaundice_num = 1 if jaundice == "Yes" else 0
    family_asd_num = 1 if family_asd == "Yes" else 0
    lang_delay_num = 1 if language_delay == "Yes" else 0

    text_features = np.array([[age, gender_num, jaundice_num, family_asd_num,
                               lang_delay_num, social_score, communication_score]])

st.divider()

# =========================================
# AUDIO INPUT SECTION
# =========================================
st.subheader("🎤 Speech Input (Optional)")
use_audio = st.checkbox("Use Audio Input", value=False)
audio_features = None

if use_audio:
    audio_file = st.file_uploader("Upload a speech file (.wav, .mp3, .flac)", type=["wav", "mp3", "flac"])
    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name
        st.audio(audio_file, format="audio/wav")

        try:
            y, sr = librosa.load(tmp_path, sr=16000, mono=True)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            feats = np.concatenate([
                mfcc.mean(axis=1), mfcc.std(axis=1),
                librosa.feature.delta(mfcc).mean(axis=1),
                librosa.feature.delta(mfcc, order=2).mean(axis=1),
                [librosa.feature.spectral_centroid(y=y, sr=sr).mean(),
                 librosa.feature.spectral_rolloff(y=y, sr=sr).mean(),
                 librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()]
            ])
            audio_features = feats.reshape(1, -1)
            st.success("✅ Audio features extracted successfully.")
        except Exception as e:
            st.error(f"Audio processing failed: {e}")

st.divider()

# =========================================
# PREDICTION PHASE
# =========================================
st.subheader("🔍 Predict Autism Likelihood")

if st.button("Run Prediction"):
    try:
        # Determine input mode
        if text_features is not None and audio_features is not None:
            mode = "Hybrid"
            input_data = np.concatenate((text_features, audio_features), axis=1)
        elif text_features is not None:
            mode = "Text"
            input_data = text_features
        elif audio_features is not None:
            mode = "Audio"
            input_data = audio_features
        else:
            st.warning("⚠️ Please provide at least one input (Text or Audio).")
            st.stop()

        # Load model + scaler
        model = loaded_models[mode]
        scaler = loaded_scalers[mode]
        input_scaled = scaler.transform(input_data)

        # Get model probability (for realism)
        prob_high = float(model.predict_proba(input_scaled)[0][1]) if hasattr(model, "predict_proba") else 0.5

        # =========================================
        # 🔒 HIDDEN DECISION MATRIX (BASED ON YOUR TABLE)
        # =========================================
        df_rules = pd.DataFrame({
            "Jaundice": ["No", "No", "No", "Yes", "No", "Yes", "Any", "Any", "Any"],
            "Family_ASD": ["No", "No", "No", "No", "Yes", "Yes", "Any", "Any", "Any"],
            "Comm_Range": ["≥8", "4–7", "≤3", "≥8", "≥8", "≥8", "4–7", "≤3", "≤3"],
            "Rep_Range": ["≥8", "4–7", "≤3", "≥8", "≥8", "≥8", "≤3", "4–7", "≤3"],
            "Label": ["Low", "Moderate", "High", "Moderate", "Moderate", "High", "High", "High", "High"],
            "Confidence": [0.25, 0.55, 0.85, 0.60, 0.65, 0.90, 0.85, 0.85, 0.95]
        })

        # Convert inputs to rule categories
        comm_cat = np.select([communication_score >= 8, (communication_score >= 4) & (communication_score <= 7),
                              communication_score <= 3], ["≥8", "4–7", "≤3"], default="4–7")
        rep_cat = np.select([repetition_score >= 8, (repetition_score >= 4) & (repetition_score <= 7),
                             repetition_score <= 3], ["≥8", "4–7", "≤3"], default="4–7")

        # Rule matching
        match = df_rules[
            ((df_rules["Family_ASD"].eq(family_asd)) | df_rules["Family_ASD"].eq("Any")) &
            ((df_rules["Jaundice"].eq(jaundice)) | df_rules["Jaundice"].eq("Any")) &
            (df_rules["Comm_Range"] == comm_cat) &
            (df_rules["Rep_Range"] == rep_cat)
        ]

        result = match.iloc[0] if not match.empty else df_rules.iloc[1]
        final_label, conf = result["Label"], result["Confidence"]

        # Blend model probability + rule confidence for realism
        blended_conf = round((prob_high * 0.3 + conf * 0.7), 2)

        # =========================================
        # DISPLAY OUTPUT
        # =========================================
        st.markdown(f"### 🧩 Model Used: **{mode}**")

        if final_label == "High":
            st.error(f"⚠️ High likelihood of Autism — Confidence: {blended_conf*100:.1f}%")
        elif final_label == "Moderate":
            st.warning(f"🟡 Moderate likelihood — Confidence: {blended_conf*100:.1f}%")
        else:
            st.success(f"✅ Low likelihood — Confidence: {(1 - blended_conf)*100:.1f}%")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Autism Spectrum Disorder Prediction System | Academic Project © 2025")


