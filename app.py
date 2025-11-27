import streamlit as st
import cv2
import numpy as np
from inference import load_model, predict
from lime_explain import generate_lime
from inference import generate_gradcam

from xai_descriptions import describe_gradcam, describe_lime
from llm_explainer import explain_with_llm

st.set_page_config(
    page_title="Truth Eye",
    page_icon="👁️",
    layout="centered"
)

st.title("👁️ Truth Eye")
st.subheader("AI-Generated Face Detection")

model = load_model()

uploaded = st.file_uploader("Upload a face image", type=["jpg","png","jpeg"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Input Image")

    with st.spinner("Analyzing image..."):
        label, confidence, face = predict(img, model)

    if label is None or confidence is None:
        st.warning("⚠️ No face detected. Please upload a clear frontal face.")
    else:
        if label == "FAKE":
            st.error(f"🚨 AI GENERATED FACE")
            st.write(label)
        else:
            st.success(f"✅ REAL FACE")
            st.write(label)

        st.metric(
            label="Confidence",
            value=f"{confidence * 100:.2f}%"
        )

        st.progress(int(confidence * 100))

        risk = "LOW"
        if confidence > 0.8:
            risk = "HIGH"
        elif confidence > 0.6:
            risk = "MODERATE"

        st.warning(f"Threat Level: {risk}")

    if label is not None:
        # Existing results output...

        st.header("🔍 Explainability (XAI)")

        st.subheader("Grad-CAM")

        # Generate Grad-CAM
        with st.spinner("Generating Grad-CAM..."):
            gradcam_img, heatmap = generate_gradcam(img, face, model)
            st.image(gradcam_img, caption="Grad-CAM Heatmap")

            # xai_text = describe_gradcam(heatmap)
            # llm_text = explain_with_llm(label, confidence, xai_text)
            #
            gradcam_text = describe_gradcam(heatmap)
            #
            # st.subheader("🧠 Model Explanation")
            # st.write(llm_text)

        st.subheader("LIME")

        # Generate LIME
        with st.spinner("Running LIME Explanation..."):
            lime_img, lime_mask, lime_contrib = generate_lime(img, face, model)
            st.image(lime_img, caption="LIME Superpixel Explanation")
            lime_text = describe_lime(lime_mask, lime_contrib)

            combined_xai_text = f"Grad-CAM: {gradcam_text}\n\nLIME: {lime_text}"

            # LLM explanation
            llm_text = explain_with_llm(label, confidence, combined_xai_text)

            st.subheader("🧠 Model Explanation")
            st.write(llm_text)


