import streamlit as st
import pandas as pd
import pickle  # dikkat: requirements.txt'te joblib var

# Modeli yükle (pipeline!)
with open("hiv_model_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="HIV Risk Tahmini", layout="centered")
st.title("🦠 HIV Risk Tahmini Aracı")
st.markdown("Aşağıdaki parametreleri girerek bireyin HIV riski tahmin edilebilir:")

sex_partneri_arama_yeri = st.selectbox(
    "Places of seeking sex partners",
    ["Bar", "Park", "Internet", "Public Bath", "Others"]
)

yas = st.number_input("Age", min_value=0, max_value=100, value=30)

egitim_gecmisi = st.selectbox(
    "Educational Background",
    ["College Degree", "Senior High School", "Junior High School", "Illiteracy", "Primary School"]
)

medeni_durum = st.selectbox(
    "Marital Status",
    ["Unmarried", "Married", "Widowed", "Divorced", "Cohabiting"]
)

uyusturucu_kullanimi = st.selectbox(
    "Drug Taking",
    ["Yes", "No"]
)

if st.button("Tahmini Hesapla"):
    # HAM veriyi, eğitimde kullandığın kolon isimleriyle veriyoruz
    input_df = pd.DataFrame({
        "Places of seeking sex partners": [sex_partneri_arama_yeri],
        "Age": [yas],
        "Educational Background": [egitim_gecmisi],
        "Marital Staus": [medeni_durum],
        "Drug- taking": [uyusturucu_kullanimi]
    })

    try:
        prediction = model.predict(input_df)[0]

        prob_pos = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            # Pozitif sınıfı makul şekilde seç
            pos_idx = None
            for i, c in enumerate(model.classes_):
                if str(c).lower() in ["1", "positive", "yes", "true"]:
                    pos_idx = i
                    break
            if pos_idx is None and len(model.classes_) > 1:
                pos_idx = 1
            if pos_idx is not None:
                prob_pos = probs[pos_idx]

        # Pozitif etiket hangisi?
        positive_label = None
        for c in model.classes_:
            if str(c).lower() in ["1", "positive", "yes", "true"]:
                positive_label = c
                break
        if positive_label is None and len(model.classes_) > 1:
            positive_label = model.classes_[1]

        if prediction == positive_label:
            if prob_pos is not None:
                st.error(f"⚠️ HIV riski yüksek! Olasılık: %{prob_pos * 100:.2f}")
            else:
                st.error("⚠️ HIV riski yüksek!")
        else:
            if prob_pos is not None:
                st.success(f"✅ HIV riski düşük. Tahmini güvenli olasılık: %{(1 - prob_pos) * 100:.2f}")
            else:
                st.success("✅ HIV riski düşük.")

    except Exception as e:
        st.error(f"Hata oluştu: {e}")
