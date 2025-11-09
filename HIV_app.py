import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="HIV Risk Tahmini", layout="centered")

# --- Modeli yükle: Pipeline (ön işleme + RF) ---
with open("hiv_model_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🦠 HIV Risk Tahmini Aracı")
st.markdown("Aşağıdaki parametreleri girerek bireyin HIV riski tahmin edilebilir:")

# Bu label'lar ekranda gözüken yazı, önemli olan aşağıda DataFrame'deki kolon isimleri
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
    "Marital Status",   # DİKKAT: Eğitimde böyleyse aynen böyle olmalı
    ["UNMARRIED", "Married", "Widowed", "Divorced", "Cohabiting"]
)

uyusturucu_kullanimi = st.selectbox(
    "Drug Taking",      # DİKKAT: Tire yok, büyük/küçük harf aynı
    ["Yes", "No"]
)

if st.button("Tahmini Hesapla"):
    # 🔴 KRİTİK NOKTA: Kolon isimleri TAM OLARAK modelin beklediği gibi:
    input_df = pd.DataFrame({
        "Places of seeking sex partners": [sex_partneri_arama_yeri],
        "Age": [yas],
        "Educational Background": [egitim_gecmisi],
        "Marital Status": [medeni_durum],
        "Drug Taking": [uyusturucu_kullanimi]
    })

    try:
        prediction = model.predict(input_df)[0]

        prob_pos = None
        positive_label = None

        # Olasılık hesapla (varsa)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]

            # Pozitif sınıfı bulmaya çalış
            pos_idx = None
            for i, c in enumerate(model.classes_):
                if str(c).lower() in ["1", "positive", "yes", "true"]:
                    pos_idx = i
                    break
            if pos_idx is None and len(model.classes_) > 1:
                pos_idx = 1

            if pos_idx is not None:
                prob_pos = probs[pos_idx]
                positive_label = model.classes_[pos_idx]

        # Pozitif label bulunamadıysa varsayılan
        if positive_label is None:
            if len(model.classes_) > 1:
                positive_label = model.classes_[1]
            else:
                positive_label = model.classes_[0]

        # Sonucu göster
        if prediction == positive_label:
            if prob_pos is not None:
                st.error(f"⚠️ HIV riski yüksek! Olasılık: %{prob_pos * 100:.2f}")
            else:
                st.error("⚠️ HIV riski yüksek!")
        else:
            if prob_pos is not None and prob_pos <= 1:
                st.success(f"✅ HIV riski düşük. Tahmini güvenli olasılık: %{(1 - prob_pos) * 100:.2f}")
            else:
                st.success("✅ HIV riski düşük.")

    except Exception as e:
        st.error(f"Hata oluştu: {e}")
