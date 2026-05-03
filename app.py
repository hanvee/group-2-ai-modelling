import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🔍",
    layout="centered"
)

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

CLASS_EMOJI = {
    "airplane": "✈️", "automobile": "🚗", "bird": "🐦",
    "cat": "🐱",      "deer": "🦌",       "dog": "🐶",
    "frog": "🐸",     "horse": "🐴",      "ship": "🚢",
    "truck": "🚛"
}

IMG_SIZE = 64

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_model.keras")

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def predict(model, img_array: np.ndarray):
    probs = model.predict(img_array, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:3]
    return probs, top_idx

# ── Header ──
st.title("🔍 CIFAR-10 Image Classifier")
st.markdown("""
Aplikasi klasifikasi gambar menggunakan model **CNN yang dioptimasi** 
dengan Transfer Learning (MobileNetV2).  
Upload gambar, dan model akan memprediksi kategori objeknya.

**Kategori:** ✈️ Airplane · 🚗 Automobile · 🐦 Bird · 🐱 Cat · 🦌 Deer 
· 🐶 Dog · 🐸 Frog · 🐴 Horse · 🚢 Ship · 🚛 Truck
""")

st.divider()

# ── Load model ──
with st.spinner("Memuat model..."):
    model = load_model()
st.success("Model berhasil dimuat!")

# ── Upload gambar ──
uploaded_file = st.file_uploader(
    "Upload gambar di sini",
    type=["jpg", "jpeg", "png", "webp"],
    help="Format yang didukung: JPG, JPEG, PNG, WEBP"
)

if uploaded_file:
    img = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Gambar Input")
        st.image(img, use_column_width=True)
        st.caption(f"Ukuran: {img.size[0]}×{img.size[1]} px")

    with st.spinner("Menganalisis gambar..."):
        img_array = preprocess_image(img)
        probs, top_idx = predict(model, img_array)

    pred_class = CLASS_NAMES[top_idx[0]]
    confidence = probs[top_idx[0]] * 100

    with col2:
        st.subheader("Hasil Prediksi")
        st.metric(
            label="Kelas Terdeteksi",
            value=f"{CLASS_EMOJI[pred_class]} {pred_class.upper()}",
            delta=f"Konfidiensi: {confidence:.1f}%"
        )

        st.markdown("**Top-3 Prediksi:**")
        for rank, idx in enumerate(top_idx):
            name  = CLASS_NAMES[idx]
            prob  = probs[idx] * 100
            emoji = CLASS_EMOJI[name]
            st.progress(int(prob), text=f"{rank+1}. {emoji} {name}: {prob:.1f}%")

    st.divider()

    # Bar chart semua kelas
    st.subheader("Distribusi Probabilitas Semua Kelas")
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#2563eb" if i == top_idx[0] else "#93c5fd" for i in range(10)]
    bars = ax.barh([f"{CLASS_EMOJI[c]} {c}" for c in CLASS_NAMES],
                   probs * 100, color=colors)
    ax.set_xlabel("Probabilitas (%)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.4)
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{prob*100:.1f}%", va="center", fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)

st.divider()
