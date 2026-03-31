"""
APLIKASI KLASIFIKASI BATU MEGALITIKUM
Sistem Klasifikasi Citra Batu Megalitikum Berbasis Deep Learning
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import json
import os
import pathlib
import requests
from fpdf import FPDF
import cv2
import time

# ==============================================
# KONFIGURASI HALAMAN - WAJIB DI AWAL
# ==============================================
st.set_page_config(
    page_title="Klasifikasi Batu Megalitikum",
    page_icon="🪨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==============================================
# INISIALISASI SESSION STATE - Mencegah re-render conflict
# ==============================================
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'image_uploaded' not in st.session_state:
    st.session_state.image_uploaded = None

# ==============================================
# KONFIGURASI GOOGLE DRIVE
# ==============================================
DRIVE_CONFIG = {
    "model_url": "https://drive.google.com/uc?export=download&id=1hRmWsJ8EmqINdMG1GCTuTjLdOfWr3JOx",
    "class_names_url": "https://drive.google.com/uc?export=download&id=1xHJ7tIuuUt-FEcGGTdxS2N5NvH03h6AK",
    "cache_dir": "/tmp/megalith_models"
}

# ==============================================
# DESKRIPSI KELAS
# ==============================================
DESKRIPSI_KELAS = {
    "Arca": "Arca adalah patung yang melambangkan nenek moyang atau dewa. Biasanya berbentuk manusia atau hewan, dan ditemukan di situs megalitik sebagai objek pemujaan.",
    "dolmen": "Dolmen adalah meja batu yang terdiri dari beberapa batu tegak yang menopang batu datar di atasnya. Digunakan sebagai tempat meletakkan sesaji atau untuk upacara.",
    "menhir": "Menhir adalah tugu batu tegak yang didirikan sebagai tanda peringatan atau simbol kekuatan. Biasanya ditemukan berdiri sendiri atau berkelompok.",
    "dakon": "Dakon adalah batu berlubang-lubang yang menyerupai papan permainan congkak. Diduga digunakan untuk ritual atau permainan tradisional.",
    "batu_datar": "Batu datar adalah batu besar berbentuk lempengan yang mungkin digunakan sebagai alas atau tempat duduk dalam upacara adat.",
    "Kubur_batu": "Kubur batu adalah peti mati yang terbuat dari batu, digunakan untuk mengubur mayat pada masa megalitik. Biasanya ditemukan di dalam tanah.",
    "Lesung_batu": "Lesung batu adalah batu berlubang yang digunakan sebagai wadah untuk menumbuk atau menghaluskan bahan makanan pada masa megalitikum."
}

# ==============================================
# KONFIGURASI
# ==============================================
CONFIDENCE_THRESHOLD = 0.60
MIN_TEXTURE_VARIANCE = 200

# ==============================================
# FUNGSI DOWNLOAD
# ==============================================
def download_file_from_drive(url, filepath):
    """Download file dari Google Drive"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if "id=" in url:
            file_id = url.split("id=")[1].split("&")[0]
            download_url = f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
        else:
            download_url = url
        
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        st.error(f"Gagal download: {str(e)}")
        return False

# ==============================================
# LOAD MODEL - Tanpa spinner yang bisa trigger re-render
# ==============================================
@st.cache_resource
def load_tflite_model():
    """Load model TFLite"""
    try:
        import tensorflow as tf
        
        cache_dir = pathlib.Path(DRIVE_CONFIG["cache_dir"])
        model_path = cache_dir / "megalitikum_model.tflite"
        
        if not model_path.exists():
            download_file_from_drive(DRIVE_CONFIG["model_url"], str(model_path))
        
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        
        return interpreter, interpreter.get_input_details(), interpreter.get_output_details()
        
    except Exception as e:
        st.error(f"Gagal load model: {str(e)}")
        return None, None, None

@st.cache_data
def load_class_names():
    """Load class names"""
    try:
        cache_dir = pathlib.Path(DRIVE_CONFIG["cache_dir"])
        class_path = cache_dir / "class_names.json"
        
        if not class_path.exists():
            download_file_from_drive(DRIVE_CONFIG["class_names_url"], str(class_path))
        
        if class_path.exists():
            with open(class_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return list(DESKRIPSI_KELAS.keys())
    except:
        return list(DESKRIPSI_KELAS.keys())

# ==============================================
# FUNGSI FILTER GAMBAR
# ==============================================
def is_megalith_image(image):
    """Filter kualitas gambar"""
    try:
        img_array = np.array(image.convert('RGB'))
        r_mean = np.mean(img_array[:,:,0])
        g_mean = np.mean(img_array[:,:,1])
        b_mean = np.mean(img_array[:,:,2])
        
        if g_mean > r_mean * 1.3 and g_mean > b_mean * 1.3:
            return False, "Gambar didominasi warna hijau"
        if b_mean > r_mean * 1.4 and b_mean > g_mean * 1.4:
            return False, "Gambar didominasi warna biru"
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        if np.var(gray) < MIN_TEXTURE_VARIANCE:
            return False, "Tekstur terlalu halus"
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 100:
            return False, "Gambar terlalu blur"
        
        brightness = np.mean(gray)
        if brightness < 40 or brightness > 220:
            return False, "Pencahayaan tidak optimal"
        
        return True, "Gambar valid"
    except:
        return False, "Error analisis"

# ==============================================
# FUNGSI PREDIKSI
# ==============================================
def predict_tflite(interpreter, input_details, output_details, image):
    """Prediksi TFLite"""
    img = image.convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    input_data = np.expand_dims(img_array, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

# ==============================================
# FUNGSI PDF
# ==============================================
def buat_pdf_hasil(nama_file, kelas, confidence, top3, deskripsi):
    """Generate PDF"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(200, 10, txt="HASIL KLASIFIKASI BATU MEGALITIKUM", ln=1, align='C')
    pdf.ln(8)
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 8, txt=f"File: {nama_file}", ln=1)
    pdf.cell(200, 8, txt=f"Hasil: {kelas}", ln=1)
    pdf.cell(200, 8, txt=f"Confidence: {confidence:.2%}", ln=1)
    pdf.ln(5)
    pdf.set_font("Arial", style='B', size=11)
    pdf.cell(200, 8, txt="Top 3 Prediksi:", ln=1)
    pdf.set_font("Arial", size=11)
    for i, (k, c) in enumerate(top3, 1):
        pdf.cell(200, 8, txt=f"{i}. {k}: {c:.2%}", ln=1)
    pdf.ln(5)
    pdf.set_font("Arial", style='B', size=11)
    pdf.cell(200, 8, txt="Deskripsi:", ln=1)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, txt=deskripsi)
    return pdf.output(dest='S').encode('latin1')

# ==============================================
# FUNGSI RESET STATE - Penting untuk hindari conflict
# ==============================================
def reset_prediction():
    st.session_state.prediction_done = False
    st.session_state.prediction_result = None

# ==============================================
# MAIN APP
# ==============================================
def main():
    # Header - gunakan native components, hindari unsafe HTML dinamis
    st.title("🪨 Klasifikasi Batu Megalitikum")
    st.caption("Sistem Identifikasi Otomatis Berbasis Deep Learning")
    
    # Load model - tanpa spinner yang bisa trigger re-render conflict
    interpreter, input_details, output_details = load_tflite_model()
    class_names = load_class_names()
    
    if interpreter is None:
        st.error("Model tidak dapat dimuat. Periksa koneksi internet.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Panduan")
        st.markdown("1. Upload atau ambil foto batu")
        st.markdown("2. Sistem verifikasi gambar")
        st.markdown("3. Klik klasifikasi")
        st.markdown("4. Lihat hasil dan download PDF")
        st.markdown("---")
        st.markdown("**Kelas:** Arca, Dolmen, Menhir, Dakon, Batu Datar, Kubur Batu, Lesung Batu")
    
    # Input section - gunakan key untuk hindari re-render conflict
    st.markdown("### 📷 Input Gambar")
    
    sumber = st.radio(
        "Sumber gambar:",
        ["Upload File", "Kamera"],
        horizontal=True,
        label_visibility="collapsed",
        key="sumber_radio"
    )
    
    gambar = None
    if sumber == "Upload File":
        gambar = st.file_uploader(
            "Pilih gambar",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed",
            key="file_uploader"
        )
    else:
        gambar = st.camera_input("Ambil foto", key="camera_input")
    
    # Reset state jika gambar berubah
    if gambar != st.session_state.image_uploaded:
        st.session_state.image_uploaded = gambar
        reset_prediction()
    
    if not gambar:
        st.info("👆 Pilih atau ambil foto untuk memulai")
        return
    
    # Proses gambar - tampilkan langsung tanpa conditional complex
    image = Image.open(gambar)
    st.image(image, caption="Gambar input", use_container_width=True)
    
    # Detail gambar dalam expander
    with st.expander("📊 Detail Gambar", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Ukuran", f"{image.size[0]} x {image.size[1]}")
        col2.metric("Format", image.format or "Unknown")
        col3.metric("Mode", image.mode)
    
    # Verifikasi gambar
    st.markdown("### 🔍 Verifikasi")
    is_valid, reason = is_megalith_image(image)
    
    if not is_valid:
        st.error(f"❌ {reason}")
        with st.expander("💡 Tips", expanded=False):
            st.markdown("- Foto batu dengan pencahayaan cukup")
            st.markdown("- Hindari background ramai")
            st.markdown("- Pastikan fokus pada tekstur batu")
        return
    
    st.success(f"✅ {reason}")
    
    # Tombol klasifikasi - gunakan key dan on_click callback
    if st.button("🚀 Klasifikasi Sekarang", type="primary", key="classify_btn", on_click=reset_prediction):
        # Lakukan prediksi
        img_enh = image.filter(ImageFilter.SHARPEN)
        img_enh = ImageEnhance.Contrast(img_enh).enhance(1.2)
        
        predictions = predict_tflite(interpreter, input_details, output_details, img_enh)
        pred_idx = int(np.argmax(predictions))
        pred_class = class_names[pred_idx] if pred_idx < len(class_names) else "Unknown"
        confidence = float(predictions[pred_idx])
        
        top_idx = np.argsort(predictions)[-3:][::-1]
        top_3 = [(class_names[i] if i < len(class_names) else "Unknown", float(predictions[i])) for i in top_idx]
        
        # Simpan hasil ke session state
        st.session_state.prediction_done = True
        st.session_state.prediction_result = {
            'pred_class': pred_class,
            'confidence': confidence,
            'top_3': top_3,
            'predictions': predictions,
            'top_idx': top_idx,
            'gambar_name': getattr(gambar, 'name', 'foto.jpg')
        }
    
    # Tampilkan hasil HANYA jika ada di session state
    # Ini mencegah re-render conflict karena komponen tidak berubah struktur
    if st.session_state.prediction_done and st.session_state.prediction_result:
        result = st.session_state.prediction_result
        pred_class = result['pred_class']
        confidence = result['confidence']
        top_3 = result['top_3']
        predictions = result['predictions']
        top_idx = result['top_idx']
        
        st.markdown("### 📊 Hasil")
        
        if confidence >= CONFIDENCE_THRESHOLD:
            # Gunakan native components, hindari dynamic HTML
            st.success(f"**{pred_class}**")
            st.metric("Confidence", f"{confidence:.1%}")
            st.info(f"**Deskripsi:** {DESKRIPSI_KELAS.get(pred_class, 'Tidak tersedia')}")
            
            st.markdown("**Prediksi Lainnya:**")
            for i, (cls, conf) in enumerate(top_3, 1):
                if i > 1:
                    st.progress(min(conf, 1.0), text=f"{i}. {cls}: {conf:.1%}")
            
            st.markdown("**Distribusi Probabilitas:**")
            chart_data = {
                "Kelas": [class_names[i] if i < len(class_names) else "Unknown" for i in top_idx],
                "Probabilitas": [float(predictions[i]) for i in top_idx]
            }
            st.bar_chart(chart_data, x="Kelas", y="Probabilitas")
            
            # PDF Download
            pdf_bytes = buat_pdf_hasil(
                result['gambar_name'],
                pred_class, confidence, top_3,
                DESKRIPSI_KELAS.get(pred_class, "")
            )
            
            st.download_button(
                "📥 Download PDF",
                data=pdf_bytes,
                file_name=f"hasil_{pred_class}.pdf",
                mime="application/pdf",
                key="pdf_download"
            )
            
        else:
            st.warning(f"**Confidence Rendah:** {confidence:.1%}")
            st.info("Coba foto dengan pencahayaan lebih baik.")
            st.markdown("**Semua Prediksi:**")
            for i, (cls, conf) in enumerate(top_3, 1):
                st.write(f"{i}. {cls}: {conf:.1%}")
    
    # Footer - static HTML, aman
    st.markdown("---")
    st.caption("Aplikasi Klasifikasi Batu Megalitikum | Deep Learning Research")

# ==============================================
# ENTRY POINT
# ==============================================
if __name__ == "__main__":
    main()
