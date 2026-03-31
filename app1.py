"""
APLIKASI KLASIFIKASI BATU MEGALITIKUM
Sistem Klasifikasi Citra Batu Megalitikum Berbasis Deep Learning
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import json
import os
import time
import pathlib
import requests
from fpdf import FPDF
import cv2

# ==============================================
# KONFIGURASI HALAMAN
# ==============================================
st.set_page_config(
    page_title="Klasifikasi Batu Megalitikum",
    page_icon="🪨",
    layout="centered"
)

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
# FUNGSI DOWNLOAD FILE DARI GOOGLE DRIVE
# ==============================================
def download_file_from_drive(url, filepath):
    """Download file dari Google Drive dengan handling redirect"""
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
        st.error(f"Gagal download file: {str(e)}")
        return False

# ==============================================
# LOAD MODEL
# ==============================================
@st.cache_resource
def load_tflite_model():
    """Load model TFLite dari cache atau download dari Drive"""
    try:
        import tensorflow as tf
        
        cache_dir = pathlib.Path(DRIVE_CONFIG["cache_dir"])
        model_path = cache_dir / "megalitikum_model.tflite"
        
        if not model_path.exists():
            with st.spinner("Mengunduh model dari Google Drive..."):
                if not download_file_from_drive(DRIVE_CONFIG["model_url"], str(model_path)):
                    return None, None, None
                st.success("✅ Model berhasil diunduh!")
        
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        return interpreter, input_details, output_details
        
    except ImportError:
        st.error("TensorFlow tidak terinstal.")
        return None, None, None
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None, None, None

@st.cache_data
def load_class_names():
    """Load class names dari cache atau download dari Drive"""
    try:
        cache_dir = pathlib.Path(DRIVE_CONFIG["cache_dir"])
        class_path = cache_dir / "class_names.json"
        
        if not class_path.exists():
            with st.spinner("Mengunduh data kelas..."):
                if not download_file_from_drive(DRIVE_CONFIG["class_names_url"], str(class_path)):
                    return list(DESKRIPSI_KELAS.keys())
                st.success("✅ Data kelas berhasil diunduh!")
        
        with open(class_path, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
            return class_names
            
    except Exception as e:
        st.warning(f"Gagal load class names: {str(e)}")
        return list(DESKRIPSI_KELAS.keys())

# ==============================================
# FUNGSI FILTER GAMBAR
# ==============================================
def is_megalith_image(image):
    """Filter untuk memastikan gambar adalah batu megalitikum"""
    try:
        img_array = np.array(image.convert('RGB'))
        
        r_mean = np.mean(img_array[:,:,0])
        g_mean = np.mean(img_array[:,:,1])
        b_mean = np.mean(img_array[:,:,2])
        
        if g_mean > r_mean * 1.3 and g_mean > b_mean * 1.3:
            return False, "Gambar didominasi warna hijau (mungkin tumbuhan)"
        
        if b_mean > r_mean * 1.4 and b_mean > g_mean * 1.4:
            return False, "Gambar didominasi warna biru (mungkin langit/air)"
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        texture_variance = np.var(gray)
        
        if texture_variance < MIN_TEXTURE_VARIANCE:
            return False, "Tekstur terlalu halus untuk dikategorikan batu"
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 100:
            return False, "Gambar terlalu blur, detail tidak jelas"
        
        brightness = np.mean(gray)
        
        if brightness < 40:
            return False, "Gambar terlalu gelap"
        if brightness > 220:
            return False, "Gambar terlalu terang (overexposed)"
        
        return True, "Gambar memenuhi kriteria analisis"
        
    except Exception as e:
        return False, f"Error saat analisis: {str(e)}"

# ==============================================
# FUNGSI PREDIKSI
# ==============================================
def predict_tflite(interpreter, input_details, output_details, image):
    """Prediksi menggunakan TFLite"""
    img = image.convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    input_data = np.expand_dims(img_array, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    return output_data[0]

# ==============================================
# FUNGSI PDF
# ==============================================
def buat_pdf_hasil(nama_file, kelas, confidence, top3, deskripsi):
    """Buat PDF hasil klasifikasi"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(200, 10, txt="HASIL KLASIFIKASI BATU MEGALITIKUM", ln=1, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"File: {nama_file}", ln=1)
    pdf.cell(200, 10, txt=f"Hasil: {kelas}", ln=1)
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2%}", ln=1)
    pdf.ln(5)
    
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, txt="Top 3 Prediksi:", ln=1)
    pdf.set_font("Arial", size=12)
    for i, (k, c) in enumerate(top3, 1):
        pdf.cell(200, 10, txt=f"{i}. {k}: {c:.2%}", ln=1)
    
    pdf.ln(5)
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, txt="Deskripsi:", ln=1)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=deskripsi)
    
    return pdf.output(dest='S').encode('latin1')

# ==============================================
# UI HEADER
# ==============================================
st.title("🪨 Klasifikasi Batu Megalitikum")
st.caption("Sistem Identifikasi Otomatis Berbasis Deep Learning")

# ==============================================
# SIDEBAR INFO
# ==============================================
with st.sidebar:
    st.markdown("### 📖 Panduan Penggunaan")
    st.markdown("""
    1. Pilih sumber gambar (upload atau kamera)
    2. Pastikan foto batu terlihat jelas
    3. Klik tombol klasifikasi
    4. Lihat hasil dan download laporan
    """)
    
    st.markdown("### 🗿 Kelas yang Didukung")
    kelas_list = ["Arca", "Dolmen", "Menhir", "Dakon", "Batu Datar", "Kubur Batu", "Lesung Batu"]
    for k in kelas_list:
        st.markdown(f"- {k}")
    
    st.markdown("---")
    st.markdown("### 📸 Kriteria Gambar")
    st.markdown("""
    - Format: JPG, JPEG, PNG
    - Pencahayaan cukup
    - Objek batu terlihat jelas
    - Hindari background ramai
    """)

# ==============================================
# LOAD MODEL
# ==============================================
with st.spinner("Memuat model..."):
    interpreter, input_details, output_details = load_tflite_model()
    class_names = load_class_names()

if interpreter is None:
    st.error("❌ Model tidak dapat dimuat. Periksa koneksi internet dan pastikan file tersedia di Google Drive.")
    st.stop()

# ==============================================
# MAIN INTERFACE
# ==============================================
st.markdown("### 📤 Upload atau Ambil Foto")

# Pilihan sumber gambar
sumber = st.radio(
    "Pilih sumber gambar:",
    ["Upload File", "Ambil Foto Kamera"],
    horizontal=True
)

gambar = None
if sumber == "Upload File":
    gambar = st.file_uploader(
        "Pilih file gambar",
        type=['jpg', 'jpeg', 'png'],
        help="Format yang didukung: JPG, JPEG, PNG"
    )
else:
    st.info("📷 Pastikan pencahayaan cukup saat mengambil foto")
    gambar = st.camera_input("Ambil foto")

if gambar:
    # Proses gambar
    image = Image.open(gambar)
    
    # Tampilkan gambar
    st.image(image, caption="Gambar yang diupload", use_container_width=True)
    
    # Detail gambar dalam expander
    with st.expander("📊 Detail Gambar"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ukuran", f"{image.size[0]} x {image.size[1]} px")
        with col2:
            st.metric("Format", image.format if image.format else "Unknown")
        with col3:
            st.metric("Mode", image.mode)
    
    # Analisis awal
    st.markdown("### 🔍 Verifikasi Gambar")
    
    with st.spinner("Memeriksa kualitas gambar..."):
        is_valid, reason = is_megalith_image(image)
    
    if not is_valid:
        st.warning(f"⚠️ {reason}")
        with st.expander("💡 Rekomendasi"):
            st.markdown("""
            - Foto batu secara langsung dengan jarak dekat
            - Gunakan pencahayaan alami atau cukup terang
            - Hindari objek lain yang mendominasi frame
            - Pastikan fokus kamera pada tekstur batu
            """)
    else:
        st.success(f"✅ {reason}")
        
        # Tombol klasifikasi
        if st.button("🚀 Mulai Klasifikasi", type="primary"):
            with st.spinner("Sedang menganalisis gambar..."):
                # Enhancement untuk prediksi lebih akurat
                img_enhanced = image.filter(ImageFilter.SHARPEN)
                enhancer = ImageEnhance.Contrast(img_enhanced)
                img_enhanced = enhancer.enhance(1.2)
                
                # Prediksi
                predictions = predict_tflite(interpreter, input_details, output_details, img_enhanced)
                pred_idx = int(np.argmax(predictions))
                pred_class = class_names[pred_idx] if pred_idx < len(class_names) else "Unknown"
                confidence = float(predictions[pred_idx])
                
                # Top 3 predictions
                top_3_idx = np.argsort(predictions)[-3:][::-1]
                top_3 = [(class_names[i] if i < len(class_names) else "Unknown", float(predictions[i])) for i in top_3_idx]
            
            # Tampilkan hasil
            st.markdown("### 📊 Hasil Klasifikasi")
            
            if confidence >= CONFIDENCE_THRESHOLD:
                # Hasil berhasil
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🏷️ Kelas Terdeteksi", pred_class)
                with col2:
                    if confidence > 0.8:
                        st.metric("📈 Confidence", f"{confidence:.1%}", delta="Tinggi")
                    else:
                        st.metric("📈 Confidence", f"{confidence:.1%}", delta="Sedang")
                
                st.markdown("#### 📝 Deskripsi")
                st.info(DESKRIPSI_KELAS.get(pred_class, "Deskripsi tidak tersedia untuk kelas ini."))
                
                # Top 3 predictions
                st.markdown("#### 🏆 Top 3 Prediksi")
                for i, (cls, conf) in enumerate(top_3[1:], 2):
                    st.progress(conf, text=f"{i}. {cls}: {conf:.1%}")
                
                # Grafik distribusi probabilitas
                st.markdown("#### 📈 Distribusi Probabilitas")
                chart_data = {
                    "Kelas": [class_names[i] if i < len(class_names) else "Unknown" for i in top_3_idx],
                    "Probabilitas": [float(predictions[i]) for i in top_3_idx]
                }
                st.bar_chart(chart_data, x="Kelas", y="Probabilitas")
                
                # Download PDF
                pdf_bytes = buat_pdf_hasil(
                    gambar.name if hasattr(gambar, 'name') else f"foto_{int(time.time())}.jpg",
                    pred_class,
                    confidence,
                    top_3,
                    DESKRIPSI_KELAS.get(pred_class, "")
                )
                
                st.download_button(
                    label="📥 Download Laporan PDF",
                    data=pdf_bytes,
                    file_name=f"klasifikasi_{pred_class}_{int(time.time())}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary"
                )
                
            else:
                # Confidence rendah
                st.warning(f"⚠️ Hasil Kurang Yakin")
                st.info(f"**Prediksi terbaik:** {pred_class}\n\n**Confidence:** {confidence:.1%}\n\n💡 Saran: Coba ambil foto dengan pencahayaan lebih baik atau sudut yang lebih jelas.")
                
                st.markdown("#### 📊 Semua Prediksi")
                for i, (cls, conf) in enumerate(top_3, 1):
                    st.progress(conf, text=f"{i}. {cls}: {conf:.1%}")

# ==============================================
# FOOTER
# ==============================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666; font-size: 0.85rem;'>Aplikasi Klasifikasi Batu Megalitikum | Berdasarkan Penelitian Deep Learning</p>",
    unsafe_allow_html=True
)
