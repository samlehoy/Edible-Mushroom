import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Load model dan data asli
model = joblib.load("model.pkl")
url = "https://raw.githubusercontent.com/dataset-machine-learning/mushroom/refs/heads/main/mushroom.csv"
original_df = pd.read_csv(url, sep=';')

# Kolom fitur yang digunakan
columns = [
    'cap-shape', 'cap-color', 'cap-surface', 'bruises',
    'stalk-shape', 'stalk-root',
    'veil-color', 'population', 'habitat'
]

# Label Indonesia
label_indonesia = {
    'cap-shape': 'Bentuk Tudung',
    'cap-color': 'Warna Tudung',
    'cap-surface': 'Permukaan Tudung',
    'bruises': 'Memar',
    'stalk-shape': 'Bentuk Batang',
    'stalk-root': 'Akar Batang',
    'veil-color': 'Warna Selubung',
    'population': 'Populasi',
    'habitat': 'Habitat',
}

# Mapping nilai ke Bahasa Indonesia
opsi_mapping = {
    'cap-shape': {'bell': 'Lonceng', 'conical': 'Kerucut', 'convex': 'Cembung', 'flat': 'Datar', 'knobbed': 'Tonjolan', 'sunken': 'Cekung'},
    'cap-surface': {'fibrous': 'Berserat', 'groovesmooth': 'Beralur', 'scaly': 'Bersisik', 'smooth': 'Halus'},
    'cap-color': {'brown': 'Coklat', 'buff': 'Coklat Kekuningan', 'cinnamon': 'Kayu Manis', 'gray': 'Abu-abu', 'green': 'Hijau', 'pink': 'Merah Muda', 'purple': 'Ungu', 'red': 'Merah', 'white': 'Putih', 'yellow': 'Kuning'},
    'bruises': {'bruises': 'Ya', 'no': 'Tidak'},
    'stalk-shape': {'enlarging': 'Membesar', 'tapering': 'Mengecil'},
    'stalk-root': {'?': '?', 'bulbous': 'Menggelembung', 'club': 'Tongkat', 'equal': 'Sama Besar', 'rooted': 'Berakar Kuat'},
    'veil-color': {'brown': 'Coklat', 'orange': 'Oranye', 'white': 'Putih', 'yellow': 'Kuning'},
    'population': {'abundant': 'Sangat Banyak', 'clustered': 'Bergerombol', 'numerous': 'Banyak', 'scattered': 'Tersebar', 'several': 'Beberapa', 'solitary': 'Sendirian'},
    'habitat': {'grasses': 'Rerumputan', 'leaves': 'Dedaunan', 'meadows': 'Padang Rumput', 'paths': 'Jalan Setapak', 'urban': 'Perkotaan', 'waste': 'Tempat Sampah', 'woods': 'Hutan'},
}

# Streamlit page config
st.set_page_config(page_title="Prediksi Jamur", page_icon="🍄")

# Sidebar Navigasi
page = st.sidebar.selectbox("Navigasi", ["🔮 Prediksi", "📊 Dashboard"])

# ================================
# Halaman PREDIKSI
# ================================
if page == "🔮 Prediksi":
    st.title("🍄 Prediksi Jamur: Bisa Dimakan atau Beracun?")

    user_input = {}
    for col in columns:
        options = sorted(original_df[col].unique())
        label = label_indonesia.get(col, col)
        if col in opsi_mapping:
            opsi_indo = [opsi_mapping[col][opt] for opt in options]
            mapping_balik = {opsi_mapping[col][opt]: opt for opt in options}
            selected = st.selectbox(label, opsi_indo)
            user_input[col] = mapping_balik[selected]
        else:
            user_input[col] = st.selectbox(label, options)

    if st.button("🔍 Prediksi Sekarang"):
        input_df = pd.DataFrame([user_input])
        input_encoded = pd.DataFrame()

        try:
            for col in columns:
                le = LabelEncoder().fit(original_df[col])
                input_encoded[col] = le.transform(input_df[col])
        except Exception as e:
            st.error(f"⚠️ Error saat encoding input: {e}")
            st.stop()

        # Pastikan urutan kolom sesuai dengan model
        input_encoded = input_encoded[model.feature_names_in_]

        # Prediksi
        prediction = model.predict(input_encoded)[0]
        proba = model.predict_proba(input_encoded)[0]

        edible_prob = proba[0] * 100
        poisonous_prob = proba[1] * 100

        # Tentukan kategori confidence berdasarkan probabilitas edible
        if edible_prob > 72:
            hasil = '✅ Jamur ini DIPERKIRAKAN AMAN DIMAKAN (Confidence Tinggi)'
        elif 50 <= edible_prob <= 72:
            hasil = '⚠ Jamur ini DIPERKIRAKAN AMAN DIMAKAN (Confidence Sedang)'
        else:
            hasil = '❌ Jamur ini TIDAK DISARANKAN UNTUK DIMAKAN (Confidence Rendah)'

        # Tampilkan hasil
        st.markdown(f"### 🧠 Hasil Prediksi:")
        st.info(hasil)

        st.markdown(f"### 📊 Probabilitas:")
        st.markdown(f"- ✅ Bisa dimakan: **{edible_prob:.2f}%**")
        st.markdown(f"- ☠️ Beracun: **{poisonous_prob:.2f}%**")


# ================================
# Halaman DASHBOARD
# ================================
elif page == "📊 Dashboard":
    st.title("📊 Dashboard Dataset & Evaluasi Model")

    # Contoh Data Asli (Belum Preprocessing)
    st.subheader("📂 Contoh Data Belum Diproses (17 Kolom Features)")
    st.dataframe(original_df.head(10))

    # Contoh Data Setelah Preprocessing
    st.subheader("⚙️ Contoh Data Setelah Preprocessing (9 Kolom Features)")
    data_preprocessed = original_df.copy()

    data_preprocessed.drop(columns=[
        'gill-spacing', 'gill-size', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
        'spore-print-color', 'gill-color', 'stalk-color-below-ring', 'stalk-color-above-ring'
    ], inplace=True, errors='ignore')

    st.dataframe(data_preprocessed.head(10))


    # Preprocessing seperti di training
    data_encoded = original_df.copy()
    le = LabelEncoder()
    for col in data_encoded.columns:
        data_encoded[col] = le.fit_transform(data_encoded[col])

    data_encoded.drop(columns=[
        'gill-spacing', 'gill-size', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
        'spore-print-color', 'gill-color', 'stalk-color-below-ring', 'stalk-color-above-ring'
    ], inplace=True)

    X = data_encoded.drop(columns=['class'])
    y = data_encoded['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)

    # Classification Report
    st.subheader("📄 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()[['precision', 'recall', 'f1-score']]

    # Ubah label agar mudah dipahami
    report_df.rename(index={
        '0': '✅ Bisa Dimakan (Edible)',
        '1': '☠️ Beracun (Poisonous)',
        'accuracy': '🎯 Akurasi',
        'macro avg': '📊 Rata-rata per Kelas',
        'weighted avg': '⚖️ Rata-rata Terbobot'
    }, inplace=True)

    # Tampilkan tabel dengan styling
    st.dataframe(report_df.style.format("{:.2f}"))
