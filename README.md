# 🍄 Prediksi Jamur: Bisa Dimakan atau Beracun?

Aplikasi web interaktif berbasis **Streamlit** untuk memprediksi apakah jamur aman dikonsumsi atau beracun, **Random Forest** sebagai model yang dilatih dari dataset Mushroom.

---

## 📦 Fitur

✅ Prediksi jamur berdasarkan 12 karakteristik  
✅ Input dropdown berbahasa Indonesia  
✅ Hasil prediksi yang mudah dipahami  
✅ Siap dijalankan secara lokal atau dideploy ke Streamlit Cloud  

---

## 🌐 Cara Deploy ke Streamlit Cloud

```
1. Push project ini ke GitHub (public atau private repo).
2. Buka https://streamlit.io/cloud.
3. Klik "New App".
4. Pilih:
      - > Repository: username/nama-repo
      - > Branch: main
      - > File: app.py
5. Klik Deploy.
````



## 🚀 Cara Menjalankan Secara Lokal

1. **Clone repositori**
```bash
git clone https://github.com/username/nama-repo.git
cd nama-repo
````

2. **(Opsional) Untuk virtual environment**
````
python -m venv env
source env/bin/activate  # Untuk Mac/Linux
env\Scripts\activate     # Untuk Windows
````

3. **Install dependency**
````
pip install -r requirements.txt
````

4. **Jalankan aplikasi**
````
streamlit run app.py
````

