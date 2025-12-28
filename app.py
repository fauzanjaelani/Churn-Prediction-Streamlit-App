import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Fungsi untuk memuat model dan scaler
@st.cache_resource
def load_models():
    """Memuat model dan scaler yang sudah disimpan"""
    try:
        model = joblib.load('gradient_boosting_churn_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"File model tidak ditemukan: {e}")
        st.info("""
        Pastikan file-file berikut ada di direktori yang sama:
        1. `gradient_boosting_churn_model.joblib`
        2. `scaler.joblib`
        
        **Catatan:** Preprocessor (`encoder.joblib`) tidak diperlukan 
        karena data sudah di-encode one-hot.
        """)
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Fungsi untuk scaling parsial sesuai dengan notebook
def apply_partial_scaling(input_array, columns_order, scaler):
    """
    Menerapkan scaling hanya pada kolom yang di-scale di notebook:
    ['age', 'number_of_dependents', 'streaming_service',
     'tech_service', 'number_of_referrals', 'satisfaction_score']
    """
    # Buat copy dari input array
    scaled_array = input_array.copy()
    
    # Definisikan kolom yang perlu di-scale sesuai notebook
    cols_to_scale = ['age', 'number_of_dependents', 'streaming_service',
                     'tech_service', 'number_of_referrals', 'satisfaction_score']
    
    try:
        # Cari indeks kolom yang perlu di-scale
        scale_indices = []
        for col in cols_to_scale:
            if col in columns_order:
                scale_indices.append(columns_order.index(col))
        
        # Debug info
        st.info(f"ℹ️ Scaling diterapkan pada {len(scale_indices)} kolom: {cols_to_scale}")
        
        # Jika scaler tidak ada, return data asli
        if scaler is None:
            st.warning("⚠️ Scaler tidak ditemukan, menggunakan data asli")
            return scaled_array
        
        # Cek kompatibilitas scaler
        if hasattr(scaler, 'n_features_in_'):
            expected_features = scaler.n_features_in_
            if expected_features != len(cols_to_scale):
                st.warning(f"⚠️ Scaler mengharapkan {expected_features} fitur, tapi kita punya {len(cols_to_scale)}")
        
        # Ekstrak hanya kolom yang perlu di-scale
        data_to_scale = input_array[:, scale_indices]
        
        # Transform menggunakan scaler
        scaled_cols = scaler.transform(data_to_scale)
        
        # Masukkan kembali ke array asli
        for i, idx in enumerate(scale_indices):
            scaled_array[0, idx] = scaled_cols[0, i]
        
        st.success("✅ Data berhasil di-scaling (partial)")
        
        # Tampilkan perbandingan untuk debugging
        with st.expander("🔍 Detail Scaling (Debug)"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Sebelum Scaling:**")
                for i, idx in enumerate(scale_indices):
                    st.write(f"{columns_order[idx]}: {input_array[0, idx]:.2f}")
            with col2:
                st.write("**Setelah Scaling:**")
                for i, idx in enumerate(scale_indices):
                    st.write(f"{columns_order[idx]}: {scaled_array[0, idx]:.2f}")
        
        return scaled_array
        
    except Exception as e:
        st.error(f"❌ Error dalam scaling: {str(e)}")
        st.warning("⚠️ Menggunakan data asli (tanpa scaling)")
        return input_array

# Sidebar untuk navigasi
st.sidebar.title("🔧 Navigasi")
app_mode = st.sidebar.selectbox(
    "Pilih Halaman",
    ["🏠 Dashboard", "📊 Prediksi Churn", "📈 Analisis Data", "ℹ️ Tentang", "🔍 Debug Info"]
)

# Header utama
st.title("📊 Customer Churn Prediction System")
st.markdown("---")

if app_mode == "🏠 Dashboard":
    st.header("Dashboard Prediksi Customer Churn")
    
    # Load model untuk info
    model, _ = load_models()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Customers", "10,000", "↑ 12%")
    
    with col2:
        st.metric("Churn Rate", "26.5%", "↓ 4%")
    
    with col3:
        if model:
            model_name = type(model).__name__
            st.metric("Model", model_name)
        else:
            st.metric("Model", "Belum dimuat")
    
    st.markdown("---")
    
    # Visualisasi
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribusi Churn")
        fig, ax = plt.subplots(figsize=(8, 4))
        data = pd.Series([73.5, 26.5], index=['Stay', 'Churn'])
        colors = ['#48a8c4', '#c0504d']
        ax.pie(data, labels=data.index, autopct='%1.1f%%', colors=colors)
        ax.set_title('Customer Churn Distribution')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Top Factors Influencing Churn")
        factors = pd.DataFrame({
            'Factor': ['Satisfaction Score', 'Number of Referrals', 'Age', 'Contract Type', 'Internet Service'],
            'Importance': [0.25, 0.20, 0.18, 0.15, 0.12]
        })
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(factors['Factor'], factors['Importance'], color='#48a8c4')
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance (Estimasi)')
        st.pyplot(fig)

elif app_mode == "📊 Prediksi Churn":
    st.header("Prediksi Customer Churn")
    
    # Load model
    model, scaler = load_models()
    
    if model is None:
        st.warning("⚠️ Model belum dimuat. Pastikan file model ada di direktori.")
        st.markdown("""
        **File yang diperlukan:**
        - `gradient_boosting_churn_model.joblib`
        - `scaler.joblib`
        
        **Cara mendapatkan file:**
        1. Jalankan notebook hingga selesai
        2. File akan otomatis tersimpan
        3. Copy file ke folder yang sama dengan app.py
        """)
    
    # Informasi penting tentang scaling
    with st.expander("ℹ️ INFORMASI PENTING: Scaling Strategy"):
        st.markdown("""
        **Scaling di notebook hanya dilakukan pada 6 kolom berikut:**
        
        ### **Kolom yang DI-SCALE:**
        1. `age` (usia customer)
        2. `number_of_dependents` (jumlah tanggungan)
        3. `streaming_service` (binary 0/1)
        4. `tech_service` (binary 0/1)
        5. `number_of_referrals` (jumlah referral)
        6. `satisfaction_score` (skor kepuasan 1-5)
        
        ### **Kolom yang TIDAK DI-SCALE:**
        - `contract_One Year` (binary 0/1)
        - `contract_Two Year` (binary 0/1)
        - `internet_type_DSL` (binary 0/1)
        - `internet_type_Fiber Optic` (binary 0/1)
        - `internet_type_No Internet Service` (binary 0/1)
        - `phone_service_x` (binary 0/1)
        - `unlimited_data` (binary 0/1)
        
        **Catatan:** Scaler hanya dilatih dengan 6 fitur, bukan 13!
        """)
    
    # Form input
    st.subheader("📝 Input Data Customer")
    
    # Baris pertama
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### **Informasi Kontrak**")
        contract_type = st.radio(
            "Contract Type",
            ["Month-to-month", "One Year", "Two Year"],
            help="Pilih jenis kontrak"
        )
        
        st.markdown("### **Tipe Internet**")
        internet_type = st.radio(
            "Internet Service Type",
            ["No Internet Service", "DSL", "Fiber Optic"],
            help="Pilih tipe layanan internet"
        )
    
    with col2:
        st.markdown("### **Informasi Demografis**")
        age = st.number_input("Age", 
                             min_value=18, max_value=100, value=45,
                             help="Usia customer")
        
        number_of_dependents = st.number_input("Number of Dependents", 
                                              min_value=0, max_value=10, value=1,
                                              help="Jumlah tanggungan (anak/orang tua)")
        
        number_of_referrals = st.number_input("Number of Referrals", 
                                             min_value=0, max_value=20, value=0,
                                             help="Jumlah customer yang direferensikan")
    
    with col3:
        st.markdown("### **Skor Kepuasan**")
        satisfaction_score = st.slider("Satisfaction Score", 
                                      min_value=1, max_value=5, value=3,
                                      help="Skor kepuasan customer (1 = sangat tidak puas, 5 = sangat puas)")
        
        st.markdown("### **Layanan Tambahan**")
        phone_service = st.radio("Phone Service", ["No", "Yes"], 
                                horizontal=True, help="Layanan telepon")
        
        streaming_service = st.radio("Streaming Service", ["No", "Yes"], 
                                    horizontal=True, help="Layanan streaming")
    
    # Baris kedua
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("### **Layanan Teknis**")
        tech_service = st.radio("Tech Support Service", ["No", "Yes"], 
                               horizontal=True, help="Layanan dukungan teknis")
        
        unlimited_data = st.radio("Unlimited Data Plan", ["No", "Yes"], 
                                 horizontal=True, help="Paket data unlimited")
    
    with col5:
        # Ringkasan input
        st.markdown("### **Ringkasan Input**")
        st.info(f"""
        **Customer Profile:**
        - Usia: {age} tahun
        - Tanggungan: {number_of_dependents} orang
        - Referral: {number_of_referrals} customer
        - Kepuasan: {satisfaction_score}/5
        
        **Scaling akan diterapkan pada:**
        - Age, Dependents, Referrals, Satisfaction Score
        - Streaming Service, Tech Service
        """)
    
    with col6:
        # Tombol prediksi
        st.markdown("### **Aksi**")
        predict_button = st.button(
            "🔮 PREDIKSI CHURN", 
            type="primary", 
            use_container_width=True,
            help="Klik untuk melakukan prediksi"
        )
    
    # Logika prediksi
    if predict_button:
        if model is None:
            st.error("Model belum dimuat. Silakan periksa file model.")
        else:
            # Encoding data
            contract_one_year = 1.0 if contract_type == "One Year" else 0.0
            contract_two_year = 1.0 if contract_type == "Two Year" else 0.0
            
            internet_dsl = 1.0 if internet_type == "DSL" else 0.0
            internet_fiber = 1.0 if internet_type == "Fiber Optic" else 0.0
            internet_none = 1.0 if internet_type == "No Internet Service" else 0.0
            
            phone_service_encoded = 1.0 if phone_service == "Yes" else 0.0
            streaming_service_encoded = 1.0 if streaming_service == "Yes" else 0.0
            tech_service_encoded = 1.0 if tech_service == "Yes" else 0.0
            unlimited_data_encoded = 1.0 if unlimited_data == "Yes" else 0.0
            
            # Buat dictionary dengan data encoded
            encoded_data = {
                'contract_One Year': contract_one_year,
                'contract_Two Year': contract_two_year,
                'internet_type_DSL': internet_dsl,
                'internet_type_Fiber Optic': internet_fiber,
                'internet_type_No Internet Service': internet_none,
                'age': float(age),
                'number_of_dependents': float(number_of_dependents),
                'phone_service_x': phone_service_encoded,
                'streaming_service': streaming_service_encoded,
                'tech_service': tech_service_encoded,
                'unlimited_data': unlimited_data_encoded,
                'number_of_referrals': float(number_of_referrals),
                'satisfaction_score': float(satisfaction_score)
            }
            
            # Tampilkan data encoded untuk debugging
            with st.expander("🔍 Data Encoded untuk Model"):
                st.write("**Data dalam format one-hot encoded:**")
                
                # Tabel data encoded
                encoded_df = pd.DataFrame([encoded_data]).T
                encoded_df.columns = ['Value']
                st.dataframe(encoded_df.style.format("{:.2f}"), use_container_width=True)
            
            try:
                # ===========================================
                # PREPARASI DATA UNTUK PREDIKSI
                # ===========================================
                
                # Urutan kolom HARUS sama dengan saat training
                columns_order = [
                    'contract_One Year', 'contract_Two Year',
                    'internet_type_DSL', 'internet_type_Fiber Optic', 'internet_type_No Internet Service',
                    'age', 'number_of_dependents', 'phone_service_x',
                    'streaming_service', 'tech_service', 'unlimited_data',
                    'number_of_referrals', 'satisfaction_score'
                ]
                
                # Buat array dengan urutan yang benar
                input_array = np.array([[encoded_data[col] for col in columns_order]])
                
                st.success(f"✅ Data siap diproses! Shape: {input_array.shape}")
                
                # Debug: Tampilkan data sebelum scaling
                with st.expander("📊 Data Sebelum Scaling"):
                    df_before = pd.DataFrame(input_array, columns=columns_order)
                    st.dataframe(df_before)
                    
                    # Highlight kolom yang akan di-scale
                    cols_to_scale = ['age', 'number_of_dependents', 'streaming_service',
                                    'tech_service', 'number_of_referrals', 'satisfaction_score']
                    
                    def highlight_scaled_columns(val):
                        if val.name in cols_to_scale:
                            return ['background-color: #e8f4f8'] * len(val)
                        return [''] * len(val)
                    
                    st.write("**Kolom yang akan di-scale:**")
                    st.dataframe(df_before.style.apply(highlight_scaled_columns, axis=0))
                
                # ===========================================
                # SCALING DATA - PARsial (hanya 6 kolom)
                # ===========================================
                scaled_data = apply_partial_scaling(input_array, columns_order, scaler)
                
                # ===========================================
                # PREDIKSI MODEL
                # ===========================================
                with st.spinner("🔄 Melakukan prediksi..."):
                    prediction = model.predict(scaled_data)[0]
                    prediction_proba = model.predict_proba(scaled_data)[0]
                
                # ===========================================
                # TAMPILKAN HASIL PREDIKSI
                # ===========================================
                st.markdown("---")
                st.subheader("🎯 **HASIL PREDIKSI**")
                
                # Hasil utama
                col_result1, col_result2 = st.columns(2)
                
                with col_result1:
                    if prediction == 1:
                        st.error(f"## ⚠️ **CHURN DETECTED**")
                        st.markdown(f"### Probability: **{prediction_proba[1]*100:.1f}%**")
                        
                        st.markdown("""
                        ### **Rekomendasi Tindakan:**
                        
                        **🎯 Prioritas Tinggi:**
                        - 📞 **Segera hubungi** customer untuk feedback
                        - 💰 **Tawarkan promo retensi** khusus
                        - 🔄 **Review contract terms** yang lebih fleksibel
                        """)
                    else:
                        st.success(f"## ✅ **NO CHURN (LOYAL)**")
                        st.markdown(f"### Probability: **{prediction_proba[0]*100:.1f}%**")
                        
                        st.markdown("""
                        ### **Rekomendasi Tindakan:**
                        
                        **🌟 Pertahankan Loyalitas:**
                        - 👍 **Pertahankan service quality** yang ada
                        - 🎁 **Tawarkan upgrade service** dengan diskon
                        - ⭐ **Program loyalitas** dengan rewards eksklusif
                        """)
                
                with col_result2:
                    # Visualisasi probabilitas
                    fig, ax = plt.subplots(figsize=(8, 5))
                    categories = ['Loyal (No Churn)', 'Churn']
                    probabilities = [prediction_proba[0], prediction_proba[1]]
                    colors = ['#48a8c4', '#c0504d']
                    
                    bars = ax.bar(categories, probabilities, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
                    ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
                    ax.set_title('Churn Prediction Probability', fontsize=14, fontweight='bold')
                    ax.set_ylim([0, 1])
                    ax.grid(axis='y', alpha=0.3)
                    
                    # Tambah nilai di atas bar
                    for bar, prob in zip(bars, probabilities):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                               f'{prob*100:.1f}%', 
                               ha='center', va='bottom',
                               fontsize=11, fontweight='bold')
                    
                    # Highlight based on prediction
                    if prediction == 1:
                        bars[1].set_alpha(1.0)
                        bars[1].set_edgecolor('red')
                        bars[1].set_linewidth(3)
                    else:
                        bars[0].set_alpha(1.0)
                        bars[0].set_edgecolor('green')
                        bars[0].set_linewidth(3)
                    
                    st.pyplot(fig)
                
                # Detail teknis
                with st.expander("📊 **Detail Teknis Prediksi**"):
                    col_tech1, col_tech2 = st.columns(2)
                    
                    with col_tech1:
                        st.metric("Predicted Class", 
                                 "⚠️ **CHURN**" if prediction == 1 else "✅ **LOYAL**")
                        st.metric("Model Confidence", 
                                 f"{max(prediction_proba)*100:.1f}%")
                    
                    with col_tech2:
                        st.metric("Probability No Churn", 
                                 f"{prediction_proba[0]*100:.2f}%")
                        st.metric("Probability Churn", 
                                 f"{prediction_proba[1]*100:.2f}%")
                
                # Alert berdasarkan threshold
                st.markdown("---")
                churn_prob_percent = prediction_proba[1] * 100
                
                if churn_prob_percent > 70:
                    st.error(f"""
                    ## 🚨 **HIGH RISK ALERT**
                    Customer memiliki **risiko churn sangat tinggi** ({churn_prob_percent:.1f}%).
                    """)
                elif churn_prob_percent > 40:
                    st.warning(f"""
                    ## ⚠️ **MEDIUM RISK ALERT**
                    Customer memiliki **risiko churn sedang** ({churn_prob_percent:.1f}%).
                    """)
                else:
                    st.success(f"""
                    ## ✅ **LOW RISK STATUS**
                    Customer **loyal** dengan risiko churn rendah ({churn_prob_percent:.1f}%).
                    """)
                
                # Simpan hasil prediksi
                with st.expander("💾 Simpan Hasil Prediksi"):
                    result_df = pd.DataFrame({
                        'Timestamp': [pd.Timestamp.now()],
                        'Age': [age],
                        'Contract_Type': [contract_type],
                        'Internet_Type': [internet_type],
                        'Satisfaction_Score': [satisfaction_score],
                        'Prediction': ['Churn' if prediction == 1 else 'Loyal'],
                        'Churn_Probability': [prediction_proba[1]],
                        'Loyal_Probability': [prediction_proba[0]]
                    })
                    
                    csv = result_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Hasil Prediksi (CSV)",
                        data=csv,
                        file_name=f"churn_prediction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
            except Exception as e:
                st.error(f"❌ **Error dalam prediksi:** {str(e)}")
                
                # Debug detail
                with st.expander("🔧 **Debug Detail Error**"):
                    st.write("**Error trace:**")
                    st.code(str(e))
                    
                    if 'input_array' in locals():
                        st.write("**Input array shape:**", input_array.shape)
                        st.write("**Columns order:**", columns_order)
                        
                        # Coba prediksi tanpa scaling untuk testing
                        try:
                            st.write("**Testing prediction without scaling...**")
                            test_pred = model.predict(input_array)
                            st.write("Test result:", test_pred)
                        except Exception as e2:
                            st.write("Error tanpa scaling:", str(e2))

elif app_mode == "📈 Analisis Data":
    st.header("Analisis Data Customer")
    
    # Upload dataset
    uploaded_file = st.file_uploader("Upload dataset CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Tampilkan preview
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Statistik dasar
            st.subheader("Dataset Information")
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                st.metric("Total Rows", len(df))
            
            with col_info2:
                st.metric("Total Columns", len(df.columns))
            
            with col_info3:
                churn_cols = [col for col in df.columns if 'churn' in col.lower()]
                if churn_cols:
                    churn_col = churn_cols[0]
                    churn_count = df[churn_col].value_counts()
                    churn_rate = (churn_count.get(1, 0) / len(df)) * 100
                    st.metric("Churn Rate", f"{churn_rate:.1f}%")
                else:
                    st.metric("Churn Column", "Not Found")
            
            # EDA - Tambahkan visualisasi di sini
            st.subheader("Exploratory Data Analysis (EDA)")
            
            # Buat tabs untuk berbagai visualisasi
            tab1, tab2, tab3 = st.tabs(["📊 Distribusi", "🔗 Korelasi", "📈 Churn Analysis"])
            
            with tab1:
                st.markdown("### Distribusi Variabel Numerik")
                
                # Pilih kolom numerik
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    selected_col = st.selectbox("Pilih kolom untuk histogram:", numeric_cols)
                    
                    # Histogram
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(df[selected_col].dropna(), bins=30, color='#48a8c4', alpha=0.7, edgecolor='black')
                    ax.set_title(f'Distribusi {selected_col}', fontsize=14, fontweight='bold')
                    ax.set_xlabel(selected_col, fontsize=12)
                    ax.set_ylabel('Frequency', fontsize=12)
                    ax.grid(alpha=0.3)
                    st.pyplot(fig)
                    
                    # Statistik deskriptif
                    st.markdown("#### Statistik Deskriptif")
                    desc_stats = df[selected_col].describe()
                    st.dataframe(desc_stats, use_container_width=True)
                else:
                    st.warning("Tidak ada kolom numerik dalam dataset")
            
            with tab2:
                st.markdown("### Heatmap Korelasi")
                
                # Pilih kolom numerik untuk korelasi
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 1:
                    # Hitung matriks korelasi
                    corr_matrix = df[numeric_cols].corr()
                    
                    # Buat heatmap
                    fig, ax = plt.subplots(figsize=(12, 8))
                    sns.heatmap(corr_matrix, 
                                annot=True, 
                                fmt='.2f', 
                                cmap='coolwarm', 
                                center=0, 
                                ax=ax,
                                square=True,
                                linewidths=0.5,
                                cbar_kws={"shrink": 0.8})
                    ax.set_title('Matriks Korelasi', fontsize=14, fontweight='bold')
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                    st.pyplot(fig)
                    
                    # Tampilkan korelasi terkuat
                    st.markdown("#### Korelasi Terkuat")
                    corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
                    # Hilangkan korelasi diri (1.0) dan duplikat
                    unique_corr_pairs = corr_pairs[~corr_pairs.index.duplicated(keep='first')]
                    top_corr = unique_corr_pairs[unique_corr_pairs < 0.999].head(10)
                    
                    corr_df = pd.DataFrame(top_corr, columns=['Correlation'])
                    st.dataframe(corr_df.style.background_gradient(cmap='coolwarm'), use_container_width=True)
                else:
                    st.warning("Perlu minimal 2 kolom numerik untuk analisis korelasi")
            
            with tab3:
                st.markdown("### Analisis Churn")

                # PRIORITAS kolom churn yang BENAR
                if 'churn_value' in df.columns:
                    churn_col = 'churn_value'
                elif 'churn_label' in df.columns:
                    churn_col = 'churn_label'
                else:
                    st.error("Kolom churn tidak ditemukan (churn_value / churn_label)")
                    st.stop()

                col_churn1, col_churn2 = st.columns(2)

                # =========================
                # PIE CHART (FIXED)
                # =========================
                with col_churn1:
                    st.markdown("#### Distribusi Churn")

                    churn_counts = df[churn_col].value_counts()

                    # Mapping label yang AMAN
                    if churn_col == 'churn_value':
                        labels = ['Tidak Churn', 'Churn']
                        values = [
                            churn_counts.get(0, 0),
                            churn_counts.get(1, 0)
                        ]
                    else:
                        churn_counts.index = churn_counts.index.astype(str)
                        labels = churn_counts.index.tolist()
                        values = churn_counts.values.tolist()

                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.pie(
                        values,
                        labels=labels,
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=['#48a8c4', '#c0504d']
                    )
                    ax.axis('equal')
                    ax.set_title("Distribusi Churn vs Non-Churn", fontsize=12, fontweight='bold')
                    st.pyplot(fig)

                # =========================
                # CHURN BY FEATURE (SAFE)
                # =========================
                with col_churn2:
                    st.markdown("#### Churn by Feature")

                    feature_options = df.columns.drop(churn_col).tolist()
                    selected_feature = st.selectbox(
                        "Pilih fitur untuk analisis churn:",
                        feature_options
                    )

                    # NUMERIK → BOXPLOT
                    if pd.api.types.is_numeric_dtype(df[selected_feature]):
                        fig, ax = plt.subplots(figsize=(8, 5))

                        data = [
                            df[df[churn_col] == 0][selected_feature],
                            df[df[churn_col] == 1][selected_feature]
                        ]

                        ax.boxplot(
                            data,
                            labels=['Tidak Churn', 'Churn'],
                            patch_artist=True,
                            boxprops=dict(facecolor='#48a8c4'),
                            medianprops=dict(color='black')
                        )

                        ax.set_title(f"{selected_feature} vs Churn")
                        ax.grid(alpha=0.3)
                        st.pyplot(fig)

                    # KATEGORIK → BAR CHART (LIMIT CATEGORY)
                    else:
                        churn_feat = (
                            df.groupby([selected_feature, churn_col])
                            .size()
                            .unstack(fill_value=0)
                            .head(10)  # 🚨 batasi kategori
                        )

                        fig, ax = plt.subplots(figsize=(10, 5))
                        churn_feat.plot(kind='bar', ax=ax, color=['#48a8c4', '#c0504d'])
                        ax.set_title(f"Churn by {selected_feature}")
                        ax.set_ylabel("Jumlah Customer")
                        plt.xticks(rotation=45, ha='right')
                        ax.grid(axis='y', alpha=0.3)
                        st.pyplot(fig)

            
            # Additional analysis
            st.subheader("Analisis Lanjutan")
            
            with st.expander("📋 Summary Statistics"):
                st.write("**Statistik Deskriptif Lengkap:**")
                st.dataframe(df.describe(include='all').T, use_container_width=True)
            
            with st.expander("🔍 Missing Values Analysis"):
                st.write("**Analisis Missing Values:**")
                missing_data = df.isnull().sum()
                missing_percent = (missing_data / len(df)) * 100
                missing_df = pd.DataFrame({
                    'Missing Values': missing_data,
                    'Percentage': missing_percent
                })
                missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values('Missing Values', ascending=False)
                
                if len(missing_df) > 0:
                    st.dataframe(missing_df.style.format({'Percentage': '{:.2f}%'}), use_container_width=True)
                    
                    # Visualisasi missing values
                    fig, ax = plt.subplots(figsize=(10, 6))
                    missing_df['Percentage'].plot(kind='bar', ax=ax, color='#c0504d', alpha=0.7)
                    ax.set_title('Percentage of Missing Values by Column', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Columns')
                    ax.set_ylabel('Missing Percentage (%)')
                    ax.grid(alpha=0.3, axis='y')
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig)
                else:
                    st.success("✅ Tidak ada missing values dalam dataset")
            
            with st.expander("📥 Download Processed Data"):
                # Opsi untuk download data yang sudah diproses
                processed_csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Data as CSV",
                    data=processed_csv,
                    file_name="processed_customer_data.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Error membaca file: {str(e)}")
            st.error("Pastikan file CSV memiliki format yang benar.")
    
    else:
        st.info("👆 Upload dataset CSV untuk melihat analisis")
        
        # Contoh struktur data yang diharapkan
        with st.expander("ℹ️ Contoh Struktur Data yang Diharapkan"):
            st.markdown("""
            **Dataset sebaiknya memiliki kolom-kolom berikut:**
            
            ### **Kolom Wajib untuk Analisis Churn:**
            - `churn` atau `Churn` atau `is_churn` (target variable, 0/1)
            
            ### **Kolom Demografis & Numerik:**
            - `age` (usia customer)
            - `number_of_dependents` (jumlah tanggungan)
            - `number_of_referrals` (jumlah referral)
            - `satisfaction_score` (skor kepuasan 1-5)
            
            ### **Kolom Layanan (binary 0/1):**
            - `phone_service_x` (layanan telepon)
            - `streaming_service` (layanan streaming)
            - `tech_service` (layanan teknis)
            - `unlimited_data` (paket data unlimited)
            
            ### **Kolom Kontrak (one-hot encoded):**
            - `contract_One Year`
            - `contract_Two Year`
            
            ### **Kolom Tipe Internet (one-hot encoded):**
            - `internet_type_DSL`
            - `internet_type_Fiber Optic`
            - `internet_type_No Internet Service`
            
            **Format:** CSV dengan header
            """)
            
            # Contoh data dummy
            example_data = pd.DataFrame({
                'age': [45, 32, 55],
                'number_of_dependents': [2, 0, 3],
                'phone_service_x': [1, 0, 1],
                'streaming_service': [1, 1, 0],
                'tech_service': [0, 1, 1],
                'unlimited_data': [1, 1, 0],
                'number_of_referrals': [3, 0, 5],
                'satisfaction_score': [4, 2, 5],
                'contract_One Year': [0, 1, 0],
                'contract_Two Year': [1, 0, 0],
                'internet_type_DSL': [1, 0, 0],
                'internet_type_Fiber Optic': [0, 1, 0],
                'internet_type_No Internet Service': [0, 0, 1],
                'churn': [0, 1, 0]
            })
            
            st.write("**Contoh Data:**")
            st.dataframe(example_data, use_container_width=True)

elif app_mode == "ℹ️ Tentang":
    st.header("📋 Tentang Aplikasi")
    
    st.markdown("""
    ### 🎯 Customer Churn Prediction System
    
    **Deskripsi:**
    Aplikasi ini menggunakan model machine learning **Gradient Boosting** untuk 
    memprediksi potensi churn (kehilangan pelanggan).
    
    **⚠️ Catatan Penting tentang Scaling:**
    - Model dilatih dengan **partial scaling** pada 6 kolom numerik/binary
    - Scaler hanya bekerja dengan 6 fitur, bukan semua 13 fitur
    - Kolom binary seperti contract dan internet type **tidak di-scale**
    """)

elif app_mode == "🔍 Debug Info":
    st.header("🔍 Debug & System Information")
    
    # Load models untuk debug
    model, scaler = load_models()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Information")
        if model:
            st.success("✅ Model loaded successfully")
            st.write(f"**Model Type:** {type(model).__name__}")
            
            # Debug scaler
            st.subheader("Scaler Debug Info")
            if scaler:
                st.success("✅ Scaler loaded successfully")
                st.write(f"**Scaler Type:** {type(scaler).__name__}")
                
                # Info fitur scaler
                if hasattr(scaler, 'n_features_in_'):
                    st.write(f"**Features expected by scaler:** {scaler.n_features_in_}")
                    st.write(f"**Note:** Scaler ini dilatih hanya untuk 6 kolom numerik/binary")
                
                if hasattr(scaler, 'mean_'):
                    st.write(f"**Scaler mean shape:** {scaler.mean_.shape}")
                    st.write(f"**Mean values:** {scaler.mean_}")
                
                if hasattr(scaler, 'scale_'):
                    st.write(f"**Scaler scale shape:** {scaler.scale_.shape}")
                    st.write(f"**Scale values:** {scaler.scale_}")
            else:
                st.warning("⚠️ Scaler not loaded")
        else:
            st.error("❌ Model not loaded")
    
    with col2:
        st.subheader("System Information")
        col_sys1, col_sys2 = st.columns(2)
        
        with col_sys1:
            st.metric("Python Version", f"{sys.version.split()[0]}")
            st.metric("Pandas Version", pd.__version__)
        
        with col_sys2:
            st.metric("NumPy Version", np.__version__)
            st.metric("Scikit-learn", "1.3.0+")
    
    # Test scaling functionality
    st.subheader("Test Scaling Functionality")
    
    if scaler:
        # Buat data test
        test_columns = ['age', 'number_of_dependents', 'streaming_service',
                       'tech_service', 'number_of_referrals', 'satisfaction_score']
        
        test_data = np.array([[45, 2, 1, 0, 3, 4]])  # Contoh data
        
        try:
            st.write("**Testing scaler with 6 features:**")
            scaled_test = scaler.transform(test_data)
            st.success(f"✅ Scaler works! Output shape: {scaled_test.shape}")
            st.write(f"Scaled values: {scaled_test[0]}")
        except Exception as e:
            st.error(f"❌ Scaler test failed: {str(e)}")
    
    # File check
    st.subheader("File Check")
    
    required_files = [
        'gradient_boosting_churn_model.joblib',
        'scaler.joblib',
        'app.py'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            st.success(f"✅ {file} - Found")
        else:
            st.error(f"❌ {file} - Not Found")

# Footer
st.markdown("---")
st.caption("© 2025 Customer Churn Prediction System | Developed by Fauzan Jaelani | Version 1.0")

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #48a8c4, #3b8ca0);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #c0504d, #a0403d);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(192, 80, 77, 0.3);
    }
    
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #48a8c4;
    }
</style>
""", unsafe_allow_html=True)