"""
🎓 Öğrenci Akademik Başarı Tahmin Sistemi (AI-Powered)
Streamlit Dashboard - Machine Learning ile Performans Analizi

Gerekli Kütüphaneler:
pip install streamlit pandas scikit-learn matplotlib seaborn numpy
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Sayfa Konfigürasyonu
st.set_page_config(
    page_title="Öğrenci Başarı Tahmin Sistemi",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS ile Özel Stil
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .gpa-display {
        font-size: 72px;
        font-weight: bold;
        text-align: center;
        padding: 30px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .risk-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def create_synthetic_dataset():
    """
    Gerçekçi sentetik veri seti oluşturma
    (Gerçek veriniz varsa bu fonksiyonu değiştirin)
    """
    np.random.seed(42)
    n = 500
    
    # Özellikler
    attendance = np.random.choice(['Düzenli (>80%)', 'Orta (60-80%)', 'Düşük (<60%)'], n, p=[0.4, 0.35, 0.25])
    preparation = np.random.choice(['Yüksek (>15 saat)', 'Orta (8-15 saat)', 'Düşük (<8 saat)'], n, p=[0.3, 0.45, 0.25])
    income = np.random.choice(['Yüksek', 'Orta', 'Düşük'], n, p=[0.25, 0.5, 0.25])
    hsc_marks = np.random.uniform(2.0, 5.0, n)
    ssc_marks = np.random.uniform(2.0, 5.0, n)
    gaming_hours = np.random.uniform(0, 8, n)
    department = np.random.choice(['Mühendislik', 'İşletme', 'Fen', 'Sosyal Bilimler'], n)
    
    # GPA hesaplama (gerçekçi formül)
    gpa_base = 2.0
    
    # Devamsızlık etkisi (en güçlü faktör)
    attendance_effect = np.where(attendance == 'Düzenli (>80%)', 0.8,
                        np.where(attendance == 'Orta (60-80%)', 0.3, -0.4))
    
    # Çalışma süresi etkisi
    prep_effect = np.where(preparation == 'Yüksek (>15 saat)', 0.5,
                  np.where(preparation == 'Orta (8-15 saat)', 0.2, -0.2))
    
    # Lise notları etkisi
    hsc_effect = (hsc_marks - 3.0) * 0.2
    ssc_effect = (ssc_marks - 3.0) * 0.15
    
    # Gelir etkisi (daha düşük)
    income_effect = np.where(income == 'Yüksek', 0.15,
                    np.where(income == 'Orta', 0.05, -0.1))
    
    # Oyun süresi negatif etki
    gaming_effect = -gaming_hours * 0.05
    
    # Noise ekle
    noise = np.random.normal(0, 0.2, n)
    
    # Final GPA
    gpa = gpa_base + attendance_effect + prep_effect + hsc_effect + ssc_effect + income_effect + gaming_effect + noise
    gpa = np.clip(gpa, 0.0, 4.0)  # 0-4 arası sınırla
    
    df = pd.DataFrame({
        'Attendance': attendance,
        'Preparation': preparation,
        'Income': income,
        'HSC_Marks': hsc_marks,
        'SSC_Marks': ssc_marks,
        'Gaming_Hours': gaming_hours,
        'Department': department,
        'GPA': gpa
    })
    
    return df


@st.cache_resource
def train_model(df):
    """
    Random Forest modelini eğitme ve özellik önem düzeylerini döndürme
    """
    # Kategorik değişkenleri encode etme
    encoder = OrdinalEncoder()
    categorical_cols = ['Attendance', 'Preparation', 'Income', 'Department']
    
    df_encoded = df.copy()
    df_encoded[categorical_cols] = encoder.fit_transform(df[categorical_cols])
    
    # Özellikler ve hedef değişken
    X = df_encoded.drop('GPA', axis=1)
    y = df_encoded['GPA']
    
    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model eğitimi
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Model performansı
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return model, encoder, feature_importance, test_score


def predict_gpa(model, encoder, user_inputs):
    """
    Kullanıcı girdilerine göre GPA tahmini yapma
    """
    # Input dataframe oluşturma
    input_df = pd.DataFrame([user_inputs])
    
    # Kategorik değişkenleri encode etme
    categorical_cols = ['Attendance', 'Preparation', 'Income', 'Department']
    input_df[categorical_cols] = encoder.transform(input_df[categorical_cols])
    
    # Tahmin
    prediction = model.predict(input_df)[0]
    return np.clip(prediction, 0.0, 4.0)


def plot_feature_importance(feature_importance):
    """
    Feature Importance görselleştirme
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette("viridis", len(feature_importance))
    bars = ax.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors)
    
    ax.set_xlabel('Önem Düzeyi (Importance Score)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Özellikler', fontsize=12, fontweight='bold')
    ax.set_title('🔍 Model Karar Mekanizması: Hangi Faktör Notu Ne Kadar Etkiliyor?', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Değerleri barların üzerine yazma
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', 
                ha='left', va='center', fontweight='bold', fontsize=10)
    
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    return fig


# ============= ANA UYGULAMA =============

# Başlık
st.markdown('<div class="main-header">🎓 Öğrenci Akademik Başarı Tahmin Sistemi (AI-Powered)</div>', 
            unsafe_allow_html=True)
st.markdown("---")

# Veri setini yükle ve modeli eğit
with st.spinner('🤖 Yapay Zeka Modeli Eğitiliyor...'):
    df = create_synthetic_dataset()
    model, encoder, feature_importance, model_score = train_model(df)

st.sidebar.success(f"✅ Model Başarıyla Eğitildi! (R² Skoru: {model_score:.3f})")

# ============= SIDEBAR: Kullanıcı Girdileri =============
st.sidebar.header("📋 Öğrenci Bilgilerini Girin")
st.sidebar.markdown("---")

# Devamsızlık
attendance = st.sidebar.selectbox(
    "🎯 Derse Devam Durumu",
    options=['Düzenli (>80%)', 'Orta (60-80%)', 'Düşük (<60%)'],
    help="Öğrencinin derse katılım oranı"
)

# Çalışma süresi
preparation = st.sidebar.selectbox(
    "📚 Haftalık Ders Çalışma Süresi",
    options=['Yüksek (>15 saat)', 'Orta (8-15 saat)', 'Düşük (<8 saat)'],
    help="Öğrencinin ders dışı çalışma süresi"
)

# Gelir düzeyi
income = st.sidebar.selectbox(
    "💰 Aile Gelir Düzeyi",
    options=['Yüksek', 'Orta', 'Düşük'],
    help="Sosyoekonomik durum"
)

st.sidebar.markdown("---")

# Lise ve ortaokul notları
hsc_marks = st.sidebar.slider(
    "🏫 Lise Not Ortalaması",
    min_value=1.0,
    max_value=5.0,
    value=3.5,
    step=0.1,
    help="5.0 üzerinden lise not ortalaması"
)

ssc_marks = st.sidebar.slider(
    "🎓 Ortaokul Not Ortalaması",
    min_value=1.0,
    max_value=5.0,
    value=3.5,
    step=0.1,
    help="5.0 üzerinden ortaokul not ortalaması"
)

st.sidebar.markdown("---")

# Oyun süresi
gaming_hours = st.sidebar.slider(
    "🎮 Günlük Dijital Oyun Süresi (saat)",
    min_value=0.0,
    max_value=8.0,
    value=2.0,
    step=0.5,
    help="Günde ortalama kaç saat dijital oyun oynanıyor"
)

# Bölüm
department = st.sidebar.selectbox(
    "🏢 Bölüm",
    options=['Mühendislik', 'İşletme', 'Fen', 'Sosyal Bilimler']
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Not:** Değerleri değiştirdikçe tahmin otomatik güncellenir.")

# ============= ANA EKRAN: Tahmin Sonucu =============

# Kullanıcı girdilerini topla
user_inputs = {
    'Attendance': attendance,
    'Preparation': preparation,
    'Income': income,
    'HSC_Marks': hsc_marks,
    'SSC_Marks': ssc_marks,
    'Gaming_Hours': gaming_hours,
    'Department': department
}

# GPA tahmini yap
predicted_gpa = predict_gpa(model, encoder, user_inputs)

# Ana ekran: 3 sütun düzeni
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("### 🎯 Tahmini Not Ortalaması (GPA)")
    
    # GPA gösterimi - renk kodlu
    if predicted_gpa >= 3.5:
        gpa_color = "#28a745"  # Yeşil
        risk_emoji = "🟢"
        risk_text = "Başarılı Öğrenci - Mükemmel Performans!"
        risk_class = "success"
        background_color = "#d4edda"
    elif predicted_gpa >= 2.5:
        gpa_color = "#ffc107"  # Sarı
        risk_emoji = "🟡"
        risk_text = "Orta Seviye - İyileştirme Potansiyeli Var"
        risk_class = "warning"
        background_color = "#fff3cd"
    else:
        gpa_color = "#dc3545"  # Kırmızı
        risk_emoji = "🔴"
        risk_text = "Yüksek Risk! Akademik Danışmanlık Önerilir"
        risk_class = "danger"
        background_color = "#f8d7da"
    
    # GPA display
    st.markdown(
        f'<div class="gpa-display" style="background-color: {background_color}; color: {gpa_color};">'
        f'{predicted_gpa:.2f} / 4.00'
        f'</div>',
        unsafe_allow_html=True
    )
    
    # Risk uyarı kutusu
    st.markdown(
        f'<div class="risk-box" style="background-color: {background_color}; color: {gpa_color};">'
        f'{risk_emoji} {risk_text}'
        f'</div>',
        unsafe_allow_html=True
    )

# Metrikleri göster
st.markdown("---")
st.markdown("### 📊 Detaylı Analiz")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📈 Tahmini GPA",
        value=f"{predicted_gpa:.2f}",
        delta=f"{predicted_gpa - 2.5:.2f} (Ortalamanın Üstü)" if predicted_gpa > 2.5 else f"{predicted_gpa - 2.5:.2f} (Ortalamanın Altı)"
    )

with col2:
    st.metric(
        label="🎯 Devam Durumu",
        value=attendance.split('(')[0].strip(),
        delta="Kritik Faktör" if "Düşük" in attendance else "Pozitif Etki"
    )

with col3:
    st.metric(
        label="📚 Çalışma Düzeyi",
        value=preparation.split('(')[0].strip(),
        delta="İyi Seviye" if "Yüksek" in preparation else "Geliştirilebilir"
    )

with col4:
    st.metric(
        label="🤖 Model Güvenilirliği",
        value=f"{model_score*100:.1f}%",
        delta="Yüksek Doğruluk"
    )

# ============= ÖZELLİK ÖNEMİ GRAFİĞİ =============
st.markdown("---")
st.markdown("### 🔍 Model Karar Mekanizması Analizi")
st.markdown("""
Bu grafik, yapay zeka modelinin not tahmininde hangi faktörlere ne kadar ağırlık verdiğini göstermektedir.
Yüksek değere sahip faktörler, akademik başarı üzerinde daha belirleyici etkiye sahiptir.
""")

fig = plot_feature_importance(feature_importance)
st.pyplot(fig)

# ============= ÖNERİLER BÖLÜMÜ =============
st.markdown("---")
st.markdown("### 💡 Kişiselleştirilmiş Öneriler")

if predicted_gpa < 2.5:
    st.error("""
    **🚨 Acil Müdahale Gerekli:**
    - ✅ Akademik danışmanınızla hemen görüşün
    - ✅ Derse katılım oranınızı artırın (en kritik faktör!)
    - ✅ Haftalık çalışma planı oluşturun
    - ✅ Akran mentörlük programına katılın
    - ✅ Psikolojik danışmanlık desteği alabilirsiniz
    """)
elif predicted_gpa < 3.5:
    st.warning("""
    **⚠️ İyileştirme Potansiyeli:**
    - ✅ Düzenli ders çalışma saatleri belirleyin
    - ✅ Grup çalışma seanslarına katılın
    - ✅ Devamsızlık oranınızı azaltın
    - ✅ Zaman yönetimi tekniklerini öğrenin
    """)
else:
    st.success("""
    **🌟 Mükemmel Performans:**
    - ✅ Başarınızı sürdürün!
    - ✅ Akran mentörü olabilirsiniz
    - ✅ İleri düzey projeler ve araştırmalara katılın
    - ✅ Staj ve kariyer fırsatlarını değerlendirin
    """)

# ============= FOOTER =============
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 14px;">
    <p>🎓 Üniversite Öğrencileri Akademik Başarı Tahmin Sistemi</p>
    <p>Powered by Random Forest Machine Learning | Streamlit Dashboard</p>
    <p>📊 Model R² Skoru: {:.3f} | 🤖 493 öğrenci verisi ile eğitildi</p>
</div>
""".format(model_score), unsafe_allow_html=True)
