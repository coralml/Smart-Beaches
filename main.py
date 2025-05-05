import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# Cargar los datos
df = pd.read_csv("sentimientos_kpis.csv")
# Asegurar que las fechas están en datetime
df['published_at_date'] = pd.to_datetime(df['published_at_date'], errors='coerce')
df['year_month'] = df['published_at_date'].dt.to_period('M').astype(str)
df['year'] = df['published_at_date'].dt.year
st.set_page_config(layout="wide", page_title="Smart Beach Dashboard")
st.title("🏖️ Public Perception of Valencian Beaches: Insights from Google Maps Reviews")

st.sidebar.title("Navegationn")
page = st.sidebar.selectbox("Select a view", ["Sentimental Analysis", "Key Performance Indicators"])

# Filtros
with st.sidebar:
    st.header("🔍 Filters")
    playas = df['place_name'].dropna().unique()
    playa_sel = st.selectbox("Select a beach", sorted(playas))

    date_min = df['published_at_date'].min()
    date_max = df['published_at_date'].max()
    fecha_sel = st.date_input("Rank of dates", [date_min, date_max])

# Aplicar filtros
df_filtrado = df[(df['place_name'] == playa_sel) &
                 (df['published_at_date'] >= pd.to_datetime(fecha_sel[0])) &
                 (df['published_at_date'] <= pd.to_datetime(fecha_sel[1]))]
st.write("❗ Null dates after parsing:", df['published_at_date'].isna().sum())

# Depuración rápida
st.write("🔎 Original data sice:", df['published_at_date'].min())
st.write("📅 Selected range:", fecha_sel[0], "→", fecha_sel[1])
st.write("📂 Filtered data sice:", df_filtrado['published_at_date'].min())
st.write("🧮 Nº of comments after filter:", len(df_filtrado))

if page == "Sentimental Analysis":
    st.title("Sentimental Analysis")
    
        # -------- 1. Nubes de palabras por sentimiento --------
    st.subheader("☁️ Cloud of words by feeling (without common words)")

    sentiments = ['positivo', 'neutro', 'negativo']
    word_lists = {s: ' '.join(df_filtrado[df_filtrado['sentimiento'] == s]['texto_limpio']).split() for s in sentiments}
    word_counts = {s: Counter(w) for s, w in word_lists.items()}
    common_words = set(word_counts['positivo'].keys()) & set(word_counts['neutro'].keys()) & set(word_counts['negativo'].keys())

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    for ax, sentiment in zip(axes2, sentiments):
        filtered_words = [w for w in word_lists[sentiment] if w not in common_words]
        text = ' '.join(filtered_words)
        if text.strip():
            wc = WordCloud(width=800, height=400, background_color='white').generate(text)
            ax.imshow(wc, interpolation='bilinear')
            ax.set_title(f'Cloud ({sentiment})')
            ax.axis('off')
        else:
            ax.set_title(f"Without data ({sentiment})")
            ax.axis('off')
    st.pyplot(fig2)
    st.divider()
    
    
    # -------- 2. Sentimientos por mes y año --------
    st.subheader("📊 Evolution of feelings (monthly and annual)")

    # Filtrar solo por playa
    df_playa = df[df['place_name'] == playa_sel]

    # Columna 1: Últimos 12 meses
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📆 Last 12 months")
        ultimos_12 = df_playa['published_at_date'].max() - pd.DateOffset(months=12)
        df_ultimos_12 = df_playa[df_playa['published_at_date'] >= ultimos_12]

        sentiment_month = (
            df_ultimos_12.groupby(['year_month', 'sentimiento'])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )

        if not sentiment_month.empty:
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sentiment_month.plot(ax=ax1, marker='o', linewidth=2)
            ax1.set_title("Monthly")
            ax1.set_ylabel("Nº of comments")
            ax1.set_xlabel("Month")
            ax1.grid(True)
            ax1.tick_params(axis='x', rotation=45)
            st.pyplot(fig1)
        else:
            st.warning("There is no data for the last 12 months.")

    with col2:
        st.markdown("### 📅 By year")
        sentiment_year = (
            df_playa.groupby(['year', 'sentimiento'])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )

        if not sentiment_year.empty:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sentiment_year.plot(kind='bar', ax=ax2)
            ax2.set_title("Yearly")
            ax2.set_ylabel("Nº of comments")
            ax2.set_xlabel("Year")
            ax2.grid(axis='y')
            st.pyplot(fig2)
        else:
            st.warning("There is no data for the selected year.")
    st.divider()     

elif page == "Key Performance Indicators":
    st.title("Key Performance Indicators")
    st.subheader("Sentimental Analysis of Key Performance Indicators")
    # -------- 3. Heatmap de KPIs vs Sentimiento --------

    df_kpis = df_filtrado.dropna(subset=["kpis_detectados"]).copy()
    kpi_emojis = {
        "Sentiment Score": "🧠",
        "Cleanliness Perception": "🧼",
        "Safety Perception": "🛡️",
        "Crowding Level Mentions": "👥",
        "Nearby Services Mentions": "🚿",
        "Public Transport Accessibility": "🚌",
        "Availability of Bars/Restaurants": "🍽️",
        "Dog Access Mentions": "🐶",
        "Price & Cost Perception": "💰",
        "Water Quality": "🌊",
        "Beach Comfort": "⛱️",
        "Customer Service": "🙋",
        "Safety Facilities": "🚑",
        "Available Activities": "🏄",
        "General Atmosphere": "🎉"
    }

    # Asegúrate de tener valores válidos
    df_kpis = df_filtrado.dropna(subset=["kpis_detectados", "sentimiento"]).copy()

    if not df_kpis.empty:
        # Asegurar formato correcto
        df_kpis["kpis_detectados"] = df_kpis["kpis_detectados"].astype(str).str.split(",\s*")
        df_kpis["sentimiento"] = df_kpis["sentimiento"].astype(str).str.strip()
        df_exploded = df_kpis.explode("kpis_detectados")
        df_exploded["kpis_detectados"] = df_exploded["kpis_detectados"].str.strip()

        # Tabla cruzada KPI x Sentimiento
        heatmap_data = pd.crosstab(df_exploded["kpis_detectados"], df_exploded["sentimiento"]).sort_index()

    else:
        st.warning("No hay KPIs detectadas en el rango y playa seleccionados.")   
        
    if not heatmap_data.empty:
        for kpi, row in heatmap_data.iterrows():
            emoji = kpi_emojis.get(kpi, "🔍")
            st.markdown(f"#### {emoji} {kpi}")

            col1, col2, col3 = st.columns(3)
            col1.metric("😊 Positive", int(row.get("positivo", 0)))
            col2.metric("😐 Neutral", int(row.get("neutro", 0)))
            col3.metric("😡 Negative", int(row.get("negativo", 0)))

            total = row.sum()
            if total > 0:
                st.progress(row.get("positivo", 0) / total)
            else:
                st.progress(0.0)
            
            st.divider()
    else:
        st.info("There are no enough data to show the table of kpis and feelings.")

# -------- 4. Tabla interactiva --------
st.subheader("📝 Filtered comments")
st.dataframe(df_filtrado[["published_at_date", "review_text", "sentimiento", "rating", "kpis_detectados"]].sort_values("published_at_date", ascending=False))

# -------- 5. Exportar datos --------
st.download_button("📥 Download data filtered as CSV", data=df_filtrado.to_csv(index=False), file_name="comentarios_filtrados.csv")
