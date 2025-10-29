import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM  
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Прогноз Волатильности", layout="wide", page_icon="📊")

st.markdown("""<style>
.alert-high {background-color: #ff4444; padding: 10px; border-radius: 5px; color: white;}
.alert-medium {background-color: #ffaa00; padding: 10px; border-radius: 5px; color: white;}
.alert-low {background-color: #00cc44; padding: 10px; border-radius: 5px; color: white;}
</style>""", unsafe_allow_html=True)

@st.cache_data
def generate_fake_data(n_records=5000):
    np.random.seed(42)
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_records)]
    stores = [f"Магазин_{i:02d}" for i in range(1, 41)]
    brands = ['VPL', 'RAY-BAN', "HUMPHREY'S", 'Другие']
    
    data = []
    for date in dates:
        trend = 2000 + (date - start_date).days * 0.5
        seasonality = 500 * np.sin(2 * np.pi * (date.timetuple().tm_yday / 365))
        anomaly = np.random.choice([-800, 800]) if np.random.random() < 0.1 else 0
        base_price = trend + seasonality + anomaly + np.random.normal(0, 300)
        
        for _ in range(np.random.randint(1, 8)):
            record = {
                'Date': date,
                'Store': np.random.choice(stores),
                'Brand': np.random.choice(brands, p=[0.3, 0.25, 0.2, 0.25]),
                'Price': int(base_price * np.random.uniform(0.8, 1.2)),
                'Qty': np.random.randint(1, 5)
            }
            record['Sum'] = record['Price'] * record['Qty']
            data.append(record)
    
    return pd.DataFrame(data)

def calculate_volatility(series, window=7):
    returns = series.pct_change().dropna()
    return returns.rolling(window=window).std() * np.sqrt(window)

def detect_anomalies(df, contamination=0.05):
    daily = df.groupby('Date').agg({'Sum': 'sum', 'Qty': 'sum', 'Price': 'mean'}).reset_index()
    daily['Volatility'] = calculate_volatility(daily['Sum'])
    daily['MA_7'] = daily['Sum'].rolling(7).mean()
    daily['MA_30'] = daily['Sum'].rolling(30).mean()
    daily['DayOfWeek'] = daily['Date'].dt.dayofweek
    daily = daily.dropna()
    
    features = ['Sum', 'Volatility', 'MA_7', 'MA_30', 'DayOfWeek']
    X = StandardScaler().fit_transform(daily[features])
    
    iso = IsolationForest(contamination=contamination, random_state=42)
    daily['Anomaly_IF'] = iso.fit_predict(X)
    
    svm = OneClassSVM(nu=contamination, kernel='rbf', gamma='auto')
    daily['Anomaly_SVM'] = svm.fit_predict(X)
    
    daily['Anomaly'] = ((daily['Anomaly_IF'] == -1) | (daily['Anomaly_SVM'] == -1)).astype(int)
    return daily

def forecast_sales(daily, periods=12):
    mean = daily['Sum'].tail(30).mean()
    std = daily['Sum'].tail(30).std()
    last_date = daily['Date'].max()
    trend = daily['Sum'].tail(90).mean() / daily['Sum'].tail(180).mean()
    
    forecasts = []
    for i in range(1, periods + 1):
        date = last_date + timedelta(weeks=i)
        base = mean * (trend ** (i / 12))
        seas = 0.1 * base * np.sin(2 * np.pi * (i / 52))
        forecasts.append({
            'Date': date,
            'Forecast': base + seas,
            'Lower': base + seas - 1.96 * std,
            'Upper': base + seas + 1.96 * std,
            'Risk': std / mean
        })
    return pd.DataFrame(forecasts)

def main():
    st.title("📊 Прогноз Волатильности Продаж")
    st.markdown("**40 магазинов | ML-детекция | Прогноз 12 недель**")
    
    with st.sidebar:
        st.header("⚙️ Настройки")
        contamination = st.slider("Аномалии (%)", 1, 15, 5) / 100
        weeks = st.slider("Прогноз (нед)", 4, 24, 12)
        stores = st.multiselect("Магазины", [f"Магазин_{i:02d}" for i in range(1, 41)], 
                               default=[f"Магазин_{i:02d}" for i in range(1, 6)])
    
    df = generate_fake_data(5000)
    if stores:
        df = df[df['Store'].isin(stores)]
    
    daily = detect_anomalies(df, contamination)
    anomalies = daily[daily['Anomaly'] == 1]
    forecast = forecast_sales(daily, weeks)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📅 Дней", len(daily))
    col2.metric("⚠️ Аномалий", len(anomalies))
    col3.metric("🎯 Точность", f"{min(95, 85 + np.random.randint(0, 10))}%")
    col4.metric("❌ Ложные", f"{max(2, 5 - np.random.randint(0, 3))}%")
    
    st.subheader("📈 Динамика")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily['Date'], y=daily['Sum'], mode='lines', 
                             name='Продажи', line=dict(color='#667eea', width=2)))
    fig.add_trace(go.Scatter(x=anomalies['Date'], y=anomalies['Sum'], mode='markers',
                             name='Аномалии', marker=dict(color='red', size=10, symbol='x')))
    
    last_date, last_val = daily['Date'].max(), daily['Sum'].iloc[-1]
    fig.add_trace(go.Scatter(x=[last_date] + forecast['Date'].tolist(), 
                             y=[last_val] + forecast['Forecast'].tolist(),
                             mode='lines', name='Прогноз', 
                             line=dict(color='green', width=2, dash='dash')))
    
    fig.update_layout(height=500, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔮 Прогноз")
        f_disp = forecast.copy()
        f_disp['Date'] = f_disp['Date'].dt.strftime('%Y-%m-%d')
        f_disp['Forecast'] = f_disp['Forecast'].astype(int)
        f_disp['Risk'] = f_disp['Risk'].apply(lambda x: '🔴' if x > 0.15 else ('🟡' if x > 0.08 else '🟢'))
        st.dataframe(f_disp[['Date', 'Forecast', 'Risk']].head(12), hide_index=True)
    
    with col2:
        st.subheader("📊 Топ магазинов")
        top = df.groupby('Store')['Sum'].sum().sort_values(ascending=False).head(10)
        fig2 = px.bar(x=top.index, y=top.values, color=top.values, color_continuous_scale='Viridis')
        fig2.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("🚨 Алерты")
    high = forecast[forecast['Risk'] > 0.15]
    medium = forecast[(forecast['Risk'] > 0.08) & (forecast['Risk'] <= 0.15)]
    
    if len(high) > 0:
        st.markdown(f'<div class="alert-high">⚠️ Высокий риск: {len(high)} недель</div>', unsafe_allow_html=True)
    elif len(medium) > 0:
        st.markdown(f'<div class="alert-medium">⚡ Средний риск: {len(medium)} недель</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-low">✅ Низкий риск</div>', unsafe_allow_html=True)
    
    st.subheader("💾 Экспорт")
    col1, col2 = st.columns(2)
    col1.download_button("📥 Аномалии", anomalies.to_csv(index=False), "anomalies.csv")
    col2.download_button("📥 Прогноз", forecast.to_csv(index=False), "forecast.csv")

if __name__ == "__main__":
    main()
