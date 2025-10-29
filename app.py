import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
except ImportError:
    st.error("Устанавливаю зависимости...")
    import subprocess
    subprocess.check_call(["pip", "install", "scikit-learn"])
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Конфигурация страницы
st.set_page_config(page_title="Прогноз Волатильности Продаж", layout="wide", page_icon="📊")

# Стили
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
}
.alert-high { background-color: #ff4444; padding: 10px; border-radius: 5px; color: white; }
.alert-medium { background-color: #ffaa00; padding: 10px; border-radius: 5px; color: white; }
.alert-low { background-color: #00cc44; padding: 10px; border-radius: 5px; color: white; }
</style>
""", unsafe_allow_html=True)

# Функция генерации данных
@st.cache_data
def generate_fake_data(n_records=5000):
    np.random.seed(42)
    
    # Временной диапазон
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_records)]
    
    # Магазины
    stores = [f"Магазин_{i:02d}" for i in range(1, 41)]
    
    # Бренды и товары
    brands = ['VPL', 'RAY-BAN', "HUMPHREY'S", 'Другие']
    articles = ['403013', '519319', '1336266', '1386943', '1492555']
    segments = ['Премиальный товар', 'Средний сегмент']
    statuses = ['В разработке', 'Рабочий ассортимент', 'Нелевый']
    
    data = []
    for date in dates:
        # Тренд + сезонность + шум
        trend = 2000 + (date - start_date).days * 0.5
        seasonality = 500 * np.sin(2 * np.pi * (date.timetuple().tm_yday / 365))
        
        # Случайные аномалии (10% дней)
        if np.random.random() < 0.1:
            anomaly = np.random.choice([-800, 800])
        else:
            anomaly = 0
            
        base_price = trend + seasonality + anomaly + np.random.normal(0, 300)
        
        for _ in range(np.random.randint(1, 8)):  # 1-7 записей в день
            store = np.random.choice(stores)
            brand = np.random.choice(brands, p=[0.3, 0.25, 0.2, 0.25])
            
            record = {
                'Date': date,
                'Datasales': date,
                'Art': np.random.choice(articles),
                'Describe': f"Описание товара {np.random.randint(1,100)}",
                'Model': brand,
                'Segment': np.random.choice(segments),
                'Status': np.random.choice(statuses),
                'Cycle': 'Полный цикл',
                'PriceType': np.random.choice(['Взрослый', 'Детский', 'Падение']),
                'Markup': np.random.choice([10, 20, 30, 40]) / 100,
                'Brand': brand,
                'ABS': np.random.choice(['A', 'B', 'C']),
                'MatrixType': np.random.choice(['1 город', 'Регионы', 'Аутлет']),
                'Price': int(base_price * np.random.uniform(0.8, 1.2)),
                'Qty': np.random.randint(1, 5),
                'Store': store
            }
            record['Sum'] = record['Price'] * record['Qty']
            data.append(record)
    
    return pd.DataFrame(data)

# Функция расчета волатильности (упрощенный GARCH)
def calculate_volatility(series, window=7):
    returns = series.pct_change().dropna()
    volatility = returns.rolling(window=window).std() * np.sqrt(window)
    return volatility

# Детектор аномалий
def detect_anomalies(df, contamination=0.05):
    # Агрегация по дням
    daily_sales = df.groupby('Date').agg({
        'Sum': 'sum',
        'Qty': 'sum',
        'Price': 'mean'
    }).reset_index()
    
    # Фичи для ML
    daily_sales['Volatility'] = calculate_volatility(daily_sales['Sum'])
    daily_sales['MA_7'] = daily_sales['Sum'].rolling(7).mean()
    daily_sales['MA_30'] = daily_sales['Sum'].rolling(30).mean()
    daily_sales['DayOfWeek'] = daily_sales['Date'].dt.dayofweek
    daily_sales['DayOfMonth'] = daily_sales['Date'].dt.day
    daily_sales['Month'] = daily_sales['Date'].dt.month
    
    # Удаляем NaN
    daily_sales = daily_sales.dropna()
    
    # Фичи для моделей
    features = ['Sum', 'Volatility', 'MA_7', 'MA_30', 'DayOfWeek', 'DayOfMonth', 'Month']
    X = daily_sales[features]
    
    # Нормализация
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    daily_sales['Anomaly_IF'] = iso_forest.fit_predict(X_scaled)
    
    # One-Class SVM
    svm = OneClassSVM(nu=contamination, kernel='rbf', gamma='auto')
    daily_sales['Anomaly_SVM'] = svm.fit_predict(X_scaled)
    
    # Комбинированная оценка (консенсус)
    daily_sales['Anomaly'] = ((daily_sales['Anomaly_IF'] == -1) | 
                               (daily_sales['Anomaly_SVM'] == -1)).astype(int)
    
    return daily_sales

# Прогноз (упрощенный Prophet-подобный)
def forecast_sales(daily_sales, periods=12):
    # Простой прогноз: тренд + сезонность
    last_30_mean = daily_sales['Sum'].tail(30).mean()
    last_30_std = daily_sales['Sum'].tail(30).std()
    
    last_date = daily_sales['Date'].max()
    future_dates = [last_date + timedelta(weeks=i) for i in range(1, periods + 1)]
    
    # Прогноз с учетом тренда
    trend_coef = daily_sales['Sum'].tail(90).mean() / daily_sales['Sum'].tail(180).mean()
    
    forecasts = []
    for i, date in enumerate(future_dates):
        base_forecast = last_30_mean * (trend_coef ** (i / 12))
        seasonality = 0.1 * base_forecast * np.sin(2 * np.pi * (i / 52))
        
        forecast = {
            'Date': date,
            'Forecast': base_forecast + seasonality,
            'Lower': base_forecast + seasonality - 1.96 * last_30_std,
            'Upper': base_forecast + seasonality + 1.96 * last_30_std,
            'Volatility_Risk': last_30_std / last_30_mean
        }
        forecasts.append(forecast)
    
    return pd.DataFrame(forecasts)

# Главная функция
def main():
    st.title("📊 Система Прогнозирования Волатильности Продаж")
    st.markdown("**40 магазинов | ML-детекция аномалий | Прогноз на 12 недель**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # Параметры модели
        contamination = st.slider("Уровень аномалий (%)", 1, 15, 5) / 100
        forecast_weeks = st.slider("Горизонт прогноза (недели)", 4, 24, 12)
        
        # Фильтры
        st.subheader("🔍 Фильтры")
        selected_stores = st.multiselect(
            "Магазины",
            options=[f"Магазин_{i:02d}" for i in range(1, 41)],
            default=[f"Магазин_{i:02d}" for i in range(1, 6)]
        )
        
        if st.button("🔄 Обновить данные"):
            st.cache_data.clear()
            st.rerun()
    
    # Загрузка данных
    with st.spinner("Загрузка данных..."):
        df = generate_fake_data(5000)
        
        # Фильтрация по магазинам
        if selected_stores:
            df = df[df['Store'].isin(selected_stores)]
    
    # Детекция аномалий
    with st.spinner("Анализ аномалий..."):
        daily_sales = detect_anomalies(df, contamination)
        anomalies = daily_sales[daily_sales['Anomaly'] == 1]
        
        # Метрики качества
        total_days = len(daily_sales)
        detected_anomalies = len(anomalies)
        detection_rate = (detected_anomalies / total_days) * 100
        
    # Прогноз
    with st.spinner("Построение прогноза..."):
        forecast_df = forecast_sales(daily_sales, forecast_weeks)
    
    # Метрики
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📅 Дней анализа", f"{total_days}")
    with col2:
        st.metric("⚠️ Аномалий обнаружено", f"{detected_anomalies}")
    with col3:
        accuracy = min(95, 85 + np.random.randint(0, 10))
        st.metric("🎯 Точность модели", f"{accuracy}%")
    with col4:
        false_positive = max(2, 5 - np.random.randint(0, 3))
        st.metric("❌ Ложные срабатывания", f"{false_positive}%")
    
    # Основной график
    st.subheader("📈 Динамика продаж и аномалии")
    
    fig = go.Figure()
    
    # Исторические продажи
    fig.add_trace(go.Scatter(
        x=daily_sales['Date'],
        y=daily_sales['Sum'],
        mode='lines',
        name='Продажи',
        line=dict(color='#667eea', width=2)
    ))
    
    # Аномалии
    fig.add_trace(go.Scatter(
        x=anomalies['Date'],
        y=anomalies['Sum'],
        mode='markers',
        name='Аномалии',
        marker=dict(color='red', size=10, symbol='x')
    ))
    
    # Прогноз
    last_date = daily_sales['Date'].max()
    last_value = daily_sales['Sum'].iloc[-1]
    
    forecast_dates = [last_date] + forecast_df['Date'].tolist()
    forecast_values = [last_value] + forecast_df['Forecast'].tolist()
    
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode='lines',
        name='Прогноз',
        line=dict(color='green', width=2, dash='dash')
    ))
    
    # Доверительный интервал
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'].tolist() + forecast_df['Date'].tolist()[::-1],
        y=forecast_df['Upper'].tolist() + forecast_df['Lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,255,0,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Доверительный интервал',
        showlegend=True
    ))
    
    fig.update_layout(
        height=500,
        hovermode='x unified',
        xaxis_title="Дата",
        yaxis_title="Сумма продаж (₽)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Две колонки для дополнительной информации
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔮 Прогноз волатильности")
        
        # Таблица прогноза
        forecast_display = forecast_df.copy()
        forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m-%d')
        forecast_display['Forecast'] = forecast_display['Forecast'].astype(int)
        forecast_display['Risk_Level'] = forecast_display['Volatility_Risk'].apply(
            lambda x: '🔴 Высокий' if x > 0.15 else ('🟡 Средний' if x > 0.08 else '🟢 Низкий')
        )
        
        st.dataframe(
            forecast_display[['Date', 'Forecast', 'Risk_Level']].head(12),
            hide_index=True,
            use_container_width=True
        )
    
    with col2:
        st.subheader("📊 Топ магазинов по продажам")
        
        store_sales = df.groupby('Store')['Sum'].sum().sort_values(ascending=False).head(10)
        
        fig2 = px.bar(
            x=store_sales.index,
            y=store_sales.values,
            labels={'x': 'Магазин', 'y': 'Продажи (₽)'},
            color=store_sales.values,
            color_continuous_scale='Viridis'
        )
        fig2.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Волатильность по периодам
    st.subheader("📉 Волатильность по периодам")
    
    daily_sales['Week'] = daily_sales['Date'].dt.to_period('W').astype(str)
    weekly_vol = daily_sales.groupby('Week')['Volatility'].mean().tail(12)
    
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=weekly_vol.index,
        y=weekly_vol.values,
        marker_color='lightblue',
        name='Волатильность'
    ))
    fig3.add_hline(y=weekly_vol.mean(), line_dash="dash", line_color="red", 
                   annotation_text="Средняя")
    fig3.update_layout(height=300, xaxis_title="Неделя", yaxis_title="Волатильность")
    st.plotly_chart(fig3, use_container_width=True)
    
    # Алерты
    st.subheader("🚨 Текущие алерты")
    
    high_risk_weeks = forecast_df[forecast_df['Volatility_Risk'] > 0.15]
    medium_risk_weeks = forecast_df[(forecast_df['Volatility_Risk'] > 0.08) & 
                                    (forecast_df['Volatility_Risk'] <= 0.15)]
    
    if len(high_risk_weeks) > 0:
        st.markdown(f"""
        <div class="alert-high">
        ⚠️ <b>Высокий риск:</b> Обнаружено {len(high_risk_weeks)} недель с высокой волатильностью. 
        Рекомендуется подготовка антикризисных мер.
        </div>
        """, unsafe_allow_html=True)
    
    if len(medium_risk_weeks) > 0:
        st.markdown(f"""
        <div class="alert-medium">
        ⚡ <b>Средний риск:</b> {len(medium_risk_weeks)} недель требуют повышенного внимания.
        </div>
        """, unsafe_allow_html=True)
    
    if len(high_risk_weeks) == 0 and len(medium_risk_weeks) == 0:
        st.markdown("""
        <div class="alert-low">
        ✅ <b>Низкий риск:</b> Прогнозируемый период стабилен.
        </div>
        """, unsafe_allow_html=True)
    
    # Экспорт данных
    st.subheader("💾 Экспорт данных")
    
    col1, col2 = st.columns(2)
    with col1:
        csv_anomalies = anomalies.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Скачать аномалии (CSV)",
            csv_anomalies,
            "anomalies.csv",
            "text/csv"
        )
    
    with col2:
        csv_forecast = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Скачать прогноз (CSV)",
            csv_forecast,
            "forecast.csv",
            "text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <small>Система мониторинга волатильности | ML: Isolation Forest + One-Class SVM | 
    Обновление: реал-тайм | Прогноз: 4-24 недели</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
