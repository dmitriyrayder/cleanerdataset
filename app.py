import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re

st.set_page_config(page_title="📊 Контроль качества данных", layout="wide")
st.title("🧠 ML + AgGrid: Расширенный контроль качества данных")

def validate_email(email):
    if pd.isna(email): return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email)))

def validate_phone(phone):
    if pd.isna(phone): return False
    cleaned = re.sub(r'[^\d]', '', str(phone))
    return len(cleaned) >= 10 and len(cleaned) <= 15

def validate_address(address):
    if pd.isna(address): return False
    addr_str = str(address).strip()
    return len(addr_str) >= 5 and any(char.isdigit() for char in addr_str)

def validate_city(city):
    if pd.isna(city): return False
    city_str = str(city).strip()
    return len(city_str) >= 2 and city_str.replace(' ', '').isalpha()

def detect_outliers_iqr(series):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return (series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)

def create_comprehensive_report(df):
    report = {
        'total_rows': len(df), 'total_columns': len(df.columns),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,
        'data_types': df.dtypes.value_counts().to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'duplicate_percentage': df.duplicated().sum() / len(df) * 100,
        'completely_empty_rows': (df.isnull().all(axis=1)).sum(),
        'rows_with_missing': (df.isnull().any(axis=1)).sum()
    }
    return report

uploaded_file = st.file_uploader("Загрузите файл данных", type=["xlsx", "xls", "csv"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        else:
            df = pd.read_excel(uploaded_file)
        
        # Парсинг дат
        if 'Datasales' in df.columns:
            try:
                df['Datasales'] = pd.to_datetime(df['Datasales'], errors='coerce')
            except:
                st.warning("Не удалось преобразовать даты")
        
        st.success("✅ Файл успешно загружен!")
        
        # === РАСШИРЕННАЯ АНАЛИТИКА ===
        st.header("📈 Расширенная аналитика датасета")
        quality_report = create_comprehensive_report(df)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("📊 Всего строк", quality_report['total_rows'])
        with col2:
            st.metric("📋 Колонок", quality_report['total_columns'])
        with col3:
            st.metric("💾 Размер", f"{quality_report['memory_usage']:.1f} MB")
        with col4:
            st.metric("🔄 Дубликаты", f"{quality_report['duplicate_rows']}")
        with col5:
            st.metric("🕳️ Пустые строки", f"{quality_report['completely_empty_rows']}")
        
        # Детальный анализ колонок
        st.subheader("🔍 Анализ по колонкам")
        column_analysis = []
        for col in df.columns:
            analysis = {
                'Колонка': col, 'Тип': str(df[col].dtype),
                'Пропущено': df[col].isnull().sum(),
                'Пропущено %': f"{df[col].isnull().sum() / len(df) * 100:.1f}%",
                'Уникальных': df[col].nunique(),
                'Дубликатов': df[col].duplicated().sum()
            }
            
            if df[col].dtype in ['int64', 'float64']:
                analysis.update({
                    'Мин': df[col].min(), 'Макс': df[col].max(),
                    'Среднее': f"{df[col].mean():.2f}",
                    'Медиана': f"{df[col].median():.2f}",
                    'Стд.откл': f"{df[col].std():.2f}"
                })
            elif df[col].dtype == 'object':
                lens = df[col].str.len()
                analysis.update({
                    'Мин длина': lens.min() if not df[col].empty else 0,
                    'Макс длина': lens.max() if not df[col].empty else 0,
                    'Ср.длина': f"{lens.mean():.1f}" if not df[col].empty else "0",
                    'Пустые': (df[col] == '').sum()
                })
            column_analysis.append(analysis)
        
        st.dataframe(pd.DataFrame(column_analysis), use_container_width=True)
        
        # === ПРОВЕРКИ КАЧЕСТВА ===
        st.header("🔍 Проверки качества данных")
        
        # Проверка обязательных колонок
        required_cols = ['Magazin', 'Adress', 'City', 'Datasales', 'Describe', 'Model', 'Segment', 'Price', 'Qty', 'Sum']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Отсутствуют колонки: {', '.join(missing_cols)}")
        else:
            st.success("✅ Все обязательные колонки присутствуют")
        
        # Математические проверки
        if all(col in df.columns for col in ['Price', 'Qty', 'Sum']):
            df['calc_sum'] = df['Price'] * df['Qty']
            df['sum_error'] = np.abs(df['calc_sum'] - df['Sum']) > 1e-2
            df['negative_price'] = df['Price'] < 0
            df['negative_qty'] = df['Qty'] < 0
            df['negative_sum'] = df['Sum'] < 0
            df['zero_price'] = df['Price'] == 0
            df['zero_qty'] = df['Qty'] == 0
            df['unrealistic_price'] = df['Price'] > 1000000
            df['unrealistic_qty'] = df['Qty'] > 10000
            
            st.subheader("🧮 Математические проверки")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("❌ Ошибки суммы", df['sum_error'].sum())
                st.metric("➖ Отриц. цены", df['negative_price'].sum())
            with col2:
                st.metric("➖ Отриц. кол-во", df['negative_qty'].sum())
                st.metric("➖ Отриц. суммы", df['negative_sum'].sum())
            with col3:
                st.metric("0️⃣ Нулевые цены", df['zero_price'].sum())
                st.metric("0️⃣ Нулевые кол-ва", df['zero_qty'].sum())
            with col4:
                st.metric("💰 Нереальные цены", df['unrealistic_price'].sum())
                st.metric("📦 Нереальные кол-ва", df['unrealistic_qty'].sum())
        
        # Проверка дат
        if 'Datasales' in df.columns:
            st.subheader("📅 Проверка дат")
            today = datetime.now()
            df['future_date'] = df['Datasales'] > today
            df['old_date'] = df['Datasales'] < (today - timedelta(days=365*5))
            df['invalid_date'] = df['Datasales'].isnull()
            df['weekend_sales'] = df['Datasales'].dt.weekday >= 5
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🔮 Будущие даты", df['future_date'].sum())
            with col2:
                st.metric("📜 Старые даты", df['old_date'].sum())
            with col3:
                st.metric("❌ Неверные даты", df['invalid_date'].sum())
            with col4:
                st.metric("📅 Выходные", df['weekend_sales'].sum())
        
        # Проверка адресов и городов
        st.subheader("🏠 Проверка адресов и городов")
        if 'Adress' in df.columns:
            df['invalid_address'] = ~df['Adress'].apply(validate_address)
            df['short_address'] = df['Adress'].str.len() < 5
            df['long_address'] = df['Adress'].str.len() > 100
        
        if 'City' in df.columns:
            df['invalid_city'] = ~df['City'].apply(validate_city)
            df['short_city'] = df['City'].str.len() < 2
            df['numeric_city'] = df['City'].str.isdigit()
        
        col1, col2, col3, col4 = st.columns(4)
        if 'Adress' in df.columns:
            with col1:
                st.metric("🏠 Неверные адреса", df['invalid_address'].sum())
            with col2:
                st.metric("📏 Короткие адреса", df['short_address'].sum())
        if 'City' in df.columns:
            with col3:
                st.metric("🏙️ Неверные города", df['invalid_city'].sum())
            with col4:
                st.metric("🔢 Цифровые города", df['numeric_city'].sum())
        
        # Проверка текстовых полей
        st.subheader("📝 Проверка текстовых полей")
        text_cols = ['Magazin', 'Describe', 'Model', 'Segment']
        
        for col in text_cols:
            if col in df.columns:
                df[f'{col}_suspicious'] = df[col].str.contains(r'[<>{}[\]\\|`~@#$%^&*]', na=False)
                df[f'{col}_too_short'] = df[col].str.len() < 2
                df[f'{col}_too_long'] = df[col].str.len() > 100
                df[f'{col}_only_digits'] = df[col].str.isdigit()
                df[f'{col}_special_chars'] = df[col].str.contains(r'[^\w\s\-]', na=False)
        
        # Отображение проблем текстовых полей
        text_problems = {}
        for col in text_cols:
            if col in df.columns:
                problems = (df[f'{col}_suspicious'].sum() + df[f'{col}_too_short'].sum() + 
                          df[f'{col}_too_long'].sum() + df[f'{col}_only_digits'].sum())
                text_problems[col] = problems
        
        if text_problems:
            col1, col2, col3, col4 = st.columns(4)
            cols = list(text_problems.keys())
            for i, (col, problems) in enumerate(text_problems.items()):
                with [col1, col2, col3, col4][i % 4]:
                    st.metric(f"📝 {col} проблемы", problems)
        
        # ML-анализ аномалий
        st.subheader("🤖 ML-анализ аномалий")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        ml_cols = [col for col in numeric_cols if not any(x in col for x in ['calc_', 'sum_', 'negative_', 'zero_', 'unrealistic_'])]
        
        if len(ml_cols) >= 2:
            try:
                X = df[ml_cols].fillna(0)
                if X.shape[1] >= 2:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Isolation Forest
                    iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=50)
                    df['ml_anomaly'] = iso.fit_predict(X_scaled) == -1
                    
                    # IQR outliers
                    for col in ['Price', 'Qty', 'Sum']:
                        if col in df.columns:
                            df[f'{col}_outlier'] = detect_outliers_iqr(df[col])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🚨 ML-аномалии", df['ml_anomaly'].sum())
                    with col2:
                        if 'Price_outlier' in df.columns:
                            price_outliers = df['Price_outlier'].sum()
                            st.metric("💰 Выбросы цен", price_outliers)
                    with col3:
                        if 'Qty_outlier' in df.columns:
                            qty_outliers = df['Qty_outlier'].sum()
                            st.metric("📦 Выбросы кол-ва", qty_outliers)
                    
                    st.success("✅ ML-анализ завершен")
                else:
                    df['ml_anomaly'] = False
                    st.warning("⚠️ Недостаточно данных для ML")
            except Exception as e:
                df['ml_anomaly'] = False
                st.error(f"❌ Ошибка ML: {str(e)[:50]}")
        else:
            df['ml_anomaly'] = False
            st.warning("⚠️ Нет числовых данных для ML")
        
        # Дополнительные проверки целостности
        st.subheader("🔗 Проверки целостности")
        
        # Проверка дубликатов по ключевым полям
        if all(col in df.columns for col in ['Magazin', 'Datasales', 'Art']):
            df['duplicate_transaction'] = df.duplicated(subset=['Magazin', 'Datasales', 'Art'], keep=False)
        
        # Проверка консистентности моделей и сегментов
        if all(col in df.columns for col in ['Model', 'Segment']):
            model_segment_map = df.groupby('Model')['Segment'].nunique()
            inconsistent_models = model_segment_map[model_segment_map > 1].index
            df['inconsistent_model_segment'] = df['Model'].isin(inconsistent_models)
        
        # Сводка всех проблем
        problem_cols = [col for col in df.columns if any(x in col for x in 
                       ['error', 'negative_', 'zero_', 'future_', 'old_', 'invalid_', 'suspicious', 
                        'too_short', 'too_long', 'only_digits', 'ml_anomaly', 'outlier', 'duplicate_', 'inconsistent'])]
        
        if problem_cols:
            df['total_problems'] = df[problem_cols].sum(axis=1)
            
            st.subheader("📋 Общая сводка")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Строк с проблемами", (df['total_problems'] > 0).sum())
            with col2:
                st.metric("✅ Чистых строк", (df['total_problems'] == 0).sum())
            with col3:
                st.metric("📈 Качество данных", f"{(df['total_problems'] == 0).sum() / len(df) * 100:.1f}%")
            with col4:
                st.metric("⚠️ Всего проблем", df[problem_cols].sum().sum())
            
            # График проблем
            st.subheader("📊 Распределение проблем")
            problem_counts = df[problem_cols].sum().sort_values(ascending=False)[:15]
            
            if not problem_counts.empty:
                fig = px.bar(x=problem_counts.values, y=problem_counts.index, orientation='h',
                           title="Топ-15 типов проблем", height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # === AgGrid ===
        st.header("📋 Интерактивное редактирование")
        
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(editable=True, filter=True, sortable=True, resizable=True)
        
        # Условная раскраска
        cell_style = JsCode("""
        function(params) {
            const data = params.data;
            if (data.sum_error || data.negative_price || data.negative_qty) return {backgroundColor: '#ffcccc'};
            if (data.ml_anomaly) return {backgroundColor: '#ffffcc'};
            if (data.future_date || data.invalid_date) return {backgroundColor: '#ffe6cc'};
            if (data.invalid_address || data.invalid_city) return {backgroundColor: '#e6ccff'};
            if (data.Price_outlier || data.Qty_outlier) return {backgroundColor: '#ffccff'};
            return null;
        }
        """)
        
        main_cols = ['Magazin', 'Adress', 'City', 'Datasales', 'Describe', 'Model', 'Segment', 'Price', 'Qty', 'Sum']
        for col in main_cols:
            if col in df.columns:
                gb.configure_column(col, cellStyle=cell_style)
        
        gb.configure_grid_options(domLayout='normal', pagination=True, paginationPageSize=50)
        gb.configure_selection(selection_mode="multiple", use_checkbox=True)
        
        # Легенда
        st.markdown("""
        **🎨 Цвета:** 🔴 Математические ошибки | 🟡 ML-аномалии | 🟠 Проблемы дат | 🟣 Адреса/города | 🟢 Выбросы
        """)
        
        grid_response = AgGrid(df, gridOptions=gb.build(), height=500, 
                              update_mode=GridUpdateMode.VALUE_CHANGED,
                              fit_columns_on_grid_load=True, allow_unsafe_jscode=True,
                              theme="alpine")
        
        updated_df = pd.DataFrame(grid_response['data'])
        
        # === Экспорт ===
        st.header("📤 Экспорт результатов")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_all = updated_df.to_csv(index=False)
            st.download_button("⬇️ Все данные", csv_all, "all_data_checked.csv", "text/csv")
        
        with col2:
            if 'total_problems' in updated_df.columns:
                clean_data = updated_df[updated_df['total_problems'] == 0]
                clean_cols = [col for col in main_cols if col in clean_data.columns]
                csv_clean = clean_data[clean_cols].to_csv(index=False)
                st.download_button("✅ Чистые данные", csv_clean, "clean_data.csv", "text/csv")
        
        with col3:
            if 'total_problems' in updated_df.columns:
                problem_data = updated_df[updated_df['total_problems'] > 0]
                if not problem_data.empty:
                    csv_problems = problem_data.to_csv(index=False)
                    st.download_button("⚠️ Проблемные данные", csv_problems, "problems.csv", "text/csv")
        
        # Финальная статистика
        st.subheader("🎯 Итоговые метрики")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Обработано", len(updated_df))
        with col2:
            if 'total_problems' in updated_df.columns:
                st.metric("✅ Качественных", (updated_df['total_problems'] == 0).sum())
        with col3:
            if 'total_problems' in updated_df.columns:
                st.metric("⚠️ С проблемами", (updated_df['total_problems'] > 0).sum())
        with col4:
            if 'total_problems' in updated_df.columns:
                quality = (updated_df['total_problems'] == 0).sum() / len(updated_df) * 100
                st.metric("🏆 Качество", f"{quality:.1f}%")
        
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")

else:
    st.info("👆 Загрузите файл для анализа")
    st.markdown("""
    **📋 Ожидаемые колонки:**
    - `Magazin` - название магазина
    - `Adress` - адрес магазина  
    - `City` - город
    - `Datasales` - дата продажи
    - `Art` - артикул товара
    - `Describe` - описание товара
    - `Model` - модель товара
    - `Segment` - сегмент товара
    - `Price` - цена за единицу
    - `Qty` - количество
    - `Sum` - общая сумма
    
    **🔍 Типы проверок:**
    - Математическая корректность (Price × Qty = Sum)
    - Валидация дат (будущие/старые даты)
    - Проверка адресов и городов
    - ML-детекция аномалий
    - Поиск выбросов и дубликатов
    - Анализ текстовых полей
    """)
