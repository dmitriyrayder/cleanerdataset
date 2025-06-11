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

st.set_page_config(page_title="📊 Контроль качества с AgGrid", layout="wide")
st.title("🧠 ML + AgGrid: Контроль качества данных")

# === Функции для проверки качества данных ===
def validate_email(email):
    """Проверка корректности email"""
    if pd.isna(email):
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email)))

def validate_phone(phone):
    """Проверка корректности телефона"""
    if pd.isna(phone):
        return False
    # Простая проверка на цифры и длину
    cleaned = re.sub(r'[^\d]', '', str(phone))
    return len(cleaned) >= 10 and len(cleaned) <= 15

def detect_outliers_iqr(series):
    """Определение выбросов методом IQR"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (series < lower_bound) | (series > upper_bound)

def analyze_text_quality(series):
    """Анализ качества текстовых данных"""
    results = {}
    results['null_count'] = series.isnull().sum()
    results['empty_count'] = (series == '').sum()
    results['whitespace_only'] = series.str.strip().eq('').sum() if series.dtype == 'object' else 0
    results['unique_count'] = series.nunique()
    results['duplicate_count'] = series.duplicated().sum()
    results['min_length'] = series.str.len().min() if series.dtype == 'object' else None
    results['max_length'] = series.str.len().max() if series.dtype == 'object' else None
    results['avg_length'] = series.str.len().mean() if series.dtype == 'object' else None
    return results

def create_data_quality_report(df):
    """Создание отчета о качестве данных"""
    report = {}
    
    # Общая информация
    report['total_rows'] = len(df)
    report['total_columns'] = len(df.columns)
    report['memory_usage'] = df.memory_usage(deep=True).sum() / 1024**2  # MB
    
    # Анализ по типам данных
    report['data_types'] = df.dtypes.value_counts().to_dict()
    
    # Пропущенные значения
    report['missing_values'] = df.isnull().sum().to_dict()
    report['missing_percentage'] = (df.isnull().sum() / len(df) * 100).to_dict()
    
    # Дубликаты
    report['duplicate_rows'] = df.duplicated().sum()
    report['duplicate_percentage'] = df.duplicated().sum() / len(df) * 100
    
    return report

# === Загрузка файла ===
uploaded_file = st.file_uploader("Загрузите Excel-файл", type=["xlsx", "xls", "csv"])

if uploaded_file:
    try:
        # Определение типа файла и загрузка
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Попытка парсинга дат
        date_columns = ['Datasales'] if 'Datasales' in df.columns else []
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                st.warning(f"Не удалось преобразовать колонку {col} в дату")
        
        st.success("✅ Файл успешно загружен!")
        
        # === РАСШИРЕННАЯ АНАЛИТИКА ДАТАСЕТА ===
        st.header("📈 Расширенная аналитика датасета")
        
        # Создание отчета о качестве данных
        quality_report = create_data_quality_report(df)
        
        # Отображение основных метрик
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Всего строк", quality_report['total_rows'])
        with col2:
            st.metric("📋 Всего колонок", quality_report['total_columns'])
        with col3:
            st.metric("💾 Размер в памяти", f"{quality_report['memory_usage']:.2f} MB")
        with col4:
            st.metric("🔄 Дубликаты", f"{quality_report['duplicate_rows']} ({quality_report['duplicate_percentage']:.1f}%)")
        
        # Детальная таблица анализа по колонкам
        st.subheader("🔍 Детальный анализ по колонкам")
        
        column_analysis = []
        for col in df.columns:
            analysis = {
                'Колонка': col,
                'Тип данных': str(df[col].dtype),
                'Пропущено': df[col].isnull().sum(),
                'Пропущено %': f"{df[col].isnull().sum() / len(df) * 100:.1f}%",
                'Уникальных': df[col].nunique(),
                'Дубликатов': df[col].duplicated().sum(),
            }
            
            if df[col].dtype in ['int64', 'float64']:
                analysis['Мин'] = df[col].min()
                analysis['Макс'] = df[col].max()
                analysis['Среднее'] = f"{df[col].mean():.2f}"
                analysis['Медиана'] = f"{df[col].median():.2f}"
                analysis['Стд. отклонение'] = f"{df[col].std():.2f}"
            elif df[col].dtype == 'object':
                analysis['Мин длина'] = df[col].str.len().min() if not df[col].empty else 0
                analysis['Макс длина'] = df[col].str.len().max() if not df[col].empty else 0
                analysis['Средняя длина'] = f"{df[col].str.len().mean():.1f}" if not df[col].empty else "0"
                analysis['Пустые строки'] = (df[col] == '').sum()
            
            column_analysis.append(analysis)
        
        analysis_df = pd.DataFrame(column_analysis)
        st.dataframe(analysis_df, use_container_width=True)
        
        # === ПРОВЕРКИ КАЧЕСТВА ДАННЫХ ===
        st.header("🔍 Проверки качества данных")
        
        # Проверка обязательных колонок
        required_columns = ['Magazin', 'Datasales', 'Describe', 'Model', 'Segment', 'Price', 'Qty', 'Sum']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Отсутствуют обязательные колонки: {', '.join(missing_columns)}")
        else:
            st.success("✅ Все обязательные колонки присутствуют")
        
        # --- Математические проверки ---
        if all(col in df.columns for col in ['Price', 'Qty', 'Sum']):
            df['calc_sum'] = df['Price'] * df['Qty']
            df['sum_diff'] = np.abs(df['calc_sum'] - df['Sum'])
            df['error_sum'] = df['sum_diff'] > 1e-2
            
            # Проверка на отрицательные значения
            df['negative_price'] = df['Price'] < 0
            df['negative_qty'] = df['Qty'] < 0
            df['negative_sum'] = df['Sum'] < 0
            
            # Проверка на нулевые значения
            df['zero_price'] = df['Price'] == 0
            df['zero_qty'] = df['Qty'] == 0
            df['zero_sum'] = df['Sum'] == 0
            
            st.subheader("🧮 Математические проверки")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("❌ Ошибки суммы", df['error_sum'].sum())
                st.metric("➖ Отрицательные цены", df['negative_price'].sum())
            with col2:
                st.metric("➖ Отрицательные количества", df['negative_qty'].sum())
                st.metric("➖ Отрицательные суммы", df['negative_sum'].sum())
            with col3:
                st.metric("0️⃣ Нулевые цены", df['zero_price'].sum())
                st.metric("0️⃣ Нулевые количества", df['zero_qty'].sum())
        
        # --- Проверка дат ---
        if 'Datasales' in df.columns:
            st.subheader("📅 Проверка дат")
            
            # Проверка на будущие даты
            today = datetime.now()
            df['future_date'] = df['Datasales'] > today
            
            # Проверка на слишком старые даты
            old_threshold = today - timedelta(days=365*10)  # 10 лет назад
            df['too_old_date'] = df['Datasales'] < old_threshold
            
            # Проверка на недействительные даты
            df['invalid_date'] = df['Datasales'].isnull()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔮 Будущие даты", df['future_date'].sum())
            with col2:
                st.metric("📜 Слишком старые даты", df['too_old_date'].sum())
            with col3:
                st.metric("❌ Недействительные даты", df['invalid_date'].sum())
        
        # --- Проверка текстовых полей ---
        st.subheader("📝 Проверка текстовых полей")
        text_columns = ['Magazin', 'Describe', 'Model', 'Segment']
        
        for col in text_columns:
            if col in df.columns:
                # Проверка на подозрительные символы
                df[f'{col}_suspicious_chars'] = df[col].str.contains(r'[<>{}[\]\\|`~]', na=False)
                
                # Проверка на слишком короткие/длинные строки
                df[f'{col}_too_short'] = df[col].str.len() < 2
                df[f'{col}_too_long'] = df[col].str.len() > 100
                
                # Проверка на только цифры (подозрительно для текстовых полей)
                df[f'{col}_only_digits'] = df[col].str.isdigit()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(f"❌ {col}: пропущено", df[col].isnull().sum())
                with col2:
                    st.metric(f"⚠️ {col}: подозрительные символы", df[f'{col}_suspicious_chars'].sum())
                with col3:
                    st.metric(f"📏 {col}: слишком короткие", df[f'{col}_too_short'].sum())
                with col4:
                    st.metric(f"🔢 {col}: только цифры", df[f'{col}_only_digits'].sum())
        
        # --- Поиск аномалий с помощью ML ---
        st.subheader("🤖 ML-анализ аномалий")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_columns) >= 2:
            try:
                # Удаляем созданные колонки проверок для ML анализа
                ml_columns = [col for col in numeric_columns if not col.startswith(('calc_', 'sum_', 'error_', 'negative_', 'zero_'))]
                
                if len(ml_columns) >= 2:
                    X = df[ml_columns].fillna(0)
                    X_scaled = StandardScaler().fit_transform(X)
                    
                    # Isolation Forest
                    iso = IsolationForest(contamination=0.05, random_state=42)
                    df['ml_anomaly'] = iso.fit_predict(X_scaled) == -1
                    
                    # IQR outliers для численных колонок
                    for col in ml_columns:
                        if col in ['Price', 'Qty', 'Sum']:
                            df[f'{col}_outlier_iqr'] = detect_outliers_iqr(df[col])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🚨 ML-аномалии", df['ml_anomaly'].sum())
                    with col2:
                        if 'Price_outlier_iqr' in df.columns:
                            total_iqr_outliers = (df['Price_outlier_iqr'] | 
                                                df['Qty_outlier_iqr'] | 
                                                df['Sum_outlier_iqr']).sum()
                            st.metric("📊 IQR-выбросы", total_iqr_outliers)
                    
                    st.success("✅ ML-анализ выполнен успешно")
                else:
                    st.warning("⚠️ Недостаточно численных колонок для ML-анализа")
                    df['ml_anomaly'] = False
            except Exception as e:
                st.error(f"❌ Ошибка ML-анализа: {e}")
                df['ml_anomaly'] = False
        else:
            st.warning("⚠️ Недостаточно численных данных для ML-анализа")
            df['ml_anomaly'] = False
        
        # --- Сводка по всем проблемам ---
        st.subheader("📋 Сводка по проблемам")
        
        # Подсчет всех проблем
        problem_columns = [col for col in df.columns if any(keyword in col for keyword in 
                          ['error_', 'negative_', 'zero_', 'future_', 'too_old_', 'invalid_', 
                           'suspicious_', 'too_short_', 'too_long_', 'only_digits_', 'ml_anomaly', 'outlier_'])]
        
        if problem_columns:
            df['total_problems'] = df[problem_columns].sum(axis=1)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Строк с проблемами", (df['total_problems'] > 0).sum())
            with col2:
                st.metric("✅ Чистых строк", (df['total_problems'] == 0).sum())
            with col3:
                st.metric("📈 Качество данных", f"{(df['total_problems'] == 0).sum() / len(df) * 100:.1f}%")
        
        # --- Визуализация проблем ---
        if problem_columns:
            st.subheader("📊 Визуализация проблем")
            
            # График распределения проблем
            problem_counts = df[problem_columns].sum().sort_values(ascending=True)
            
            fig = px.bar(
                x=problem_counts.values,
                y=problem_counts.index,
                orientation='h',
                title="Распределение типов проблем в данных",
                labels={'x': 'Количество проблем', 'y': 'Тип проблемы'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # === AgGrid: интерактивное редактирование ===
        st.header("📋 Интерактивное редактирование данных")
        
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(editable=True, filter=True, sortable=True, resizable=True)
        
        # Улучшенная условная раскраска
        cell_style_jscode = JsCode("""
        function(params) {
            const data = params.data;
            let backgroundColor = null;
            
            // Математические ошибки - красный
            if (data.error_sum || data.negative_price || data.negative_qty || data.negative_sum) {
                backgroundColor = '#ffcccc';
            }
            // ML аномалии - желтый
            else if (data.ml_anomaly) {
                backgroundColor = '#ffffcc';
            }
            // Проблемы с датами - оранжевый
            else if (data.future_date || data.too_old_date || data.invalid_date) {
                backgroundColor = '#ffe6cc';
            }
            // Проблемы с текстом - голубой
            else if (data.Magazin_suspicious_chars || data.Describe_suspicious_chars || 
                     data.Model_suspicious_chars || data.Segment_suspicious_chars) {
                backgroundColor = '#cce6ff';
            }
            // Выбросы - розовый
            else if (data.Price_outlier_iqr || data.Qty_outlier_iqr || data.Sum_outlier_iqr) {
                backgroundColor = '#ffccff';
            }
            
            return backgroundColor ? { backgroundColor: backgroundColor } : null;
        }
        """)
        
        # Применяем стили к основным колонкам
        main_columns = ['Magazin', 'Datasales', 'Describe', 'Model', 'Segment', 'Price', 'Qty', 'Sum']
        for col in main_columns:
            if col in df.columns:
                gb.configure_column(col, cellStyle=cell_style_jscode)
        
        # Настройка специальных колонок
        boolean_columns = [col for col in df.columns if df[col].dtype == 'bool']
        for col in boolean_columns:
            gb.configure_column(col, type=["booleanColumn"], editable=False)
        
        gb.configure_grid_options(domLayout='normal', pagination=True, paginationPageSize=50)
        gb.configure_selection(selection_mode="multiple", use_checkbox=True)
        
        grid_options = gb.build()
        
        # Добавляем легенду для цветов
        st.markdown("""
        **Легенда цветов:**
        - 🔴 Красный: математические ошибки (неверные суммы, отрицательные значения)
        - 🟡 Желтый: ML-аномалии
        - 🟠 Оранжевый: проблемы с датами
        - 🔵 Голубой: проблемы с текстом
        - 🟣 Розовый: статистические выбросы
        """)
        
        grid_response = AgGrid(
            df,
            gridOptions=grid_options,
            height=600,
            width='100%',
            update_mode=GridUpdateMode.VALUE_CHANGED,
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            theme="alpine",
            enable_enterprise_modules=True
        )
        
        updated_df = pd.DataFrame(grid_response['data'])
        
        # === Экспорт данных ===
        st.header("📤 Экспорт данных")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Экспорт всех данных
            csv_all = updated_df.to_csv(index=False)
            st.download_button(
                "⬇️ Скачать все данные (CSV)",
                csv_all,
                file_name="all_data_with_quality_checks.csv",
                mime="text/csv"
            )
        
        with col2:
            # Экспорт только чистых данных
            if 'total_problems' in updated_df.columns:
                clean_data = updated_df[updated_df['total_problems'] == 0]
                main_cols = [col for col in main_columns if col in clean_data.columns]
                clean_export = clean_data[main_cols]
                
                csv_clean = clean_export.to_csv(index=False)
                st.download_button(
                    "✅ Скачать чистые данные (CSV)",
                    csv_clean,
                    file_name="clean_data_only.csv",
                    mime="text/csv"
                )
        
        with col3:
            # Экспорт отчета о проблемах
            if problem_columns:
                problem_data = updated_df[updated_df['total_problems'] > 0]
                if not problem_data.empty:
                    csv_problems = problem_data.to_csv(index=False)
                    st.download_button(
                        "⚠️ Скачать проблемные данные (CSV)",
                        csv_problems,
                        file_name="problem_data.csv",
                        mime="text/csv"
                    )
        
        # Финальная статистика
        st.subheader("📊 Финальная статистика")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Обработано строк", len(updated_df))
        with col2:
            if 'total_problems' in updated_df.columns:
                clean_count = (updated_df['total_problems'] == 0).sum()
                st.metric("✅ Чистых строк", clean_count)
        with col3:
            if 'total_problems' in updated_df.columns:
                problem_count = (updated_df['total_problems'] > 0).sum()
                st.metric("⚠️ Проблемных строк", problem_count)
        with col4:
            if 'total_problems' in updated_df.columns:
                quality_score = (updated_df['total_problems'] == 0).sum() / len(updated_df) * 100
                st.metric("🎯 Оценка качества", f"{quality_score:.1f}%")
        
    except Exception as e:
        st.error(f"❌ Ошибка обработки файла: {e}")
        st.stop()

else:
    st.info("👆 Загрузите Excel или CSV файл для начала анализа")
    st.markdown("""
    **Ожидаемые колонки:**
    - `Magazin` - название магазина
    - `Datasales` - дата продажи
    - `Art` - артикул (опционально)
    - `Describe` - описание товара
    - `Model` - модель товара
    - `Segment` - сегмент товара
    - `Price` - цена за единицу
    - `Qty` - количество
    - `Sum` - общая сумма
    """)
