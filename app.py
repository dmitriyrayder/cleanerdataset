import streamlit as st
import pandas as pd
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode
import plotly.express as px
from datetime import datetime, timedelta
import re

st.set_page_config(page_title="📊 Контроль качества данных", layout="wide")
st.title("🔍 Контроль качества данных с AgGrid")

def validate_email(email):
    """Проверка email"""
    if pd.isna(email): return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email)))

def validate_phone(phone):
    """Проверка телефона"""
    if pd.isna(phone): return False
    cleaned = re.sub(r'[^\d]', '', str(phone))
    return len(cleaned) >= 10 and len(cleaned) <= 15

def detect_outliers_iqr(series):
    """IQR выбросы"""
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return (series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))

def create_quality_report(df):
    """Отчет качества данных"""
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,
        'data_types': df.dtypes.value_counts().to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'duplicate_percentage': df.duplicated().sum() / len(df) * 100 if len(df) > 0 else 0
    }
    return report

def check_data_consistency(df):
    """Проверка согласованности данных"""
    issues = {}
    
    # Проверка математической согласованности
    if all(col in df.columns for col in ['Price', 'Qty', 'Sum']):
        df['calc_sum'] = df['Price'] * df['Qty']
        df['sum_error'] = np.abs(df['calc_sum'] - df['Sum']) > 0.01
        issues['math_errors'] = df['sum_error'].sum()
    
    # Проверка отрицательных значений
    numeric_cols = ['Price', 'Qty', 'Sum']
    for col in numeric_cols:
        if col in df.columns:
            df[f'{col.lower()}_negative'] = df[col] < 0
            issues[f'{col}_negative'] = df[f'{col.lower()}_negative'].sum()
    
    # Проверка нулевых значений
    for col in numeric_cols:
        if col in df.columns:
            df[f'{col.lower()}_zero'] = df[col] == 0
            issues[f'{col}_zero'] = df[f'{col.lower()}_zero'].sum()
    
    return issues

def check_text_quality(df):
    """Проверка текстовых полей"""
    text_columns = ['Magazin', 'Adress', 'City', 'Describe', 'Model', 'Segment', 'Art']
    issues = {}
    
    for col in text_columns:
        if col in df.columns:
            # Пустые значения
            df[f'{col.lower()}_empty'] = df[col].isnull() | (df[col].str.strip() == '')
            issues[f'{col}_empty'] = df[f'{col.lower()}_empty'].sum()
            
            # Подозрительные символы
            df[f'{col.lower()}_suspicious'] = df[col].str.contains(r'[<>{}[\]\\|`~@#$%^&*]', na=False)
            issues[f'{col}_suspicious'] = df[f'{col.lower()}_suspicious'].sum()
            
            # Слишком короткие/длинные
            df[f'{col.lower()}_short'] = df[col].str.len() < 2
            df[f'{col.lower()}_long'] = df[col].str.len() > 100
            issues[f'{col}_short'] = df[f'{col.lower()}_short'].sum()
            issues[f'{col}_long'] = df[f'{col.lower()}_long'].sum()
            
            # Только цифры (подозрительно для текста)
            df[f'{col.lower()}_digits_only'] = df[col].str.match(r'^\d+$', na=False)
            issues[f'{col}_digits_only'] = df[f'{col.lower()}_digits_only'].sum()
            
            # Повторяющиеся символы
            df[f'{col.lower()}_repeated'] = df[col].str.contains(r'(.)\1{3,}', na=False)
            issues[f'{col}_repeated'] = df[f'{col.lower()}_repeated'].sum()
    
    return issues

def check_date_quality(df):
    """Проверка дат"""
    issues = {}
    if 'Datasales' in df.columns:
        today = datetime.now()
        
        # Недействительные даты
        df['date_invalid'] = df['Datasales'].isnull()
        issues['date_invalid'] = df['date_invalid'].sum()
        
        # Будущие даты
        df['date_future'] = df['Datasales'] > today
        issues['date_future'] = df['date_future'].sum()
        
        # Слишком старые даты (>20 лет)
        old_threshold = today - timedelta(days=365*20)
        df['date_too_old'] = df['Datasales'] < old_threshold
        issues['date_too_old'] = df['date_too_old'].sum()
        
        # Выходные дни (подозрительно для продаж)
        df['date_weekend'] = df['Datasales'].dt.weekday >= 5
        issues['date_weekend'] = df['date_weekend'].sum()
    
    return issues

def check_business_logic(df):
    """Проверка бизнес-логики"""
    issues = {}
    
    # Дубликаты по ключевым полям
    key_fields = ['Magazin', 'Datasales', 'Art', 'Model']
    available_fields = [f for f in key_fields if f in df.columns]
    if len(available_fields) >= 2:
        df['business_duplicate'] = df.duplicated(subset=available_fields, keep=False)
        issues['business_duplicates'] = df['business_duplicate'].sum()
    
    # Несоответствие цены и сегмента
    if all(col in df.columns for col in ['Price', 'Segment']):
        segment_prices = df.groupby('Segment')['Price'].agg(['mean', 'std']).fillna(0)
        df['price_segment_outlier'] = False
        
        for segment in segment_prices.index:
            mask = df['Segment'] == segment
            mean_price = segment_prices.loc[segment, 'mean']
            std_price = segment_prices.loc[segment, 'std']
            
            if std_price > 0:
                z_score = np.abs((df.loc[mask, 'Price'] - mean_price) / std_price)
                df.loc[mask, 'price_segment_outlier'] = z_score > 3
        
        issues['price_segment_outliers'] = df['price_segment_outlier'].sum()
    
    return issues

# Загрузка файла
uploaded_file = st.file_uploader("Загрузите файл данных", type=["xlsx", "xls", "csv"])

if uploaded_file:
    try:
        # Загрузка данных
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        else:
            df = pd.read_excel(uploaded_file)
        
        # Обработка дат
        if 'Datasales' in df.columns:
            df['Datasales'] = pd.to_datetime(df['Datasales'], errors='coerce')
        
        st.success("✅ Файл загружен успешно!")
        
        # === РАСШИРЕННАЯ АНАЛИТИКА ===
        st.header("📊 Расширенная аналитика датасета")
        
        quality_report = create_quality_report(df)
        
        # Основные метрики
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Всего строк", quality_report['total_rows'])
        with col2:
            st.metric("📋 Колонок", quality_report['total_columns'])
        with col3:
            st.metric("💾 Размер (MB)", f"{quality_report['memory_usage']:.2f}")
        with col4:
            st.metric("🔄 Дубликаты", f"{quality_report['duplicate_rows']} ({quality_report['duplicate_percentage']:.1f}%)")
        
        # Детальный анализ колонок
        st.subheader("🔍 Анализ по колонкам")
        
        column_stats = []
        for col in df.columns:
            stats = {
                'Колонка': col,
                'Тип': str(df[col].dtype),
                'Пропуски': df[col].isnull().sum(),
                'Пропуски %': f"{df[col].isnull().sum() / len(df) * 100:.1f}%",
                'Уникальных': df[col].nunique(),
                'Дубликатов': df[col].duplicated().sum()
            }
            
            if df[col].dtype in ['int64', 'float64']:
                stats.update({
                    'Мин': df[col].min(),
                    'Макс': df[col].max(),
                    'Среднее': f"{df[col].mean():.2f}",
                    'Медиана': f"{df[col].median():.2f}",
                    'Стд.откл': f"{df[col].std():.2f}"
                })
            elif df[col].dtype == 'object':
                lengths = df[col].str.len()
                stats.update({
                    'Мин длина': lengths.min() if not lengths.empty else 0,
                    'Макс длина': lengths.max() if not lengths.empty else 0,
                    'Сред длина': f"{lengths.mean():.1f}" if not lengths.empty else "0",
                    'Пустые': (df[col] == '').sum()
                })
            
            column_stats.append(stats)
        
        stats_df = pd.DataFrame(column_stats)
        st.dataframe(stats_df, use_container_width=True)
        
        # === ПРОВЕРКИ КАЧЕСТВА ===
        st.header("🔍 Проверки качества данных")
        
        # Проверка обязательных колонок
        required_cols = ['Magazin', 'Datasales', 'Describe', 'Model', 'Segment', 'Price', 'Qty', 'Sum']
        missing_required = [col for col in required_cols if col not in df.columns]
        
        if missing_required:
            st.error(f"❌ Отсутствуют обязательные колонки: {', '.join(missing_required)}")
        else:
            st.success("✅ Все обязательные колонки присутствуют")
        
        # Выполнение всех проверок
        consistency_issues = check_data_consistency(df)
        text_issues = check_text_quality(df)
        date_issues = check_date_quality(df)
        business_issues = check_business_logic(df)
        
        # Математические проверки
        st.subheader("🧮 Математические проверки")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("❌ Ошибки сумм", consistency_issues.get('math_errors', 0))
            st.metric("➖ Отрицательные цены", consistency_issues.get('Price_negative', 0))
        with col2:
            st.metric("➖ Отрицательные кол-ва", consistency_issues.get('Qty_negative', 0))
            st.metric("➖ Отрицательные суммы", consistency_issues.get('Sum_negative', 0))
        with col3:
            st.metric("0️⃣ Нулевые цены", consistency_issues.get('Price_zero', 0))
            st.metric("0️⃣ Нулевые кол-ва", consistency_issues.get('Qty_zero', 0))
        
        # Проверки дат
        if date_issues:
            st.subheader("📅 Проверка дат")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("❌ Недействительные", date_issues.get('date_invalid', 0))
            with col2:
                st.metric("🔮 Будущие даты", date_issues.get('date_future', 0))
            with col3:
                st.metric("📜 Слишком старые", date_issues.get('date_too_old', 0))
            with col4:
                st.metric("📅 Выходные дни", date_issues.get('date_weekend', 0))
        
        # Проверки текста
        st.subheader("📝 Проверка текстовых полей")
        text_cols = ['Magazin', 'Adress', 'City', 'Describe', 'Model', 'Segment']
        
        for col in text_cols:
            if col in df.columns:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(f"🗑️ {col}: пустые", text_issues.get(f'{col}_empty', 0))
                with col2:
                    st.metric(f"⚠️ {col}: подозрительные", text_issues.get(f'{col}_suspicious', 0))
                with col3:
                    st.metric(f"📏 {col}: короткие", text_issues.get(f'{col}_short', 0))
                with col4:
                    st.metric(f"🔢 {col}: только цифры", text_issues.get(f'{col}_digits_only', 0))
        
        # Бизнес-логика
        if business_issues:
            st.subheader("💼 Бизнес-логика")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🔄 Бизнес-дубликаты", business_issues.get('business_duplicates', 0))
            with col2:
                st.metric("💰 Аномалии цена-сегмент", business_issues.get('price_segment_outliers', 0))
        
        # Подсчет всех проблем
        problem_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                       ['error', 'negative', 'zero', 'future', 'old', 'invalid', 'suspicious', 
                        'short', 'long', 'digits_only', 'repeated', 'weekend', 'duplicate', 'outlier', 'empty'])]
        
        if problem_cols:
            df['total_problems'] = df[problem_cols].sum(axis=1)
            
            st.subheader("📋 Сводка проблем")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Строк с проблемами", (df['total_problems'] > 0).sum())
            with col2:
                st.metric("✅ Чистых строк", (df['total_problems'] == 0).sum())
            with col3:
                quality_pct = (df['total_problems'] == 0).sum() / len(df) * 100
                st.metric("📈 Качество данных", f"{quality_pct:.1f}%")
            
            # График проблем
            problem_counts = df[problem_cols].sum().sort_values(ascending=False).head(15)
            if not problem_counts.empty:
                fig = px.bar(
                    x=problem_counts.values,
                    y=problem_counts.index,
                    orientation='h',
                    title="Топ-15 типов проблем в данных"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # === AgGrid ===
        st.header("📋 Интерактивное редактирование")
        
        # Настройка AgGrid
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(editable=True, filter=True, sortable=True, resizable=True)
        
        # Стили для раскраски проблемных строк
        cell_style = JsCode("""
        function(params) {
            const data = params.data;
            if (data.sum_error || data.price_negative || data.qty_negative || data.sum_negative) {
                return { backgroundColor: '#ffcccc' };
            }
            if (data.date_invalid || data.date_future) {
                return { backgroundColor: '#ffe6cc' };
            }
            if (data.magazin_empty || data.describe_empty || data.model_empty) {
                return { backgroundColor: '#fff2cc' };
            }
            if (data.magazin_suspicious || data.describe_suspicious) {
                return { backgroundColor: '#cce6ff' };
            }
            return null;
        }
        """)
        
        # Применение стилей к основным колонкам
        main_cols = ['Magazin', 'Adress', 'City', 'Datasales', 'Art', 'Describe', 'Model', 'Segment', 'Price', 'Qty', 'Sum']
        for col in main_cols:
            if col in df.columns:
                gb.configure_column(col, cellStyle=cell_style)
        
        # Скрытие служебных колонок
        service_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                       ['error', 'negative', 'zero', 'invalid', 'suspicious', 'empty', 'short', 'long', 'calc'])]
        for col in service_cols:
            gb.configure_column(col, hide=True)
        
        gb.configure_grid_options(domLayout='normal', pagination=True, paginationPageSize=50)
        gb.configure_selection(selection_mode="multiple", use_checkbox=True)
        
        # Легенда
        st.markdown("""
        **Цветовая схема:**
        - 🔴 Красный: математические ошибки
        - 🟠 Оранжевый: проблемы с датами  
        - 🟡 Желтый: пустые обязательные поля
        - 🔵 Голубой: подозрительные символы
        """)
        
        grid_response = AgGrid(
            df,
            gridOptions=gb.build(),
            height=500,
            update_mode=GridUpdateMode.VALUE_CHANGED,
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            theme="alpine"
        )
        
        updated_df = pd.DataFrame(grid_response['data'])
        
        # === ЭКСПОРТ ===
        st.header("📤 Экспорт данных")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Все данные
            csv_all = updated_df.to_csv(index=False)
            st.download_button("⬇️ Все данные", csv_all, "all_data.csv", "text/csv")
        
        with col2:
            # Чистые данные
            if 'total_problems' in updated_df.columns:
                clean_data = updated_df[updated_df['total_problems'] == 0]
                clean_cols = [col for col in main_cols if col in clean_data.columns]
                csv_clean = clean_data[clean_cols].to_csv(index=False)
                st.download_button("✅ Чистые данные", csv_clean, "clean_data.csv", "text/csv")
        
        with col3:
            # Проблемные данные
            if 'total_problems' in updated_df.columns:
                problem_data = updated_df[updated_df['total_problems'] > 0]
                if not problem_data.empty:
                    csv_problems = problem_data.to_csv(index=False)
                    st.download_button("⚠️ Проблемные данные", csv_problems, "problems.csv", "text/csv")
        
        # Финальная статистика
        st.subheader("📊 Итоговая статистика")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Обработано", len(updated_df))
        with col2:
            if 'total_problems' in updated_df.columns:
                clean_count = (updated_df['total_problems'] == 0).sum()
                st.metric("✅ Чистых", clean_count)
        with col3:
            if 'total_problems' in updated_df.columns:
                problem_count = (updated_df['total_problems'] > 0).sum()
                st.metric("⚠️ С проблемами", problem_count)
        with col4:
            if 'total_problems' in updated_df.columns:
                quality = (updated_df['total_problems'] == 0).sum() / len(updated_df) * 100
                st.metric("🎯 Качество", f"{quality:.1f}%")
        
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")

else:
    st.info("👆 Загрузите файл для анализа")
    st.markdown("""
    **Ожидаемые колонки:**
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
    """)
