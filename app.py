import streamlit as st
import pandas as pd
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode

st.set_page_config(page_title="📊 Контроль качества с AgGrid", layout="wide")
st.title("📊 AgGrid: Контроль качества данных")

uploaded_file = st.file_uploader("Загрузите Excel-файл", type=["xlsx", "xls"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        if 'Datasales' in df.columns:
            df['Datasales'] = pd.to_datetime(df['Datasales'], errors='coerce')
        st.success(f"Файл успешно загружен! Строк: {len(df)}")
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        st.stop()

    # Базовые проверки
    df['calc_sum'] = df['Price'] * df['Qty']
    df['sum_diff'] = np.abs(df['calc_sum'] - df['Sum'])
    df['error_sum'] = df['sum_diff'] > 0.01
    
    # Дополнительные проверки
    df['negative_price'] = df['Price'] < 0
    df['negative_qty'] = df['Qty'] < 0
    df['zero_price'] = df['Price'] == 0
    df['zero_qty'] = df['Qty'] == 0
    df['high_price'] = df['Price'] > df['Price'].quantile(0.99)
    df['low_price'] = df['Price'] < df['Price'].quantile(0.01)
    df['empty_magazin'] = df['Magazin'].isnull() | (df['Magazin'].astype(str).str.strip() == '')
    df['empty_city'] = df['City'].isnull() | (df['City'].astype(str).str.strip() == '') if 'City' in df.columns else False
    df['empty_art'] = df['Art'].isnull() | (df['Art'].astype(str).str.strip() == '') if 'Art' in df.columns else False
    df['empty_describe'] = df['Describe'].isnull() | (df['Describe'].astype(str).str.strip() == '')
    df['empty_model'] = df['Model'].isnull() | (df['Model'].astype(str).str.strip() == '')
    df['empty_segment'] = df['Segment'].isnull() | (df['Segment'].astype(str).str.strip() == '')
    df['invalid_date'] = df['Datasales'].isnull() if 'Datasales' in df.columns else False
    
    # Проверка дубликатов
    duplicate_cols = ['Magazin', 'Art', 'Datasales'] if all(col in df.columns for col in ['Magazin', 'Art', 'Datasales']) else []
    if duplicate_cols:
        df['is_duplicate'] = df.duplicated(subset=duplicate_cols, keep=False)
    else:
        df['is_duplicate'] = False

    st.subheader("📋 Редактирование данных (AgGrid)")
    
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(editable=True, filter=True, sortable=True, resizable=True)
    
    # Подкраска пустых ячеек желтым и других ошибок
    cell_style_jscode = JsCode("""
    function(params) {
        var value = params.value;
        var colId = params.colDef.field;
        
        // Подкраска пустых ячеек желтым
        if (value === null || value === undefined || value === '' || 
            (typeof value === 'string' && value.trim() === '')) {
            return { backgroundColor: '#ffff99' };
        }
        
        // Ошибки суммы - красный
        if (params.data.error_sum && ['Price', 'Qty', 'Sum', 'calc_sum'].includes(colId)) {
            return { backgroundColor: '#ffe6e6' };
        }
        
        // Отрицательные значения - оранжевый
        if ((params.data.negative_price && colId === 'Price') || 
            (params.data.negative_qty && colId === 'Qty')) {
            return { backgroundColor: '#ffcc99' };
        }
        
        // Дубликаты - фиолетовый
        if (params.data.is_duplicate) {
            return { backgroundColor: '#e6ccff' };
        }
        
        return null;
    }
    """)
    
    # Применение стилей ко всем колонкам
    for col in df.columns:
        if col not in ['error_sum', 'negative_price', 'negative_qty', 'zero_price', 'zero_qty', 
                      'high_price', 'low_price', 'empty_magazin', 'empty_city', 'empty_art',
                      'empty_describe', 'empty_model', 'empty_segment', 'invalid_date', 'is_duplicate']:
            gb.configure_column(col, cellStyle=cell_style_jscode)
    
    # Конфигурация служебных колонок
    gb.configure_column("error_sum", headerName="Ошибка суммы", type=["booleanColumn"], editable=False, hide=True)
    gb.configure_column("negative_price", headerName="Цена<0", type=["booleanColumn"], editable=False, hide=True)
    gb.configure_column("negative_qty", headerName="Кол-во<0", type=["booleanColumn"], editable=False, hide=True)
    gb.configure_column("is_duplicate", headerName="Дубликат", type=["booleanColumn"], editable=False, hide=True)
    
    gb.configure_grid_options(domLayout='normal', pagination=True, paginationPageSize=50)
    grid_options = gb.build()
    
    grid_response = AgGrid(
        df,
        gridOptions=grid_options,
        height=600,
        width='100%',
        update_mode=GridUpdateMode.VALUE_CHANGED,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        theme="alpine"
    )
    
    new_df = pd.DataFrame(grid_response['data'])

    # === РАСШИРЕННАЯ АНАЛИТИКА ДАТАСЕТА ===
    st.subheader("📊 Расширенная аналитика датасета")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Всего строк", len(df))
        st.metric("Строк с ошибками", df['error_sum'].sum())
        st.metric("Дубликатов", df['is_duplicate'].sum())
        
    with col2:
        empty_rows = df.isnull().any(axis=1).sum()
        st.metric("Строк с пустыми значениями", empty_rows)
        st.metric("Отрицательные цены", df['negative_price'].sum())
        st.metric("Отрицательные количества", df['negative_qty'].sum())
        
    with col3:
        total_cells = len(df) * len(df.columns)
        empty_cells = df.isnull().sum().sum()
        st.metric("Заполненность (%)", f"{((total_cells-empty_cells)/total_cells*100):.1f}")
        st.metric("Уникальных магазинов", df['Magazin'].nunique())
        if 'City' in df.columns:
            st.metric("Уникальных городов", df['City'].nunique())

    # Детальная статистика по колонкам
    st.subheader("📋 Детальная статистика по колонкам")
    
    stats_data = []
    for col in df.columns:
        if col not in ['calc_sum', 'sum_diff', 'error_sum', 'negative_price', 'negative_qty', 
                      'zero_price', 'zero_qty', 'high_price', 'low_price', 'empty_magazin',
                      'empty_city', 'empty_art', 'empty_describe', 'empty_model', 'empty_segment',
                      'invalid_date', 'is_duplicate']:
            stats_data.append({
                'Колонка': col,
                'Тип': str(df[col].dtype),
                'Пустых': df[col].isnull().sum(),
                'Заполнено (%)': f"{((len(df)-df[col].isnull().sum())/len(df)*100):.1f}",
                'Уникальных': df[col].nunique(),
                'Пример значения': str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else 'Нет данных'
            })
    
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True)

    # Проблемы в данных
    st.subheader("⚠️ Обнаруженные проблемы")
    
    problems = []
    if df['error_sum'].any():
        problems.append(f"❌ Несоответствие сумм: {df['error_sum'].sum()} строк")
    if df['negative_price'].any():
        problems.append(f"❌ Отрицательные цены: {df['negative_price'].sum()} строк")
    if df['negative_qty'].any():
        problems.append(f"❌ Отрицательные количества: {df['negative_qty'].sum()} строк")
    if df['is_duplicate'].any():
        problems.append(f"❌ Дубликаты: {df['is_duplicate'].sum()} строк")
    if df['zero_price'].any():
        problems.append(f"⚠️ Нулевые цены: {df['zero_price'].sum()} строк")
    if df['zero_qty'].any():
        problems.append(f"⚠️ Нулевые количества: {df['zero_qty'].sum()} строк")
    if df['empty_magazin'].any():
        problems.append(f"⚠️ Пустые названия магазинов: {df['empty_magazin'].sum()} строк")
    if 'Datasales' in df.columns and df['invalid_date'].any():
        problems.append(f"⚠️ Некорректные даты: {df['invalid_date'].sum()} строк")
    
    if problems:
        for problem in problems:
            st.markdown(problem)
    else:
        st.success("✅ Критических проблем не обнаружено!")

    # Числовая статистика
    if any(col in df.columns for col in ['Price', 'Qty', 'Sum']):
        st.subheader("📈 Числовая статистика")
        numeric_cols = [col for col in ['Price', 'Qty', 'Sum'] if col in df.columns]
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)

    # Топ значения
    st.subheader("🔝 Топ значения")
    
    top_col1, top_col2 = st.columns(2)
    
    with top_col1:
        if 'Magazin' in df.columns:
            st.markdown("**Топ-5 магазинов по количеству записей:**")
            top_magazin = df['Magazin'].value_counts().head().to_frame()
            top_magazin.columns = ['Количество']
            st.dataframe(top_magazin)
            
    with top_col2:
        if 'Segment' in df.columns:
            st.markdown("**Топ-5 сегментов:**")
            top_segment = df['Segment'].value_counts().head().to_frame()
            top_segment.columns = ['Количество']
            st.dataframe(top_segment)

    # Сохранение
    st.subheader("💾 Сохранение данных")
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Скачать очищенные данные (CSV)",
            new_df.drop(columns=[col for col in new_df.columns if col.startswith(('calc_', 'sum_', 'error_', 'negative_', 'zero_', 'high_', 'low_', 'empty_', 'invalid_', 'is_'))]).to_csv(index=False),
            file_name="cleaned_data.csv",
            mime="text/csv"
        )
    
    with col2:
        # Отчет о качестве
        quality_report = f"""ОТЧЕТ О КАЧЕСТВЕ ДАННЫХ
=========================
Файл: {uploaded_file.name}
Дата анализа: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

ОБЩАЯ ИНФОРМАЦИЯ:
- Всего строк: {len(df)}
- Всего колонок: {len([col for col in df.columns if not col.startswith(('calc_', 'sum_', 'error_', 'negative_', 'zero_', 'high_', 'low_', 'empty_', 'invalid_', 'is_'))])}
- Заполненность: {((total_cells-empty_cells)/total_cells*100):.1f}%

ПРОБЛЕМЫ:
{chr(10).join(problems) if problems else '✅ Проблем не найдено'}

РЕКОМЕНДАЦИИ:
- Заполнить пустые ячейки (выделены желтым)
- Проверить строки с ошибками сумм (красные)
- Устранить дубликаты (фиолетовые)
- Проверить отрицательные значения (оранжевые)
"""
        
        st.download_button(
            "📄 Скачать отчет о качестве",
            quality_report,
            file_name="quality_report.txt",
            mime="text/plain"
        )

else:
    st.info("📁 Загрузите Excel-файл для начала анализа")
    st.markdown("""
    **Ожидаемые колонки:**
    - Magazin, Adress, City, Datasales, Art, Describe, Model, Segment, Price, Qty, Sum
    
    **Возможности системы:**
    - 🟡 Желтый - пустые ячейки
    - 🔴 Красный - ошибки в суммах  
    - 🟠 Оранжевый - отрицательные значения
    - 🟣 Фиолетовый - дубликаты
    """)
