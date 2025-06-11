import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode

st.set_page_config(page_title="📊 Контроль качества с AgGrid", layout="wide")
st.title("🧠 ML + AgGrid: Контроль качества данных")

# === Загрузка файла ===
uploaded_file = st.file_uploader("Загрузите CSV-файл", type=["xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["Datasales"])
    st.success("Файл загружен!")

    # --- Обработка: логические проверки ---
    df['calc_sum'] = df['Price'] * df['Qty']
    df['sum_diff'] = np.abs(df['calc_sum'] - df['Sum'])
    df['error_sum'] = df['sum_diff'] > 1e-2

    # --- Аномалии ---
    try:
        X_scaled = StandardScaler().fit_transform(df[['Price', 'Qty', 'Sum']])
        iso = IsolationForest(contamination=0.01, random_state=42)
        df['anomaly'] = iso.fit_predict(X_scaled) == -1
    except Exception as e:
        st.error(f"Ошибка аномального анализа: {e}")
        df['anomaly'] = False

    # --- Проверка текстовых колонок ---
    st.subheader("📌 Проверка колонок")
    for col in ['Magazin', 'Datasales', 'Describe', 'Model', 'Segment']:
        st.markdown(f"**{col}**: пропущено — {df[col].isnull().sum()}, уникальных — {df[col].nunique()}")

    # --- AgGrid: интерактивное редактирование ---
    st.subheader("📋 Редактирование данных (AgGrid)")

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(editable=True, filter=True, sortable=True, resizable=True)

    # Условная раскраска для ошибок
    cell_style_jscode = JsCode("""
    function(params) {
        if (params.data.error_sum) {
            return { backgroundColor: '#ffe6e6' };
        }
        if (params.data.anomaly) {
            return { backgroundColor: '#ffffcc' };
        }
        return null;
    }
    """)

    for col in ['Price', 'Qty', 'Sum', 'calc_sum', 'sum_diff']:
        gb.configure_column(col, cellStyle=cell_style_jscode)

    gb.configure_column("error_sum", headerName="Ошибка суммы", type=["booleanColumn"], editable=False)
    gb.configure_column("anomaly", headerName="Аномалия", type=["booleanColumn"], editable=False)
    gb.configure_grid_options(domLayout='normal', pagination=True)

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

    # --- Кнопка сохранения ---
    st.subheader("📤 Сохранение")
    st.download_button("⬇️ Скачать обновлённый датасет", new_df.to_csv(index=False), file_name="cleaned_data_aggrid.csv")

else:
    st.warning("Пожалуйста, загрузите CSV-файл с колонками: Magazin, Datasales, Art, Describe, Model, Segment, Price, Qty, Sum")
