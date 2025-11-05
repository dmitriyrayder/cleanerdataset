import dash
from dash import dcc, html, Input, Output, State, dash_table, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import requests
from io import BytesIO
import base64
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ========================
# ИНИЦИАЛИЗАЦИЯ ПРИЛОЖЕНИЯ
# ========================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "DDMRP Система управления остатками"

# ========================
# ФУНКЦИИ ЗАГРУЗКИ ДАННЫХ
# ========================

def download_google_sheet(sheet_url):
    """Загрузка торговой матрицы из Google Sheets"""
    try:
        if '/edit' in sheet_url:
            csv_url = sheet_url.replace('/edit?gid=', '/export?format=csv&gid=')
            csv_url = csv_url.split('#')[0]
        else:
            csv_url = sheet_url
        
        response = requests.get(csv_url)
        if response.status_code == 200:
            df = pd.read_csv(BytesIO(response.content))
            return df, None
        else:
            return None, f"Ошибка загрузки: {response.status_code}"
    except Exception as e:
        return None, f"Ошибка при загрузке Google Sheets: {str(e)}"


def load_stock_file(contents, filename):
    """Загрузка файла остатков Excel"""
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_excel(BytesIO(decoded))
        
        # Маппинг колонок
        column_mapping = {
            'Art': 'Article',
            'Magazin': 'Store_ID',
            'Describe': 'Describe',
            'к-во': 'Current_Stock',
            'Model': 'Model'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Проверка обязательных колонок
        required_cols = ['Article', 'Store_ID', 'Describe', 'Current_Stock']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            return None, f"Отсутствуют колонки: {missing_cols}"
        
        # Очистка данных
        df['Current_Stock'] = pd.to_numeric(df['Current_Stock'], errors='coerce').fillna(0)
        df['Store_ID'] = df['Store_ID'].astype(str).str.strip()
        df['Article'] = df['Article'].astype(str).str.strip()
        
        return df, None
    
    except Exception as e:
        return None, f"Ошибка при загрузке Excel: {str(e)}"


def validate_matrix(df):
    """Валидация торговой матрицы"""
    required_cols = ['Article', 'Describe', 'Store_ID', 'Red_Zone', 'Yellow_Zone', 'Green_Zone']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return False, f"В торговой матрице отсутствуют колонки: {missing_cols}"
    
    # Проверка типов данных
    df['Red_Zone'] = pd.to_numeric(df['Red_Zone'], errors='coerce')
    df['Yellow_Zone'] = pd.to_numeric(df['Yellow_Zone'], errors='coerce')
    df['Green_Zone'] = pd.to_numeric(df['Green_Zone'], errors='coerce')
    df['Store_ID'] = df['Store_ID'].astype(str).str.strip()
    df['Article'] = df['Article'].astype(str).str.strip()
    
    return True, None


# ========================
# DDMRP ЛОГИКА
# ========================

def calculate_ddmrp_status(matrix_df, stock_df):
    """Расчет статуса буферов DDMRP"""
    merged = matrix_df.merge(
        stock_df[['Article', 'Store_ID', 'Current_Stock', 'Model']],
        on=['Article', 'Store_ID'],
        how='left'
    )
    
    merged['Current_Stock'] = merged['Current_Stock'].fillna(0)
    
    if 'Retail_Price' in merged.columns:
        merged['Retail_Price'] = pd.to_numeric(merged['Retail_Price'], errors='coerce').fillna(0)
        merged['Stock_Value'] = merged['Retail_Price'] * merged['Current_Stock']
    else:
        merged['Stock_Value'] = 0
    
    merged['Top_of_Green'] = merged['Red_Zone'] + merged['Yellow_Zone'] + merged['Green_Zone']
    merged['Red_Zone_Max'] = merged['Red_Zone']
    merged['Yellow_Zone_Max'] = merged['Red_Zone'] + merged['Yellow_Zone']
    merged['Green_Zone_Max'] = merged['Top_of_Green']
    
    def get_buffer_status(row):
        stock = row['Current_Stock']
        if stock <= row['Red_Zone_Max']:
            return 'RED'
        elif stock <= row['Yellow_Zone_Max']:
            return 'YELLOW'
        elif stock <= row['Green_Zone_Max']:
            return 'GREEN'
        else:
            return 'EXCESS'
    
    merged['Buffer_Status'] = merged.apply(get_buffer_status, axis=1)
    merged['Buffer_Fill_Percent'] = (merged['Current_Stock'] / merged['Top_of_Green'] * 100).round(1)
    
    def calculate_order_qty(row):
        if row['Buffer_Status'] in ['RED', 'YELLOW']:
            order_qty = row['Top_of_Green'] - row['Current_Stock']
            return max(0, order_qty)
        return 0
    
    merged['Order_Qty'] = merged.apply(calculate_order_qty, axis=1)
    
    priority_map = {'RED': 1, 'YELLOW': 2, 'GREEN': 3, 'EXCESS': 4}
    merged['Priority'] = merged['Buffer_Status'].map(priority_map)
    
    if 'Avg_Daily_Usage' in merged.columns:
        merged['Avg_Daily_Usage'] = pd.to_numeric(merged['Avg_Daily_Usage'], errors='coerce').fillna(0)
        merged['Days_Until_Stockout'] = np.where(
            merged['Avg_Daily_Usage'] > 0,
            (merged['Current_Stock'] / merged['Avg_Daily_Usage']).round(1),
            np.inf
        )
    else:
        merged['Days_Until_Stockout'] = np.nan
    
    return merged


def generate_order_report(ddmrp_df):
    """Генерация отчета по заказам"""
    orders = ddmrp_df[ddmrp_df['Order_Qty'] > 0].copy()
    
    if orders.empty:
        return pd.DataFrame()
    
    orders = orders.sort_values(['Priority', 'Store_ID', 'Article'])
    
    report_columns = [
        'Store_ID', 'Article', 'Describe', 'Brand', 'Model',
        'Current_Stock', 'Stock_Value', 'Top_of_Green', 'Order_Qty', 
        'Buffer_Status', 'Priority', 'Days_Until_Stockout'
    ]
    
    available_columns = [col for col in report_columns if col in orders.columns]
    
    return orders[available_columns].reset_index(drop=True)


# ========================
# ВИЗУАЛИЗАЦИЯ
# ========================

def create_buffer_status_chart(ddmrp_df):
    """График распределения статусов буферов"""
    if ddmrp_df is None or ddmrp_df.empty:
        return go.Figure()
    
    status_counts = ddmrp_df['Buffer_Status'].value_counts()
    
    colors = {
        'RED': '#FF4444',
        'YELLOW': '#FFD700',
        'GREEN': '#44FF44',
        'EXCESS': '#4444FF'
    }
    
    fig = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title='Распределение статусов буферов',
        color=status_counts.index,
        color_discrete_map=colors
    )
    
    return fig


def create_store_summary_chart(ddmrp_df):
    """График сводки по магазинам"""
    if ddmrp_df is None or ddmrp_df.empty:
        return go.Figure()
    
    store_summary = ddmrp_df.groupby(['Store_ID', 'Buffer_Status']).size().reset_index(name='Count')
    
    fig = px.bar(
        store_summary,
        x='Store_ID',
        y='Count',
        color='Buffer_Status',
        title='Статусы буферов по магазинам',
        color_discrete_map={
            'RED': '#FF4444',
            'YELLOW': '#FFD700',
            'GREEN': '#44FF44',
            'EXCESS': '#4444FF'
        },
        barmode='stack'
    )
    
    fig.update_layout(xaxis_title='Магазин', yaxis_title='Количество товаров')
    
    return fig


# ========================
# РАСШИРЕННАЯ АНАЛИТИКА
# ========================

def calculate_lost_sales(ddmrp_df):
    """Расчет упущенной прибыли от дефицита"""
    if ddmrp_df is None or ddmrp_df.empty:
        return pd.DataFrame()
    
    # Фильтруем RED и YELLOW позиции
    deficit_items = ddmrp_df[ddmrp_df['Buffer_Status'].isin(['RED', 'YELLOW'])].copy()
    
    if deficit_items.empty:
        return pd.DataFrame()
    
    # Расчет упущенных продаж
    if 'Avg_Daily_Usage' in deficit_items.columns and 'Retail_Price' in deficit_items.columns:
        deficit_items['Avg_Daily_Usage'] = pd.to_numeric(deficit_items['Avg_Daily_Usage'], errors='coerce').fillna(0)
        deficit_items['Retail_Price'] = pd.to_numeric(deficit_items['Retail_Price'], errors='coerce').fillna(0)
        
        # Дефицит = Red_Zone - Current_Stock (если остаток ниже красной зоны)
        deficit_items['Deficit_Qty'] = np.maximum(0, deficit_items['Red_Zone'] - deficit_items['Current_Stock'])
        
        # Упущенная прибыль = Дефицит × Цена × Дни (предполагаем 7 дней до пополнения)
        deficit_items['Lost_Sales_7days'] = deficit_items['Deficit_Qty'] * deficit_items['Retail_Price']
        
        # Упущенная прибыль в день
        deficit_items['Daily_Lost_Sales'] = deficit_items['Avg_Daily_Usage'] * deficit_items['Retail_Price']
        
        # Приоритет по упущенной прибыли
        deficit_items['Lost_Sales_Priority'] = deficit_items['Lost_Sales_7days'].rank(ascending=False, method='dense').astype(int)
    else:
        deficit_items['Lost_Sales_7days'] = 0
        deficit_items['Daily_Lost_Sales'] = 0
        deficit_items['Lost_Sales_Priority'] = 0
    
    # Сортировка по упущенной прибыли
    deficit_items = deficit_items.sort_values('Lost_Sales_7days', ascending=False)
    
    return deficit_items[['Store_ID', 'Article', 'Describe', 'Brand', 'Current_Stock', 
                          'Red_Zone', 'Buffer_Status', 'Deficit_Qty', 'Lost_Sales_7days', 
                          'Daily_Lost_Sales', 'Lost_Sales_Priority']]


def dynamic_buffer_adjustment(ddmrp_df, adjustment_factor=1.2, seasonal_factor=1.0):
    """Динамическая корректировка буферов на основе спроса"""
    if ddmrp_df is None or ddmrp_df.empty:
        return ddmrp_df
    
    adjusted_df = ddmrp_df.copy()
    
    # Если есть фактический спрос, корректируем буферы
    if 'Avg_Daily_Usage' in adjusted_df.columns:
        adjusted_df['Avg_Daily_Usage'] = pd.to_numeric(adjusted_df['Avg_Daily_Usage'], errors='coerce').fillna(0)
        
        # Корректировка на основе фактического использования
        # Если товар часто уходит в RED - увеличиваем буферы
        adjusted_df['Suggested_Red_Zone'] = np.ceil(adjusted_df['Red_Zone'] * adjustment_factor * seasonal_factor)
        adjusted_df['Suggested_Yellow_Zone'] = np.ceil(adjusted_df['Yellow_Zone'] * adjustment_factor * seasonal_factor)
        adjusted_df['Suggested_Green_Zone'] = np.ceil(adjusted_df['Green_Zone'] * adjustment_factor * seasonal_factor)
        
        adjusted_df['Adjustment_Recommended'] = adjusted_df['Buffer_Status'].isin(['RED', 'YELLOW'])
    else:
        adjusted_df['Suggested_Red_Zone'] = adjusted_df['Red_Zone']
        adjusted_df['Suggested_Yellow_Zone'] = adjusted_df['Yellow_Zone']
        adjusted_df['Suggested_Green_Zone'] = adjusted_df['Green_Zone']
        adjusted_df['Adjustment_Recommended'] = False
    
    return adjusted_df


def get_critical_alerts(ddmrp_df):
    """Генерация критических алертов"""
    if ddmrp_df is None or ddmrp_df.empty:
        return pd.DataFrame()
    
    alerts = []
    
    # 1. Критические позиции (RED)
    red_items = ddmrp_df[ddmrp_df['Buffer_Status'] == 'RED']
    for _, row in red_items.iterrows():
        alerts.append({
            'Priority': 1,
            'Type': '🔴 КРИТИЧНО',
            'Store_ID': row['Store_ID'],
            'Article': row['Article'],
            'Describe': row['Describe'],
            'Current_Stock': row['Current_Stock'],
            'Red_Zone': row['Red_Zone'],
            'Message': f"Остаток {row['Current_Stock']} ниже критического уровня {row['Red_Zone']}"
        })
    
    # 2. Товары близкие к stockout (остаток < 20% от красной зоны)
    near_stockout = ddmrp_df[
        (ddmrp_df['Current_Stock'] > 0) & 
        (ddmrp_df['Current_Stock'] < ddmrp_df['Red_Zone'] * 0.2)
    ]
    for _, row in near_stockout.iterrows():
        days_left = row.get('Days_Until_Stockout', 'N/A')
        alerts.append({
            'Priority': 2,
            'Type': '⚠️ БЛИЗОК К НУЛЮ',
            'Store_ID': row['Store_ID'],
            'Article': row['Article'],
            'Describe': row['Describe'],
            'Current_Stock': row['Current_Stock'],
            'Red_Zone': row['Red_Zone'],
            'Message': f"Осталось {row['Current_Stock']} шт (дней: {days_left})"
        })
    
    # 3. Полный stockout (остаток = 0)
    zero_stock = ddmrp_df[ddmrp_df['Current_Stock'] == 0]
    for _, row in zero_stock.iterrows():
        alerts.append({
            'Priority': 1,
            'Type': '❌ НЕТ В НАЛИЧИИ',
            'Store_ID': row['Store_ID'],
            'Article': row['Article'],
            'Describe': row['Describe'],
            'Current_Stock': 0,
            'Red_Zone': row['Red_Zone'],
            'Message': 'Товар полностью отсутствует в магазине!'
        })
    
    if not alerts:
        return pd.DataFrame()
    
    alerts_df = pd.DataFrame(alerts).sort_values('Priority')
    return alerts_df


def generate_order_file(orders_df):
    """Генерация файла заказа для поставщикам"""
    if orders_df is None or orders_df.empty:
        return None
    
    # Группировка по поставщикам (если есть такая колонка)
    if 'Brand' in orders_df.columns:
        order_summary = orders_df.groupby(['Brand', 'Article', 'Describe']).agg({
            'Order_Qty': 'sum',
            'Store_ID': lambda x: ', '.join(x.astype(str))
        }).reset_index()
        order_summary.rename(columns={'Store_ID': 'Stores'}, inplace=True)
    else:
        order_summary = orders_df[['Article', 'Describe', 'Order_Qty', 'Store_ID']].copy()
    
    return order_summary


def calculate_seasonal_factor():
    """Расчет сезонного коэффициента на основе текущего месяца"""
    current_month = datetime.now().month
    
    # Пример сезонных коэффициентов (можно настроить под бизнес)
    seasonal_factors = {
        1: 0.9,   # Январь - после праздников
        2: 0.85,  # Февраль - низкий сезон
        3: 0.95,  # Март
        4: 1.0,   # Апрель
        5: 1.05,  # Май - начало сезона
        6: 1.1,   # Июнь - высокий сезон
        7: 1.15,  # Июль - пик
        8: 1.1,   # Август
        9: 1.0,   # Сентябрь
        10: 1.05, # Октябрь
        11: 1.1,  # Ноябрь - предновогодний
        12: 1.2   # Декабрь - праздники
    }
    
    return seasonal_factors.get(current_month, 1.0)


# ========================
# LAYOUT
# ========================

app.layout = dbc.Container([
    # Хранилище данных
    dcc.Store(id='ddmrp-data'),
    dcc.Store(id='orders-data'),
    dcc.Store(id='lost-sales-data'),
    dcc.Store(id='alerts-data'),
    dcc.Store(id='adjusted-buffers-data'),
    
    # Заголовок
    dbc.Row([
        dbc.Col([
            html.H1("📊 DDMRP: Система управления остатками", className="text-center mb-2"),
            html.P("Динамическое управление буферами запасов по методологии DDMRP", 
                   className="text-center text-muted"),
            html.Hr()
        ])
    ]),
    
    # Боковая панель и основной контент
    dbc.Row([
        # Боковая панель
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📂 Загрузка данных"),
                dbc.CardBody([
                    html.Label("Google Sheets URL (торговая матрица):"),
                    dbc.Input(id='google-sheet-url', type='text', placeholder='Вставьте URL...', className="mb-3"),
                    
                    html.Label("Загрузите Excel с остатками:"),
                    dcc.Upload(
                        id='upload-stock',
                        children=dbc.Button("Выбрать файл", color="secondary", className="w-100 mb-3"),
                        multiple=False
                    ),
                    
                    html.Div(id='upload-status', className="mb-3"),
                    
                    dbc.Button("🔄 Загрузить и рассчитать", id='load-button', color="primary", className="w-100 mb-3"),
                    
                    html.Div(id='load-status'),
                    
                    html.Hr(),
                    
                    html.H6("📖 Легенда статусов"),
                    html.P("🔴 RED - Критический уровень", className="mb-1"),
                    html.P("🟡 YELLOW - Требуется заказ", className="mb-1"),
                    html.P("🟢 GREEN - Норма", className="mb-1"),
                    html.P("🔵 EXCESS - Излишек", className="mb-1"),
                ])
            ])
        ], width=3),
        
        # Основной контент
        dbc.Col([
            # Метрики
            html.Div(id='metrics-row'),
            
            html.Hr(),
            
            # Вкладки
            dbc.Tabs([
                dbc.Tab(label="🚨 Алерты", tab_id="tab-alerts"),
                dbc.Tab(label="📋 Заказы", tab_id="tab-orders"),
                dbc.Tab(label="💰 Упущенная прибыль", tab_id="tab-lost-sales"),
                dbc.Tab(label="🔧 Корректировка буферов", tab_id="tab-buffer-adjust"),
                dbc.Tab(label="📊 Все товары", tab_id="tab-all"),
                dbc.Tab(label="🏪 По магазинам", tab_id="tab-stores"),
                dbc.Tab(label="📈 Аналитика", tab_id="tab-analytics"),
                dbc.Tab(label="⚙️ Детали", tab_id="tab-details"),
            ], id="tabs", active_tab="tab-alerts"),
            
            html.Div(id='tab-content', className="mt-3")
        ], width=9)
    ])
], fluid=True)


# ========================
# CALLBACKS
# ========================

@callback(
    Output('upload-status', 'children'),
    Input('upload-stock', 'filename')
)
def update_upload_status(filename):
    if filename:
        return dbc.Alert(f"Файл выбран: {filename}", color="info", dismissable=True)
    return ""


@callback(
    [Output('ddmrp-data', 'data'),
     Output('orders-data', 'data'),
     Output('lost-sales-data', 'data'),
     Output('alerts-data', 'data'),
     Output('adjusted-buffers-data', 'data'),
     Output('load-status', 'children')],
    Input('load-button', 'n_clicks'),
    [State('google-sheet-url', 'value'),
     State('upload-stock', 'contents'),
     State('upload-stock', 'filename')],
    prevent_initial_call=True
)
def load_and_calculate(n_clicks, sheet_url, contents, filename):
    if not sheet_url:
        return None, None, None, None, None, dbc.Alert("❌ Укажите URL Google Sheets", color="danger")
    
    if not contents:
        return None, None, None, None, None, dbc.Alert("❌ Загрузите Excel файл с остатками", color="danger")
    
    # Загрузка торговой матрицы
    matrix_df, error = download_google_sheet(sheet_url)
    if error:
        return None, None, None, None, None, dbc.Alert(f"❌ {error}", color="danger")
    
    # Валидация
    valid, error = validate_matrix(matrix_df)
    if not valid:
        return None, None, None, None, None, dbc.Alert(f"❌ {error}", color="danger")
    
    # Загрузка остатков
    stock_df, error = load_stock_file(contents, filename)
    if error:
        return None, None, None, None, None, dbc.Alert(f"❌ {error}", color="danger")
    
    # Расчет DDMRP
    ddmrp_df = calculate_ddmrp_status(matrix_df, stock_df)
    orders_df = generate_order_report(ddmrp_df)
    
    # Расчет упущенной прибыли
    lost_sales_df = calculate_lost_sales(ddmrp_df)
    
    # Генерация алертов
    alerts_df = get_critical_alerts(ddmrp_df)
    
    # Динамическая корректировка буферов
    seasonal_factor = calculate_seasonal_factor()
    adjusted_buffers_df = dynamic_buffer_adjustment(ddmrp_df, adjustment_factor=1.2, seasonal_factor=seasonal_factor)
    
    return (ddmrp_df.to_dict('records'), 
            orders_df.to_dict('records'),
            lost_sales_df.to_dict('records'),
            alerts_df.to_dict('records'),
            adjusted_buffers_df.to_dict('records'),
            dbc.Alert("✅ Расчеты выполнены успешно!", color="success"))


@callback(
    Output('metrics-row', 'children'),
    Input('ddmrp-data', 'data')
)
def update_metrics(ddmrp_data):
    if not ddmrp_data:
        return dbc.Alert("👆 Загрузите данные для начала работы", color="info")
    
    df = pd.DataFrame(ddmrp_data)
    
    total_items = len(df)
    red_count = len(df[df['Buffer_Status'] == 'RED'])
    yellow_count = len(df[df['Buffer_Status'] == 'YELLOW'])
    green_count = len(df[df['Buffer_Status'] == 'GREEN'])
    total_order_qty = df['Order_Qty'].sum()
    total_stock_value = df['Stock_Value'].sum() if 'Stock_Value' in df.columns else 0
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("📦 Всего позиций"),
                    html.H3(f"{total_items}")
                ])
            ])
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("🔴 Критичных"),
                    html.H3(f"{red_count}")
                ])
            ])
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("🟡 Требуют заказа"),
                    html.H3(f"{yellow_count}")
                ])
            ])
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("🟢 В норме"),
                    html.H3(f"{green_count}")
                ])
            ])
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("📋 К заказу (шт)"),
                    html.H3(f"{int(total_order_qty)}")
                ])
            ])
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("💰 Остатки (₴)"),
                    html.H3(f"{total_stock_value:,.0f}")
                ])
            ])
        ], width=2)
    ])


@callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'active_tab'),
     Input('ddmrp-data', 'data'),
     Input('orders-data', 'data'),
     Input('lost-sales-data', 'data'),
     Input('alerts-data', 'data'),
     Input('adjusted-buffers-data', 'data')]
)
def render_tab_content(active_tab, ddmrp_data, orders_data, lost_sales_data, alerts_data, adjusted_buffers_data):
    if not ddmrp_data:
        return ""
    
    ddmrp_df = pd.DataFrame(ddmrp_data)
    orders_df = pd.DataFrame(orders_data) if orders_data else pd.DataFrame()
    lost_sales_df = pd.DataFrame(lost_sales_data) if lost_sales_data else pd.DataFrame()
    alerts_df = pd.DataFrame(alerts_data) if alerts_data else pd.DataFrame()
    adjusted_buffers_df = pd.DataFrame(adjusted_buffers_data) if adjusted_buffers_data else pd.DataFrame()
    
    # ============ НОВАЯ ВКЛАДКА: АЛЕРТЫ ============
    if active_tab == "tab-alerts":
        if alerts_df.empty:
            return dbc.Alert("✅ Нет критических алертов! Все в порядке.", color="success", className="mt-3")
        
        # Подсчет алертов по типам
        alert_counts = alerts_df['Type'].value_counts()
        
        return html.Div([
            html.H4("🚨 Критические алерты"),
            
            # Счетчики алертов
            dbc.Row([
                dbc.Col([
                    dbc.Alert([
                        html.H5(f"🔴 Критичных: {alert_counts.get('🔴 КРИТИЧНО', 0)}"),
                        html.P("Остаток ниже критического уровня", className="mb-0")
                    ], color="danger")
                ], width=4),
                dbc.Col([
                    dbc.Alert([
                        html.H5(f"⚠️ Близко к нулю: {alert_counts.get('⚠️ БЛИЗОК К НУЛЮ', 0)}"),
                        html.P("Остаток < 20% от красной зоны", className="mb-0")
                    ], color="warning")
                ], width=4),
                dbc.Col([
                    dbc.Alert([
                        html.H5(f"❌ Stockout: {alert_counts.get('❌ НЕТ В НАЛИЧИИ', 0)}"),
                        html.P("Товар полностью отсутствует", className="mb-0")
                    ], color="dark")
                ], width=4)
            ], className="mb-3"),
            
            # Таблица алертов
            dash_table.DataTable(
                data=alerts_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in alerts_df.columns],
                page_size=25,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '10px', 'fontSize': '13px'},
                style_header={'backgroundColor': '#dc3545', 'color': 'white', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {'if': {'filter_query': '{Priority} = 1'}, 'backgroundColor': '#FFE6E6'},
                    {'if': {'filter_query': '{Priority} = 2'}, 'backgroundColor': '#FFF9E6'}
                ],
                sort_action='native',
                filter_action='native'
            )
        ])
    
    # ============ НОВАЯ ВКЛАДКА: УПУЩЕННАЯ ПРИБЫЛЬ ============
    elif active_tab == "tab-lost-sales":
        if lost_sales_df.empty:
            return dbc.Alert("✅ Упущенной прибыли нет! Все товары в достаточном количестве.", color="success", className="mt-3")
        
        # Общая упущенная прибыль
        total_lost_sales = lost_sales_df['Lost_Sales_7days'].sum()
        total_daily_lost = lost_sales_df['Daily_Lost_Sales'].sum()
        
        # Топ-10 товаров по упущенной прибыли
        top_10_lost = lost_sales_df.head(10)
        
        # График
        fig_lost_sales = px.bar(
            top_10_lost,
            x='Lost_Sales_7days',
            y='Describe',
            color='Buffer_Status',
            title='Топ-10 товаров по упущенной прибыли (7 дней)',
            orientation='h',
            labels={'Lost_Sales_7days': 'Упущенная прибыль (₴)', 'Describe': 'Товар'},
            color_discrete_map={'RED': '#FF4444', 'YELLOW': '#FFD700'}
        )
        fig_lost_sales.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        return html.Div([
            html.H4("💰 Анализ упущенной прибыли"),
            
            # Метрики
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("💸 Упущенная прибыль (7 дней)"),
                            html.H3(f"{total_lost_sales:,.0f} ₴", className="text-danger")
                        ])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("📉 Упущенная прибыль в день"),
                            html.H3(f"{total_daily_lost:,.0f} ₴", className="text-warning")
                        ])
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("📊 Товаров с потерями"),
                            html.H3(f"{len(lost_sales_df)}", className="text-info")
                        ])
                    ])
                ], width=4)
            ], className="mb-3"),
            
            # График
            dcc.Graph(figure=fig_lost_sales, className="mb-3"),
            
            # Таблица
            html.H5("Полный список товаров с упущенной прибылью"),
            dash_table.DataTable(
                data=lost_sales_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in lost_sales_df.columns],
                page_size=20,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={'backgroundColor': '#ffc107', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {'if': {'filter_query': '{Buffer_Status} = "RED"'}, 'backgroundColor': '#FFE6E6'},
                    {'if': {'column_id': 'Lost_Sales_7days'}, 'fontWeight': 'bold', 'color': '#dc3545'}
                ],
                sort_action='native',
                filter_action='native'
            )
        ])
    
    # ============ НОВАЯ ВКЛАДКА: КОРРЕКТИРОВКА БУФЕРОВ ============
    elif active_tab == "tab-buffer-adjust":
        if adjusted_buffers_df.empty:
            return dbc.Alert("⚠️ Нет данных для корректировки буферов", color="warning", className="mt-3")
        
        # Товары, требующие корректировки
        items_to_adjust = adjusted_buffers_df[adjusted_buffers_df['Adjustment_Recommended'] == True]
        
        seasonal_factor = calculate_seasonal_factor()
        current_month = datetime.now().strftime('%B %Y')
        
        return html.Div([
            html.H4("🔧 Динамическая корректировка буферов"),
            
            # Информация о сезонности
            dbc.Alert([
                html.H5("📅 Сезонная информация"),
                html.P(f"Текущий период: {current_month}"),
                html.P(f"Сезонный коэффициент: {seasonal_factor:.2f}x"),
                html.P("Рекомендуется корректировка буферов для товаров с частыми дефицитами", className="mb-0")
            ], color="info", className="mb-3"),
            
            # Метрики
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("🔧 Товаров требуют корректировки"),
                            html.H3(f"{len(items_to_adjust)}")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("✅ Буферов в норме"),
                            html.H3(f"{len(adjusted_buffers_df) - len(items_to_adjust)}")
                        ])
                    ])
                ], width=6)
            ], className="mb-3"),
            
            # Таблица с рекомендациями
            html.H5("Рекомендуемые корректировки буферов"),
            dash_table.DataTable(
                data=items_to_adjust[['Store_ID', 'Article', 'Describe', 'Current_Stock', 
                                     'Red_Zone', 'Yellow_Zone', 'Green_Zone',
                                     'Suggested_Red_Zone', 'Suggested_Yellow_Zone', 'Suggested_Green_Zone',
                                     'Buffer_Status']].to_dict('records'),
                columns=[{"name": i, "id": i} for i in ['Store_ID', 'Article', 'Describe', 'Current_Stock', 
                                                        'Red_Zone', 'Yellow_Zone', 'Green_Zone',
                                                        'Suggested_Red_Zone', 'Suggested_Yellow_Zone', 'Suggested_Green_Zone',
                                                        'Buffer_Status']],
                page_size=20,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '10px', 'fontSize': '12px'},
                style_header={'backgroundColor': '#17a2b8', 'color': 'white', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {'if': {'column_id': ['Suggested_Red_Zone', 'Suggested_Yellow_Zone', 'Suggested_Green_Zone']}, 
                     'backgroundColor': '#E8F8F5', 'fontWeight': 'bold'}
                ],
                sort_action='native',
                filter_action='native'
            )
        ])
    
    elif active_tab == "tab-orders":
        if orders_df.empty:
            return dbc.Alert("🎉 Все товары в норме! Заказов не требуется.", color="success")
        
        return html.Div([
            html.H4("📋 Список товаров для заказа"),
            dash_table.DataTable(
                data=orders_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in orders_df.columns],
                page_size=20,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {'if': {'filter_query': '{Buffer_Status} = "RED"'}, 'backgroundColor': '#FFE6E6'},
                    {'if': {'filter_query': '{Buffer_Status} = "YELLOW"'}, 'backgroundColor': '#FFF9E6'}
                ]
            )
        ])
    
    elif active_tab == "tab-all":
        return html.Div([
            html.H4("📊 Полный список товаров и статусы буферов"),
            dash_table.DataTable(
                data=ddmrp_df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in ddmrp_df.columns],
                page_size=20,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
            )
        ])
    
    elif active_tab == "tab-stores":
        stores = sorted(ddmrp_df['Store_ID'].unique())
        return html.Div([
            html.H4("🏪 Анализ по магазинам"),
            dcc.Dropdown(
                id='store-selector',
                options=[{'label': f'Магазин {s}', 'value': s} for s in stores],
                value=stores[0] if stores else None,
                className="mb-3"
            ),
            html.Div(id='store-details')
        ])
    
    elif active_tab == "tab-analytics":
        fig1 = create_buffer_status_chart(ddmrp_df)
        fig2 = create_store_summary_chart(ddmrp_df)
        
        return html.Div([
            html.H4("📈 Аналитические графики"),
            dbc.Row([
                dbc.Col([dcc.Graph(figure=fig1)], width=6),
                dbc.Col([dcc.Graph(figure=fig2)], width=6)
            ])
        ])
    
    elif active_tab == "tab-details":
        return html.Div([
            dbc.Card([
                dbc.CardHeader("⚙️ Методология расчета DDMRP"),
                dbc.CardBody([
                    html.H5("Зоны буфера:"),
                    html.Ul([
                        html.Li("🔴 Красная зона (Red Zone): Критический минимум запаса"),
                        html.Li("🟡 Желтая зона (Yellow Zone): Зона пополнения"),
                        html.Li("🟢 Зеленая зона (Green Zone): Целевой запас"),
                        html.Li("🔵 Излишек (Excess): Запас выше Top of Green")
                    ]),
                    html.Hr(),
                    html.H5("Расчет Top of Green:"),
                    html.Code("Top of Green = Red Zone + Yellow Zone + Green Zone"),
                    html.Hr(),
                    html.H5("Расчет количества для заказа:"),
                    html.Code("Order Qty = Top of Green - Current Stock"),
                    html.P("(только для RED и YELLOW статусов)", className="text-muted mt-2")
                ])
            ])
        ])
    
    return ""


@callback(
    Output('store-details', 'children'),
    [Input('store-selector', 'value'),
     Input('ddmrp-data', 'data')]
)
def update_store_details(selected_store, ddmrp_data):
    if not selected_store or not ddmrp_data:
        return ""
    
    ddmrp_df = pd.DataFrame(ddmrp_data)
    store_data = ddmrp_df[ddmrp_df['Store_ID'] == selected_store]
    
    red_store = len(store_data[store_data['Buffer_Status'] == 'RED'])
    yellow_store = len(store_data[store_data['Buffer_Status'] == 'YELLOW'])
    order_qty_store = store_data['Order_Qty'].sum()
    
    return html.Div([
        dbc.Row([
            dbc.Col([dbc.Card([dbc.CardBody([html.H6("Всего SKU"), html.H4(len(store_data))])])], width=3),
            dbc.Col([dbc.Card([dbc.CardBody([html.H6("🔴 Критичных"), html.H4(red_store)])])], width=3),
            dbc.Col([dbc.Card([dbc.CardBody([html.H6("🟡 Требуют заказа"), html.H4(yellow_store)])])], width=3),
            dbc.Col([dbc.Card([dbc.CardBody([html.H6("К заказу (шт)"), html.H4(int(order_qty_store))])])], width=3)
        ], className="mb-3"),
        
        dash_table.DataTable(
            data=store_data.to_dict('records'),
            columns=[{"name": i, "id": i} for i in store_data.columns],
            page_size=15,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
        )
    ])


# ========================
# ЗАПУСК ПРИЛОЖЕНИЯ
# ========================

if __name__ == '__main__':
    app.run(debug=False)
