import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import pearsonr, spearmanr
from datetime import datetime

st.set_page_config(page_title="Анализ продаж по сегментам", layout="wide")

st.title("📊 Анализ продаж: Сегменты и Магазины")

# Загрузка файла
uploaded_file = st.file_uploader("Загрузите Excel файл с продажами", type=['xlsx', 'xls'])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df['Datasales'] = pd.to_datetime(df['Datasales'], errors='coerce')
    
    # ИСПРАВЛЕНИЕ: более строгая валидация данных
    initial_rows = len(df)
    df = df.dropna(subset=['Datasales', 'Sum', 'Segment', 'Magazin'])
    df = df[df['Sum'] > 0]
    df['Qty'] = df['Qty'].fillna(1).astype(int)  # ИСПРАВЛЕНИЕ: заполняем пустые Qty
    df = df.sort_values('Datasales')
    
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        st.warning(f"⚠️ Удалено {removed_rows} некорректных записей ({removed_rows/initial_rows*100:.1f}%)")
    
    if len(df) == 0:
        st.error("❌ Нет данных после очистки")
        st.stop()
    
    # Проверка распределения данных по годам
    df['Year'] = df['Datasales'].dt.year
    data_by_year = df.groupby('Year')['Sum'].agg(['count', 'sum']).reset_index()
    data_by_year.columns = ['Год', 'Записей', 'Сумма продаж']
    
    st.success(f"✅ Загружено {len(df):,} записей | Период: {df['Datasales'].min().date()} — {df['Datasales'].max().date()}")
    
    # НОВОЕ: KPI дашборд в самом начале
    st.markdown("### 📌 Ключевые показатели")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_sales = df['Sum'].sum()
    total_qty = df['Qty'].sum()
    num_transactions = len(df)
    avg_transaction = total_sales / num_transactions if num_transactions > 0 else 0
    num_segments = df['Segment'].nunique()
    num_magazins = df['Magazin'].nunique()
    
    with col1:
        st.metric("💰 Общие продажи", f"{total_sales:,.0f}")
    with col2:
        st.metric("🛒 Транзакций", f"{num_transactions:,}")
    with col3:
        st.metric("📦 Единиц", f"{total_qty:,}")
    with col4:
        st.metric("💳 Средний чек", f"{avg_transaction:,.0f}")
    with col5:
        st.metric("🏪 Магазинов", f"{num_magazins}")
    
    with st.expander("📊 Распределение данных по годам"):
        st.dataframe(data_by_year, hide_index=True, use_container_width=True)
        
        if len(data_by_year) > 1:
            year_diff = data_by_year['Год'].max() - data_by_year['Год'].min() + 1
            if len(data_by_year) < year_diff:
                missing_years = set(range(data_by_year['Год'].min(), data_by_year['Год'].max() + 1)) - set(data_by_year['Год'])
                st.warning(f"⚠️ Пропущены годы: {sorted(missing_years)}")
    
    # Фильтр по годам
    available_years = sorted(df['Year'].unique())
    selected_years = st.multiselect(
        "Выберите годы для анализа",
        available_years,
        default=available_years
    )
    
    if not selected_years:
        st.error("❌ Выберите хотя бы один год")
        st.stop()
    
    df = df[df['Year'].isin(selected_years)]
    
    # Выбор типа анализа
    analysis_type = st.radio("Что анализируем?", ["Сегменты", "Магазины"], horizontal=True)
    
    st.markdown("---")
    
    if analysis_type == "Сегменты":
        st.header("📈 Анализ по сегментам")
        
        # Агрегация по сегментам
        df['Month'] = df['Datasales'].dt.to_period('M')
        df['Quarter'] = df['Datasales'].dt.to_period('Q')
        
        # Выбор периода агрегации
        period = st.selectbox("Период агрегации", ["День", "Неделя", "Месяц", "Квартал"])
        
        if period == "День":
            df_grouped = df.groupby(['Datasales', 'Segment'])['Sum'].sum().reset_index()
            df_pivot = df_grouped.pivot(index='Datasales', columns='Segment', values='Sum')
        elif period == "Неделя":
            df['Period'] = df['Datasales'].dt.to_period('W')
            df_grouped = df.groupby(['Period', 'Segment'])['Sum'].sum().reset_index()
            df_grouped['Period'] = df_grouped['Period'].dt.to_timestamp()
            df_pivot = df_grouped.pivot(index='Period', columns='Segment', values='Sum')
        elif period == "Месяц":
            df_grouped = df.groupby(['Month', 'Segment'])['Sum'].sum().reset_index()
            df_grouped['Month'] = df_grouped['Month'].dt.to_timestamp()
            df_pivot = df_grouped.pivot(index='Month', columns='Segment', values='Sum')
        else:  # Квартал
            df_grouped = df.groupby(['Quarter', 'Segment'])['Sum'].sum().reset_index()
            df_grouped['Quarter'] = df_grouped['Quarter'].dt.to_timestamp()
            df_pivot = df_grouped.pivot(index='Quarter', columns='Segment', values='Sum')
        
        df_pivot = df_pivot.dropna(how='all')
        
        # 1. ВРЕМЕННЫЕ РЯДЫ СЕГМЕНТОВ
        st.subheader("1️⃣ Динамика продаж по сегментам")
        
        fig = go.Figure()
        for segment in df_pivot.columns:
            fig.add_trace(go.Scatter(
                x=df_pivot.index,
                y=df_pivot[segment],
                name=segment,
                mode='lines+markers',
                connectgaps=False
            ))
        
        fig.update_layout(
            xaxis_title='Дата',
            yaxis_title='Продажи',
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 2. КОРРЕЛЯЦИЯ МЕЖДУ СЕГМЕНТАМИ
        st.subheader("2️⃣ Корреляция между сегментами")
        
        df_pivot_corr = df_pivot.dropna()
        
        if len(df_pivot_corr) < 10:
            st.warning(f"⚠️ Мало данных для корреляции (только {len(df_pivot_corr)} периодов). Результаты могут быть неточными.")
        
        corr_matrix = df_pivot_corr.corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Корреляция")
        ))
        
        fig_corr.update_layout(
            title='Матрица корреляции сегментов',
            height=500
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # НОВОЕ: Анализ сильных корреляций
        if len(corr_matrix) > 1:
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Сегмент 1': corr_matrix.columns[i],
                        'Сегмент 2': corr_matrix.columns[j],
                        'Корреляция': corr_matrix.iloc[i, j]
                    })
            corr_df = pd.DataFrame(corr_pairs).sort_values('Корреляция', key=abs, ascending=False)
            
            st.info("💡 Положительная корреляция (красный) = сегменты растут/падают вместе. Отрицательная (синий) = обратная зависимость.")
            
            with st.expander("📊 Топ-5 связанных пар сегментов"):
                st.dataframe(corr_df.head(), hide_index=True, use_container_width=True)
        
        # 3. СЕЗОННОСТЬ ПО МЕСЯЦАМ
        st.subheader("3️⃣ Сезонность: какой сегмент когда продается")
        
        df['MonthName'] = df['Datasales'].dt.month
        seasonal_data = df.groupby(['MonthName', 'Segment'])['Sum'].sum().reset_index()
        
        if len(seasonal_data) == 0:
            st.warning("⚠️ Недостаточно данных для анализа сезонности")
        else:
            seasonal_pivot = seasonal_data.pivot(index='MonthName', columns='Segment', values='Sum')
            seasonal_pivot_filled = seasonal_pivot.fillna(0)
            segment_totals = seasonal_pivot_filled.sum(axis=0)
            segment_totals = segment_totals.replace(0, np.nan)
            seasonal_pct = seasonal_pivot_filled.div(segment_totals, axis=1) * 100
            seasonal_pct = seasonal_pct.fillna(0)
            
            month_names = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
            x_labels = [month_names[i-1] for i in seasonal_pivot.index if 1 <= i <= 12]
            
            fig_seasonal = go.Figure()
            for segment in seasonal_pct.columns:
                fig_seasonal.add_trace(go.Bar(
                    x=x_labels,
                    y=seasonal_pct[segment],
                    name=segment
                ))
            
            fig_seasonal.update_layout(
                title='% продаж сегмента по месяцам (от годовых)',
                xaxis_title='Месяц',
                yaxis_title='% от годовых продаж',
                barmode='group',
                height=500
            )
            st.plotly_chart(fig_seasonal, use_container_width=True)
        
        # НОВОЕ: Индекс сезонности
        with st.expander("📈 Индекс сезонности по сегментам"):
            st.markdown("**Индекс > 100** = месяц сильнее среднего, **< 100** = слабее")
            seasonal_index = seasonal_pivot_filled.div(seasonal_pivot_filled.mean(axis=0), axis=1) * 100
            seasonal_index = seasonal_index.round(0)
            seasonal_index.index = [month_names[i-1] for i in seasonal_index.index if 1 <= i <= 12]
            st.dataframe(seasonal_index, use_container_width=True)
        
        # 4. ДОЛИ СЕГМЕНТОВ
        st.subheader("4️⃣ Структура продаж по сегментам")
        
        col1, col2 = st.columns(2)
        
        with col1:
            segment_totals = df.groupby('Segment')['Sum'].sum().sort_values(ascending=False)
            fig_pie = go.Figure(data=[go.Pie(
                labels=segment_totals.index,
                values=segment_totals.values,
                hole=0.3
            )])
            fig_pie.update_layout(title='Общая доля продаж', height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            segment_stats = df.groupby('Segment').agg({
                'Sum': ['sum', 'mean', 'std'],
                'Qty': 'sum'
            }).round(0)
            segment_stats.columns = ['Общая сумма', 'Средняя', 'Ст. отклонение', 'Единиц']
            segment_stats['Доля %'] = (segment_stats['Общая сумма'] / segment_stats['Общая сумма'].sum() * 100).round(1)
            
            # ИСПРАВЛЕНИЕ: Коэффициент вариации
            segment_stats['CV %'] = ((segment_stats['Ст. отклонение'] / segment_stats['Средняя']) * 100).round(1)
            segment_stats = segment_stats.sort_values('Общая сумма', ascending=False)
            
            st.dataframe(segment_stats[['Общая сумма', 'Доля %', 'CV %', 'Единиц']], use_container_width=True)
            st.caption("CV % = коэффициент вариации (стабильность продаж)")
        
        # 5. ЛУЧШИЕ/ХУДШИЕ ПЕРИОДЫ ДЛЯ КАЖДОГО СЕГМЕНТА
        st.subheader("5️⃣ Лучшие и худшие месяцы по сегментам")
        
        for segment in df['Segment'].unique():
            segment_monthly = df[df['Segment'] == segment].groupby('Month')['Sum'].sum()
            if len(segment_monthly) > 0:
                best_month = segment_monthly.idxmax()
                worst_month = segment_monthly.idxmin()
                avg_month = segment_monthly.mean()
                
                best_value = segment_monthly[best_month]
                worst_value = segment_monthly[worst_month]
                
                # Процент от среднего
                best_pct = ((best_value / avg_month - 1) * 100) if avg_month > 0 else 0
                worst_pct = ((worst_value / avg_month - 1) * 100) if avg_month > 0 else 0
                
                # Разница между лучшим и худшим
                diff_abs = best_value - worst_value
                diff_pct = ((best_value / worst_value - 1) * 100) if worst_value > 0 else 0
                
                # Форматирование дат
                best_month_str = best_month.strftime('%B %Y') if hasattr(best_month, 'strftime') else str(best_month)
                worst_month_str = worst_month.strftime('%B %Y') if hasattr(worst_month, 'strftime') else str(worst_month)
                
                # Визуализация
                col1, col2, col3, col4 = st.columns([2, 2, 2, 3])
                
                with col1:
                    st.metric(
                        f"**{segment}**",
                        f"{segment_monthly.sum():,.0f}",
                        f"Ср./мес: {avg_month:,.0f}"
                    )
                
                with col2:
                    st.success(f"🔥 **Лучший:** {best_month_str}")
                    st.write(f"💰 {best_value:,.0f}")
                    st.write(f"📈 +{best_pct:,.0f}% от среднего")
                
                with col3:
                    st.error(f"📉 **Худший:** {worst_month_str}")
                    st.write(f"💰 {worst_value:,.0f}")
                    st.write(f"📉 {worst_pct:,.0f}% от среднего")
                
                with col4:
                    st.info(f"**📊 Разброс**")
                    st.write(f"Разница: {diff_abs:,.0f}")
                    st.write(f"В {diff_pct/100 + 1:.1f}х раз")
                    
                    # Мини-бар для визуализации
                    fig_mini = go.Figure()
                    fig_mini.add_trace(go.Bar(
                        x=['Худший', 'Средний', 'Лучший'],
                        y=[worst_value, avg_month, best_value],
                        marker_color=['red', 'gray', 'green'],
                        text=[f'{worst_value:,.0f}', f'{avg_month:,.0f}', f'{best_value:,.0f}'],
                        textposition='outside'
                    ))
                    fig_mini.update_layout(
                        height=150,
                        margin=dict(l=0, r=0, t=0, b=0),
                        showlegend=False,
                        yaxis_visible=False
                    )
                    st.plotly_chart(fig_mini, use_container_width=True)
                
                st.markdown("---")
        
        # 6. ТРЕНДЫ И РОСТ
        st.subheader("6️⃣ Тренды: рост/падение сегментов")
        
        df_sorted = df.sort_values('Datasales')
        split_point = len(df_sorted) // 3
        
        if split_point < 1:
            st.warning("⚠️ Недостаточно данных для анализа трендов")
        else:
            first_period = df_sorted.iloc[:split_point].groupby('Segment')['Sum'].sum()
            last_period = df_sorted.iloc[-split_point:].groupby('Segment')['Sum'].sum()
            common_segments = first_period.index.intersection(last_period.index)
            
            if len(common_segments) == 0:
                st.warning("⚠️ Нет общих сегментов для сравнения периодов")
            else:
                growth = ((last_period[common_segments] - first_period[common_segments]) / first_period[common_segments] * 100)
                growth = growth.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
                
                fig_growth = go.Figure(data=[
                    go.Bar(x=growth.index, y=growth.values, 
                           marker_color=['green' if x > 0 else 'red' for x in growth.values])
                ])
                fig_growth.update_layout(
                    title='Изменение продаж: начало vs конец периода (%)',
                    xaxis_title='Сегмент',
                    yaxis_title='Рост/падение %',
                    height=400
                )
                st.plotly_chart(fig_growth, use_container_width=True)
        
        # НОВОЕ: ABC-анализ сегментов
        st.subheader("7️⃣ ABC-анализ сегментов")
        
        segment_abc = df.groupby('Segment')['Sum'].sum().sort_values(ascending=False)
        segment_abc_df = pd.DataFrame({
            'Сегмент': segment_abc.index,
            'Продажи': segment_abc.values,
            'Доля %': (segment_abc.values / segment_abc.sum() * 100).round(1),
            'Накопительная %': (segment_abc.values.cumsum() / segment_abc.sum() * 100).round(1)
        })
        
        # Классификация ABC
        segment_abc_df['Категория'] = segment_abc_df['Накопительная %'].apply(
            lambda x: 'A (топ 80%)' if x <= 80 else ('B (80-95%)' if x <= 95 else 'C (остальное)')
        )
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(segment_abc_df, hide_index=True, use_container_width=True)
        with col2:
            category_counts = segment_abc_df['Категория'].value_counts()
            st.write("**Распределение по категориям:**")
            for cat, count in category_counts.items():
                st.write(f"{cat}: {count} сегм.")
        
        # НОВОЕ: Выводы и рекомендации по сегментам
        st.markdown("---")
        st.header("🎯 Выводы и рекомендации по сегментам")
        
        # ==================== ГЛУБОКИЙ АНАЛИЗ ====================
        
        # Базовые метрики
        total_sales = df['Sum'].sum()
        top_segment = segment_abc_df.iloc[0]['Сегмент']
        top_share = segment_abc_df.iloc[0]['Доля %']
        top_segment_sales = segment_abc_df.iloc[0]['Продажи']
        
        # Анализ роста
        growing_segments = growth[growth > 10].sort_values(ascending=False) if 'growth' in locals() and len(growth) > 0 else pd.Series()
        declining_segments = growth[growth < -10].sort_values() if 'growth' in locals() and len(growth) > 0 else pd.Series()
        
        # Анализ стабильности
        if 'segment_stats' in locals():
            stable_segments = segment_stats[segment_stats['CV %'] < 50].sort_values('CV %')
            volatile_segments = segment_stats[segment_stats['CV %'] > 100].sort_values('CV %', ascending=False)
        
        # Концентрация рисков
        a_category_count = len(segment_abc_df[segment_abc_df['Категория'] == 'A (топ 80%)'])
        a_category_share = segment_abc_df[segment_abc_df['Категория'] == 'A (топ 80%)']['Доля %'].sum()
        
        # ==================== ЭКСПРЕСС-ДИАГНОСТИКА ====================
        
        st.subheader("📊 Экспресс-диагностика бизнеса")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Показатель 1: Концентрация
        concentration_status = "🔴 Критично" if top_share > 50 else ("🟡 Внимание" if top_share > 35 else "🟢 Норма")
        with col1:
            st.metric("Концентрация", f"{top_share:.0f}%", concentration_status)
            st.caption("Доля топ-сегмента")
        
        # Показатель 2: Рост
        growth_count = len(growing_segments)
        decline_count = len(declining_segments)
        growth_status = "🟢 Растем" if growth_count > decline_count else ("🔴 Падаем" if decline_count > growth_count else "🟡 Стабильно")
        with col2:
            st.metric("Динамика", f"+{growth_count} / -{decline_count}", growth_status)
            st.caption("Растущие/падающие")
        
        # Показатель 3: Стабильность
        stable_count = len(stable_segments) if 'stable_segments' in locals() else 0
        total_segments = len(segment_abc_df)
        stability_status = "🟢 Стабильно" if stable_count / total_segments > 0.5 else ("🟡 Умеренно" if stable_count / total_segments > 0.3 else "🔴 Волатильно")
        with col3:
            st.metric("Стабильность", f"{stable_count}/{total_segments}", stability_status)
            st.caption("Стабильные сегменты")
        
        # Показатель 4: Диверсификация
        diversification_status = "🟢 Хорошо" if a_category_count >= 3 else ("🟡 Средне" if a_category_count == 2 else "🔴 Риск")
        with col4:
            st.metric("ABC категория A", f"{a_category_count} сегм.", diversification_status)
            st.caption("Ключевые сегменты")
        
        st.markdown("---")
        
        # ==================== ДЕТАЛЬНЫЙ АНАЛИЗ ====================
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ Сильные стороны")
            
            st.write(f"**1. Лидер продаж: {top_segment}**")
            st.write(f"   💰 Продажи: {top_segment_sales:,.0f} ({top_share:.1f}%)")
            st.write(f"   📊 Статус: {'Доминирующий' if top_share > 50 else 'Ключевой'} сегмент")
            
            if len(growing_segments) > 0:
                st.write(f"\n**2. Растущие сегменты** ({len(growing_segments)} шт):")
                for i, (seg, val) in enumerate(growing_segments.head(3).items(), 1):
                    seg_sales = segment_abc_df[segment_abc_df['Сегмент'] == seg]['Продажи'].values[0]
                    st.write(f"   {i}. **{seg}**: +{val:.0f}% (💰 {seg_sales:,.0f})")
            
            if len(stable_segments) > 0:
                st.write(f"\n**3. Стабильные сегменты** (CV < 50%):")
                for i, seg in enumerate(stable_segments.head(3).index, 1):
                    cv = stable_segments.loc[seg, 'CV %']
                    st.write(f"   {i}. **{seg}**: CV = {cv:.0f}% (предсказуемые продажи)")
        
        with col2:
            st.subheader("⚠️ Зоны внимания")
            
            if len(declining_segments) > 0:
                st.write(f"**1. Падающие сегменты** ({len(declining_segments)} шт):")
                total_decline_value = 0
                for i, (seg, val) in enumerate(declining_segments.head(3).items(), 1):
                    seg_sales = segment_abc_df[segment_abc_df['Сегмент'] == seg]['Продажи'].values[0]
                    decline_loss = seg_sales * abs(val) / 100
                    total_decline_value += decline_loss
                    st.write(f"   {i}. **{seg}**: {val:.0f}% (💸 потеря ~{decline_loss:,.0f})")
                st.write(f"   ⚡ Общая потенциальная потеря: **{total_decline_value:,.0f}**")
            
            if len(volatile_segments) > 0:
                st.write(f"\n**2. Нестабильные сегменты** (CV > 100%):")
                for i, seg in enumerate(volatile_segments.head(3).index, 1):
                    cv = volatile_segments.loc[seg, 'CV %']
                    st.write(f"   {i}. **{seg}**: CV = {cv:.0f}% (непредсказуемые)")
            
            if a_category_share > 80:
                st.write(f"\n**3. Риск концентрации:**")
                st.write(f"   📊 {a_category_share:.0f}% продаж в {a_category_count} сегментах")
                st.write(f"   ⚠️ Высокая зависимость от топа")
        
        st.markdown("---")
        
        # ==================== ПРИОРИТИЗИРОВАННЫЕ РЕКОМЕНДАЦИИ ====================
        
        st.subheader("💡 Приоритизированный план действий")
        
        recommendations = []
        
        # ПРИОРИТЕТ 1: Критические проблемы
        if len(declining_segments) > 0:
            top_decliner = declining_segments.index[0]
            decline_rate = declining_segments.iloc[0]
            decliner_sales = segment_abc_df[segment_abc_df['Сегмент'] == top_decliner]['Продажи'].values[0]
            potential_loss = decliner_sales * abs(decline_rate) / 100
            
            recommendations.append({
                'priority': '🔴 КРИТИЧНО',
                'title': f'Остановить падение: {top_decliner}',
                'problem': f'Падение на {decline_rate:.0f}% за период',
                'why': f'Потенциальная потеря: {potential_loss:,.0f} ({abs(decline_rate):.0f}% от {decliner_sales:,.0f})',
                'action': [
                    '1. Проанализировать причины: конкуренты, цены, качество, маркетинг',
                    '2. Опросить клиентов и отток',
                    '3. Запустить специальные акции на 30 дней',
                    '4. Пересмотреть ассортимент и позиционирование'
                ],
                'metric': f'Целевой рост: +{abs(decline_rate/2):.0f}% за квартал',
                'impact': 'Высокий',
                'effort': 'Средний',
                'roi': f'Возврат ~{potential_loss * 0.5:,.0f} за квартал'
            })
        
        # ПРИОРИТЕТ 2: Быстрые победы (рост)
        if len(growing_segments) > 0:
            top_grower = growing_segments.index[0]
            growth_rate = growing_segments.iloc[0]
            grower_sales = segment_abc_df[segment_abc_df['Сегмент'] == top_grower]['Продажи'].values[0]
            grower_share = segment_abc_df[segment_abc_df['Сегмент'] == top_grower]['Доля %'].values[0]
            potential_gain = grower_sales * 0.2  # Консервативно 20% дополнительного роста
            
            recommendations.append({
                'priority': '🟢 БЫСТРАЯ ПОБЕДА',
                'title': f'Ускорить рост: {top_grower}',
                'problem': f'Уже растет на +{growth_rate:.0f}%, но есть потенциал',
                'why': f'Текущие продажи: {grower_sales:,.0f} ({grower_share:.1f}% доли)',
                'action': [
                    f'1. Увеличить маркетинговый бюджет на {top_grower} на 30%',
                    '2. Расширить ассортимент в категории',
                    '3. Обучить персонал активным продажам',
                    '4. Создать программу лояльности для сегмента'
                ],
                'metric': f'Целевой рост: +{growth_rate * 1.5:.0f}% (ускорение в 1.5х)',
                'impact': 'Высокий',
                'effort': 'Низкий',
                'roi': f'Доп. выручка ~{potential_gain:,.0f} при инвестициях ~{potential_gain * 0.3:,.0f}'
            })
        
        # ПРИОРИТЕТ 3: Диверсификация
        if top_share > 40:
            second_segment = segment_abc_df.iloc[1]['Сегмент']
            second_share = segment_abc_df.iloc[1]['Доля %']
            gap = top_share - second_share
            
            recommendations.append({
                'priority': '🟡 СТРАТЕГИЯ',
                'title': 'Снизить концентрацию рисков',
                'problem': f'{top_segment} = {top_share:.1f}% (разрыв с #{2}: {gap:.0f}%)',
                'why': 'Высокая зависимость от одного сегмента = риск при проблемах',
                'action': [
                    f'1. Развивать {second_segment} (сейчас {second_share:.1f}%)',
                    '2. Инвестировать в сегменты категории B',
                    '3. Тестировать новые ниши',
                    f'4. Цель: довести топ-2-3 сегмента до 60% (сейчас {top_share:.0f}%)'
                ],
                'metric': f'Целевое распределение: топ сегмент < 40% за год',
                'impact': 'Средний',
                'effort': 'Высокий',
                'roi': 'Снижение бизнес-рисков + рост на 10-15%'
            })
        
        # ПРИОРИТЕТ 4: Сезонность
        if 'seasonal_index' in locals():
            seasonal_recommendations = []
            for segment in seasonal_index.columns[:3]:
                peak_month = seasonal_index[segment].idxmax()
                peak_value = seasonal_index[segment].max()
                low_month = seasonal_index[segment].idxmin()
                low_value = seasonal_index[segment].min()
                
                if peak_value > 150:  # Сильная сезонность
                    seg_sales = segment_abc_df[segment_abc_df['Сегмент'] == segment]['Продажи'].values[0]
                    peak_potential = seg_sales * (peak_value / 100 - 1) * 0.1  # 10% улучшение пика
                    
                    seasonal_recommendations.append({
                        'segment': segment,
                        'peak_month': peak_month,
                        'peak_index': peak_value,
                        'low_month': low_month,
                        'low_index': low_value,
                        'potential': peak_potential
                    })
            
            if seasonal_recommendations:
                best_seasonal = max(seasonal_recommendations, key=lambda x: x['potential'])
                
                recommendations.append({
                    'priority': '🟠 ТАКТИКА',
                    'title': f'Оптимизация сезонности: {best_seasonal["segment"]}',
                    'problem': f'Индекс {best_seasonal["peak_month"]} = {best_seasonal["peak_index"]:.0f}, {best_seasonal["low_month"]} = {best_seasonal["low_index"]:.0f}',
                    'why': f'Резкие колебания спроса → упущенная выручка в пик или затоваривание',
                    'action': [
                        f'1. За 2 месяца до {best_seasonal["peak_month"]}: увеличить запасы на 50%',
                        f'2. В {best_seasonal["low_month"]}: запустить стимулирующие акции',
                        '3. Настроить динамическое ценообразование',
                        '4. Сгладить спрос: предзаказы со скидкой в слабые месяцы'
                    ],
                    'metric': f'Цель: поднять {best_seasonal["low_month"]} с индекса {best_seasonal["low_index"]:.0f} до 80',
                    'impact': 'Средний',
                    'effort': 'Низкий',
                    'roi': f'Доп. выручка ~{best_seasonal["potential"]:,.0f}'
                })
        
        # ПРИОРИТЕТ 5: Кросс-продажи
        if len(corr_df) > 0 and corr_df.iloc[0]['Корреляция'] > 0.7:
            seg1 = corr_df.iloc[0]['Сегмент 1']
            seg2 = corr_df.iloc[0]['Сегмент 2']
            corr_value = corr_df.iloc[0]['Корреляция']
            
            seg1_sales = segment_abc_df[segment_abc_df['Сегмент'] == seg1]['Продажи'].values[0]
            seg2_sales = segment_abc_df[segment_abc_df['Сегмент'] == seg2]['Продажи'].values[0]
            cross_sell_potential = min(seg1_sales, seg2_sales) * 0.15  # 15% кросс-продаж
            
            recommendations.append({
                'priority': '🟢 БЫСТРАЯ ПОБЕДА',
                'title': f'Кросс-продажи: {seg1} × {seg2}',
                'problem': f'Корреляция {corr_value:.2f} - клиенты часто покупают вместе',
                'why': f'Потенциал: {cross_sell_potential:,.0f} (15% от меньшего сегмента)',
                'action': [
                    '1. Создать комплектные предложения со скидкой 10-15%',
                    f'2. При покупке {seg1} → рекомендовать {seg2} (и наоборот)',
                    '3. Разместить товары рядом в магазинах',
                    '4. Настроить email-цепочки с кросс-офферами'
                ],
                'metric': f'Цель: 15% покупателей {seg1} покупают и {seg2}',
                'impact': 'Средний',
                'effort': 'Низкий',
                'roi': f'Доп. выручка ~{cross_sell_potential:,.0f} при минимальных затратах'
            })
        
        # ПРИОРИТЕТ 6: Стабилизация волатильности
        if len(volatile_segments) > 0:
            top_volatile = volatile_segments.index[0]
            cv_value = volatile_segments.iloc[0]['CV %']
            
            recommendations.append({
                'priority': '🟡 СТРАТЕГИЯ',
                'title': f'Стабилизировать: {top_volatile}',
                'problem': f'CV = {cv_value:.0f}% (очень высокая волатильность)',
                'why': 'Непредсказуемые продажи → сложно планировать запасы и маркетинг',
                'action': [
                    '1. Проанализировать факторы волатильности',
                    '2. Ввести регулярные акции (каждую неделю)',
                    '3. Программа подписок/абонементов для регулярных покупок',
                    '4. Договориться с ключевыми клиентами о плановых закупках'
                ],
                'metric': f'Цель: снизить CV с {cv_value:.0f}% до < 80% за полгода',
                'impact': 'Низкий',
                'effort': 'Средний',
                'roi': 'Улучшение планирования → экономия 5-10% на складах'
            })
        
        # Сортируем по приоритету
        priority_order = {'🔴 КРИТИЧНО': 1, '🟢 БЫСТРАЯ ПОБЕДА': 2, '🟠 ТАКТИКА': 3, '🟡 СТРАТЕГИЯ': 4}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 5))
        
        # Вывод рекомендаций
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"**{rec['priority']} | #{i}: {rec['title']}**", expanded=i<=2):
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**📋 Проблема:**")
                    st.write(rec['problem'])
                    
                    st.markdown(f"**🎯 Почему важно:**")
                    st.write(rec['why'])
                    
                    st.markdown(f"**⚡ План действий:**")
                    for action in rec['action']:
                        st.write(action)
                
                with col2:
                    st.markdown(f"**📊 Метрики успеха:**")
                    st.info(rec['metric'])
                    
                    st.markdown(f"**💼 Оценка:**")
                    st.write(f"Влияние: **{rec['impact']}**")
                    st.write(f"Усилия: **{rec['effort']}**")
                    
                    st.markdown(f"**💰 ROI:**")
                    st.success(rec['roi'])
        
        # ==================== ФИНАНСОВАЯ ОЦЕНКА ====================
        
        st.markdown("---")
        st.subheader("💰 Финансовая оценка потенциала")
        
        total_potential = 0
        
        # Считаем потенциал от падающих (остановить потери)
        if len(declining_segments) > 0:
            decline_potential = sum([
                segment_abc_df[segment_abc_df['Сегмент'] == seg]['Продажи'].values[0] * abs(val) / 200  # 50% от потерь
                for seg, val in declining_segments.items()
            ])
            total_potential += decline_potential
        else:
            decline_potential = 0
        
        # Считаем потенциал от растущих (ускорить рост)
        if len(growing_segments) > 0:
            growth_potential = sum([
                segment_abc_df[segment_abc_df['Сегмент'] == seg]['Продажи'].values[0] * 0.2
                for seg in growing_segments.index[:2]  # топ-2
            ])
            total_potential += growth_potential
        else:
            growth_potential = 0
        
        # Потенциал от сезонности
        seasonal_potential = best_seasonal['potential'] if 'best_seasonal' in locals() else 0
        total_potential += seasonal_potential
        
        # Потенциал от кросс-продаж
        crosssell_potential = cross_sell_potential if 'cross_sell_potential' in locals() else 0
        total_potential += crosssell_potential
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "💾 Остановка потерь",
                f"{decline_potential:,.0f}",
                f"{decline_potential/total_sales*100:.1f}% от текущих продаж"
            )
        
        with col2:
            st.metric(
                "🚀 Ускорение роста",
                f"{growth_potential:,.0f}",
                f"{growth_potential/total_sales*100:.1f}% от текущих продаж"
            )
        
        with col3:
            st.metric(
                "📅 Сезонность + кросс",
                f"{seasonal_potential + crosssell_potential:,.0f}",
                f"{(seasonal_potential + crosssell_potential)/total_sales*100:.1f}% от текущих продаж"
            )
        
        with col4:
            st.metric(
                "💎 ИТОГО потенциал",
                f"{total_potential:,.0f}",
                f"+{total_potential/total_sales*100:.1f}% к обороту",
                delta_color="normal"
            )
        
        st.success(f"**🎯 При реализации всех рекомендаций прогнозируемый рост выручки: {total_potential:,.0f} (+{total_potential/total_sales*100:.1f}%)**")
        
        st.info("💡 **Рекомендация по приоритетам:** Начните с 🔴 критичных и 🟢 быстрых побед (первые 1-2 рекомендации). Они дадут 70% эффекта при 30% усилий.")
    
    else:  # Анализ по магазинам
        st.header("🏪 Анализ по магазинам")
        
        all_magazins = sorted(df['Magazin'].unique())
        selected_magazins = st.multiselect(
            "Выберите магазины для сравнения (до 10)",
            all_magazins,
            default=all_magazins[:min(5, len(all_magazins))]
        )
        
        if len(selected_magazins) > 10:
            st.warning("⚠️ Выбрано больше 10 магазинов, оставлены первые 10")
            selected_magazins = selected_magazins[:10]
        
        if not selected_magazins:
            st.error("Выберите хотя бы один магазин")
            st.stop()
        
        df_filtered = df[df['Magazin'].isin(selected_magazins)]
        
        period = st.selectbox("Период агрегации", ["День", "Неделя", "Месяц"])
        
        if period == "День":
            df_grouped = df_filtered.groupby(['Datasales', 'Magazin'])['Sum'].sum().reset_index()
            df_pivot = df_grouped.pivot(index='Datasales', columns='Magazin', values='Sum')
        elif period == "Неделя":
            df_filtered['Period'] = df_filtered['Datasales'].dt.to_period('W')
            df_grouped = df_filtered.groupby(['Period', 'Magazin'])['Sum'].sum().reset_index()
            df_grouped['Period'] = df_grouped['Period'].dt.to_timestamp()
            df_pivot = df_grouped.pivot(index='Period', columns='Magazin', values='Sum')
        else:
            df_filtered['Month'] = df_filtered['Datasales'].dt.to_period('M')
            df_grouped = df_filtered.groupby(['Month', 'Magazin'])['Sum'].sum().reset_index()
            df_grouped['Month'] = df_grouped['Month'].dt.to_timestamp()
            df_pivot = df_grouped.pivot(index='Month', columns='Magazin', values='Sum')
        
        df_pivot = df_pivot.dropna(how='all')
        
        # 1. ДИНАМИКА МАГАЗИНОВ
        st.subheader("1️⃣ Динамика продаж по магазинам")
        
        fig = go.Figure()
        for magazin in df_pivot.columns:
            fig.add_trace(go.Scatter(
                x=df_pivot.index,
                y=df_pivot[magazin],
                name=magazin,
                mode='lines+markers',
                connectgaps=False
            ))
        
        fig.update_layout(
            xaxis_title='Дата',
            yaxis_title='Продажи',
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 2. КОРРЕЛЯЦИЯ МЕЖДУ МАГАЗИНАМИ
        st.subheader("2️⃣ Корреляция между магазинами")
        
        if len(selected_magazins) > 1:
            df_pivot_corr = df_pivot.dropna()
            
            if len(df_pivot_corr) < 10:
                st.warning(f"⚠️ Мало данных для корреляции (только {len(df_pivot_corr)} периодов)")
            
            corr_matrix = df_pivot_corr.corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig_corr.update_layout(title='Матрица корреляции магазинов', height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # 3. СРАВНЕНИЕ МАГАЗИНОВ
        st.subheader("3️⃣ Сравнительная таблица магазинов")
        
        # ИСПРАВЛЕНИЕ: считаем количество транзакций для среднего чека
        magazin_stats = df_filtered.groupby('Magazin').agg({
            'Sum': ['sum', 'mean', 'std', 'count'],  # count = количество транзакций
            'Qty': 'sum'
        }).round(0)
        magazin_stats.columns = ['Общая сумма', 'Средняя за транзакцию', 'Ст. отклонение', 'Транзакций', 'Единиц продано']
        
        # Средний чек = общая сумма / количество транзакций (уже есть в 'Средняя за транзакцию')
        magazin_stats['Средний чек'] = magazin_stats['Средняя за транзакцию']
        magazin_stats['Единиц за транзакцию'] = (magazin_stats['Единиц продано'] / magazin_stats['Транзакций']).round(1)
        
        # НОВОЕ: Производительность на транзакцию
        magazin_stats = magazin_stats.sort_values('Общая сумма', ascending=False)
        
        st.dataframe(magazin_stats[['Общая сумма', 'Транзакций', 'Средний чек', 'Единиц за транзакцию']], use_container_width=True)
        
        # 4. СТРУКТУРА ПРОДАЖ МАГАЗИНОВ ПО СЕГМЕНТАМ
        st.subheader("4️⃣ Что продают магазины: структура по сегментам")
        
        for magazin in selected_magazins[:3]:
            magazin_segments = df_filtered[df_filtered['Magazin'] == magazin].groupby('Segment')['Sum'].sum()
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write(f"**{magazin}**")
                fig_pie = go.Figure(data=[go.Pie(
                    labels=magazin_segments.index,
                    values=magazin_segments.values,
                    hole=0.4
                )])
                fig_pie.update_layout(height=250, margin=dict(t=30, b=0, l=0, r=0))
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                segment_pct = (magazin_segments / magazin_segments.sum() * 100).round(1)
                segment_df = pd.DataFrame({
                    'Сегмент': segment_pct.index,
                    'Сумма': magazin_segments.values.astype(int),
                    'Доля %': segment_pct.values
                }).sort_values('Доля %', ascending=False)
                st.dataframe(segment_df, hide_index=True, use_container_width=True)
        
        # 5. РЕЙТИНГ МАГАЗИНОВ
        st.subheader("5️⃣ Рейтинг магазинов")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🏆 Топ по продажам**")
            top_magazins = magazin_stats.nlargest(10, 'Общая сумма')[['Общая сумма', 'Средний чек']]
            st.dataframe(top_magazins, use_container_width=True)
        
        with col2:
            st.write("**📊 Топ по количеству транзакций**")
            top_qty = magazin_stats.nlargest(10, 'Транзакций')[['Транзакций', 'Средний чек']]
            st.dataframe(top_qty, use_container_width=True)
        
        # НОВОЕ: Эффективность магазинов
        st.subheader("6️⃣ Эффективность магазинов")
        
        # Scatter plot: транзакции vs средний чек
        fig_efficiency = px.scatter(
            magazin_stats.reset_index(),
            x='Транзакций',
            y='Средний чек',
            size='Общая сумма',
            hover_name='Magazin',
            title='Эффективность: Объем vs Средний чек',
            labels={'Транзакций': 'Количество транзакций', 'Средний чек': 'Средний чек'},
            height=500
        )
        fig_efficiency.update_traces(marker=dict(sizemode='diameter'))
        st.plotly_chart(fig_efficiency, use_container_width=True)
        
        st.info("💡 Правый верхний угол = лидеры (много транзакций + высокий чек). Левый нижний = зона роста.")
        
        # НОВОЕ: Выводы и рекомендации по магазинам
        st.markdown("---")
        st.header("🎯 Выводы и рекомендации по магазинам")
        
        # ==================== ГЛУБОКИЙ АНАЛИЗ ====================
        
        # Базовые метрики
        total_magazins = len(magazin_stats)
        total_sales_mag = magazin_stats['Общая сумма'].sum()
        avg_check_overall = magazin_stats['Средний чек'].mean()
        avg_transactions = magazin_stats['Транзакций'].mean()
        
        # Топ и аутсайдеры
        top_magazin = magazin_stats.index[0]
        top_magazin_sales = magazin_stats.iloc[0]['Общая сумма']
        top_magazin_share = (top_magazin_sales / total_sales_mag * 100)
        
        bottom_magazins = magazin_stats.nsmallest(max(3, int(total_magazins * 0.2)), 'Общая сумма')
        
        # Анализ среднего чека
        high_check_stores = magazin_stats[magazin_stats['Средний чек'] > avg_check_overall * 1.2].sort_values('Средний чек', ascending=False)
        low_check_stores = magazin_stats[magazin_stats['Средний чек'] < avg_check_overall * 0.8].sort_values('Средний чек')
        
        # Анализ эффективности (продажи на транзакцию)
        magazin_stats['Эффективность'] = magazin_stats['Общая сумма'] / magazin_stats['Транзакций']
        high_efficiency = magazin_stats.nlargest(5, 'Эффективность')
        low_efficiency = magazin_stats.nsmallest(5, 'Эффективность')
        
        # ==================== ЭКСПРЕСС-ДИАГНОСТИКА ====================
        
        st.subheader("📊 Экспресс-диагностика сети магазинов")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Разброс по среднему чеку
        check_variance = (magazin_stats['Средний чек'].std() / avg_check_overall * 100)
        check_status = "🟢 Однородная сеть" if check_variance < 20 else ("🟡 Есть разброс" if check_variance < 40 else "🔴 Сильный разброс")
        with col1:
            st.metric("Разброс чека", f"{check_variance:.0f}%", check_status)
            st.caption("CV среднего чека")
        
        # Концентрация
        top_3_share = (magazin_stats.nlargest(3, 'Общая сумма')['Общая сумма'].sum() / total_sales_mag * 100)
        conc_status = "🟢 Распределено" if top_3_share < 40 else ("🟡 Умеренно" if top_3_share < 60 else "🔴 Концентрация")
        with col2:
            st.metric("Топ-3 магазина", f"{top_3_share:.0f}%", conc_status)
            st.caption("Доля в продажах")
        
        # Проблемные магазины
        problem_stores = len(low_check_stores) + len(bottom_magazins)
        problem_status = "🟢 Мало" if problem_stores <= total_magazins * 0.2 else ("🟡 Средне" if problem_stores <= total_magazins * 0.3 else "🔴 Много")
        with col3:
            st.metric("Слабых точек", f"{problem_stores}", problem_status)
            st.caption(f"Из {total_magazins} магазинов")
        
        # Средний чек vs топ
        if len(high_check_stores) > 0:
            best_check = high_check_stores.iloc[0]['Средний чек']
            check_gap = ((best_check / avg_check_overall - 1) * 100)
            gap_status = "🟢 Малый" if check_gap < 30 else ("🟡 Средний" if check_gap < 50 else "🔴 Большой")
        else:
            check_gap = 0
            gap_status = "🟡 Нет данных"
        
        with col4:
            st.metric("Разрыв с лучшим", f"+{check_gap:.0f}%", gap_status)
            st.caption("Потенциал роста")
        
        st.markdown("---")
        
        # ==================== ДЕТАЛЬНЫЙ АНАЛИЗ ====================
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ Лучшие практики")
            
            st.write(f"**1. Лидер продаж: {top_magazin}**")
            st.write(f"   💰 Продажи: {top_magazin_sales:,.0f} ({top_magazin_share:.1f}%)")
            st.write(f"   💳 Средний чек: {magazin_stats.loc[top_magazin, 'Средний чек']:,.0f}")
            st.write(f"   🛒 Транзакций: {magazin_stats.loc[top_magazin, 'Транзакций']:,.0f}")
            
            if len(high_check_stores) > 0:
                st.write(f"\n**2. Высокий средний чек** ({len(high_check_stores)} магазинов):")
                for i, store in enumerate(high_check_stores.head(3).index, 1):
                    check = high_check_stores.loc[store, 'Средний чек']
                    vs_avg = ((check / avg_check_overall - 1) * 100)
                    st.write(f"   {i}. **{store}**: {check:,.0f} (+{vs_avg:.0f}% к среднему)")
            
            if len(high_efficiency) > 0:
                st.write(f"\n**3. Эффективные магазины:**")
                for i, store in enumerate(high_efficiency.head(3).index, 1):
                    eff = high_efficiency.loc[store, 'Эффективность']
                    st.write(f"   {i}. **{store}**: {eff:,.0f} за транзакцию")
        
        with col2:
            st.subheader("⚠️ Точки роста")
            
            if len(low_check_stores) > 0:
                total_low_check_loss = sum([
                    (avg_check_overall - row['Средний чек']) * row['Транзакций']
                    for idx, row in low_check_stores.iterrows()
                ])
                
                st.write(f"**1. Низкий средний чек** ({len(low_check_stores)} магазинов):")
                for i, store in enumerate(low_check_stores.head(3).index, 1):
                    check = low_check_stores.loc[store, 'Средний чек']
                    transactions = low_check_stores.loc[store, 'Транзакций']
                    loss = (avg_check_overall - check) * transactions
                    st.write(f"   {i}. **{store}**: {check:,.0f} (💸 потеря ~{loss:,.0f})")
                st.write(f"   ⚡ Общая потенциальная потеря: **{total_low_check_loss:,.0f}**")
            
            if len(bottom_magazins) > 0:
                st.write(f"\n**2. Слабые по продажам** ({len(bottom_magazins)} магазинов):")
                for i, store in enumerate(bottom_magazins.index[:3], 1):
                    sales = bottom_magazins.loc[store, 'Общая сумма']
                    st.write(f"   {i}. **{store}**: {sales:,.0f}")
                st.write(f"   📊 Средний по сети: {magazin_stats['Общая сумма'].mean():,.0f}")
            
            if top_3_share > 50:
                st.write(f"\n**3. Концентрация продаж:**")
                st.write(f"   📊 Топ-3 = {top_3_share:.0f}% всех продаж")
                st.write(f"   ⚠️ Высокий риск зависимости")
        
        st.markdown("---")
        
        # ==================== ПРИОРИТИЗИРОВАННЫЕ РЕКОМЕНДАЦИИ ====================
        
        st.subheader("💡 Приоритизированный план действий")
        
        recommendations_mag = []
        
        # ПРИОРИТЕТ 1: Поднять средний чек в слабых магазинах
        if len(low_check_stores) > 0:
            total_low_check_potential = sum([
                (avg_check_overall - row['Средний чек']) * row['Транзакций'] * 0.5  # 50% от разрыва
                for idx, row in low_check_stores.iterrows()
            ])
            
            worst_store = low_check_stores.index[0]
            worst_check = low_check_stores.iloc[0]['Средний чек']
            worst_transactions = low_check_stores.iloc[0]['Транзакций']
            
            recommendations_mag.append({
                'priority': '🟢 БЫСТРАЯ ПОБЕДА',
                'title': f'Увеличить средний чек в слабых точках',
                'problem': f'{len(low_check_stores)} магазинов с чеком < {avg_check_overall * 0.8:,.0f} (на 20% ниже среднего)',
                'why': f'Потенциал: {total_low_check_potential:,.0f} при достижении среднего уровня',
                'action': [
                    f'1. Анализ лучших: изучить технику продаж в {high_check_stores.index[0]} (чек {high_check_stores.iloc[0]["Средний чек"]:,.0f})',
                    f'2. Обучение персонала: допродажи, cross-sell, up-sell',
                    f'3. Мотивация: премия за средний чек > {avg_check_overall:,.0f}',
                    f'4. Пилот в {worst_store}: комбо-предложения, "товар дня"',
                    '5. Мерчандайзинг: импульсные товары у кассы'
                ],
                'metric': f'Цель: поднять средний чек с {worst_check:,.0f} до {avg_check_overall:,.0f} за 2-3 месяца',
                'impact': 'Высокий',
                'effort': 'Низкий',
                'roi': f'Доп. выручка ~{total_low_check_potential:,.0f} при затратах на обучение ~{total_low_check_potential * 0.05:,.0f}'
            })
        
        # ПРИОРИТЕТ 2: Тиражирование лучших практик
        if len(high_check_stores) > 0:
            best_store = high_check_stores.index[0]
            best_check = high_check_stores.iloc[0]['Средний чек']
            
            # Потенциал если все магазины достигнут 80% от лучшего
            target_check = best_check * 0.8
            replication_potential = sum([
                max(0, target_check - row['Средний чек']) * row['Транзакций']
                for idx, row in magazin_stats.iterrows()
                if row['Средний чек'] < target_check
            ])
            
            recommendations_mag.append({
                'priority': '🟡 СТРАТЕГИЯ',
                'title': f'Тиражировать опыт лучших магазинов',
                'problem': f'{best_store} показывает чек {best_check:,.0f} (на {check_gap:.0f}% выше среднего)',
                'why': f'Если поднять все магазины до 80% от лучшего: потенциал {replication_potential:,.0f}',
                'action': [
                    f'1. Бенчмаркинг: выявить "секреты" {best_store}',
                    '2. Создать чек-лист успешных практик',
                    f'3. Стажировки персонала других магазинов в {best_store}',
                    '4. Видео-инструкции по лучшим техникам продаж',
                    '5. Ежемесячный конкурс магазинов по среднему чеку'
                ],
                'metric': f'Цель: 70% магазинов достигают чека > {target_check:,.0f} за полгода',
                'impact': 'Очень высокий',
                'effort': 'Средний',
                'roi': f'Потенциал {replication_potential:,.0f} (около {replication_potential/total_sales_mag*100:.0f}% от текущих продаж)'
            })
        
        # ПРИОРИТЕТ 3: Аудит и оптимизация слабых точек
        if len(bottom_magazins) > 0:
            bottom_total_sales = bottom_magazins['Общая сумма'].sum()
            bottom_share = (bottom_total_sales / total_sales_mag * 100)
            avg_magazin_sales = magazin_stats['Общая сумма'].mean()
            
            # Потенциал если слабые магазины достигнут 70% от среднего
            bottom_potential = sum([
                max(0, avg_magazin_sales * 0.7 - row['Общая сумма'])
                for idx, row in bottom_magazins.iterrows()
            ])
            
            recommendations_mag.append({
                'priority': '🔴 КРИТИЧНО',
                'title': f'Аудит слабых магазинов',
                'problem': f'{len(bottom_magazins)} магазинов в нижней части ({bottom_share:.0f}% продаж)',
                'why': f'Либо закрыть, либо исправить. Потенциал улучшения: {bottom_potential:,.0f}',
                'action': [
                    '1. Диагностика каждого: локация, трафик, конкуренты, персонал, ассортимент',
                    '2. План на 3 месяца: конкретные KPI для каждого магазина',
                    '3. Если локация плохая → рассмотреть переезд или закрытие',
                    '4. Если персонал слабый → замена или усиленное обучение',
                    '5. Если ассортимент не тот → адаптация под район'
                ],
                'metric': f'Цель: рост слабых точек на 30% за квартал ИЛИ закрытие убыточных',
                'impact': 'Высокий',
                'effort': 'Высокий',
                'roi': f'Либо +{bottom_potential:,.0f} выручки, либо экономия на убыточных точках'
            })
        
        # ПРИОРИТЕТ 4: Специализация магазинов
        magazin_specialization = df_filtered.groupby(['Magazin', 'Segment'])['Sum'].sum().reset_index()
        magazin_specialization = magazin_specialization.sort_values(['Magazin', 'Sum'], ascending=[True, False])
        top_segment_per_store = magazin_specialization.groupby('Magazin').first()
        
        # Находим магазины где топ-сегмент > 50%
        magazin_segment_share = magazin_specialization.pivot(index='Magazin', columns='Segment', values='Sum').fillna(0)
        magazin_segment_share_pct = magazin_segment_share.div(magazin_segment_share.sum(axis=1), axis=0) * 100
        
        specialized_stores = []
        for store in magazin_segment_share_pct.index:
            max_share = magazin_segment_share_pct.loc[store].max()
            if max_share > 50:
                top_seg = magazin_segment_share_pct.loc[store].idxmax()
                specialized_stores.append({'store': store, 'segment': top_seg, 'share': max_share})
        
        if len(specialized_stores) > 0:
            specialization_potential = sum([
                magazin_stats.loc[s['store'], 'Общая сумма'] * 0.15  # 15% рост за счет углубления специализации
                for s in specialized_stores
                if s['store'] in magazin_stats.index
            ])
            
            recommendations_mag.append({
                'priority': '🟠 ТАКТИКА',
                'title': f'Усилить специализацию магазинов',
                'problem': f'{len(specialized_stores)} магазинов уже специализированы (1 сегмент > 50%)',
                'why': f'Углубление специализации → экспертиза → +15% продаж = {specialization_potential:,.0f}',
                'action': [
                    '1. Идентифицировать профиль каждого магазина по топ-сегменту',
                    '2. Расширить ассортимент в профильном сегменте на 20-30%',
                    '3. Обучить персонал как экспертов в своем сегменте',
                    '4. Маркетинг: позиционировать магазин как специализированный',
                    '5. Примеры специализаций: "Магазин #1 по Премиальным товарам"'
                ],
                'metric': f'Цель: увеличить долю профильного сегмента с 50% до 60% за полгода',
                'impact': 'Средний',
                'effort': 'Средний',
                'roi': f'Потенциал {specialization_potential:,.0f} + повышение лояльности клиентов'
            })
        
        # ПРИОРИТЕТ 5: Конкуренция между магазинами
        if total_magazins >= 5:
            competition_potential = total_sales_mag * 0.08  # 8% рост за счет здоровой конкуренции
            
            recommendations_mag.append({
                'priority': '🟢 БЫСТРАЯ ПОБЕДА',
                'title': f'Запустить соревнование магазинов',
                'problem': f'Нет явной системы мотивации и сравнения {total_magazins} магазинов',
                'why': f'Здоровая конкуренция → рост 5-10% = потенциал {competition_potential:,.0f}',
                'action': [
                    '1. Создать публичный рейтинг магазинов (доска почета)',
                    '2. KPI: средний чек, количество транзакций, NPS, conversion',
                    '3. Ежемесячные призы: лучший магазин, лучший рост',
                    '4. Бонусы команде победителя',
                    '5. Ежеквартальный съезд: обмен опытом и награждение'
                ],
                'metric': f'Цель: минимум 50% магазинов улучшают показатели каждый месяц',
                'impact': 'Высокий',
                'effort': 'Низкий',
                'roi': f'Рост продаж ~{competition_potential:,.0f} при минимальных затратах на призы'
            })
        
        # Сортируем по приоритету
        priority_order = {'🔴 КРИТИЧНО': 1, '🟢 БЫСТРАЯ ПОБЕДА': 2, '🟠 ТАКТИКА': 3, '🟡 СТРАТЕГИЯ': 4}
        recommendations_mag.sort(key=lambda x: priority_order.get(x['priority'], 5))
        
        # Вывод рекомендаций
        for i, rec in enumerate(recommendations_mag, 1):
            with st.expander(f"**{rec['priority']} | #{i}: {rec['title']}**", expanded=i<=2):
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**📋 Ситуация:**")
                    st.write(rec['problem'])
                    
                    st.markdown(f"**🎯 Почему важно:**")
                    st.write(rec['why'])
                    
                    st.markdown(f"**⚡ План действий:**")
                    for action in rec['action']:
                        st.write(action)
                
                with col2:
                    st.markdown(f"**📊 Метрики успеха:**")
                    st.info(rec['metric'])
                    
                    st.markdown(f"**💼 Оценка:**")
                    st.write(f"Влияние: **{rec['impact']}**")
                    st.write(f"Усилия: **{rec['effort']}**")
                    
                    st.markdown(f"**💰 ROI:**")
                    st.success(rec['roi'])
        
        # ==================== ФИНАНСОВАЯ ОЦЕНКА ====================
        
        st.markdown("---")
        st.subheader("💰 Финансовая оценка потенциала по магазинам")
        
        # Считаем потенциалы
        check_potential = total_low_check_potential if 'total_low_check_potential' in locals() else 0
        replication_potential_val = replication_potential if 'replication_potential' in locals() else 0
        bottom_potential_val = bottom_potential if 'bottom_potential' in locals() else 0
        specialization_potential_val = specialization_potential if 'specialization_potential' in locals() else 0
        competition_potential_val = competition_potential if 'competition_potential' in locals() else 0
        
        total_mag_potential = check_potential + replication_potential_val * 0.5 + bottom_potential_val * 0.5 + specialization_potential_val + competition_potential_val
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "💳 Рост среднего чека",
                f"{check_potential:,.0f}",
                f"{check_potential/total_sales_mag*100:.1f}% от текущих продаж"
            )
        
        with col2:
            st.metric(
                "🏆 Тиражирование + аудит",
                f"{(replication_potential_val * 0.5 + bottom_potential_val * 0.5):,.0f}",
                f"{(replication_potential_val * 0.5 + bottom_potential_val * 0.5)/total_sales_mag*100:.1f}% от текущих продаж"
            )
        
        with col3:
            st.metric(
                "🎯 Специализация + мотивация",
                f"{specialization_potential_val + competition_potential_val:,.0f}",
                f"{(specialization_potential_val + competition_potential_val)/total_sales_mag*100:.1f}% от текущих продаж"
            )
        
        st.success(f"**🎯 При реализации всех рекомендаций прогнозируемый рост выручки: {total_mag_potential:,.0f} (+{total_mag_potential/total_sales_mag*100:.1f}%)**")
        
        # ==================== ИТОГОВАЯ МАТРИЦА ПРИОРИТЕТОВ ====================
        
        st.markdown("---")
        st.subheader("📋 Матрица приоритетов: с чего начать")
        
        priority_matrix = pd.DataFrame({
            'Рекомендация': [rec['title'] for rec in recommendations_mag],
            'Приоритет': [rec['priority'] for rec in recommendations_mag],
            'Влияние': [rec['impact'] for rec in recommendations_mag],
            'Усилия': [rec['effort'] for rec in recommendations_mag],
            'Сроки': ['1 месяц' if 'БЫСТРАЯ' in rec['priority'] else ('3 месяца' if 'КРИТИЧНО' in rec['priority'] or 'ТАКТИКА' in rec['priority'] else '6 месяцев') for rec in recommendations_mag]
        })
        
        st.dataframe(priority_matrix, hide_index=True, use_container_width=True)
        
        st.info("💡 **Рекомендуемый порядок внедрения:** 1) 🔴 Критично → 2) 🟢 Быстрые победы → 3) 🟠 Тактика → 4) 🟡 Стратегия. Начните с первых 2-3 инициатив.")

else:
    st.info("👆 Загрузите Excel файл для начала анализа")
    st.markdown("""
    ### Что анализирует приложение:
    
    **По сегментам:**
    - Динамика продаж каждого сегмента
    - Корреляция между сегментами
    - Сезонность и индексы
    - ABC-анализ
    - Структура и тренды
    - **Выводы и рекомендации**
    
    **По магазинам:**
    - Динамика и корреляция
    - Сравнительная аналитика
    - Эффективность магазинов
    - Специализация по сегментам
    - Рейтинги
    - **Выводы и рекомендации**
    """)
