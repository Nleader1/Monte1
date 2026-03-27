import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Настройка страницы
st.set_page_config(page_title="MC Equity Sim", layout="wide")

# --- СТИЛИЗАЦИЯ (Темная тема) ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stSidebar { background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

# --- БОКОВАЯ ПАНЕЛЬ (Параметры со скриншота) ---
with st.sidebar:
    st.title("MC Equity Sim")
    
    st.header("UPLOAD CSV FILES")
    uploaded_file = st.file_uploader("Drag & drop CSV files here", type="csv")
    
    st.header("TRADE COSTS")
    col1, col2 = st.columns(2)
    comm = col1.number_input("Commission ($)", value=6.0)
    slip = col2.number_input("Slippage (pips)", value=0.2)
    avg_sl = col1.number_input("Avg SL (pips)", value=14.1)
    fixed_risk = col2.number_input("Fixed Risk ($)", value=1000)

    st.header("MONTE CARLO")
    n_sims = st.number_input("Simulations", value=2000)
    horizon = st.number_input("Horizon (trades)", value=150)
    std_dev_type = st.radio("Std Dev divisor", ["N-1 (sample)", "N (population)"])

    st.header("STRESS FACTORS")
    r_jitter = st.number_input("R Jitter (±)", value=0.1)
    bad_slip_prob = st.number_input("Bad slip prob", value=0.05)
    bad_slip_mult = st.number_input("Bad slip mult", value=1.5)
    missed_win = st.number_input("Missed win prob", value=0.05)
    human_error = st.number_input("Human error %", value=0.0)

# --- ЛОГИКА СИМУЛЯЦИИ ---
def run_simulation(data, n_sims, horizon):
    # Преобразуем PnL в R-multiple (кратность риску) для гибкости
    returns = data['PNL'].values / fixed_risk
    
    all_runs = []
    for _ in range(n_sims):
        # Случайная выборка сделок
        sim_returns = np.random.choice(returns, size=horizon, replace=True)
        
        # Применяем стресс-факторы (упрощенно)
        # 1. R-Jitter (шум в результатах)
        sim_returns += np.random.uniform(-r_jitter, r_jitter, size=horizon)
        
        # 2. Пропуск прибыльных сделок
        mask_missed = (sim_returns > 0) & (np.random.rand(horizon) < missed_win)
        sim_returns[mask_missed] = 0
        
        # Считаем накопленную кривую
        equity_curve = np.cumsum(sim_returns * fixed_risk)
        all_runs.append(equity_curve)
        
    return np.array(all_runs)

# --- ОСНОВНОЙ ЭКРАН ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if 'PNL' in df.columns:
        results = run_simulation(df, n_sims, horizon)
        
        # Визуализация
        fig = go.Figure()
        x_axis = np.arange(horizon)
        
        # Рисуем первые 100 линий для производительности
        for i in range(min(n_sims, 100)):
            fig.add_trace(go.Scatter(y=results[i], mode='lines', 
                         line=dict(width=0.5, color='rgba(100, 150, 255, 0.3)'),
                         showlegend=False))

        # Средняя линия
        mean_curve = np.mean(results, axis=0)
        fig.add_trace(go.Scatter(y=mean_curve, mode='lines', 
                     line=dict(color='yellow', width=2), name="Mean Equity"))

        fig.update_layout(title="Monte Carlo Equity Projection",
                          template="plotly_dark",
                          xaxis_title="Trades",
                          yaxis_title="Profit ($)")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Расчет SQN (Van Tharp)
        # Formula: SQN = (Average R / StdDev R) * sqrt(N)
        avg_r = np.mean(df['PNL'] / fixed_risk)
        std_r = np.std(df['PNL'] / fixed_risk)
        sqn = (avg_r / std_r) * np.sqrt(min(len(df), horizon))
        
        st.metric("Historical SQN", f"{sqn:.2f}")
    else:
        st.error("CSV должен содержать колонку 'PNL'")
else:
    st.info("Загрузите CSV файл с колонкой 'PNL', чтобы начать симуляцию.")
