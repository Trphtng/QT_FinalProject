import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import time
import os

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="PPO Portfolio Allocation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD HOẶC TẠO DỮ LIỆU ---
@st.cache_data
def load_data():
    file_path = "weights_history.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        # Tạo dữ liệu giả lập (dummy data) nếu không tìm thấy file CSV
        np.random.seed(42)
        dates = pd.date_range(end=pd.Timestamp.today(), periods=50)
        data = []
        
        # Tạo hiệu ứng chuyển động mượt mà (random walk)
        weights = np.random.dirichlet(np.ones(5))
        for d in dates:
            noise = np.random.normal(0, 0.05, 5)
            weights = weights + noise
            weights = np.clip(weights, 0.01, 1) # Không để tỷ trọng < 1% để chart đẹp
            weights = weights / weights.sum()   # Chuẩn hóa tổng = 1
            data.append([d.strftime("%Y-%m-%d")] + list(weights))
            
        df = pd.DataFrame(data, columns=['Date', 'AAPL', 'MSFT', 'TSLA', 'AMZN', 'CASH'])
    return df

df = load_data()
assets = [col for col in df.columns if col != 'Date']

# Xác định cột tiền mặt (nếu có)
cash_col = next((col for col in assets if col.upper() == 'CASH'), None)

# --- QUẢN LÝ TRẠNG THÁI (SESSION STATE) ---
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False
if "step" not in st.session_state:
    st.session_state.step = 0

def play():
    st.session_state.is_playing = True

def pause():
    st.session_state.is_playing = False

# --- SIDEBAR: ĐIỀU KHIỂN ---
st.sidebar.title("🎮 Controls")
st.sidebar.markdown("Điều khiển timeline để xem mô hình thay đổi phân bổ vốn.")

col1, col2 = st.sidebar.columns(2)
with col1:
    st.button("▶️ Play", on_click=play, use_container_width=True)
with col2:
    st.button("⏸️ Pause", on_click=pause, use_container_width=True)

# Slider chọn ngày
step = st.sidebar.slider(
    "Thời gian (Index/Ngày)", 
    0, len(df) - 1, 
    key="step"
)

current_date = df.iloc[step]['Date']
st.sidebar.success(f"**📅 Ngày đang chọn:** {current_date}")
st.sidebar.markdown("---")
st.sidebar.caption("Dashboard mô phỏng cách PPO Actor-Critic phân bổ vốn tự động.")

# --- MAIN CONTENT ---
st.title("📈 PPO Actor-Critic: Portfolio Allocation Dashboard")
st.markdown("Theo dõi cách mô hình **Deep Reinforcement Learning** thay đổi chiến lược đầu tư và quản lý rủi ro qua từng giai đoạn.")

# Lấy dữ liệu của ngày hiện tại
current_weights = df.iloc[step][assets]
highest_asset = current_weights.idxmax()
highest_weight = current_weights.max()
cash_weight = current_weights.get(cash_col, 0) if cash_col else 0

# Tính số lượng tài sản thực sự được đầu tư (Tỷ trọng > 1% và không phải CASH)
invested_assets = sum((current_weights > 0.01) & (current_weights.index != cash_col))

# --- KPI / BONUS METRICS ---
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric(
    "🏆 Cổ phiếu ưu tiên nhất", 
    f"{highest_asset}", 
    f"{highest_weight:.1%} vốn"
)
kpi2.metric(
    "💵 Tỷ trọng Tiền mặt (CASH)", 
    f"{cash_weight:.1%}", 
    "Phòng thủ cao" if cash_weight > 0.3 else "Bình thường"
)
kpi3.metric(
    "📊 Số mã đang đầu tư", 
    f"{invested_assets} / {len(assets) - (1 if cash_col else 0)} mã",
    "Đa dạng hóa" if invested_assets > 2 else "Tập trung"
)

st.divider()

# --- INSIGHTS ĐƠN GIẢN ---
def get_insight(weights):
    insights = []
    max_a = weights.idxmax()
    cash_w = weights.get(cash_col, 0) if cash_col else 0
    
    # Logic sinh insight
    if cash_col and cash_w > 0.4:
        insights.append("🛡️ **Cash weight tăng cao:** Mô hình đang ở trạng thái phòng thủ (Bearish), hạn chế rủi ro.")
    elif cash_col and cash_w < 0.1:
        insights.append("⚔️ **Cash weight thấp:** Mô hình tự tin giải ngân tối đa vào thị trường (Bullish).")
        
    if weights[max_a] > 0.5 and max_a != cash_col:
        insights.append(f"🎯 **Tập trung cao độ:** Mô hình đang đánh giá cao và Overweight mã **{max_a}**.")
        
    std_dev = np.std(weights)
    if std_dev < 0.15:
        insights.append("⚖️ **Phân bổ đều:** Danh mục đang được đa dạng hóa đồng đều giữa các tài sản.")
        
    if not insights:
        insights.append("🔄 Mô hình đang duy trì trạng thái phân bổ cơ bản, không có đột biến.")
        
    return " | ".join(insights)

st.info(f"🧠 **Model Insight:** {get_insight(current_weights)}")

# --- BIỂU ĐỒ (CHARTS) ---
col_pie, col_bar = st.columns(2)

plot_df = pd.DataFrame({
    'Tài sản': current_weights.index,
    'Tỷ trọng': current_weights.values
})

# Bảng màu đẹp cho biểu đồ
custom_colors = px.colors.qualitative.Pastel

with col_pie:
    st.subheader("Trạng thái danh mục (Pie Chart)")
    fig_pie = px.pie(
        plot_df, 
        values='Tỷ trọng', 
        names='Tài sản', 
        hole=0.45, # Donut chart nhìn hiện đại hơn
        color='Tài sản',
        color_discrete_sequence=custom_colors
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=False
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_bar:
    st.subheader("Phân bổ chi tiết (Bar Chart)")
    fig_bar = px.bar(
        plot_df, 
        x='Tài sản', 
        y='Tỷ trọng', 
        color='Tài sản',
        color_discrete_sequence=custom_colors,
        text_auto='.1%'
    )
    fig_bar.update_layout(
        yaxis=dict(tickformat=".0%", range=[0, 1]), 
        showlegend=False, 
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# --- BẢNG SỐ LIỆU ---
with st.expander("📋 Xem bảng tỷ trọng chi tiết", expanded=True):
    table_df = plot_df.set_index('Tài sản').T
    st.dataframe(table_df.style.format("{:.2%}"), use_container_width=True)

# --- VÒNG LẶP ANIMATION ---
# Nếu đang ở trạng thái play, tiến lên 1 bước và tự động rerun lại ứng dụng
if st.session_state.is_playing:
    if st.session_state.step < len(df) - 1:
        time.sleep(0.5) # Tốc độ chạy (0.5 giây/frame)
        st.session_state.step += 1
        st.rerun()
    else:
        st.session_state.is_playing = False # Dừng lại khi đến cuối
