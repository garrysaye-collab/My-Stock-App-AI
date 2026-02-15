import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import google.generativeai as genai
from duckduckgo_search import DDGS
import time

# ==========================================
# 🔧 核心設定
# ==========================================
st.set_page_config(page_title="股市動能掃描 AI (經理人版)", page_icon="📈", layout="wide")

# 初始化 Session State 用於對話紀錄
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 🌐 聯網搜尋與 AI 分析 (大腦)
# ==========================================
def search_latest_news(ticker_name):
    """自主搜尋最新股息、PE與重大新聞"""
    try:
        with DDGS() as ddgs:
            # 針對股息、PE、新聞進行三位一體搜尋
            query = f"{ticker_name} dividend yield PE ratio news 2026"
            results = [f"內容: {r['body']}" for r in ddgs.text(query, max_results=6)]
        return "\n".join(results)
    except:
        return "暫時無法取得即時聯網數據，將以基本面資料進行分析。"

def chat_with_manager(api_key, user_input, stock_data, backtest_log, search_news):
    if not api_key: return "⚠️ 請在左側輸入 Google API Key 以啟動 AI 經理人。"
    
    genai.configure(api_key=api_key)
    
    # 這裡就是您要求的【性格與流程】設定
    system_instruction = """
    你是一群專業股票基金經理人。你擁有查閱即時資訊、新聞、財報與宏觀經濟的權限。
    你的任務是對用戶提供的量化回測數據進行「二次審計」。
    
    【性格與流程】
    1. 獨立辯證：用戶提供的回測數據僅是參考。若數據差，主動從即時新聞中尋找『護城河』或『高配息』等轉機。
    2. 兩方對立：對話中必須包含「多方(基本面)」與「空方(籌碼/技術面)」的激烈碰撞。
    3. 暗黑兵法：莊家團隊須以寓言方式揭示市場陷阱（例如：目前的止損訊號是否是莊家在收割散戶）。
    4. 巴菲特裁定：最後由巴菲特總結，決定是否參與並預估效益。
    
    請務必引用搜尋到的真實數字（股息率、PE、新聞日期）來說話。
    """
    
    full_prompt = f"""
    標的：{user_input}
    量化指標：{stock_data}
    回測紀錄：{backtest_log}
    即時聯網資訊：{search_news}
    
    請開始你們經理人團隊的辯證。
    """

    try:
        # 使用您帳號中可用的最新模型
        model = genai.GenerativeModel("models/gemini-2.5-flash", system_instruction=system_instruction)
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"AI 經理人離線中: {str(e)}"

# ==========================================
# 📊 量化回測邏輯 (底層)
# ==========================================
@st.cache_data(ttl=300)
def get_data_and_analyze(stock_id):
    stock_id = stock_id.strip().upper()
    if stock_id.isdigit(): stock_id = f"{stock_id}.TW"
    elif not any(suffix in stock_id for suffix in [".TW", ".TWO", ".HK", ".US", ".SS", ".SZ"]):
        if not (stock_id.isalpha() and len(stock_id) <= 4): stock_id = f"{stock_id}.TW"
    try:
        df = yf.download(stock_id, period="1y", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.empty: return None, stock_id, "查無資料"
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        # 簡易 RSI 計算
        delta = df['Close'].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean(); rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        return df.dropna(), stock_id, None
    except Exception as e: return None, stock_id, str(e)

def run_backtest(df):
    log = []
    holding = False; entry_price = 0; total_ret = 0
    for i in range(1, len(df)):
        r = df.iloc[i]; prev = df.iloc[i-1]
        if not holding and r['Close'] > r['MA20'] and r['Close'] > prev['High']:
            holding = True; entry_price = r['Close']
        elif holding and (r['Close'] < r['MA20'] or r['RSI'] > 80):
            holding = False; p = (r['Close'] - entry_price) / entry_price * 100
            total_ret += p
            log.append({"日期": df.index[i].strftime('%Y-%m-%d'), "獲利%": round(p, 2)})
    return log, round(total_ret, 2)

# ==========================================
# 🖥️ UI 介面
# ==========================================
st.title("📡 經理人級別：股市動能戰情室")

with st.sidebar:
    st.header("🔑 權限驗證")
    api_key = st.text_input("輸入 Google API Key", type="password")
    ticker = st.text_input("輸入標的 (如 2330, NVDA)", value="2330")
    run_btn = st.button("啟動專業分析", type="primary")

if run_btn:
    with st.spinner("經理人正在查閱即時財報與新聞..."):
        # 1. 量化回測
        df, real_id, err = get_data_and_analyze(ticker)
        if df is not None:
            backtest_log, total_ret = run_backtest(df)
            latest = df.iloc[-1]
            stock_info = f"價格: {latest['Close']:.2f}, RSI: {latest['RSI']:.2f}, MA20: {latest['MA20']:.2f}"
            
            # 2. 聯網搜尋 (經理人權限)
            news_context = search_latest_news(ticker)
            
            # 3. 儀表板展示
            c1, c2 = st.columns(2)
            c1.metric("量化回測累計報酬", f"{total_ret}%")
            c2.info(f"當前標的: {real_id}")
            
            # 4. AI 經理人辯證 (核心)
            st.divider()
            st.subheader("🕵️ 經理人團隊辯證報告")
            
            analysis_report = chat_with_manager(api_key, real_id, stock_info, backtest_log, news_context)
            st.markdown(analysis_report)
            
            # 保存至對話紀錄
            st.session_state.messages.append({"role": "assistant", "content": analysis_report})
            
            st.line_chart(df['Close'])
        else:
            st.error(err)

# 追問功能
if st.session_state.messages:
    if prompt := st.chat_input("對經理人團隊進一步質詢..."):
        st.chat_message("user").write(prompt)
        with st.spinner("團隊討論中..."):
            # 這裡簡單簡化，實際可帶入更多上下文
            res = chat_with_manager(api_key, prompt, "續前數據", "續前紀錄", "重新搜尋中...")
            st.chat_message("assistant").write(res)
