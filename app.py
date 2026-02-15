import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import google.generativeai as genai
import datetime

# ==========================================
# 🔧 系統設定與狀態初始化
# ==========================================
st.set_page_config(page_title="Gemini 2.5 Pro 戰情室", page_icon="📈", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "data_context" not in st.session_state:
    st.session_state.data_context = None

# ==========================================
# 📊 進階量化計算與回測
# ==========================================
def calculate_advanced_metrics(df, log_df):
    """計算夏普比率與最大回撤"""
    if log_df.empty:
        return 0, 0
    
    # 簡單 Sharpe Ratio 估算 (年化)
    returns = log_df['獲利%'] / 100
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 1 else 0
    
    # 最大回撤 (MDD)
    cum_rets = (1 + returns).cumprod()
    peak = cum_rets.cummax()
    mdd = ((cum_rets - peak) / peak).min() * 100
    
    return round(sharpe, 2), round(mdd, 2)

@st.cache_data(ttl=300)
def get_data_engine(symbol):
    symbol = symbol.strip().upper()
    if symbol.isdigit(): symbol = f"{symbol}.TW"
    elif not any(s in symbol for s in [".TW", ".TWO", ".HK", ".US", ".SS", ".SZ"]):
        if not (symbol.isalpha() and len(symbol) <= 4): symbol = f"{symbol}.TW"
    
    try:
        t = yf.Ticker(symbol)
        df = t.history(period="2y")
        if df.empty: return None, None, symbol, "查無數據"
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # 指標補完
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['DIF'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['MACD'] = df['DIF'].ewm(span=9).mean()
        df['OSC'] = df['DIF'] - df['MACD']
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df.dropna(), t.info.get('longName', symbol), symbol, None
    except Exception as e:
        return None, None, symbol, str(e)

# ==========================================
# 🤖 Gemini 2.5 Flash 智能核心 (優先調用)
# ==========================================
def get_gemini_25_response(api_key, messages_history):
    genai.configure(api_key=api_key)
    
    # 強制優先級：2.5 Flash > 2.0 Flash > 1.5 Pro
    priority_models = [
        "gemini-2.5-flash", 
        "gemini-2.0-flash", 
        "gemini-1.5-pro"
    ]
    
    # 偵測可用模型
    available = [m.name.replace("models/", "") for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    target_model = next((m for m in priority_models if m in available), "gemini-1.5-flash")

    sys_instruction = f"""
    現在是 {datetime.datetime.now().strftime("%Y-%m-%d")}。你是配備 Gemini 2.5 Flash 的頂級對沖基金經理。
    
    【能力】
    1. 使用 Google Search 抓取今日即時財報、法說會訊息與國際局勢。
    2. 使用 GEM (Global Economic Monitor) 框架：
       - **🌊 資金浪潮**：分析籌碼面與市場流動性。
       - **🌋 結構風險**：利用空方視角指出潛在技術崩潰點。
       - **🎭 莊家博弈**：拆解大戶洗盤或誘多陷阱。
       - **🏆 最終勝算**：綜合量化得分，給出「投資建議評等」。
    """

    try:
        model = genai.GenerativeModel(
            model_name=target_model,
            tools=[{"google_search_retrieval": {}}],
            system_instruction=sys_instruction
        )
        
        # 轉換歷史訊息格式
        history = []
        for m in messages_history[:-1]:
            role = "user" if m["role"] == "user" else "model"
            history.append({"role": role, "parts": [m["content"]]})
        
        chat = model.start_chat(history=history)
        # Gemini 2.5 Flash 回應
        response = chat.send_message(messages_history[-1]["content"])
        return response.text, target_model
    except Exception as e:
        return f"❌ AI 服務異常: {str(e)}", "N/A"

# ==========================================
# 🖥️ UI 介面
# ==========================================
with st.sidebar:
    st.title("🏦 智庫控制中心")
    key = st.text_input("Gemini API Key", type="password")
    ticker = st.text_input("股票代號", value="AAPL")
    scan_btn = st.button("啟動全數據掃描", type="primary", use_container_width=True)
    if st.button("重置對話"): st.session_state.messages = []; st.rerun()

if scan_btn and key:
    with st.spinner("🚀 Gemini 2.5 正在調閱全球數據與回測分析..."):
        df, name, sid, err = get_data_engine(ticker)
        if df is not None:
            # 量化評分與回測 (沿用你的邏輯但補強)
            from __main__ import detailed_scoring, comprehensive_backtest # 確保能抓到下方定義
            score, score_df = detailed_scoring(df)
            bt_log = comprehensive_backtest(df)
            sharpe, mdd = calculate_advanced_metrics(df, bt_log)
            
            # 存入環境
            st.session_state.data_context = {
                "df": df, "name": name, "sid": sid, "score": score, 
                "score_df": score_df, "bt_log": bt_log, "sharpe": sharpe, "mdd": mdd
            }
            
            # 初始 Prompt
            prompt = f"分析 {name} ({sid})。評分:{score}/10, 夏普值:{sharpe}, 最大回撤:{mdd}%。請聯網檢索今日最新重大新聞。"
            ai_resp, used_model = get_gemini_25_response(key, [{"role": "user", "content": prompt}])
            
            st.session_state.messages = [
                {"role": "user", "content": f"啟動 {sid} 深度掃描報告"},
                {"role": "assistant", "content": ai_resp}
            ]
            st.session_state.used_model = used_model
        else:
            st.error(err)

# 顯示看板
if st.session_state.data_context:
    ctx = st.session_state.data_context
    st.header(f"📊 {ctx['name']} ({ctx['sid']})")
    
    # 頂部指標
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("動能評分", f"{ctx['score']} / 10")
    m2.metric("夏普比率 (Sharpe)", ctx['sharpe'])
    m3.metric("最大回撤 (MDD)", f"{ctx['mdd']}%")
    m4.metric("使用模型", st.session_state.used_model)

    # 圖表區
    st.line_chart(ctx['df'][['Close', 'MA20']].tail(120))
    
    with st.expander("📝 查看詳細指標與回測對帳單"):
        c_a, c_b = st.columns(2)
        c_a.table(ctx['score_df'])
        c_b.dataframe(ctx['bt_log'])

    st.divider()
    
    # 對話區
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if query := st.chat_input("詢問 AI 經理人..."):
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("智庫辯證中..."):
                resp, _ = get_gemini_25_response(key, st.session_state.messages)
                st.markdown(resp)
                st.session_state.messages.append({"role": "assistant", "content": resp})

# 原有函數需放在同一文件或 import
def slope(series, n=3):
    y = series.tail(n).dropna()
    if len(y) < n: return 0
    return np.polyfit(np.arange(len(y)), y, 1)[0]

def detailed_scoring(df):
    r = df.iloc[-1]
    details = []
    total_score = 0
    checks = [
        (r['MA5'] > r['MA10'] > r['MA20'], 3, "均線多頭", "MA5>10>20"),
        (slope(df['DIF']) > 0 and r['OSC'] > 0, 2, "MACD轉強", "DIF斜率>0"),
        (r['Close'] > r['MA20'], 1, "站上月線", "Close > MA20"),
        (r['Volume'] > df['Volume'].tail(5).mean(), 1, "量能增溫", "Vol > 5MA")
    ]
    for cond, pts, rule, desc in checks:
        s = pts if cond else 0
        details.append({"準則": rule, "得分": s, "狀態": "✅" if cond else "❌"})
        total_score += s
    return total_score, pd.DataFrame(details)

def comprehensive_backtest(df):
    log = []
    holding = False; entry_p = 0; entry_d = None
    for i in range(20, len(df)):
        r = df.iloc[i]; prev = df.iloc[i-1]
        if not holding and r['Close'] > r['MA20'] and r['OSC'] > 0:
            holding = True; entry_p = r['Close']; entry_d = df.index[i]
        elif holding and (r['Close'] < r['MA20'] or r['RSI'] > 85):
            log.append({"進場": entry_d.date(), "出場": df.index[i].date(), "買入": entry_p, "賣出": r['Close'], "獲利%": (r['Close']-entry_p)/entry_p*100})
            holding = False
    return pd.DataFrame(log)
