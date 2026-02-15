import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import google.generativeai as genai
import datetime

# ==========================================
# 🔧 1. 系統設定與狀態初始化
# ==========================================
st.set_page_config(page_title="Gemini 2.5 智庫戰情室", page_icon="📈", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "data_context" not in st.session_state:
    st.session_state.data_context = None

# ==========================================
# 📊 2. 量化核心函數 (優先定義，解決 Import 錯誤)
# ==========================================
def slope(series, n=3):
    """計算指標斜率"""
    y = series.tail(n).dropna()
    if len(y) < n: return 0
    return np.polyfit(np.arange(len(y)), y, 1)[0]

def detailed_scoring(df):
    """量化動能評分系統"""
    r = df.iloc[-1]
    details = []
    total_score = 0
    
    # 評分邏輯
    checks = [
        (r['MA5'] > r['MA10'] > r['MA20'], 3, "均線多頭排列", "MA 5>10>20"),
        (slope(df['DIF']) > 0 and r['OSC'] > 0, 2, "MACD 能量轉強", "DIF斜率>0"),
        (r['Close'] > r['MA20'], 1, "站上月線關鍵位", "Close > MA20"),
        (r['Volume'] > df['Volume'].tail(5).mean(), 1, "成交量增溫", "Vol > 5MA")
    ]
    
    for cond, pts, rule, desc in checks:
        s = pts if cond else 0
        details.append({"準則": rule, "得分": s, "狀態": "✅" if cond else "❌"})
        total_score += s
        
    return total_score, pd.DataFrame(details)

def comprehensive_backtest(df):
    """回測邏輯 (2年數據)"""
    log = []
    holding = False; entry_p = 0; entry_d = None
    
    # 從第 20 天開始模擬 (預留 MA20 計算空間)
    for i in range(20, len(df)):
        r = df.iloc[i]
        
        # 進場條件：站上月線 + MACD 紅柱
        if not holding and r['Close'] > r['MA20'] and r['OSC'] > 0:
            holding = True; entry_p = r['Close']; entry_d = df.index[i]
            
        # 出場條件：跌破月線 或 RSI 過熱(>85)
        elif holding and (r['Close'] < r['MA20'] or r['RSI'] > 85):
            log.append({
                "進場日期": entry_d.date(), 
                "出場日期": df.index[i].date(), 
                "買入價": round(entry_p, 2), 
                "賣出價": round(r['Close'], 2), 
                "獲利%": round((r['Close']-entry_p)/entry_p*100, 2)
            })
            holding = False
            
    return pd.DataFrame(log)

def calculate_advanced_metrics(log_df):
    """計算夏普值與最大回撤"""
    if log_df.empty: return 0, 0
    
    # 年化夏普值 (簡易估算)
    returns = log_df['獲利%'] / 100
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 1 and returns.std() != 0 else 0
    
    # 最大回撤 (MDD)
    cum_rets = (1 + returns).cumprod()
    if cum_rets.empty: return 0, 0
    mdd = ((cum_rets - cum_rets.cummax()) / cum_rets.cummax()).min() * 100
    
    return round(sharpe, 2), round(mdd, 2)

# ==========================================
# 🕸️ 3. 數據引擎
# ==========================================
@st.cache_data(ttl=300)
def get_data_engine(symbol):
    """數據獲取與指標預算"""
    symbol = symbol.strip().upper()
    # 簡易代號處理
    if symbol.isdigit(): symbol = f"{symbol}.TW"
    elif not any(s in symbol for s in [".TW", ".TWO", ".HK", ".US", ".SS", ".SZ"]):
        if not (symbol.isalpha() and len(symbol) <= 4): symbol = f"{symbol}.TW"
    
    try:
        t = yf.Ticker(symbol)
        df = t.history(period="2y")
        if df.empty: return None, None, symbol, "查無數據"
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # 計算技術指標
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
# 🤖 4. AI 智能核心 (Gemini 2.5 Flash 優先)
# ==========================================
def get_gemini_25_response(api_key, messages_history):
    genai.configure(api_key=api_key)
    
    # ✅ 優先級設定：2.5 Flash 第一
    priority_models = [
        "gemini-2.5-flash", 
        "gemini-2.0-flash", 
        "gemini-1.5-pro"
    ]
    
    # 嘗試獲取可用模型
    try:
        available = [m.name.replace("models/", "") for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        target_model = next((m for m in priority_models if m in available), "gemini-1.5-flash")
    except:
        target_model = "gemini-2.5-flash" # Fallback

    sys_instruction = f"""
    現在是 {datetime.datetime.now().strftime("%Y-%m-%d")}。你是配備 Gemini 2.5 Flash 的頂級對沖基金經理。
    
    【核心任務】
    1. **即時資訊**：使用 Google Search 工具檢索該公司「今日/本週」的最新財報、法說會與新聞。
    2. **GEM 分析框架**：
       - **🌊 資金浪潮**：分析外資與主力動向。
       - **🌋 結構風險**：利用空方視角指出潛在崩盤點。
       - **🎭 莊家博弈**：拆解大戶洗盤或誘多陷阱。
       - **🏆 最終勝算**：綜合量化得分，給出明確的「投資建議評等」(買進/持有/賣出)。
    """

    try:
        model = genai.GenerativeModel(
            model_name=target_model,
            tools=[{"google_search_retrieval": {}}],
            system_instruction=sys_instruction
        )
        
        # 轉換歷史格式
        history = []
        for m in messages_history[:-1]:
            role = "user" if m["role"] == "user" else "model"
            history.append({"role": role, "parts": [m["content"]]})
        
        chat = model.start_chat(history=history)
        response = chat.send_message(messages_history[-1]["content"])
        return response.text, target_model
    except Exception as e:
        return f"❌ AI 服務異常: {str(e)}", "N/A"

# ==========================================
# 🖥️ 5. UI 介面與主程序
# ==========================================
with st.sidebar:
    st.title("🏦 智庫控制中心")
    st.caption("Powered by Gemini 2.5 Flash")
    key = st.text_input("Gemini API Key", type="password")
    ticker = st.text_input("股票代號", value="2330")
    scan_btn = st.button("🚀 啟動深度掃描", type="primary", use_container_width=True)
    if st.button("🗑️ 清除對話"): st.session_state.messages = []; st.rerun()

if scan_btn and key:
    with st.spinner("🚀 Gemini 2.5 正在調閱全球數據與回測分析..."):
        df, name, sid, err = get_data_engine(ticker)
        
        if df is not None:
            # ✅ 直接調用已定義的函數
            score, score_df = detailed_scoring(df)
            bt_log = comprehensive_backtest(df)
            sharpe, mdd = calculate_advanced_metrics(bt_log)
            
            # 存入 Session Context
            st.session_state.data_context = {
                "df": df, "name": name, "sid": sid, "score": score, 
                "score_df": score_df, "bt_log": bt_log, "sharpe": sharpe, "mdd": mdd
            }
            
            # 生成 AI Prompt
            prompt = (
                f"分析 {name} ({sid})。目前技術面評分:{score}/10, "
                f"過去兩年趨勢策略回測夏普值:{sharpe}, 最大回撤:{mdd}%。"
                f"請聯網搜尋今日最新重大新聞，並進行多空辯證分析。"
            )
            
            ai_resp, used_model = get_gemini_25_response(key, [{"role": "user", "content": prompt}])
            
            st.session_state.messages = [
                {"role": "user", "content": f"啟動 {sid} 深度掃描報告"},
                {"role": "assistant", "content": ai_resp}
            ]
            st.session_state.used_model = used_model
            st.rerun() # 強制刷新以顯示結果
        else:
            st.error(err)

# 顯示結果看板
if st.session_state.data_context:
    ctx = st.session_state.data_context
    st.header(f"📊 {ctx['name']} ({ctx['sid']})")
    
    # 頂部關鍵指標
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("動能評分", f"{ctx['score']} / 10")
    m2.metric("夏普比率 (Sharpe)", ctx['sharpe'])
    m3.metric("最大回撤 (MDD)", f"{ctx['mdd']}%")
    m4.metric("AI 模型", st.session_state.get('used_model', 'Pending'))

    # 股價走勢圖
    st.line_chart(ctx['df'][['Close', 'MA20']].tail(120))
    
    # 詳細數據摺疊區
    with st.expander("📝 查看詳細指標評分與回測對帳單"):
        c_a, c_b = st.columns([1, 2])
        c_a.table(ctx['score_df'])
        if not ctx['bt_log'].empty:
            c_b.dataframe(ctx['bt_log'], use_container_width=True)
        else:
            c_b.info("過去兩年無觸發進場訊號")

    st.divider()
    
    # 對話區
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # 用戶輸入區
    if query := st.chat_input("詢問 AI 經理人更多細節..."):
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)
        
        with st.chat_message("assistant"):
            with st.spinner("智庫辯證中..."):
                resp, _ = get_gemini_25_response(key, st.session_state.messages)
                st.markdown(resp)
                st.session_state.messages.append({"role": "assistant", "content": resp})
