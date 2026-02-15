import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import google.generativeai as genai
import datetime

# ==========================================
# 🔧 系統設定與狀態初始化
# ==========================================
st.set_page_config(page_title="專業量化與 AI 經理人戰情室", page_icon="🏦", layout="wide")

# 初始化 Session State 用於存放對話紀錄與數據
if "messages" not in st.session_state:
    st.session_state.messages = []
if "data_context" not in st.session_state:
    st.session_state.data_context = None

# ==========================================
# 🕵️ 核心數據邏輯 (抓取、指標計算、回測)
# ==========================================
@st.cache_data(ttl=300)
def get_verified_data(symbol):
    """抓取股票數據並標準化代號"""
    symbol = symbol.strip().upper()
    if symbol.isdigit(): 
        symbol = f"{symbol}.TW"
    elif not any(s in symbol for s in [".TW", ".TWO", ".HK", ".US", ".SS", ".SZ"]):
        if not (symbol.isalpha() and len(symbol) <= 4): 
            symbol = f"{symbol}.TW"
    
    try:
        t = yf.Ticker(symbol)
        df = t.history(period="2y")
        if df.empty: 
            return None, None, symbol, "查無數據，請檢查代號"
        
        # 處理 MultiIndex 問題 (新版 yfinance 常見)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        info = t.info
        full_name = info.get('longName') or info.get('shortName') or symbol
        return df, full_name, symbol, None
    except Exception as e:
        return None, None, symbol, str(e)

def slope(series, n=3):
    """計算指標斜率判斷動能"""
    y = series.tail(n).dropna()
    if len(y) < n: return 0
    return np.polyfit(np.arange(len(y)), y, 1)[0]

def detailed_scoring(df):
    """量化動能評分系統"""
    r = df.iloc[-1]
    details = []
    total_score = 0
    
    macd_slope = slope(df['DIF'], 3)
    rsi_slope = slope(df['RSI'], 3)
    vwap_approx = (r['High'] + r['Low'] + r['Close']) / 3

    # 評分規則矩陣
    checks = [
        (r['MA5'] > r['MA10'] > r['MA20'], 3, "均線多頭排列", "MA5 > MA10 > MA20"),
        (macd_slope > 0 and r['OSC'] > 0, 2, "MACD 能量轉強", "DIF斜率 > 0 且 OSC > 0"),
        (r['Close'] > vwap_approx, 2, "價格位於均價上方", "Close > 日均估計值"),
        (r['Close'] > r['MA20'], 1, "站上月線關鍵位", "Close > MA20"),
        (rsi_slope > 0, 1, "RSI 動能向上", "RSI 近期斜率為正"),
        (r['Volume'] > df['Volume'].tail(5).mean(), 1, "量能大於均量", "今日成交量 > 5日均量")
    ]
    
    for cond, pts, rule, desc in checks:
        s = pts if cond else 0
        details.append({"準則": rule, "條件": desc, "狀態": "✅ 通過" if cond else "❌ 未達成", "得分": s})
        total_score += s

    # 異常扣分：波動過大
    day_range = r['High'] - r['Low']
    cond_vol = day_range > 1.8 * r['ATR']
    s_vol = -2 if cond_vol else 0
    details.append({"準則": "⚠️ 波動過熱風險", "條件": "震幅 > 1.8倍 ATR", "狀態": "🚩 觸發" if cond_vol else "⚪ 正常", "得分": s_vol})
    total_score += s_vol
    
    return max(0, total_score), pd.DataFrame(details)

def comprehensive_backtest(df):
    """簡單趨勢跟隨策略歷史回測"""
    log = []
    holding = False; entry_price = 0; entry_date = None; highest_after_entry = 0
    
    for i in range(20, len(df)):
        r = df.iloc[i]; prev = df.iloc[i-1]; curr_date = df.index[i]
        
        if not holding:
            # 入場條件：站上月線 + MACD翻紅 + 突破前高
            if r['Close'] > r['MA20'] and r['OSC'] > 0 and r['Close'] > prev['High']:
                holding = True; entry_price = r['Close']; entry_date = curr_date; highest_after_entry = r['Close']
        elif holding:
            highest_after_entry = max(highest_after_entry, r['Close'])
            # 出場條件：跌破月線 或 RSI極度過熱
            if r['Close'] < r['MA20'] or r['RSI'] > 85:
                profit_pct = (r['Close'] - entry_price) / entry_price * 100
                log.append({
                    "進場日期": entry_date.strftime('%Y-%m-%d'),
                    "出場日期": curr_date.strftime('%Y-%m-%d'),
                    "持股天數": (curr_date - entry_date).days,
                    "買入價": round(entry_price, 2),
                    "賣出價": round(r['Close'], 2),
                    "獲利%": round(profit_pct, 2),
                    "最高浮盈%": round((highest_after_entry - entry_price)/entry_price*100, 2),
                    "原因": "趨勢反轉" if r['Close'] < r['MA20'] else "過熱獲利"
                })
                holding = False
    return pd.DataFrame(log)

# ==========================================
# 🤖 AI 智能核心 (GEM 架構 + 自動模型適配)
# ==========================================
def get_ai_response(api_key, messages_history):
    """自動偵測可用模型並執行聯網分析"""
    genai.configure(api_key=api_key)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # GEM 架構指令
    sys_instruction = f"""
    現在時間是 {current_time}。你是「全球智庫投資團隊」。
    你必須使用 Google Search 工具來獲取最新的新聞與財報。

    你的對話包含四個視角：
    1. **【多方經理人】**：專注基本面優勢、成長動能、產業護城河。
    2. **【空方分析師】**：專注技術面背離、籌碼鬆動、宏觀經濟風險。
    3. **【莊家/暗黑視角】**：揭示盤面上的心理陷阱與洗盤行為。
    4. **【巴菲特決策】**：最後總結，給出具體的投資評等 (強力買進/觀望/避開) 與預期風險。

    請務必查閱最新股價新聞與配息資訊，不要只重複用戶給的歷史數據。
    """

    try:
        # 1. 自動偵測可用模型 (解決 404 問題)
        available_models = [m.name.replace("models/", "") for m in genai.list_models() 
                           if 'generateContent' in m.supported_generation_methods]
        
        priority_list = ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"]
        selected_model = next((m for m in priority_list if m in available_models), available_models[0])

        # 2. 初始化模型與工具
        model = genai.GenerativeModel(
            model_name=selected_model,
            tools=[{"google_search_retrieval": {}}], # 啟用聯網
            system_instruction=sys_instruction
        )

        # 3. 發送對話
        chat = model.start_chat(history=[])
        formatted_history = []
        for m in messages_history:
            role = "user" if m["role"] == "user" else "model"
            formatted_history.append({"role": role, "parts": [m["content"]]})
        
        # 取最後一條作為當前輸入，前面的作為 context
        last_msg = formatted_history.pop()["parts"][0]
        response = model.generate_content(last_msg) # 簡化調用以確保穩定性
        
        return response.text

    except Exception as e:
        return f"⚠️ AI 服務異常: {str(e)}"

# ==========================================
# 🖥️ UI 介面
# ==========================================
with st.sidebar:
    st.header("🔑 投資控制台")
    api_key = st.text_input("Gemini API Key", type="password")
    ticker_input = st.text_input("輸入代號 (台/美/陸股)", value="2330")
    run_btn = st.button("啟動全數據掃描", type="primary", use_container_width=True)
    
    if st.button("🗑️ 清除對話紀錄"):
        st.session_state.messages = []
        st.rerun()

# 主邏輯執行
if run_btn:
    if not api_key:
        st.error("請先輸入 API Key")
    else:
        with st.spinner(f"正在分析 {ticker_input} 並查閱即時新聞..."):
            df, full_name, real_id, err = get_verified_data(ticker_input)
            
            if df is not None:
                # 指標計算
                df['MA5'] = df['Close'].rolling(5).mean()
                df['MA10'] = df['Close'].rolling(10).mean()
                df['MA20'] = df['Close'].rolling(20).mean()
                ema12 = df['Close'].ewm(span=12).mean()
                ema26 = df['Close'].ewm(span=26).mean()
                df['DIF'] = ema12 - ema26
                df['MACD'] = df['DIF'].ewm(span=9).mean()
                df['OSC'] = df['DIF'] - df['MACD']
                df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
                delta = df['Close'].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean(); rs = gain / loss.replace(0, np.nan)
                df['RSI'] = 100 - (100 / (1 + rs))

                # 評分與回測
                score, score_df = detailed_scoring(df)
                bt_log = comprehensive_backtest(df)
                
                # 存入 Session
                st.session_state.data_context = {
                    "df": df, "name": full_name, "id": real_id, 
                    "score": score, "score_df": score_df, "bt_log": bt_log
                }
                
                # 初始 AI 指令
                prompt = f"請分析 {full_name} ({real_id})。技術面分數為 {score}/10。請結合搜尋到的最新財報與產業新聞進行辯證。"
                ai_report = get_ai_response(api_key, [{"role": "user", "content": prompt}])
                st.session_state.messages = [
                    {"role": "user", "content": f"啟動 {real_id} 深度分析"},
                    {"role": "assistant", "content": ai_report}
                ]
            else:
                st.error(err)

# 渲染戰情室面板
if st.session_state.data_context:
    ctx = st.session_state.data_context
    st.title(f"🏛️ {ctx['name']} ({ctx['id']}) 戰情看板")
    
    c1, c2 = st.columns([4, 6])
    with c1:
        st.subheader("🎯 量化動能檢視")
        st.metric("核心評分", f"{ctx['score']} / 10")
        st.table(ctx['score_df'])
        
    with c2:
        st.subheader("📈 價格與趨勢 (MA20)")
        st.line_chart(ctx['df'][['Close', 'MA20']].tail(120))

    with st.expander("📂 查看歷史回測統計 (近兩年趨勢策略)"):
        if not ctx['bt_log'].empty:
            st.dataframe(ctx['bt_log'], use_container_width=True)
            win_rate = (len(ctx['bt_log'][ctx['bt_log']['獲利%']>0]) / len(ctx['bt_log'])) * 100
            st.info(f"回測勝率: {win_rate:.1f}% | 總交易次數: {len(ctx['bt_log'])}")
        else:
            st.write("該標的在過去兩年未觸發本策略進場訊號。")

    st.divider()
    st.subheader("💬 AI 智庫經理人連線")
    
    # 對話紀錄顯示
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # 對話輸入
    if user_q := st.chat_input("詢問更多關於此股的細節 (例如：現在適合長抱嗎？)..."):
        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"): st.markdown(user_q)
        
        with st.chat_message("assistant"):
            with st.spinner("智庫討論中..."):
                resp = get_ai_response(api_key, st.session_state.messages)
                st.markdown(resp)
                st.session_state.messages.append({"role": "assistant", "content": resp})
else:
    st.info("👈 請在左側輸入股票代號並按下「啟動全數據掃描」開始。")
