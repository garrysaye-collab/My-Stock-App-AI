import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import google.generativeai as genai
import datetime

# ==========================================
# 🔧 1. 系統設定與狀態初始化
# ==========================================
st.set_page_config(page_title="專業量化與 AI 經理人戰情室", page_icon="🏦", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "data_context" not in st.session_state:
    st.session_state.data_context = None

# ==========================================
# 📈 2. 核心量化函數 (保持不變)
# ==========================================
def slope(series, n=3):
    y = series.tail(n).dropna()
    if len(y) < n: return 0
    return np.polyfit(np.arange(len(y)), y, 1)[0]

def detailed_scoring(df):
    """細緻評分邏輯"""
    r = df.iloc[-1]
    details = []
    total_score = 0
    
    macd_slope = slope(df['DIF'], 3)
    rsi_slope = slope(df['RSI'], 3)
    vwap_approx = (r['High'] + r['Low'] + r['Close']) / 3

    checks = [
        (r['MA5'] > r['MA10'] > r['MA20'], 3, "均線多頭排列", "MA5>MA10>MA20"),
        (macd_slope > 0 and r['OSC'] > 0, 2, "MACD 轉強", "DIF斜率>0, OSC>0"),
        (r['Close'] > vwap_approx, 2, "價格優勢", "收盤價 > VWAP"),
        (r['Close'] > r['MA20'], 1, "站上月線", "收盤價 > MA20"),
        (rsi_slope > 0, 1, "RSI 動能", "RSI 斜率 > 0"),
        (r['Volume'] > df['Volume'].tail(5).mean(), 1, "量能增溫", "今日量 > 5日均量")
    ]
    
    for cond, pts, rule, desc in checks:
        s = pts if cond else 0
        details.append({"準則": rule, "條件": desc, "狀態": "✅ 通過" if cond else "❌ 未達成", "得分": s})
        total_score += s

    # 扣分項：波動過大
    day_range = r['High'] - r['Low']
    cond_vol = day_range > 1.8 * r['ATR']
    s_vol = -2 if cond_vol else 0
    details.append({"準則": "⚠️ 波動過熱", "條件": ">1.8倍ATR", "狀態": "🚩 觸發" if cond_vol else "⚪ 正常", "得分": s_vol})
    total_score += s_vol
    
    return max(0, total_score), pd.DataFrame(details)

def comprehensive_backtest(df):
    """歷史交易回測"""
    log = []
    holding = False; entry_price = 0; entry_date = None
    
    for i in range(1, len(df)):
        r = df.iloc[i]; prev = df.iloc[i-1]
        curr_date = df.index[i]

        if not holding:
            # 買入訊號: 站上月線 + MACD紅柱 + 突破前高
            if r['Close'] > r['MA20'] and r['OSC'] > 0 and r['Close'] > prev['High']:
                holding = True; entry_price = r['Close']; entry_date = curr_date
        elif holding:
            # 賣出訊號: 跌破月線 或 RSI過熱
            if r['Close'] < r['MA20'] or r['RSI'] > 85:
                profit_pct = (r['Close'] - entry_price) / entry_price * 100
                log.append({
                    "進場日期": entry_date.date(),
                    "出場日期": curr_date.date(),
                    "進場價": round(entry_price, 2),
                    "出場價": round(r['Close'], 2),
                    "獲利%": round(profit_pct, 2),
                    "出場原因": "趨勢反轉" if r['Close'] < r['MA20'] else "過熱獲利"
                })
                holding = False
    return pd.DataFrame(log)

@st.cache_data(ttl=300)
def get_verified_data(symbol):
    symbol = symbol.strip().upper()
    if symbol.isdigit(): symbol = f"{symbol}.TW"
    try:
        t = yf.Ticker(symbol)
        df = t.history(period="2y")
        if df.empty: return None, None, symbol, "查無數據"
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        info = t.info
        full_name = info.get('longName') or info.get('shortName') or symbol
        return df, full_name, symbol, None
    except Exception as e:
        return None, None, symbol, str(e)

# ==========================================
# 🧠 3. AI 經理人核心 (修改重點：加入 GEM 架構與原生搜尋)
# ==========================================
def get_ai_response(api_key, messages_history):
    genai.configure(api_key=api_key)
    
    # 1. 獲取當前時間，強制時間對齊
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 2. 定義 GEM 架構 System Instruction
    sys_instruction = f"""
    現在時間是：{current_time}。
    你們是一群專業的股票基金經理人，具備使用 Google Search 查閱即時資訊、新聞、財報與宏觀經濟的最高權限。

    【性格與流程】
    1. **獨立辯證**：用戶提供的「歷史回測數據」僅是參考。如果回測數據慘淡，不要直接判死刑，請**主動使用工具搜尋**該標的是否有『高額配息』、『資產重組』或『產業護城河』被忽視了。
    2. **兩方對立**：必須給出多方（價值/基本面）與空方（技術/籌碼）的激烈碰撞。
    3. **暗黑兵法**：莊家團隊須以寓言方式揭示市場陷阱（例如：回測止損可能是為了收割散戶恐慌盤）。
    4. **巴菲特裁定**：最後由巴菲特决定是否參與，並預估投資效益。

    【聯網要求】
    每次對話前，請**自主使用 Google Search 工具搜尋**該股的最新股息率、PE位階及最近一個月的重大新聞，用搜尋到的真實數字說話。不要重複用戶給出的文字。
    """

    try:
        # 3. 初始化模型 (啟用 Google Search 工具)
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash", # 建議使用支援搜尋的新模型
            tools='google_search_retrieval',
            system_instruction=sys_instruction
        )
        
        gemini_hist = []
        for m in messages_history:
            role = "user" if m["role"] == "user" else "model"
            gemini_hist.append({"role": role, "parts": [m["content"]]})
            
        response = model.generate_content(gemini_hist)
        
        # 4. 處理回傳結果與來源標註
        final_text = response.text
        if hasattr(response.candidates[0], 'grounding_metadata') and \
           response.candidates[0].grounding_metadata.search_entry_point:
            search_html = response.candidates[0].grounding_metadata.search_entry_point.rendered_content
            final_text += "\n\n🔍 **資料來源與即時驗證：**\n" + search_html
            
        return final_text
    except Exception as e:
        return f"AI 經理人連線錯誤 (請確認 API Key 支援 Google Search): {str(e)}"

# ==========================================
# 🖥️ 4. UI 介面與主邏輯
# ==========================================
with st.sidebar:
    st.header("🔑 戰情室控制台")
    api_key = st.text_input("Google API Key", type="password")
    ticker_input = st.text_input("股票代號", value="2330")
    run_btn = st.button("啟動全數據掃描", type="primary")
    
    if st.button("🗑️ 清除對話紀錄"):
        st.session_state.messages = []
        st.rerun()

if run_btn and api_key:
    with st.spinner(f"正在調閱 {ticker_input} 檔案與聯網數據..."):
        df, full_name, real_symbol, err = get_verified_data(ticker_input)

        if df is not None:
            # 計算指標
            df['MA5'] = df['Close'].rolling(5).mean()
            df['MA10'] = df['Close'].rolling(10).mean()
            df['MA20'] = df['Close'].rolling(20).mean()
            df['DIF'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
            df['MACD'] = df['DIF'].ewm(span=9).mean()
            df['OSC'] = df['DIF'] - df['MACD']
            df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
            delta = df['Close'].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean(); rs = gain / loss.replace(0, np.nan)
            df['RSI'] = 100 - (100 / (1 + rs))

            score, score_details = detailed_scoring(df)
            bt_log = comprehensive_backtest(df)
            
            # 修改：移除 DuckDuckGo 預先搜尋，改為只提供量化數據，讓 AI 自己去查即時資訊
            system_prompt = f"""
            【量化技術面數據輸入】
            標的：{full_name} ({real_symbol})
            
            【技術面診斷】：
            - 核心動能得分：{score}/10
            - 詳細指標狀態：\n{score_details.to_string()}
            
            【歷史回測統計 (過去2年)】：
            - 總交易次數: {len(bt_log)} 次
            - 策略勝率: {((len(bt_log[bt_log['獲利%']>0])/len(bt_log)*100) if not bt_log.empty else 0):.1f}%
            - 累計報酬: {bt_log['獲利%'].sum() if not bt_log.empty else 0:.1f}%
            
            請根據上述「技術與量化數據」，並立刻使用你的 Google Search 工具查詢該公司的「最新財報」、「最新股息」與「產業新聞」，開始第一輪的多空辯證分析。
            """

            initial_response = get_ai_response(api_key, [{"role": "user", "content": system_prompt}])
            
            st.session_state.data_context = {
                "df": df, "name": full_name, "symbol": real_symbol,
                "score": score, "score_details": score_details, "bt_log": bt_log
            }
            # 注意：這裡將 user prompt 簡化存入歷史，避免太長
            st.session_state.messages = [
                {"role": "user", "content": f"分析 {full_name} 的量化數據與最新基本面"}, 
                {"role": "assistant", "content": initial_response}
            ]
        else:
            st.error(err)

# --- 顯示儀表板 ---
if st.session_state.data_context:
    ctx = st.session_state.data_context
    st.title(f"🏛️ {ctx['name']} ({ctx['symbol']}) 戰情室")
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("🎯 量化評分")
        st.metric("核心動能總分", f"{ctx['score']} / 10")
        st.dataframe(ctx['score_details'], use_container_width=True)
    
    with c2:
        st.subheader("📈 價格走勢 (120D)")
        st.line_chart(ctx['df'][['Close', 'MA20']].tail(120))

    with st.expander("📜 查看歷史回測日誌"):
        if not ctx['bt_log'].empty:
            st.dataframe(ctx['bt_log'], use_container_width=True)
        else:
            st.write("無交易紀錄")
        
    st.divider()
    st.subheader("💬 專家經理人對話")
    
    # 對話過濾與顯示
    for msg in st.session_state.messages:
        # 只顯示這段 User 簡化後的指令，隱藏原始長 Prompt
        if "分析" in msg['content'] and "量化數據" in msg['content']:
             with st.chat_message(msg["role"]): st.markdown(msg["content"])
        elif "你是一群專業投資經理人團隊" in msg['content']: 
            continue
        elif "【量化技術面數據輸入】" in msg['content']:
             continue # 隱藏最原始的 Prompt
        else:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("詢問更多細節..."):
        with st.chat_message("user"): st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("討論中 (正在聯網檢索)..."):
                response = get_ai_response(api_key, st.session_state.messages)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
elif not run_btn:
    st.info("👈 請在左側輸入代號並啟動掃描")
