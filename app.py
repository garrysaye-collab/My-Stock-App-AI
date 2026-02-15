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
# 新增一個狀態來儲存動態生成的 System Instruction (含當前股票數據)
if "current_system_instruction" not in st.session_state:
    st.session_state.current_system_instruction = ""

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
            # 買入訊號
            if r['Close'] > r['MA20'] and r['OSC'] > 0 and r['Close'] > prev['High']:
                holding = True; entry_price = r['Close']; entry_date = curr_date
        elif holding:
            # 賣出訊號
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
# 🧠 3. AI 對話核心 (針對 Streamlit Cloud 的修復版)
# ==========================================
def chat_with_gemini(api_key, prompt_text, system_instruction):
    if not api_key: return "⚠️ 請先輸入 API Key。"
    
    try:
        genai.configure(api_key=api_key)
        
        # 🔧 修正重點：新版 Google Search 工具寫法
        tools_configuration = [
            {
                "google_search_retrieval": {
                    "dynamic_retrieval_config": {
                        "mode": "dynamic",
                        "dynamic_threshold": 0.3, # 讓 AI 自己決定何時搜尋
                    }
                }
            }
        ]
        
        # 優先嘗試 2.0-flash，失敗則降級
        model_name = 'gemini-2.0-flash' 
        
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction,
                tools=tools_configuration
            )
            chat = model.start_chat(history=[])
        except:
            # 備用方案：使用 1.5-flash
            model = genai.GenerativeModel(
                model_name='gemini-1.5-flash',
                system_instruction=system_instruction,
                tools=tools_configuration
            )
            chat = model.start_chat(history=[])

        # 重建歷史訊息 (過濾掉系統數據提示，避免 token 浪費或混淆)
        history = []
        for msg in st.session_state.messages:
            role = "user" if msg["role"] == "user" else "model"
            # 這裡過濾掉我們在 UI 顯示的某些標記，確保傳給 AI 的是乾淨對話
            if "【系統數據已載入】" not in msg["content"]: 
                history.append({"role": role, "parts": [msg["content"]]})
        
        chat.history = history
        
        # 發送訊息
        response = chat.send_message(prompt_text)
        
        # 處理搜尋來源顯示
        final_text = response.text
        if hasattr(response.candidates[0], 'grounding_metadata') and \
           response.candidates[0].grounding_metadata.search_entry_point:
            search_html = response.candidates[0].grounding_metadata.search_entry_point.rendered_content
            final_text += "\n\n🔍 **資料來源與即時驗證：**\n" + search_html
            
        return final_text

    except Exception as e:
        return f"❌ AI 連線錯誤: {str(e)} \n(建議：請在 Streamlit 後台點擊 'Reboot app' 以強制更新環境)"

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
        st.session_state.current_system_instruction = ""
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
            
            # 構建 System Instruction (包含 GEM 角色與量化數據背景)
            # 這樣做的好處是：即便對話很長，AI 永遠知道現在在討論哪支股票的什麼數據
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            st.session_state.current_system_instruction = f"""
            現在時間：{current_time}。
            你是一群專業的股票基金經理人 (GEM 架構)，具備 Google Search 最高權限。
            
            【當前分析標的數據】
            - 股票：{full_name} ({real_symbol})
            - 量化動能得分：{score}/10
            - 指標詳情：{score_details.to_string()}
            - 歷史回測(2年)：勝率 {((len(bt_log[bt_log['獲利%']>0])/len(bt_log)*100) if not bt_log.empty else 0):.1f}%，總報酬 {bt_log['獲利%'].sum() if not bt_log.empty else 0:.1f}%

            【你的性格與任務】
            1. **獨立辯證**：量化數據僅供參考。若回測差，請主動搜尋是否有高配息或轉機新聞被忽略。
            2. **兩方對立**：必須呈現「基本面(多)」vs「技術籌碼(空)」的對立觀點。
            3. **強制聯網**：回答前必須使用 Google Search 工具搜尋該股的「最新財報 EPS」、「最新股息公告」及「本月重大新聞」。
            4. **巴菲特裁定**：最後以巴菲特口吻給出總結。
            """

            # 初始 Prompt (觸發 AI 開始分析)
            initial_prompt = f"請根據上述量化數據，並立刻搜尋 {full_name} 的最新基本面新聞，開始第一輪深度多空辯證分析。"

            # 呼叫 AI
            response_text = chat_with_gemini(
                api_key, 
                initial_prompt, 
                st.session_state.current_system_instruction
            )
            
            # 更新狀態
            st.session_state.data_context = {
                "df": df, "name": full_name, "symbol": real_symbol,
                "score": score, "score_details": score_details, "bt_log": bt_log
            }
            # 為了介面整潔，我們可以只顯示「分析報告」而不顯示那一大串數據 Prompt
            st.session_state.messages = [
                {"role": "user", "content": f"📊 【系統數據已載入】分析 {full_name} ({real_symbol})"},
                {"role": "assistant", "content": response_text}
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
    
    # 對話顯示
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("詢問更多細節 (如：外資看法、風險點)..."):
        # 1. 顯示使用者輸入
        with st.chat_message("user"): st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 2. AI 回答
        with st.chat_message("assistant"):
            with st.spinner("經理人正在查閱資料與思考..."):
                # 這裡傳入 stored_system_instruction 確保 AI 記得他是誰以及現在在聊哪支股票
                response = chat_with_gemini(
                    api_key, 
                    prompt, 
                    st.session_state.current_system_instruction
                )
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

elif not run_btn:
    st.info("👈 請在左側輸入代號並啟動掃描")
