import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from datetime import datetime

# ==========================================
# 🔧 設定頁面與 Session
# ==========================================
st.set_page_config(
    page_title="股票基金大師團隊 AI (完全解封版)", 
    page_icon="🏦", 
    layout="wide"
)

# 初始化 Session State
if "messages" not in st.session_state: st.session_state.messages = []
if "stock_data" not in st.session_state: st.session_state.stock_data = None
if "backtest_log" not in st.session_state: st.session_state.backtest_log = None
if "quant_score" not in st.session_state: st.session_state.quant_score = None
if "score_details" not in st.session_state: st.session_state.score_details = ""
if "vwap" not in st.session_state: st.session_state.vwap = 0

# ==========================================
# 🧮 數學與指標計算工具
# ==========================================
def slope(series, n=3):
    """計算斜率"""
    y = series.tail(n).dropna()
    if len(y) < n: return 0
    x = np.arange(len(y))
    try: return np.polyfit(x, y, 1)[0]
    except: return 0

def calc_vwap(stock_id):
    """抓取今日 15分K 計算當日均價 (VWAP)"""
    try:
        df_intra = yf.download(stock_id, period="5d", interval="15m", progress=False)
        if isinstance(df_intra.columns, pd.MultiIndex):
            df_intra.columns = df_intra.columns.get_level_values(0)
            
        if df_intra.empty: return None
        
        last_date = df_intra.index[-1].date()
        df_today = df_intra[df_intra.index.date == last_date]
        
        if df_today.empty: return None
        
        vwap = (df_today['Close'] * df_today['Volume']).sum() / df_today['Volume'].sum()
        return vwap
    except:
        return None

# ==========================================
# 📊 核心數據下載與指標計算
# ==========================================
@st.cache_data(ttl=300)
def get_data_with_indicators(stock_id):
    stock_id = stock_id.strip().upper()
    if stock_id.isdigit(): stock_id = f"{stock_id}.TW"
    elif not any(x in stock_id for x in [".TW", ".TWO", ".HK", ".US", ".SS", ".SZ"]): stock_id = f"{stock_id}.TW"

    try:
        df = yf.download(stock_id, start="2020-01-01", progress=False)
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
            
        if df.empty: return None, stock_id, "查無資料，請確認代號正確 (如 2330.TW, AAPL)"
        
        if 'Adj Close' in df.columns: df['Close'] = df['Adj Close']
        df = df.dropna()

        # --- 計算指標 ---
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        df['MA250'] = df['Close'].rolling(250).mean()
        
        # ATR
        hl = df['High'] - df['Low']
        hc = (df['High'] - df['Close'].shift()).abs()
        lc = (df['Low'] - df['Close'].shift()).abs()
        df['ATR'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

        # KD
        low_min = df['Low'].rolling(9).min()
        high_max = df['High'].rolling(9).max()
        df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
        df['K'] = df['RSV'].ewm(com=2).mean()
        df['D'] = df['K'].ewm(com=2).mean()
        
        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['DIF'] = ema12 - ema26
        df['MACD'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['OSC'] = df['DIF'] - df['MACD']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['Vol_MA'] = df['Volume'].rolling(5).mean()

        return df.dropna(), stock_id, None
    except Exception as e:
        return None, stock_id, str(e)

# ==========================================
# 📈 策略回測邏輯
# ==========================================
def run_backtest(df):
    trade_log = []
    holding = False
    entry_price = 0
    entry_date = None
    entry_type = ""
    stop_loss = 0
    highest = 0
    
    test_data = df.tail(1000) if len(df) > 1000 else df
    
    for i in range(1, len(test_data)):
        r = test_data.iloc[i]
        prev = test_data.iloc[i-1]
        curr_date = test_data.index[i]
        
        # 進場條件
        buy_main = (r['Close'] > r['MA20']) and (r['OSC'] > 0) and (r['Close'] > prev['High'])
        buy_dip = (r['Close'] < r['MA60']) and (r['RSI'] < 40) and (r['Close'] > prev['Close'])

        # 出場條件
        sell_signal = False
        reason = ""

        if holding:
            highest = max(highest, r['Close'])
            if r['Close'] < stop_loss: 
                sell_signal = True; reason = "🛡️破底(停損)"
            elif entry_type == "🟡主升段":
                if r['Close'] < r['MA20']: sell_signal = True; reason = "📉趨勢結束"
                elif r['RSI'] > 80: sell_signal = True; reason = "🔴極度過熱"
            elif entry_type == "🟢撿籌碼":
                if r['Close'] > r['MA60'] and r['RSI'] > 75: sell_signal = True; reason = "🔴極度過熱"
                elif r['Close'] < highest * 0.9: sell_signal = True; reason = "📉趨勢結束"

        if not holding:
            if buy_main:
                holding = True; entry_price = r['Close']; entry_date = curr_date; entry_type = "🟡主升段"
                stop_loss = r['MA20'] * 0.98; highest = r['Close']
            elif buy_dip:
                holding = True; entry_price = r['Close']; entry_date = curr_date; entry_type = "🟢撿籌碼"
                stop_loss = r['Close'] - 1.5*r['ATR']; highest = r['Close']
        elif holding and sell_signal:
            holding = False
            profit = (r['Close'] - entry_price) / entry_price * 100
            trade_log.append({
                '買入日期': entry_date.strftime('%Y-%m-%d'), '進場': entry_type,
                '買入價': entry_price, '賣出日期': curr_date.strftime('%Y-%m-%d'),
                '賣出價': r['Close'], '獲利%': round(profit, 2), '賣出原因': reason
            })

    if holding:
        curr_price = test_data.iloc[-1]['Close']
        profit = (curr_price - entry_price) / entry_price * 100
        trade_log.append({
            '買入日期': entry_date.strftime('%Y-%m-%d'), '進場': entry_type,
            '買入價': entry_price, '賣出日期': "持倉中",
            '賣出價': curr_price, '獲利%': round(profit, 2), '賣出原因': "📢持有中"
        })
    return pd.DataFrame(trade_log)

def calculate_quant_score(df, vwap_val):
    score = 0; reasons = []
    r = df.iloc[-1]
    
    macd_slope = slope(df['DIF'], 4)
    rsi_slope = slope(df['RSI'], 4)
    vol_slope = slope(df['Vol_MA'], 4)
    vwap_compare = vwap_val if vwap_val else r['Close']

    if r['MA5'] > r['MA10'] > r['MA20']: score += 3; reasons.append("★均線多排(+3)")
    if macd_slope > 0: score += 2; reasons.append("MACD轉強(+2)")
    if r['Close'] > vwap_compare: score += 2; reasons.append("價>VWAP(+2)")
    if r['Close'] > r['MA20']: score += 1; reasons.append("站上月線(+1)")
    if rsi_slope > 0: score += 1; reasons.append("RSI向上(+1)")
    if vol_slope > 0: score += 1; reasons.append("量能增溫(+1)")
    if (r['High'] - r['Low']) > 1.8 * r['ATR']: score -= 2; reasons.append("⚠️波動過大(-2)")

    return max(0, min(10, score)), " | ".join(reasons)

# ==========================================
# 🧠 AI 對話核心 (包含安全解鎖與搜尋修正)
# ==========================================
def chat_with_gemini(api_key, prompt_text, system_instruction):
    if not api_key: return "⚠️ 請先輸入 API Key。"
    
    try:
        genai.configure(api_key=api_key)
        
        # 1. 工具設定：使用 google_search_retrieval (符合新版 SDK)
        tools_config = [
            {
                "google_search_retrieval": {
                    "dynamic_retrieval_config": {
                        "mode": "dynamic",
                        "dynamic_threshold": 0.3,
                    }
                }
            }
        ]
        
        # 2. 安全設定：解除封印 (關鍵！)
        # 允許 AI 談論高風險話題，展現 "個性"
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        
        # 3. 建立模型 (優先 2.0 -> 1.5 備援)
        model_name = 'gemini-2.0-flash'
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction,
                tools=tools_config,
                safety_settings=safety_settings
            )
            chat = model.start_chat(history=[])
        except:
            model = genai.GenerativeModel(
                model_name='gemini-1.5-flash',
                system_instruction=system_instruction,
                tools=tools_config,
                safety_settings=safety_settings
            )
            chat = model.start_chat(history=[])

        # 4. 歷史訊息處理
        history = []
        for msg in st.session_state.messages:
            role = "user" if msg["role"] == "user" else "model"
            # 避免將過長的系統數據重複餵給 AI，浪費 Token
            if "【系統傳入數據" not in msg["content"]:
                history.append({"role": role, "parts": [msg["content"]]})
        
        chat.history = history
        
        response = chat.send_message(prompt_text)
        return response.text

    except Exception as e:
        return f"❌ AI 連線錯誤: {str(e)} \n(請檢查 API Key 或網路狀態)"

# ==========================================
# 🖥️ 網頁介面佈局
# ==========================================
st.title("🏦 股票基金大師團隊 AI (聯網解封版)")
st.caption("自主聯網查證 × 莊家暗黑兵法 × 歷史策略回測")

with st.sidebar:
    st.header("⚙️ 控制台")
    api_key = st.text_input("Google API Key", type="password")
    st.divider()
    
    default_prompt = """你們是一群專業的股票基金經理人，擁有使用 Google Search 查閱即時資訊、新聞、財報與宏觀經濟的權限。

【團隊性格】
1. **多頭總司令**：樂觀，看好產業前景，尋找價值低估。
2. **空軍總司令**：悲觀，講話刻薄，專門找財報漏洞與主力出貨痕跡，喜歡用「割韭菜」、「接盤俠」等詞。
3. **巴菲特仲裁者**：理性，最後做總結，重視安全邊際。

【任務流程】
1. **獨立查證**：不要輕信數據。請務必使用 Google Search 搜尋該股最新的「配息」、「財報」與「利空新聞」。
2. **激烈辯論**：多空雙方必須吵架。
3. **結論**：給出明確操作建議。"""
    
    system_prompt = st.text_area("大師團隊指令", value=default_prompt, height=250)
    st.divider()
    
    ticker = st.text_input("輸入股票代號", value="2330.TW")
    
    if st.button("📊 完整分析 (含聯網查證)", type="primary", use_container_width=True):
        st.session_state.messages = [] 
        
        with st.spinner("🔄 正在執行量化回測並召集大師團隊..."):
            df, real_id, err = get_data_with_indicators(ticker)
            
            if df is not None:
                st.session_state.stock_data = df
                
                # 回測
                backtest_df = run_backtest(df)
                st.session_state.backtest_log = backtest_df
                
                if not backtest_df.empty:
                    wins = len(backtest_df[backtest_df['獲利%'] > 0])
                    total = len(backtest_df)
                    win_rate = (wins/total)*100
                    total_return = backtest_df['獲利%'].sum()
                    backtest_summary = f"歷史回測(近4年)共交易 {total} 次，勝率 {win_rate:.1f}%，累計獲利 {total_return:.1f}%。"
                else:
                    backtest_summary = "歷史回測無交易訊號。"

                # 即時指標
                vwap = calc_vwap(real_id)
                st.session_state.vwap = vwap if vwap else 0
                score, details = calculate_quant_score(df, vwap)
                st.session_state.quant_score = score
                st.session_state.score_details = details
                
                # 準備 AI Prompt
                latest = df.iloc[-1]
                vwap_str = f"{vwap:.2f}" if vwap else "N/A"
                
                first_msg = f"""
                【系統傳入數據 - {real_id}】
                1. 最新收盤: {latest['Close']:.2f} (MA20: {latest['MA20']:.2f}, MA60: {latest['MA60']:.2f})
                2. 即時 VWAP: {vwap_str}
                3. 量化評分: {score}分 (理由: {details})
                4. RSI: {latest['RSI']:.2f}
                5. 【重要回測結果】{backtest_summary}
                
                【大師團隊任務】
                請開始分析。**必須先聯網搜尋**：
                1. 這家公司最近一週有沒有重大新聞？
                2. 最近的股利政策如何？
                3. 用搜尋到的事實來佐證或反駁上面的技術指標。
                """
                st.session_state.messages.append({"role": "user", "content": first_msg})
                
                response = chat_with_gemini(api_key, first_msg, system_prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.error(err)

# ==========================================
# 📊 主畫面呈現
# ==========================================
if st.session_state.stock_data is not None:
    df = st.session_state.stock_data
    score = st.session_state.quant_score
    details = st.session_state.score_details
    backtest_df = st.session_state.backtest_log
    vwap = st.session_state.vwap
    last_price = df.iloc[-1]['Close']
    
    # 儀表板
    st.subheader(f"📊 {ticker} 策略動能儀表板")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1: st.metric("最新股價 / VWAP", f"{last_price:.2f}", f"均價: {vwap:.2f}" if vwap else "N/A")
    with c2: st.metric("量化總分", f"{score} 分", "🚀強勢" if score >= 8 else "😐盤整" if score >= 5 else "🐻弱勢")
    with c3: st.info(f"**得分詳情:** {details}")
    st.divider()

    # 回測紀錄
    st.subheader("📜 歷史交易回測紀錄")
    if backtest_df is not None and not backtest_df.empty:
        total_return = backtest_df['獲利%'].sum()
        m1, m2, m3 = st.columns(3)
        m1.metric("總交易次數", f"{len(backtest_df)} 次")
        m2.metric("策略勝率", f"{(len(backtest_df[backtest_df['獲利%'] > 0])/len(backtest_df)*100):.1f} %")
        m3.metric("技術累計報酬", f"{total_return:.1f} %", delta_color="inverse" if total_return < 0 else "normal")
        
        # 樣式化表格
        def color_profit(val):
            color = '#ff4b4b' if val < 0 else '#00c853'
            return f'color: {color}'
        
        st.dataframe(
            backtest_df.style.format({"買入價": "{:.2f}", "賣出價": "{:.2f}", "獲利%": "{:.2f}%"})
                             .map(color_profit, subset=['獲利%']), 
            use_container_width=True
        )
    else:
        st.warning("⚠️ 此策略在過去區間內無觸發交易訊號。")
    st.divider()

    # 圖表
    with st.expander("📈 查看 K 線圖與原始數據"):
        st.line_chart(df['Close'].tail(200))

# --- AI 對話區 ---
st.subheader("💬 大師團隊對話室")
for msg in st.session_state.messages:
    if "【系統傳入數據" in msg["content"]: continue
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("詢問大師團隊 (例如：這家公司最近有什麼負面新聞？)..."):
    if not api_key:
        st.error("請輸入 API Key")
    else:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("大師正在聯網搜尋真相..."):
                response = chat_with_gemini(api_key, prompt, system_prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
