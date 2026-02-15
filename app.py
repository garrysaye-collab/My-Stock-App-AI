import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time

# ==========================================
# 🔧 設定頁面與 Session
# ==========================================
st.set_page_config(
    page_title="股票基金大師團隊 AI (穩定版)", 
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
# 🧮 基礎計算函數
# ==========================================
def slope(series, n=3):
    y = series.tail(n).dropna()
    if len(y) < n: return 0
    x = np.arange(len(y))
    try: return np.polyfit(x, y, 1)[0]
    except: return 0

def calc_vwap(stock_id):
    try:
        df_intra = yf.download(stock_id, period="5d", interval="15m", progress=False)
        if isinstance(df_intra.columns, pd.MultiIndex):
            df_intra.columns = df_intra.columns.get_level_values(0)
        if df_intra.empty: return None
        last_date = df_intra.index[-1].date()
        df_today = df_intra[df_intra.index.date == last_date]
        if df_today.empty: return None
        return (df_today['Close'] * df_today['Volume']).sum() / df_today['Volume'].sum()
    except: return None

# ==========================================
# 📊 核心數據處理
# ==========================================
@st.cache_data(ttl=300)
def get_data_with_indicators(stock_id):
    stock_id = stock_id.strip().upper()
    if stock_id.isdigit(): stock_id = f"{stock_id}.TW"
    elif not any(x in stock_id for x in [".TW", ".TWO", ".HK", ".US", ".SS"]): stock_id = f"{stock_id}.TW"

    try:
        df = yf.download(stock_id, start="2020-01-01", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.empty: return None, stock_id, "查無資料"
        
        if 'Adj Close' in df.columns: df['Close'] = df['Adj Close']
        df = df.dropna()

        # 指標
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        
        # ATR
        hl = df['High'] - df['Low']
        hc = (df['High'] - df['Close'].shift()).abs()
        lc = (df['Low'] - df['Close'].shift()).abs()
        df['ATR'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['DIF'] = ema12 - ema26
        df['MACD'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['OSC'] = df['DIF'] - df['MACD']
        
        df['Vol_MA'] = df['Volume'].rolling(5).mean()

        return df.dropna(), stock_id, None
    except Exception as e:
        return None, stock_id, str(e)

# ==========================================
# 📈 簡易回測
# ==========================================
def run_backtest(df):
    trade_log = []
    holding = False
    entry_price = 0
    entry_date = None
    
    test_data = df.tail(800) # 只測最近幾年以節省資源
    
    for i in range(1, len(test_data)):
        r = test_data.iloc[i]
        curr_date = test_data.index[i]
        
        # 簡單策略範例：MA20翻揚且RSI強勢
        buy_signal = (r['Close'] > r['MA20']) and (r['RSI'] > 50) and (test_data.iloc[i-1]['Close'] < test_data.iloc[i-1]['MA20'])
        sell_signal = (r['Close'] < r['MA20'])

        if not holding and buy_signal:
            holding = True; entry_price = r['Close']; entry_date = curr_date
        elif holding and sell_signal:
            holding = False
            profit = (r['Close'] - entry_price) / entry_price * 100
            trade_log.append({
                '買入日期': entry_date.strftime('%Y-%m-%d'), 
                '買入價': entry_price, 
                '賣出日期': curr_date.strftime('%Y-%m-%d'), 
                '賣出價': r['Close'], 
                '獲利%': round(profit, 2)
            })

    return pd.DataFrame(trade_log)

def calculate_quant_score(df, vwap_val):
    score = 0; reasons = []
    r = df.iloc[-1]
    
    if r['Close'] > r['MA20']: score += 2; reasons.append("站上月線(+2)")
    if r['Close'] > r['MA60']: score += 2; reasons.append("多頭排列(+2)")
    if slope(df['RSI'], 3) > 0: score += 1; reasons.append("RSI向上(+1)")
    if slope(df['MACD'], 3) > 0: score += 1; reasons.append("MACD翻揚(+1)")
    if vwap_val and r['Close'] > vwap_val: score += 2; reasons.append("價>VWAP(+2)")
    if r['RSI'] > 80: score -= 2; reasons.append("⚠️過熱(-2)")
    
    return max(0, min(10, score)), " | ".join(reasons)

# ==========================================
# 🧠 AI 核心 (針對 429 錯誤的防禦性寫法)
# ==========================================
def chat_with_gemini(api_key, prompt_text, system_instruction):
    if not api_key: return "⚠️ 請先輸入 API Key。"
    
    try:
        genai.configure(api_key=api_key)
        
        # 1. 安全設定全開 (保留個性)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

        # 2. 處理歷史訊息 (Token 優化：只保留對話，不重複傳送舊數據)
        history = []
        for msg in st.session_state.messages:
            role = "user" if msg["role"] == "user" else "model"
            # 關鍵過濾：避免每次都傳送龐大的「系統數據」導致 429
            if "【系統數據】" not in msg["content"]:
                history.append({"role": role, "parts": [msg["content"]]})

        # 3. 嘗試使用 1.5 Flash + 搜尋工具 (最穩定)
        tools_config = [
            {"google_search_retrieval": {"dynamic_retrieval_config": {"mode": "dynamic", "dynamic_threshold": 0.3}}}
        ]
        
        try:
            model = genai.GenerativeModel(
                model_name='gemini-1.5-flash', # 強制使用 1.5 (配額較多)
                system_instruction=system_instruction,
                tools=tools_config,
                safety_settings=safety_settings
            )
            chat = model.start_chat(history=history)
            response = chat.send_message(prompt_text)
            return response.text

        except Exception as e:
            # 4. 降級處理：如果搜尋工具失敗或 429，嘗試「無工具」模式
            if "429" in str(e) or "quota" in str(e).lower():
                time.sleep(2) # 稍微緩衝
                model_backup = genai.GenerativeModel(
                    model_name='gemini-1.5-flash',
                    system_instruction=system_instruction,
                    safety_settings=safety_settings
                    # 移除 tools 以節省資源
                )
                chat_backup = model_backup.start_chat(history=history)
                response = chat_backup.send_message(prompt_text + "\n(系統提示：因網路繁忙，此回應暫時關閉聯網搜尋功能，僅基於內建知識庫回答)")
                return response.text
            else:
                return f"❌ 發生未知錯誤: {str(e)}"

    except Exception as e:
        return f"❌ API 連線失敗: {str(e)}"

# ==========================================
# 🖥️ UI 介面
# ==========================================
st.title("🏦 股票基金大師團隊 AI (穩定版)")
st.caption("Gemini 1.5 Flash | 自主聯網 | 策略回測")

with st.sidebar:
    st.header("⚙️ 控制台")
    api_key = st.text_input("Google API Key", type="password")
    
    default_prompt = """你現在是「股票基金大師團隊」，由三種人格組成：
1. **多頭總司令**：擅長挖掘價值，看好未來。
2. **空軍總司令**：極度悲觀，講話刻薄，專門找財報漏洞與主力出貨痕跡，喜歡嘲諷「韭菜」。
3. **巴菲特仲裁者**：最後做決策，理性客觀。

【最高權限指令】
- 回答問題前，請優先使用 Google Search 查詢該股票最新的「新聞」、「財報」、「配息」。
- 如果系統忙碌無法搜尋，請根據你的專業知識回答，但要註明資料可能不是最新的。
- 講話風格要像華爾街狼群一樣，用詞犀利、專業。"""
    
    system_prompt = st.text_area("大師指令", value=default_prompt, height=200)
    ticker = st.text_input("股票代號", value="2330")
    
    if st.button("🚀 啟動大師分析", type="primary"):
        st.session_state.messages = []
        with st.spinner("大師團隊正在調閱資料..."):
            df, real_id, err = get_data_with_indicators(ticker)
            if df is not None:
                st.session_state.stock_data = df
                trades = run_backtest(df)
                st.session_state.backtest_log = trades
                
                # 計算摘要
                win_rate = 0
                total_ret = 0
                if not trades.empty:
                    win_rate = len(trades[trades['獲利%']>0]) / len(trades) * 100
                    total_ret = trades['獲利%'].sum()
                
                latest = df.iloc[-1]
                vwap = calc_vwap(real_id)
                st.session_state.vwap = vwap if vwap else 0
                score, details = calculate_quant_score(df, vwap)
                st.session_state.quant_score = score
                st.session_state.score_details = details
                
                msg = f"""
                【系統數據】{real_id}
                - 收盤價: {latest['Close']:.2f}
                - RSI: {latest['RSI']:.2f} | MACD: {latest['MACD']:.2f}
                - 量化評分: {score} ({details})
                - 歷史策略回測: 勝率 {win_rate:.1f}%, 總報酬 {total_ret:.1f}%
                
                請大師團隊開始分析。請盡量聯網搜尋這檔股票最近的新聞與配息狀況。
                """
                st.session_state.messages.append({"role": "user", "content": msg})
                res = chat_with_gemini(api_key, msg, system_prompt)
                st.session_state.messages.append({"role": "assistant", "content": res})
            else:
                st.error(err)

# 主畫面
if st.session_state.stock_data is not None:
    df = st.session_state.stock_data
    latest = df.iloc[-1]
    
    # 1. 儀表板
    c1, c2, c3 = st.columns(3)
    c1.metric("最新價", f"{latest['Close']:.2f}", f"VWAP: {st.session_state.vwap:.2f}" if st.session_state.vwap else "")
    c2.metric("RSI", f"{latest['RSI']:.1f}")
    c3.metric("量化評分", f"{st.session_state.quant_score}", st.session_state.score_details)
    
    # 2. 圖表
    st.line_chart(df['Close'].tail(200))
    
    # 3. 回測表
    if st.session_state.backtest_log is not None and not st.session_state.backtest_log.empty:
        with st.expander("查看歷史回測細節"):
            st.dataframe(st.session_state.backtest_log.style.format({'獲利%': '{:.2f}%'}))

# 對話區
for msg in st.session_state.messages:
    if "【系統數據】" not in msg["content"]:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

if user_input := st.chat_input("問問大師團隊 (例如：最近外資在賣什麼？)..."):
    if not api_key: st.error("請輸入 API Key")
    else:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("assistant"):
            with st.spinner("大師思考中..."):
                response = chat_with_gemini(api_key, user_input, system_prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
