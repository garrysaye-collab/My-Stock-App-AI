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
    page_title="股票基金大師團隊 AI (工具修復版)", 
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
if "valid_model_name" not in st.session_state: st.session_state.valid_model_name = None

# ==========================================
# 🧮 基礎計算函數 (維持不變)
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
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        
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
        
        return df.dropna(), stock_id, None
    except Exception as e:
        return None, stock_id, str(e)

def run_backtest(df):
    trade_log = []
    holding = False
    entry_price = 0
    entry_date = None
    test_data = df.tail(800) 
    
    for i in range(1, len(test_data)):
        r = test_data.iloc[i]
        curr_date = test_data.index[i]
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
# 🧠 AI 核心 (修復 Tool Name 錯誤)
# ==========================================
def find_valid_model(api_key):
    genai.configure(api_key=api_key)
    try:
        # 優先嘗試最穩定的模型
        priority_models = [
            'gemini-1.5-flash',
            'gemini-1.5-flash-latest',
            'gemini-pro'
        ]
        return priority_models[0] # 先強行回傳 flash，通常都支援
    except:
        return 'gemini-1.5-flash'

def chat_with_gemini(api_key, prompt_text, system_instruction):
    if not api_key: return "⚠️ 請先輸入 API Key。"
    
    # 1. 初始化模型設定
    model_name = 'gemini-1.5-flash' # 強制指定
    genai.configure(api_key=api_key)
    
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    # 2. 準備對話歷史 (過濾系統數據以節省 Token)
    history = []
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "model"
        if "【系統數據】" not in msg["content"]:
            history.append({"role": role, "parts": [msg["content"]]})

    # ==========================================
    # 🚨 關鍵修復：嘗試建立聊天 Session
    # ==========================================
    
    # 方案 A: 帶有正確工具名稱的模式 (google_search)
    try:
        # 修正這裡：使用新的工具定義方式，移除 dynamic_retrieval_config
        tools_config = [
            {"google_search": {}} 
        ]
        
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction,
            tools=tools_config,
            safety_settings=safety_settings
        )
        chat = model.start_chat(history=history)
        response = chat.send_message(prompt_text)
        return response.text

    except Exception as e_tool:
        # 如果方案 A 失敗 (400 Tool Error 或 429 Quota Error)，自動切換到方案 B
        print(f"Tool mode failed: {e_tool}")
        
        # 方案 B: 純文字模式 (無搜尋工具，保證不死機)
        try:
            time.sleep(1) # 緩衝
            model_backup = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction,
                safety_settings=safety_settings
                # 這裡不放 tools
            )
            chat_backup = model_backup.start_chat(history=history)
            
            # 附加提示告訴使用者目前狀況
            fallback_msg = "\n(系統提示：由於搜尋工具連線異常，以下回應基於內建知識庫分析)"
            response = chat_backup.send_message(prompt_text + fallback_msg)
            return response.text
            
        except Exception as e_final:
            return f"❌ 最終連線失敗: {str(e_final)}\n請檢查 API Key 是否正確或配額是否已滿。"

# ==========================================
# 🖥️ UI 介面
# ==========================================
st.title("🏦 股票基金大師團隊 AI (工具修復版)")

with st.sidebar:
    st.header("⚙️ 設定")
    
    # 顯示套件版本，確認環境
    try:
        st.caption(f"GenAI Lib Version: {genai.__version__}")
    except:
        pass

    api_key = st.text_input("Google API Key", type="password")

    default_prompt = """你現在是「股票基金大師團隊」。
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
    
    c1, c2, c3 = st.columns(3)
    c1.metric("最新價", f"{latest['Close']:.2f}", f"VWAP: {st.session_state.vwap:.2f}" if st.session_state.vwap else "")
    c2.metric("RSI", f"{latest['RSI']:.1f}")
    c3.metric("量化評分", f"{st.session_state.quant_score}", st.session_state.score_details)
    
    st.line_chart(df['Close'].tail(200))
    
    if st.session_state.backtest_log is not None and not st.session_state.backtest_log.empty:
        with st.expander("查看歷史回測細節"):
            st.dataframe(st.session_state.backtest_log.style.format({'獲利%': '{:.2f}%'}))

# 對話區
for msg in st.session_state.messages:
    if "【系統數據】" not in msg["content"]:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

if user_input := st.chat_input("問問大師團隊..."):
    if not api_key: st.error("請輸入 API Key")
    else:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("assistant"):
            with st.spinner("大師思考中..."):
                response = chat_with_gemini(api_key, user_input, system_prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
