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
    page_title="股票基金大師團隊 AI (自動偵測版)", 
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
# 🧠 AI 核心 (自動搜尋模型版)
# ==========================================
def find_valid_model(api_key):
    """
    自動查詢 API Key 權限下可用的模型，避免 404 錯誤
    """
    genai.configure(api_key=api_key)
    try:
        # 列出所有可用模型
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        # 優先順序策略
        priority_order = [
            'models/gemini-1.5-flash',
            'models/gemini-1.5-flash-latest',
            'models/gemini-1.5-flash-001',
            'models/gemini-pro',
            'models/gemini-1.0-pro'
        ]
        
        # 1. 先找優先清單裡有的
        for model in priority_order:
            if model in available_models:
                return model
        
        # 2. 如果都沒有，隨便找一個名字裡有 flash 的
        for model in available_models:
            if 'flash' in model:
                return model

        # 3. 再沒有，隨便找一個 gemini 的
        for model in available_models:
            if 'gemini' in model:
                return model
                
        return None # 真的找不到
        
    except Exception as e:
        return f"Error: {str(e)}"

def chat_with_gemini(api_key, prompt_text, system_instruction):
    if not api_key: return "⚠️ 請先輸入 API Key。"
    
    # 1. 確保有可用的模型名稱
    if not st.session_state.valid_model_name:
        found_model = find_valid_model(api_key)
        if not found_model or "Error" in found_model:
            # 如果自動尋找失敗 (通常是套件版本太舊)，回退到最原始的設定
            st.session_state.valid_model_name = "gemini-pro"
        else:
            st.session_state.valid_model_name = found_model

    current_model = st.session_state.valid_model_name
    genai.configure(api_key=api_key)
    
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    history = []
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "model"
        if "【系統數據】" not in msg["content"]:
            history.append({"role": role, "parts": [msg["content"]]})

    # 嘗試生成
    try:
        # 嘗試帶工具
        tools_config = [{"google_search_retrieval": {"dynamic_retrieval_config": {"mode": "dynamic", "dynamic_threshold": 0.3}}}]
        model = genai.GenerativeModel(
            model_name=current_model,
            system_instruction=system_instruction,
            tools=tools_config,
            safety_settings=safety_settings
        )
        chat = model.start_chat(history=history)
        response = chat.send_message(prompt_text)
        return response.text

    except Exception as e:
        # 錯誤處理 (429 or 404)
        err_msg = str(e).lower()
        if "429" in err_msg or "quota" in err_msg:
             # 降級：不帶工具
            try:
                time.sleep(1)
                model_backup = genai.GenerativeModel(current_model, system_instruction=system_instruction, safety_settings=safety_settings)
                chat_backup = model_backup.start_chat(history=history)
                return chat_backup.send_message(prompt_text + " (流量限制模式)").text
            except Exception as e2:
                return f"❌ 流量超限且重試失敗: {str(e2)}"
        elif "404" in err_msg or "not found" in err_msg:
             return f"❌ 模型找不到 ({current_model})。請確認你的 google-generativeai 套件版本是否 >=0.8.3"
        else:
            return f"❌ 未知錯誤: {str(e)}"

# ==========================================
# 🖥️ UI 介面
# ==========================================
st.title("🏦 股票基金大師團隊 AI (版本診斷版)")
st.caption("自動偵測模型 | 環境診斷")

with st.sidebar:
    st.header("⚙️ 控制台")
    
    # === 版本檢查診斷區 ===
    try:
        lib_ver = genai.__version__
        st.write(f"📚 GenAI 套件版本: `{lib_ver}`")
        ver_parts = lib_ver.split('.')
        if int(ver_parts[1]) < 8 and int(ver_parts[0]) == 0:
             st.error("❌ 版本過舊！請更新到 0.8.3 以上才能使用 Flash 模型。")
             st.code("pip install -U google-generativeai", language="bash")
        else:
             st.success("✅ 版本檢查通過")
    except:
        st.error("⚠️ 無法讀取版本號，環境可能異常")
    # ====================

    api_key = st.text_input("Google API Key", type="password")
    
    if st.button("🔍 測試 API 連線與模型", type="secondary"):
        if not api_key:
            st.error("請先輸入 API Key")
        else:
            with st.spinner("正在向 Google 查詢可用模型..."):
                valid = find_valid_model(api_key)
                if valid and "Error" not in valid:
                    st.session_state.valid_model_name = valid
                    st.success(f"✅ 成功連線！將使用模型: {valid}")
                else:
                    st.error(f"❌ 連線失敗或無可用模型: {valid}")

    default_prompt = """你現在是「股票基金大師團隊」。
【最高權限指令】
- 回答問題前，請優先使用 Google Search 查詢該股票最新的「新聞」、「財報」、「配息」。
- 如果系統忙碌無法搜尋，請根據你的專業知識回答，但要註明資料可能不是最新的。
- 講話風格要像華爾街狼群一樣，用詞犀利、專業。"""
    
    system_prompt = st.text_area("大師指令", value=default_prompt, height=200)
    ticker = st.text_input("股票代號", value="2330")
    
    if st.button("🚀 啟動大師分析", type="primary"):
        st.session_state.messages = []
        with st.spinner("大師團隊正在建立連線..."):
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
