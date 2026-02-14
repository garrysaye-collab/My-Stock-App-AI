import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import google.generativeai as genai  # 👈 新增這個庫

# ==========================================
# 🔧 設定頁面
# ==========================================
st.set_page_config(page_title="股市動能 AI", page_icon="📈", layout="wide")

# ==========================================
# 🧠 AI 分析模組 (新增)
# ==========================================
def ask_gemini(api_key, stock_id, df, score, reasons):
    if not api_key:
        return "⚠️ 請先在側邊欄輸入 Google API Key 才能啟動 AI 分析。"
    
    try:
        # 設定 API
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')

        # 準備餵給 AI 的數據
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        prompt = f"""
        你是一位專業的華爾街股票分析師。請根據以下技術指標數據，對 {stock_id} 進行簡短且犀利的分析。
        
        【技術數據】
        1. 最新收盤價: {latest['Close']:.2f} (漲跌: {latest['Close'] - prev['Close']:.2f})
        2. 量化評分: {score}/10 分
        3. 觸發訊號: {', '.join(reasons)}
        4. RSI (14): {latest['RSI']:.2f}
        5. MACD柱狀體: {latest['OSC']:.4f}
        6. 是否站上月線(MA20): {'是' if latest['Close'] > latest['MA20'] else '否'}
        
        【你的任務】
        請不要解釋指標定義，直接給出：
        1. 目前的多空趨勢判斷（強多、偏多、盤整、偏空、強空）。
        2. 給操作者的具體建議（例如：適合進場、續抱、或設停損）。
        3. 風險提示。
        請用繁體中文回答，語氣專業且自信。
        """
        
        with st.spinner('🤖 AI 分析師正在撰寫報告...'):
            response = model.generate_content(prompt)
            return response.text
    except Exception as e:
        return f"❌ AI 分析失敗: {str(e)}"

# ====== 核心邏輯 (保留原本的) ======
@st.cache_data(ttl=300)
def get_data_and_analyze(stock_id):
    stock_id = stock_id.strip().upper()
    original_id = stock_id
    if stock_id.isdigit(): stock_id = f"{stock_id}.TW"
    elif not any(x in stock_id for x in [".TW", ".TWO", ".HK", ".US", ".SS", ".SZ"]): stock_id = f"{stock_id}.TW"

    try:
        df = yf.download(stock_id, start="2020-01-01", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.empty: return None, stock_id, "查無資料"
        if 'Adj Close' in df.columns: df['Close'] = df['Adj Close']
        df = df.dropna()

        # 指標計算
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        hl = df['High'] - df['Low']
        hc = (df['High'] - df['Close'].shift()).abs()
        lc = (df['Low'] - df['Close'].shift()).abs()
        df['ATR'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['DIF'] = ema12 - ema26
        df['MACD'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['OSC'] = df['DIF'] - df['MACD']
        df['Vol_MA'] = df['Volume'].rolling(5).mean()

        return df.dropna(), stock_id, None
    except Exception as e:
        return None, original_id, str(e)

def slope(series, n=3):
    y = series.tail(n).dropna()
    if len(y) < n: return 0
    x = np.arange(len(y))
    try: return np.polyfit(x, y, 1)[0]
    except: return 0

def calculate_score(df):
    score = 0; reasons = []
    r = df.iloc[-1]
    macd_slope = slope(df['DIF'], 4)
    rsi_slope = slope(df['RSI'], 4)
    vol_slope = slope(df['Vol_MA'], 4)
    vwap_approx = (r['High'] + r['Low'] + r['Close']) / 3

    if r['MA5'] > r['MA10'] > r['MA20']: score += 3; reasons.append("均線多排")
    if macd_slope > 0: score += 2; reasons.append("MACD轉強")
    if r['Close'] > vwap_approx: score += 2; reasons.append("價>日均")
    if r['Close'] > r['MA20']: score += 1; reasons.append("站上月線")
    if rsi_slope > 0: score += 1; reasons.append("RSI向上")
    if vol_slope > 0: score += 1; reasons.append("量能增溫")
    
    day_range = r['High'] - r['Low']
    if day_range > 1.8 * r['ATR']: score -= 2; reasons.append("波動過大")

    return max(0, min(10, score)), reasons

# ==========================================
# 🖥️ 網頁介面
# ==========================================
st.title("🚀 股市動能 AI")

with st.sidebar:
    st.header("🔍 設定")
    # 👇 這裡新增一個輸入框讓用戶填 API Key
    api_key = st.text_input("Google API Key (選填)", type="password", help="去 Google AI Studio 申請免費 Key，填入後可啟用 AI 分析")
    ticker = st.text_input("股票代號", value="2330")
    run_btn = st.button("開始分析", type="primary")

if run_btn:
    with st.spinner(f"正在分析 {ticker} ..."):
        df, real_id, err = get_data_and_analyze(ticker)
        
        if df is None:
            st.error(f"❌ 錯誤: {err}")
        else:
            score, reasons = calculate_score(df)
            last_price = df.iloc[-1]['Close']
            
            # 1. 顯示基本數據
            c1, c2, c3 = st.columns(3)
            c1.metric("最新股價", f"{last_price:.2f}")
            c2.metric("量化評分", f"{score} 分")
            c3.markdown(f"**訊號:** {', '.join(reasons)}")
            st.progress(score / 10)
            
            # 2. 🤖 AI 分析區塊 (重點)
            st.divider()
            st.subheader("🤖 Gemini AI 分析師觀點")
            if api_key:
                # 呼叫我們上面寫的函數
                ai_comment = ask_gemini(api_key, real_id, df, score, reasons)
                st.info(ai_comment)
            else:
                st.warning("👉 請在左側輸入 Google API Key，即可解鎖 AI 自動解盤功能！")

            # 3. 圖表
            st.line_chart(df['Close'].tail(100))