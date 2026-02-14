import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import google.generativeai as genai

# ==========================================
# 🔧 設定頁面
# ==========================================
st.set_page_config(page_title="股市動能 AI", page_icon="📈", layout="wide")

# ==========================================
# 🧠 AI 分析模組 (已升級為 Gemini 2.5 Flash)
# ==========================================
def ask_gemini(api_key, stock_id, df, score, reasons):
    if not api_key:
        return None
    
    try:
        # 設定 API
        genai.configure(api_key=api_key)
        
        # ✅ 修正點：改用診斷出的最新模型
        model = genai.GenerativeModel('gemini-2.5-flash')

        # 準備餵給 AI 的數據
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 構建提示詞 (Prompt)
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
        請用繁體中文，針對投資人給出以下結構的建議（語氣要專業、果斷）：
        
        1. **📊 多空趨勢判斷**：(一句話判斷目前是強多、偏多、盤整、偏空還是強空)
        2. **💡 操作建議**：(具體建議，例如：適合進場、建議觀望、或是設好停損續抱)
        3. **⚠️ 風險提示**：(指出目前最需要注意的一個風險點，例如乖離過大或量能不足)
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"❌ AI 分析發生錯誤: {str(e)} (請檢查 API Key 是否正確)"

# ==========================================
# 📊 核心數據邏輯 (抓取與計算)
# ==========================================
@st.cache_data(ttl=300)
def get_data_and_analyze(stock_id):
    stock_id = stock_id.strip().upper()
    original_id = stock_id
    
    # 台股代號處理
    if stock_id.isdigit(): 
        stock_id = f"{stock_id}.TW"
    elif not any(x in stock_id for x in [".TW", ".TWO", ".HK", ".US", ".SS", ".SZ"]): 
        stock_id = f"{stock_id}.TW"

    try:
        # 下載資料
        df = yf.download(stock_id, period="1y", progress=False)
        
        # 處理 MultiIndex 欄位問題
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.empty: 
            return None, stock_id, "查無資料，請確認代號是否正確"
            
        if 'Adj Close' in df.columns: 
            df['Close'] = df['Adj Close']
            
        df = df.dropna()

        # 計算指標
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ATR
        hl = df['High'] - df['Low']
        hc = (df['High'] - df['Close'].shift()).abs()
        lc = (df['Low'] - df['Close'].shift()).abs()
        df['ATR'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

        # MACD
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
    score = 0
    reasons = []
    r = df.iloc[-1]
    
    macd_slope = slope(df['DIF'], 4)
    rsi_slope = slope(df['RSI'], 4)
    vol_slope = slope(df['Vol_MA'], 4)
    vwap_approx = (r['High'] + r['Low'] + r['Close']) / 3

    # 評分規則
    if r['MA5'] > r['MA10'] > r['MA20']: score += 3; reasons.append("均線多排")
    if macd_slope > 0: score += 2; reasons.append("MACD轉強")
    if r['Close'] > vwap_approx: score += 2; reasons.append("價>日均")
    if r['Close'] > r['MA20']: score += 1; reasons.append("站上月線")
    if rsi_slope > 0: score += 1; reasons.append("RSI向上")
    if vol_slope > 0: score += 1; reasons.append("量能增溫")
    
    # 扣分項
    day_range = r['High'] - r['Low']
    if day_range > 1.8 * r['ATR']: score -= 2; reasons.append("波動過大(風險)")

    return max(0, min(10, score)), reasons

# ==========================================
# 🖥️ 網頁介面
# ==========================================
st.title("🚀 股市動能 AI 分析儀")
st.caption("結合量化數據與 Gemini 2.5 AI 的智慧分析")

with st.sidebar:
    st.header("⚙️ 設定與輸入")
    
    # API Key 輸入框
    api_key = st.text_input("🔑 Google Gemini API Key", type="password", help="請輸入您的 API Key 以啟用 AI 分析功能")
    
    if not api_key:
        st.warning("👉 請輸入 API Key 才能看到 AI 的詳細解盤喔！")
    
    st.divider()
    
    ticker = st.text_input("股票代號", value="2330", help="支援台股(2330)、美股(AAPL)、陸股(600519.SS)")
    run_btn = st.button("🔍 開始分析", type="primary", use_container_width=True)

if run_btn:
    with st.spinner(f"正在抓取 {ticker} 資料..."):
        df, real_id, err = get_data_and_analyze(ticker)
        
        if df is None:
            st.error(f"❌ 錯誤: {err}")
        else:
            # 計算分數
            score, reasons = calculate_score(df)
            last_price = df.iloc[-1]['Close']
            change = last_price - df.iloc[-2]['Close']
            pct_change = (change / df.iloc[-2]['Close']) * 100
            
            # --- 1. 顯示核心指標 ---
            st.subheader(f"📊 {real_id} 分析結果")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("最新股價", f"{last_price:.2f}", f"{change:.2f} ({pct_change:.2f}%)")
            col2.metric("量化動能評分", f"{score} 分", delta_color="normal")
            col3.write("**觸發訊號:**")
            for r in reasons:
                st.write(f"- {r}")
                
            st.progress(score / 10)
            
            # --- 2. AI 分析區塊 ---
            st.divider()
            st.subheader("🤖 Gemini AI 觀點")
            
            if api_key:
                with st.spinner("🤖 Gemini 2.5 正在閱讀線圖，請稍候..."):
                    ai_response = ask_gemini(api_key, real_id, df, score, reasons)
                    if ai_response:
                        st.success("分析完成！")
                        st.markdown(ai_response)
                    else:
                        st.error("AI 無法回應，請檢查 API Key。")
            else:
                st.info("💡 輸入 Google API Key 即可解鎖 AI 具體操作建議。")

            # --- 3. 走勢圖 ---
            st.divider()
            st.subheader("📈 近期走勢圖")
            st.line_chart(df['Close'].tail(100))
