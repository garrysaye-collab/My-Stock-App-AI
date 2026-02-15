import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import google.generativeai as genai
from duckduckgo_search import DDGS  # 引入搜尋功能
import time

# ==========================================
# 🔧 設定頁面
# ==========================================
st.set_page_config(page_title="全球股市 AI 戰情室", page_icon="📡", layout="wide")

# 初始化 Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stock_cache" not in st.session_state:
    st.session_state.stock_cache = None

# ==========================================
# 🌐 網路搜尋功能 (解決無法查詢外部資訊問題)
# ==========================================
def search_web(keyword, max_results=5):
    """使用 DuckDuckGo 搜尋即時財經新聞"""
    try:
        results = []
        with DDGS() as ddgs:
            # 搜尋關鍵字加上 "stock news" 或 "股價新聞" 以提高精準度
            search_query = f"{keyword} stock news finance"
            ddgs_gen = ddgs.text(search_query, max_results=max_results)
            for r in ddgs_gen:
                results.append(f"標題: {r['title']}\n連結: {r['href']}\n摘要: {r['body']}")
        
        return "\n\n".join(results) if results else "查無相關即時新聞。"
    except Exception as e:
        return f"搜尋功能暫時無法使用: {str(e)}"

# ==========================================
# 📊 數據獲取與計算
# ==========================================
def slope(series, n=3):
    y = series.tail(n).dropna()
    if len(y) < n: return 0
    x = np.arange(len(y))
    try: return np.polyfit(x, y, 1)[0]
    except: return 0

def calculate_technical_indicators(df):
    """計算技術指標"""
    # 均線
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = ema12 - ema26
    df['MACD'] = df['DIF'].ewm(span=9, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # KD
    low_min = df['Low'].rolling(9).min()
    high_max = df['High'].rolling(9).max()
    df['RSV'] = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = df['RSV'].ewm(com=2).mean()
    
    return df

@st.cache_data(ttl=300)
def get_stock_data(ticker):
    """下載股價並獲取公司名稱"""
    ticker = ticker.strip().upper()
    
    # 智慧判斷後綴 (解決 600900.SS 錯誤問題)
    # 如果純數字，預設為台股，除非使用者自己輸入了後綴
    if ticker.isdigit():
        ticker = f"{ticker}.TW"
    
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        
        if df.empty:
            return None, None, "❌ 查無此股票數據，請確認代號 (如: 2330.TW, 600900.SS, AAPL)"
        
        # 嘗試獲取真實公司名稱
        try:
            info = stock.info
            company_name = info.get('longName') or info.get('shortName') or ticker
            currency = info.get('currency', 'Unknown')
        except:
            company_name = ticker
            currency = "?"

        df = calculate_technical_indicators(df)
        return df, {"name": company_name, "currency": currency, "ticker": ticker}, None
        
    except Exception as e:
        return None, None, str(e)

# ==========================================
# 🧠 AI 核心
# ==========================================
def chat_with_gemini(api_key, user_input, stock_context, news_context, system_prompt):
    if not api_key: return "⚠️ 請輸入 Google API Key"
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash', system_instruction=system_prompt) 
        # 註: 建議使用 gemini-2.0-flash 或 1.5-flash 速度較快
        
        # 構建包含「即時數據」與「新聞」的完整 Prompt
        full_prompt = f"""
        【使用者問題】: {user_input}
        
        【當前股票即時數據】:
        {stock_context}
        
        【網路搜尋到的即時新聞/市場消息】(這是真實的外部資訊，請依此分析):
        {news_context}
        
        請根據以上真實數據與新聞，進行專業團隊的辯證與分析。
        """
        
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"❌ AI 發生錯誤: {str(e)}"

# ==========================================
# 🖥️ UI 介面
# ==========================================
st.title("📡 全球股市 AI 戰情室 (聯網版)")
st.caption("結合 yfinance 數據 + DuckDuckGo 即時新聞搜尋 + Gemini AI 分析")

with st.sidebar:
    st.header("⚙️ 設定")
    api_key = st.text_input("Google API Key", type="password")
    st.markdown("[取得 Google API Key](https://aistudio.google.com/app/apikey)")
    
    st.divider()
    st.subheader("🔍 股票代號範例")
    st.code("台積電: 2330\n工行: 600900.SS\n蘋果: AAPL\n騰訊: 0700.HK")
    ticker_input = st.text_input("輸入代號", value="600900.SS")
    
    if st.button("🚀 啟動分析", type="primary"):
        st.session_state.messages = [] # 清空舊對話
        with st.spinner(f"正在連線交易所與搜尋 {ticker_input} 最新新聞..."):
            df, info, err = get_stock_data(ticker_input)
            
            if df is not None:
                # 1. 搜尋網路新聞
                news_text = search_web(f"{info['name']} {info['ticker']}")
                
                # 2. 整理數據文本
                latest = df.iloc[-1]
                stock_context_str = f"""
                股票: {info['name']} ({info['ticker']})
                幣別: {info['currency']}
                收盤價: {latest['Close']:.2f}
                MA5: {latest['MA5']:.2f} | MA20: {latest['MA20']:.2f} | MA60: {latest['MA60']:.2f}
                RSI: {latest['RSI']:.2f} | MACD: {latest['MACD']:.2f}
                """
                
                # 3. 存入 Session
                st.session_state.stock_cache = {
                    "df": df,
                    "info": info,
                    "news": news_text,
                    "context_str": stock_context_str
                }
                
                # 4. 觸發 AI 第一句話
                initial_prompt = "請根據傳入的數據與新聞，對這檔股票進行一次完整的「莊家團隊」多角度分析。"
                
                # 系統 Prompt 設定
                system_instruction = """
                你是一個由「總體經濟師、技術分析師、量化專家、莊家操盤手、巴菲特」組成的投資團隊。
                
                重要規則：
                1. 必須基於提供的【即時數據】和【網路新聞】進行分析，不要捏造數據。
                2. 如果新聞中提到具體的利好或利空（如財報、政策、收購），請務必引用並納入分析。
                3. 「莊家操盤手」需用陰謀論視角解讀新聞（例如：這是為了出貨發布的假利好）。
                4. 最後由「巴菲特」給出買入、觀望或賣出的明確建議。
                """
                
                ai_reply = chat_with_gemini(api_key, initial_prompt, stock_context_str, news_text, system_instruction)
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                
            else:
                st.error(err)

# === 主要顯示區 ===

if st.session_state.stock_cache:
    cache = st.session_state.stock_cache
    df = cache['df']
    info = cache['info']
    
    # 顯示基本資訊
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric(f"{info['name']}", f"{df.iloc[-1]['Close']:.2f}", f"{df.iloc[-1]['Close'] - df.iloc[-2]['Close']:.2f}")
    with col2:
        st.info(f"📰 **已獲取最新網路情報**：\n{cache['news'][:150]}... (已傳送給 AI 進行分析)")

    # 顯示圖表
    st.line_chart(df['Close'])

    # 對話區
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 使用者輸入
    if user_input := st.chat_input("追問 AI (例如：這則新聞對明天股價有什麼影響？)"):
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("AI 團隊正在根據新聞辯證中..."):
            system_instruction = "你是一個專業股票分析團隊，請根據已有的數據與新聞回答用戶問題。"
            response = chat_with_gemini(api_key, user_input, cache['context_str'], cache['news'], system_instruction)
            
            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("👈 請在左側輸入 API Key 與 股票代號並點擊「啟動分析」")
