import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import google.generativeai as genai
from duckduckgo_search import DDGS
import time

# ==========================================
# 🔧 設定頁面
# ==========================================
st.set_page_config(page_title="全球股市 AI 戰情室 (終極修復版)", page_icon="📡", layout="wide")

# 初始化 Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stock_cache" not in st.session_state:
    st.session_state.stock_cache = None

# ==========================================
# 🌐 網路搜尋功能
# ==========================================
def search_web(keyword, max_results=5):
    """使用 DuckDuckGo 搜尋即時財經新聞"""
    try:
        results = []
        with DDGS() as ddgs:
            search_query = f"{keyword} stock news finance"
            ddgs_gen = ddgs.text(search_query, max_results=max_results)
            for r in ddgs_gen:
                results.append(f"標題: {r['title']}\n連結: {r['href']}\n摘要: {r['body']}")
        
        return "\n\n".join(results) if results else "查無相關即時新聞。"
    except Exception as e:
        return f"搜尋功能暫時無法使用 (可能是頻率限制): {str(e)}"

# ==========================================
# 📊 數據獲取與計算
# ==========================================
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
    
    return df

@st.cache_data(ttl=300)
def get_stock_data(ticker):
    """下載股價並獲取公司名稱"""
    ticker = ticker.strip().upper()
    
    # 智慧判斷後綴
    if ticker.isdigit():
        ticker = f"{ticker}.TW"
    
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        
        if df.empty:
            return None, None, "❌ 查無此股票數據，請確認代號 (如: 2330.TW, 600900.SS, AAPL)"
        
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
# 🧠 AI 核心 (含自動模型切換與重試)
# ==========================================
def chat_with_gemini(api_key, user_input, stock_context, news_context, system_prompt):
    if not api_key: return "⚠️ 請輸入 Google API Key"
    
    genai.configure(api_key=api_key)
    
    full_prompt = f"""
    【使用者問題】: {user_input}
    
    【當前股票即時數據】:
    {stock_context}
    
    【網路搜尋到的即時新聞/市場消息】:
    {news_context}
    
    請根據以上真實數據與新聞，進行專業團隊的辯證與分析。
    """

    # 定義模型優先順序：先試 Flash (快且新)，不行就換 Pro (舊但穩定)
    models_to_try = ['gemini-1.5-flash', 'gemini-pro']
    
    last_error = ""

    for model_name in models_to_try:
        try:
            # 建立模型
            model = genai.GenerativeModel(model_name, system_instruction=system_prompt)
            
            # 重試機制 (處理 429 Rate Limit)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = model.generate_content(full_prompt)
                    return response.text # 成功就直接回傳
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg: # 如果是配額限制
                        time.sleep(5 * (attempt + 1)) # 等待後重試
                        continue
                    elif "404" in error_msg or "not found" in error_msg.lower():
                        # 如果是找不到模型 (404)，跳出內層迴圈，讓外層迴圈換下一個模型
                        raise Exception("Model Not Found") 
                    else:
                        raise e # 其他錯誤直接拋出

        except Exception as e:
            last_error = str(e)
            if "Model Not Found" in str(e):
                print(f"⚠️ 模型 {model_name} 無法使用，嘗試切換至下一個模型...")
                continue # 換下一個模型
            else:
                # 如果是其他嚴重錯誤，就不換模型了，直接回報
                return f"❌ AI 分析錯誤 ({model_name}): {str(e)}"
    
    return f"❌ 所有模型嘗試皆失敗。最後錯誤: {last_error}"

# ==========================================
# 🖥️ UI 介面
# ==========================================
st.title("📡 全球股市 AI 戰情室")
st.caption("自動切換模型版：優先使用 Flash，自動降級至 Pro")

with st.sidebar:
    st.header("⚙️ 設定")
    api_key = st.text_input("Google API Key", type="password")
    
    st.divider()
    ticker_input = st.text_input("輸入代號 (如 2330, AAPL)", value="2330")
    
    if st.button("🚀 啟動分析", type="primary"):
        if not api_key:
            st.error("請先輸入 API Key！")
        else:
            st.session_state.messages = [] 
            st.session_state.stock_cache = None 
            
            with st.spinner(f"正在連線交易所與搜尋 {ticker_input} 最新新聞..."):
                df, info, err = get_stock_data(ticker_input)
                
                if df is not None:
                    # 1. 搜尋網路新聞
                    news_text = search_web(f"{info['name']} {info['ticker']}")
                    
                    # 2. 整理數據文本
                    latest = df.iloc[-1]
                    stock_context_str = f"""
                    股票: {info['name']} ({info['ticker']})
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
                    
                    # 4. 觸發 AI
                    initial_prompt = "請根據傳入的數據與新聞，對這檔股票進行一次完整的「莊家團隊」多角度分析。"
                    system_instruction = "你是一個由「總體經濟師、技術分析師、莊家操盤手、巴菲特」組成的投資團隊。請引用新聞並用陰謀論視角解讀。"
                    
                    ai_reply = chat_with_gemini(api_key, initial_prompt, stock_context_str, news_text, system_instruction)
                    st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                    
                else:
                    st.error(err)

# === 主要顯示區 ===

if st.session_state.stock_cache:
    cache = st.session_state.stock_cache
    df = cache['df']
    info = cache['info']
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric(f"{info['name']}", f"{df.iloc[-1]['Close']:.2f}")
    with col2:
        st.info(f"📰 **網路新聞摘要**：\n{cache['news'][:150]}...")

    st.line_chart(df['Close'])

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("追問 AI..."):
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("AI 團隊思考中..."):
            system_instruction = "你是一個專業股票分析團隊。"
            response = chat_with_gemini(api_key, user_input, cache['context_str'], cache['news'], system_instruction)
            
            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
