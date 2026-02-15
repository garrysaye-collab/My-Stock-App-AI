import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import google.generativeai as genai
from duckduckgo_search import DDGS
import time

# ==========================================
# 🔧 系統設定與狀態初始化
# ==========================================
st.set_page_config(page_title="專業量化與 AI 經理人戰情室", page_icon="🏦", layout="wide")

# 初始化 Session State (讓資料在對話時不會消失)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "data_context" not in st.session_state:
    st.session_state.data_context = None

# ==========================================
# 🕵️ 核心功能函數 (數據、計算、AI)
# ==========================================
def get_verified_data(symbol):
    symbol = symbol.strip().upper()
    if not symbol: return None, None, None, "請輸入代號"
    
    try:
        t = yf.Ticker(symbol)
        df = t.history(period="2y") # 抓兩年確保均線計算
        if df.empty: return None, None, symbol, "查無數據"
        
        # 處理 MultiIndex 欄位 (yfinance 新版常見問題)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 抓取公司名稱
        info = t.info
        full_name = info.get('longName') or info.get('shortName') or symbol
        
        return df, full_name, symbol, None
    except Exception as e:
        return None, None, symbol, str(e)

def slope(series, n=3):
    y = series.tail(n).dropna()
    if len(y) < n: return 0
    return np.polyfit(np.arange(len(y)), y, 1)[0]

def detailed_scoring(df):
    """細緻評分邏輯"""
    r = df.iloc[-1]
    prev = df.iloc[-2]
    details = []
    total_score = 0
    
    macd_slope = slope(df['DIF'], 3)
    rsi_slope = slope(df['RSI'], 3)
    vwap_approx = (r['High'] + r['Low'] + r['Close']) / 3

    # 1. 均線多頭排列
    cond1 = r['MA5'] > r['MA10'] > r['MA20']
    s1 = 3 if cond1 else 0
    details.append({"準則": "均線多頭排列", "條件": "MA5 > MA10 > MA20", "狀態": "✅ 通過" if cond1 else "❌ 未達成", "得分": s1})
    total_score += s1

    # 2. MACD 動能
    cond2 = macd_slope > 0 and r['OSC'] > 0
    s2 = 2 if cond2 else 0
    details.append({"準則": "MACD 轉強", "條件": "DIF斜率 > 0 且 OSC > 0", "狀態": "✅ 通過" if cond2 else "❌ 未達成", "得分": s2})
    total_score += s2

    # 3. 價在均價之上
    cond3 = r['Close'] > vwap_approx
    s3 = 2 if cond3 else 0
    details.append({"準則": "價格優勢", "條件": "收盤價 > VWAP", "狀態": "✅ 通過" if cond3 else "❌ 未達成", "得分": s3})
    total_score += s3

    # 4. 站上月線
    cond4 = r['Close'] > r['MA20']
    s4 = 1 if cond4 else 0
    details.append({"準則": "站上月線", "條件": "收盤價 > MA20", "狀態": "✅ 通過" if cond4 else "❌ 未達成", "得分": s4})
    total_score += s4

    # 5. RSI 向上
    cond5 = rsi_slope > 0
    s5 = 1 if cond5 else 0
    details.append({"準則": "RSI 動能", "條件": "RSI 斜率 > 0", "狀態": "✅ 通過" if cond5 else "❌ 未達成", "得分": s5})
    total_score += s5

    # 6. 量能爆發
    vol_ma5 = df['Volume'].tail(5).mean()
    cond6 = r['Volume'] > vol_ma5
    s6 = 1 if cond6 else 0
    details.append({"準則": "量能增溫", "條件": "今日量 > 5日均量", "狀態": "✅ 通過" if cond6 else "❌ 未達成", "得分": s6})
    total_score += s6

    # 7. 扣分項：波動過大
    day_range = r['High'] - r['Low']
    cond7 = day_range > 1.8 * r['ATR']
    s7 = -2 if cond7 else 0
    details.append({"準則": "⚠️ 波動過熱", "條件": ">1.8倍 ATR", "狀態": "🚩 觸發扣分" if cond7 else "⚪ 正常", "得分": s7})
    total_score += s7

    return max(0, total_score), pd.DataFrame(details)

def comprehensive_backtest(df):
    log = []
    holding = False; entry_price = 0; entry_date = None; highest_after_entry = 0

    for i in range(20, len(df)):
        r = df.iloc[i]; prev = df.iloc[i-1]
        curr_date = df.index[i]

        if not holding:
            if r['Close'] > r['MA20'] and r['OSC'] > 0 and r['Close'] > prev['High']:
                holding = True; entry_price = r['Close']; entry_date = curr_date
                highest_after_entry = r['Close']
        elif holding:
            highest_after_entry = max(highest_after_entry, r['Close'])
            if r['Close'] < r['MA20'] or r['RSI'] > 85:
                profit_pct = (r['Close'] - entry_price) / entry_price * 100
                log.append({
                    "進場日期": entry_date, "出場日期": curr_date,
                    "進場價": round(entry_price, 2), "出場價": round(r['Close'], 2),
                    "獲利%": round(profit_pct, 2),
                    "出場原因": "趨勢反轉" if r['Close'] < r['MA20'] else "過熱獲利"
                })
                holding = False
    return pd.DataFrame(log)

def get_ai_response(api_key, messages_history):
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel("gemini-1.5-flash") # 修正 model 名稱
        gemini_hist = []
        for m in messages_history:
            role = "user" if m["role"] == "user" else "model"
            gemini_hist.append({"role": role, "parts": [m["content"]]})
        response = model.generate_content(gemini_hist)
        return response.text
    except Exception as e:
        return f"AI 經理人連線錯誤: {str(e)}"

# ==========================================
# 🖥️ UI 介面與主邏輯
# ==========================================
with st.sidebar:
    st.header("🔑 戰情室控制台")
    api_key = st.text_input("Google API Key", type="password")
    ticker_input = st.text_input("股票代號", value="2330.TW") # 台股需加 .TW
    run_btn = st.button("啟動全數據掃描", type="primary")
    
    if st.button("🗑️ 清除對話紀錄"):
        st.session_state.messages = []
        st.rerun()

# --- 1. 執行掃描與分析 ---
if run_btn:
    if not api_key:
        st.error("請先輸入 API Key")
    else:
        with st.spinner(f"正在調閱 {ticker_input} 檔案與聯網數據..."):
            df, full_name, real_symbol, err = get_verified_data(ticker_input)
            
            if df is not None:
                # 技術指標計算
                df['MA5'] = df['Close'].rolling(5).mean()
                df['MA10'] = df['Close'].rolling(10).mean()
                df['MA20'] = df['Close'].rolling(20).mean()
                df['DIF'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
                df['MACD'] = df['DIF'].ewm(span=9).mean()
                df['OSC'] = df['DIF'] - df['MACD']
                df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss.replace(0, np.nan)
                df['RSI'] = 100 - (100 / (1 + rs))

                score, score_details = detailed_scoring(df)
                bt_log = comprehensive_backtest(df)
                
                # 聯網搜尋 (模擬經理人收集情報)
                try:
                    with DDGS() as ddgs:
                        news = list(ddgs.text(f"{full_name} {real_symbol} 財報展望 2026", max_results=3))
                    news_text = "\n".join([f"- {n['title']}: {n['body']}" for n in news])
                except:
                    news_text = "聯網搜尋暫時不可用。"

                system_prompt = f"""你是一群專業投資經理人(總經分析、暗黑操盤手、巴菲特)。
                標的：{full_name} ({real_symbol})
                量化得分：{score}分。
                歷史勝率：{(len(bt_log[bt_log['獲利%']>0])/len(bt_log)*100) if not bt_log.empty else 0:.1f}%。
                即時新聞：{news_text}
                請給出深度辯證報告。"""

                initial_response = get_ai_response(api_key, [{"role": "user", "content": system_prompt}])
                
                st.session_state.data_context = {
                    "df": df, "name": full_name, "symbol": real_symbol,
                    "score": score, "score_details": score_details, "bt_log": bt_log
                }
                st.session_state.messages = [
                    {"role": "user", "content": system_prompt},
                    {"role": "assistant", "content": initial_response}
                ]
            else:
                st.error(err)

# --- 2. 顯示儀表板 ---
if st.session_state.data_context:
    ctx = st.session_state.data_context
    st.title(f"🏛️ {ctx['name']} ({ctx['symbol']}) 戰情室")
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("🎯 量化評分")
        st.metric("核心動能總分", f"{ctx['score']} / 10")
        st.table(ctx['score_details'])
    with c2:
        st.subheader("📈 價格走勢")
        st.line_chart(ctx['df'][['Close', 'MA20']].tail(100))

    st.subheader("📜 歷史回測日誌")
    if not ctx['bt_log'].empty:
        st.dataframe(ctx['bt_log'], use_container_width=True)
    else:
        st.info("過去兩年內未觸發完整交易訊號。")
        
    st.divider()

    # --- 3. 對話區域 ---
    st.subheader("💬 與經理人團隊對話")
    for msg in st.session_state.messages:
        if "你是一群專業投資經理人" in msg['content']: continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if chat_input := st.chat_input("詢問專家意見..."):
        st.session_state.messages.append({"role": "user", "content": chat_input})
        with st.chat_message("user"): st.markdown(chat_input)
        with st.chat_message("assistant"):
            response = get_ai_response(api_key, st.session_state.messages)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
elif not run_btn:
    st.info("👈 請在左側輸入代號並點擊「啟動全數據掃描」")
