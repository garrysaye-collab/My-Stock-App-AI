import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import google.generativeai as genai
from duckduckgo_search import DDGS

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
    if symbol.isdigit(): symbol = f"{symbol}.TW"
    elif not any(s in symbol for s in [".TW", ".TWO", ".HK", ".US", ".SS", ".SZ"]):
        if not (symbol.isalpha() and len(symbol) <= 4): symbol = f"{symbol}.TW"
    
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

def slope(series, n=3):
    y = series.tail(n).dropna()
    if len(y) < n: return 0
    return np.polyfit(np.arange(len(y)), y, 1)[0]

def detailed_scoring(df):
    r = df.iloc[-1]; prev = df.iloc[-2]
    details = []; total_score = 0
    
    macd_slope = slope(df['DIF'], 3)
    rsi_slope = slope(df['RSI'], 3)
    vwap_approx = (r['High'] + r['Low'] + r['Close']) / 3

    # 評分邏輯 (簡潔版，保持邏輯不變)
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

    # 扣分
    day_range = r['High'] - r['Low']
    cond_vol = day_range > 1.8 * r['ATR']
    s_vol = -2 if cond_vol else 0
    details.append({"準則": "⚠️ 波動過熱", "條件": ">1.8倍ATR", "狀態": "🚩 觸發" if cond_vol else "⚪ 正常", "得分": s_vol})
    total_score += s_vol
    
    return max(0, total_score), pd.DataFrame(details)

def comprehensive_backtest(df):
    log = []
    holding = False; entry_price = 0; entry_date = None; highest_after_entry = 0
    
    for i in range(1, len(df)):
        r = df.iloc[i]; prev = df.iloc[i-1]; curr_date = df.index[i]
        
        if not holding:
            if r['Close'] > r['MA20'] and r['OSC'] > 0 and r['Close'] > prev['High']:
                holding = True; entry_price = r['Close']; entry_date = curr_date; highest_after_entry = r['Close']
        elif holding:
            highest_after_entry = max(highest_after_entry, r['Close'])
            if r['Close'] < r['MA20'] or r['RSI'] > 85:
                profit_pct = (r['Close'] - entry_price) / entry_price * 100
                log.append({
                    "進場日期": entry_date.strftime('%Y-%m-%d'),
                    "出場日期": curr_date.strftime('%Y-%m-%d'),
                    "持股天數": (curr_date - entry_date).days,
                    "買入價格": round(entry_price, 2),
                    "賣出價格": round(r['Close'], 2),
                    "獲利%": round(profit_pct, 2),
                    "最高浮盈%": round((highest_after_entry - entry_price)/entry_price*100, 2),
                    "出場原因": "趨勢反轉" if r['Close'] < r['MA20'] else "過熱獲利"
                })
                holding = False
    return pd.DataFrame(log)

def get_ai_response(api_key, messages_history):
    """處理對話請求"""
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        # 將對話歷史轉換為 Gemini 格式
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
    ticker_input = st.text_input("股票代號", value="2330")
    run_btn = st.button("啟動全數據掃描", type="primary")
    
    if st.button("🗑️ 清除對話紀錄"):
        st.session_state.messages = []
        st.rerun()

# --- 1. 按下按鈕時：執行分析並儲存狀態 ---
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

            # 執行量化與回測
            score, score_details = detailed_scoring(df)
            bt_log = comprehensive_backtest(df)
            
            # 聯網搜尋
            try:
                with DDGS() as ddgs:
                    news = list(ddgs.text(f"{full_name} {real_symbol} 股息 PE 財報 新聞 2026", max_results=5))
                news_text = "\n".join([f"- {n['title']}: {n['body']}" for n in news])
            except:
                news_text = "聯網搜尋失敗，請依據現有技術面回答。"

            # 構建初始 System Prompt
            system_prompt = f"""
            你是一群專業投資經理人團隊 (總經分析、暗黑操盤手、巴菲特)。
            標的：{full_name} ({real_symbol})
            
            【最新量化得分】：{score}分
            {score_details.to_string()}
            
            【歷史回測統計】：
            總交易: {len(bt_log)} 次
            勝率: {(len(bt_log[bt_log['獲利%']>0])/len(bt_log)*100) if not bt_log.empty else 0:.1f}%
            累計報酬: {bt_log['獲利%'].sum() if not bt_log.empty else 0:.1f}%
            
            【即時聯網情報】：
            {news_text}
            
            請根據以上數據，給出第一份詳盡的辯證報告。
            """

            # 呼叫 AI 產生第一份報告
            initial_response = get_ai_response(api_key, [{"role": "user", "content": system_prompt}])
            
            # === 將數據存入 Session State ===
            st.session_state.data_context = {
                "df": df,
                "name": full_name,
                "symbol": real_symbol,
                "score": score,
                "score_details": score_details,
                "bt_log": bt_log
            }
            
            # 更新對話紀錄 (只保留 System Prompt 概念作為背景，不顯示給用戶看，直接顯示 AI 回答)
            st.session_state.messages = [
                {"role": "user", "content": system_prompt}, # 這一條隱藏的 context
                {"role": "assistant", "content": initial_response}
            ]
        else:
            st.error(err)

# --- 2. 顯示儀表板 (只要有資料就顯示) ---
if st.session_state.data_context:
    ctx = st.session_state.data_context
    
    st.title(f"🏛️ {ctx['name']} ({ctx['symbol']}) 戰情室")
    
    # 顯示圖表與數據
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("🎯 量化評分")
        st.metric("核心動能總分", f"{ctx['score']} / 10")
        st.dataframe(ctx['score_details'], use_container_width=True)
    
    with col2:
        st.subheader("📈 價格走勢")
        st.line_chart(ctx['df'][['Close', 'MA20']].tail(120))

    st.subheader("📜 歷史回測日誌")
    if not ctx['bt_log'].empty:
        st.dataframe(ctx['bt_log'], use_container_width=True)
    else:
        st.info("無交易紀錄")
        
    st.divider()

    # --- 3. 對話區域 (Chat Interface) ---
    st.subheader("💬 與經理人團隊對話")
    
    # 顯示歷史訊息 (排除第一條 User System Prompt，因為太長且是用戶看不懂的 raw data)
    for msg in st.session_state.messages:
        if msg == st.session_state.messages[0] and "你是一群專業投資經理人團隊" in msg['content']:
            continue # 跳過系統預設的第一條 Prompt 顯示
        
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 接收用戶新輸入
    if prompt := st.chat_input("向巴菲特或暗黑操盤手提問 (例如：莊家最近有在洗盤嗎？)..."):
        if not api_key:
            st.error("請先輸入 API Key")
        else:
            # 1. 顯示用戶問題
            with st.chat_message("user"):
                st.markdown(prompt)
            # 2. 加入歷史紀錄
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # 3. 呼叫 AI 回答
            with st.spinner("經理人團隊討論中..."):
                response = get_ai_response(api_key, st.session_state.messages)
            
            # 4. 顯示 AI 回答
            with st.chat_message("assistant"):
                st.markdown(response)
            # 5. 加入歷史紀錄
            st.session_state.messages.append({"role": "assistant", "content": response})

elif not run_btn:
    st.info("👈 請在左側輸入代號並點擊「啟動全數據掃描」")
