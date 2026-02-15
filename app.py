import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import google.generativeai as genai
import datetime

# ==========================================
# 🔧 系統設定與狀態初始化
# ==========================================
st.set_page_config(page_title="專業量化與 AI 經理人戰情室", page_icon="🏦", layout="wide")

# 初始化 Session State
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

    # 評分邏輯
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
    """
    處理對話請求：啟用 Google Search Grounding 與 動態時間注入
    """
    genai.configure(api_key=api_key)
    
    # 1. 獲取精確的當前時間，強制對齊時間軸
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 2. 定義系統指令 (System Instruction) - 強制聯網與角色扮演
    sys_instruction = f"""
    現在時間是：{current_time}。
    你是【全能基金經理人團隊】，包含：
    1. 總經分析師 (關注宏觀、利率、地緣政治)
    2. 暗黑操盤手 (關注籌碼、騙線、養套殺)
    3. 巴菲特 (關注護城河、現金流、價值)

    【最高指令】：
    - 你擁有 Google Search 工具。針對用戶的每個問題，必須**優先使用工具**查閱該股票的「最新股價」、「最新新聞 (24小時內)」與「最新財報」。
    - **嚴禁**使用你訓練資料中的舊數據來回答關於「現價」、「本益比」或「近期趨勢」的問題。
    - 如果回測數據 (用戶提供) 與即時新聞 (你搜尋到的) 衝突，以**即時新聞**為準，並指出市場是否發生了結構性改變。
    - 回答風格：激烈辯證，數據說話，最後由巴菲特給出結論。
    """

    try:
        # 3. 初始化模型時啟用 Search 工具 (建議使用 gemini-1.5-pro 以獲得最佳搜尋推理能力)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro", 
            tools='google_search_retrieval', 
            system_instruction=sys_instruction
        )

        # 4. 轉換對話歷史
        gemini_hist = []
        for m in messages_history:
            role = "user" if m["role"] == "user" else "model"
            gemini_hist.append({"role": role, "parts": [str(m["content"])]})
            
        # 5. 生成回應 (啟用自動搜尋)
        response = model.generate_content(gemini_hist)
        
        # 6. 解析回應 (包含搜尋來源引用)
        final_text = response.text
        
        # 檢查是否有 Grounding Metadata (搜尋來源)
        if response.candidates[0].grounding_metadata.search_entry_point:
            search_html = response.candidates[0].grounding_metadata.search_entry_point.rendered_content
            final_text += "\n\n" + "🔍 **資料來源與即時驗證：**\n" + search_html
        
        return final_text

    except Exception as e:
        return f"AI 經理人連線或搜尋錯誤: {str(e)}\n(請確認 API Key 是否有效，並支援 Google Search Grounding)"

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
            
            # --- 重點修改：這裡不再使用 DDGS，而是準備純數據給 AI ---
            
            # 構建初始 Prompt (只提供技術面數據，要求 AI 自己去查基本面)
            initial_data_prompt = f"""
            【量化技術面數據輸入】
            標的：{full_name} ({real_symbol})
            
            【技術面診斷】：
            - 核心動能得分：{score}/10
            - 詳細指標狀態：\n{score_details.to_string()}
            
            【歷史回測統計 (過去2年)】：
            - 總交易次數: {len(bt_log)} 次
            - 策略勝率: {(len(bt_log[bt_log['獲利%']>0])/len(bt_log)*100) if not bt_log.empty else 0:.1f}%
            - 累計報酬: {bt_log['獲利%'].sum() if not bt_log.empty else 0:.1f}%
            
            請根據上述「技術與量化數據」，並立刻使用你的 Google Search 工具查詢該公司的「最新財報」、「除權息消息」與「產業新聞」，開始第一輪的多空辯證分析。
            """

            # 呼叫 AI 產生第一份報告 (AI 會在這裡觸發搜尋)
            initial_response = get_ai_response(api_key, [{"role": "user", "content": initial_data_prompt}])
            
            # === 將數據存入 Session State ===
            st.session_state.data_context = {
                "df": df,
                "name": full_name,
                "symbol": real_symbol,
                "score": score,
                "score_details": score_details,
                "bt_log": bt_log
            }
            
            # 更新對話紀錄
            st.session_state.messages = [
                {"role": "user", "content": initial_data_prompt}, 
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
    
    # 顯示歷史訊息 (排除第一條 User System Prompt)
    for msg in st.session_state.messages:
        if msg == st.session_state.messages[0] and "【量化技術面數據輸入】" in str(msg['content']):
            continue 
        
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
            
            # 3. 呼叫 AI 回答 (這裡也會觸發 Google Search)
            with st.spinner("經理人團隊討論中 (正在聯網檢索)..."):
                response = get_ai_response(api_key, st.session_state.messages)
            
            # 4. 顯示 AI 回答
            with st.chat_message("assistant"):
                st.markdown(response)
            # 5. 加入歷史紀錄
            st.session_state.messages.append({"role": "assistant", "content": response})

elif not run_btn:
    st.info("👈 請在左側輸入代號並點擊「啟動全數據掃描」")
