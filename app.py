import numpy as np
import google.generativeai as genai
from duckduckgo_search import DDGS
import time

# ==========================================
# 🔧 系統設定
# 🔧 系統設定與狀態初始化
# ==========================================
st.set_page_config(page_title="專業量化與 AI 經理人戰情室", page_icon="🏦", layout="wide")

# 初始化 Session State (讓資料在對話時不會消失)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "data_context" not in st.session_state:
    st.session_state.data_context = None

# ==========================================
# 🕵️ 公司身份與數據抓取
# 🕵️ 核心功能函數 (數據、計算、AI)
# ==========================================
def get_verified_data(symbol):
    symbol = symbol.strip().upper()
@@ -25,110 +27,65 @@ def get_verified_data(symbol):

    try:
        t = yf.Ticker(symbol)
        df = t.history(period="2y") # 抓兩年確保均線計算
        df = t.history(period="2y")
        if df.empty: return None, None, symbol, "查無數據"
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        # 處理 MultiIndex 欄位
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 抓取公司名稱
        info = t.info
        full_name = info.get('longName') or info.get('shortName') or symbol
        
        return df, full_name, symbol, None
    except Exception as e:
        return None, None, symbol, str(e)

# ==========================================
# 📈 技術指標與【不簡化】評分系統
# ==========================================
def slope(series, n=3):
    y = series.tail(n).dropna()
    if len(y) < n: return 0
    return np.polyfit(np.arange(len(y)), y, 1)[0]

def detailed_scoring(df):
    """
    極細緻評分邏輯：返回 (總分, 詳情列表)
    """
    r = df.iloc[-1]
    prev = df.iloc[-2]
    details = []
    total_score = 0
    r = df.iloc[-1]; prev = df.iloc[-2]
    details = []; total_score = 0

    # 指標計算
    macd_slope = slope(df['DIF'], 3)
    rsi_slope = slope(df['RSI'], 3)
    vwap_approx = (r['High'] + r['Low'] + r['Close']) / 3

    # 1. 均線多頭排列
    cond1 = r['MA5'] > r['MA10'] > r['MA20']
    s1 = 3 if cond1 else 0
    details.append({"準則": "均線多頭排列", "條件": "MA5 > MA10 > MA20", "實際數值": f"{r['MA5']:.2f}>{r['MA10']:.2f}>{r['MA20']:.2f}", "狀態": "✅ 通過" if cond1 else "❌ 未達成", "得分": s1})
    total_score += s1

    # 2. MACD 動能
    cond2 = macd_slope > 0 and r['OSC'] > 0
    s2 = 2 if cond2 else 0
    details.append({"準則": "MACD 轉強", "條件": "DIF斜率 > 0 且 OSC > 0", "實際數值": f"斜率:{macd_slope:.4f}, OSC:{r['OSC']:.2f}", "狀態": "✅ 通過" if cond2 else "❌ 未達成", "得分": s2})
    total_score += s2

    # 3. 價在均價之上
    cond3 = r['Close'] > vwap_approx
    s3 = 2 if cond3 else 0
    details.append({"準則": "價格優勢", "條件": "收盤價 > 當日均價(VWAP)", "實際數值": f"{r['Close']:.2f} > {vwap_approx:.2f}", "狀態": "✅ 通過" if cond3 else "❌ 未達成", "得分": s3})
    total_score += s3

    # 4. 站上月線
    cond4 = r['Close'] > r['MA20']
    s4 = 1 if cond4 else 0
    details.append({"準則": "站上月線", "條件": "收盤價 > MA20", "實際數值": f"{r['Close']:.2f} > {r['MA20']:.2f}", "狀態": "✅ 通過" if cond4 else "❌ 未達成", "得分": s4})
    total_score += s4

    # 5. RSI 向上
    cond5 = rsi_slope > 0
    s5 = 1 if cond5 else 0
    details.append({"準則": "RSI 動能", "條件": "RSI 斜率 > 0", "實際數值": f"RSI:{r['RSI']:.2f}, 斜率:{rsi_slope:.2f}", "狀態": "✅ 通過" if cond5 else "❌ 未達成", "得分": s5})
    total_score += s5

    # 6. 量能爆發
    vol_ma5 = df['Volume'].tail(5).mean()
    cond6 = r['Volume'] > vol_ma5
    s6 = 1 if cond6 else 0
    details.append({"準則": "量能增溫", "條件": "今日成交量 > 5日均量", "實際數值": f"{r['Volume']:.0f} > {vol_ma5:.0f}", "狀態": "✅ 通過" if cond6 else "❌ 未達成", "得分": s6})
    total_score += s6

    # 7. 扣分項：波動過大
    day_range = r['High'] - r['Low']
    cond7 = day_range > 1.8 * r['ATR']
    s7 = -2 if cond7 else 0
    details.append({"準則": "⚠️ 波動過熱(扣分)", "條件": "高低震幅 > 1.8倍 ATR", "實際數值": f"{day_range:.2f} > {1.8*r['ATR']:.2f}", "狀態": "🚩 觸發扣分" if cond7 else "⚪ 正常", "得分": s7})
    total_score += s7
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

# ==========================================
# 📜 完整歷史交易回測紀錄
# ==========================================
def comprehensive_backtest(df):
    log = []
    holding = False; entry_price = 0; entry_date = None; highest_after_entry = 0

    for i in range(1, len(df)):
        r = df.iloc[i]; prev = df.iloc[i-1]
        curr_date = df.index[i]
        r = df.iloc[i]; prev = df.iloc[i-1]; curr_date = df.index[i]

        # 買入訊號: 站上月線 + MACD紅柱 + 突破前高
        if not holding:
            if r['Close'] > r['MA20'] and r['OSC'] > 0 and r['Close'] > prev['High']:
                holding = True; entry_price = r['Close']; entry_date = curr_date
                highest_after_entry = r['Close']
        
        # 持有中判斷賣出
                holding = True; entry_price = r['Close']; entry_date = curr_date; highest_after_entry = r['Close']
        elif holding:
            highest_after_entry = max(highest_after_entry, r['Close'])
            # 賣出訊號: 跌破月線 或 RSI過熱(85)
            if r['Close'] < r['MA20'] or r['RSI'] > 85:
                profit_pct = (r['Close'] - entry_price) / entry_price * 100
                log.append({
@@ -142,100 +99,163 @@ def comprehensive_backtest(df):
                    "出場原因": "趨勢反轉" if r['Close'] < r['MA20'] else "過熱獲利"
                })
                holding = False

    return pd.DataFrame(log)

# ==========================================
# 🧠 AI 經理人團隊 (連網與辯證)
# ==========================================
def ai_manager_report(api_key, company, symbol, score_df, backtest_df):
def get_ai_response(api_key, messages_history):
    """處理對話請求"""
    genai.configure(api_key=api_key)
    
    # 聯網搜尋即時數據
    with DDGS() as ddgs:
        news = list(ddgs.text(f"{company} {symbol} 股息 PE 財報 新聞 2026", max_results=5))
    
    news_text = "\n".join([f"- {n['title']}: {n['body']}" for n in news])
    
    system_prompt = f"""
    你是一群由「總體經濟師、暗黑操盤手、價值投資者(巴菲特)」組成的專家小組。
    正在審計標的：{company} ({symbol})。
    
    【你的工作手冊】
    1. 數據穿透：用戶提供的回測僅代表過去。請結合聯網新聞中的『配息率』、『產業前景』進行翻案。
    2. 多空激戰：分析報告必須包含兩方觀點的激烈辯論。
    3. 莊家思維：用寓言解釋該股最近的波動是否為「洗盤」或「誘多」。
    4. 最終裁決：由巴菲特給出 1-10 分的投資意願。
    """
    
    prompt = f"""
    標的：{company} ({symbol})
    最新量化得分詳情：\n{score_df.to_string()}
    
    歷史回測完整紀錄：\n{backtest_df.to_string()}
    
    聯網即時情報：\n{news_text}
    
    請開始你們的專家辯證報告。
    """
    
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash", system_instruction=system_prompt)
        res = model.generate_content(prompt)
        return res.text
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        # 將對話歷史轉換為 Gemini 格式
        gemini_hist = []
        for m in messages_history:
            role = "user" if m["role"] == "user" else "model"
            gemini_hist.append({"role": role, "parts": [m["content"]]})
            
        response = model.generate_content(gemini_hist)
        return response.text
    except Exception as e:
        return f"AI 團隊討論中斷: {e}"
        return f"AI 經理人連線錯誤: {str(e)}"

# ==========================================
# 🖥️ UI 介面
# 🖥️ UI 介面與主邏輯
# ==========================================
with st.sidebar:
    st.header("🔑 系統權限")
    st.header("🔑 戰情室控制台")
    api_key = st.text_input("Google API Key", type="password")
    ticker_input = st.text_input("股票代號", value="2330")
    run_btn = st.button("啟動全數據掃描", type="primary")
    
    if st.button("🗑️ 清除對話紀錄"):
        st.session_state.messages = []
        st.rerun()

if run_btn:
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

        st.header(f"🏛️ {full_name} ({real_symbol}) 深度分析報告")
        
        # 1. 量化得分詳情
        st.subheader("🎯 量化得分審計追蹤")
        score, score_details = detailed_scoring(df)
        st.metric("核心動能總分", f"{score} / 10")
        st.table(score_details) # 使用 table 確保不被簡化
# --- 1. 按下按鈕時：執行分析並儲存狀態 ---
if run_btn and api_key:
    with st.spinner(f"正在調閱 {ticker_input} 檔案與聯網數據..."):
        df, full_name, real_symbol, err = get_verified_data(ticker_input)

        # 2. 歷史回測完整紀錄
        st.subheader("📜 歷史交易回測完整日誌")
        bt_log = comprehensive_backtest(df)
        if not bt_log.empty:
            st.dataframe(bt_log, use_container_width=True)
            col1, col2 = st.columns(2)
            col1.metric("歷史勝率", f"{(len(bt_log[bt_log['獲利%']>0])/len(bt_log)*100):.1f}%")
            col2.metric("累計報酬率", f"{bt_log['獲利%'].sum():.1f}%")
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
            st.info("過去一年內該策略未觸發任何完整交易訊號。")

        # 3. AI 經理人報告
        st.divider()
        st.subheader("🕵️ 專家經理人團隊：即時辯證分析")
        with st.spinner("經理人們正在針對數據進行激烈討論..."):
            report = ai_manager_report(api_key, full_name, real_symbol, score_details, bt_log)
            st.markdown(report)
        
        st.line_chart(df[['Close', 'MA20']])
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
        st.error(err)
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
