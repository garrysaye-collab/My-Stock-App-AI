import streamlit as st
import google.generativeai as genai
import yfinance as yf

st.title("測試連線：Gemini Stock App")

# 1. 設定 API Key
api_key = st.text_input("輸入 Google API Key", type="password")

# 2. 顯示套件版本 (這是除錯關鍵)
st.write(f"目前使用的 Google GenAI 套件版本: {genai.__version__}")

if st.button("測試連線"):
    if not api_key:
        st.error("請輸入 Key")
    else:
        try:
            genai.configure(api_key=api_key)
            
            # 使用最舊、最穩定的模型測試
            model = genai.GenerativeModel('gemini-pro') 
            
            with st.spinner("正在呼叫 AI..."):
                response = model.generate_content("你好，請用一句話證明你沒有壞掉，並告訴我現在股價分析能不能做？")
                st.success("✅ 連線成功！系統沒有阻擋您。")
                st.write(response.text)
                
        except Exception as e:
            st.error(f"錯誤詳情: {e}")
