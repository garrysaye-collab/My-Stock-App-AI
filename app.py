import streamlit as st
import google.generativeai as genai
import os

st.title("👨‍⚕️ Gemini 模型診斷室")

# 1. 讓你在網頁上輸入 Key，避免 Key 寫死在程式碼裡
api_key = st.text_input("請輸入 Google API Key", type="password")

if st.button("開始診斷"):
    if not api_key:
        st.error("❌ 請先輸入 API Key")
    else:
        try:
            # 設定 Key
            genai.configure(api_key=api_key)
            
            st.info(f"正在檢查可用的模型列表...")
            
            # 列出所有模型
            available_models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
            
            if available_models:
                st.success(f"✅ 成功連線！共找到 {len(available_models)} 個可用模型：")
                st.code("\n".join(available_models)) # 這裡會直接把正確名稱印出來
                
                # 自動測試：嘗試用第一個找到的 Flash 模型寫一句話
                flash_models = [m for m in available_models if 'flash' in m]
                if flash_models:
                    target_model = flash_models[0] # 抓第一個能用的 Flash
                    st.divider()
                    st.write(f"🚀 正在嘗試使用 **{target_model}** 進行測試...")
                    
                    model = genai.GenerativeModel(target_model)
                    response = model.generate_content("你好，請回應「測試成功」四個字。")
                    st.write("🤖 AI 回應：", response.text)
                    st.balloons()
                else:
                    st.warning("⚠️ 雖然連線成功，但清單中沒有看到 'flash' 相關的模型。")
            else:
                st.error("⚠️ 連線成功，但沒有找到任何支援 generateContent 的模型。")
                
        except Exception as e:
            st.error(f"❌ 發生錯誤：\n{e}")
            st.write("---")
            st.write("💡 常見原因：")
            st.write("1. API Key 無效或沒有權限。")
            st.write("2. 所在的地區 (IP) 被 Google 封鎖 (Streamlit 主機有時在被擋的地區)。")
