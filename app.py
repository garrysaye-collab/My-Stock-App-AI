import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="API 模型診斷工具", page_icon="🩺")

st.title("🩺 Google API 診斷室")
st.write("這個工具會直接詢問 Google：『我的 API Key 到底能用哪些模型？』")

# 1. 輸入 API Key
api_key = st.text_input("請輸入 Google API Key", type="password")

# 2. 顯示環境中的套件版本
st.info(f"目前 Streamlit 環境中的 google-generativeai 版本: `{genai.__version__}`")

if st.button("🔍 開始診斷"):
    if not api_key:
        st.error("❌ 請輸入 API Key")
    else:
        try:
            # 設定 Key
            genai.configure(api_key=api_key)
            
            # 嘗試列出所有模型
            st.write("正在連線 Google 伺服器讀取清單...")
            
            # 獲取模型列表
            models_iter = genai.list_models()
            available_models = []
            
            for m in models_iter:
                # 只列出支援 'generateContent' (對話生成) 的模型
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
            
            if available_models:
                st.success(f"✅ 連線成功！您的 API Key 可以使用以下 {len(available_models)} 個模型：")
                
                # 顯示列表
                st.json(available_models)
                
                st.markdown("### 💡 下一步建議：")
                st.write("請記下上面列表中的名稱。")
                st.write("例如，如果您看到 `models/gemini-1.5-flash`，請在您的程式碼中精確地使用這個字串。")
                
                # 直接嘗試用第一個可用的模型打招呼
                first_model = available_models[0]
                st.divider()
                st.write(f"正在嘗試使用 `{first_model}` 進行測試生成...")
                
                test_model = genai.GenerativeModel(first_model)
                response = test_model.generate_content("你好，如果你看到這句話，代表連線完全正常。")
                st.balloons()
                st.write("🤖 AI 回應：")
                st.success(response.text)
                
            else:
                st.warning("⚠️ 連線成功，但這個 API Key 似乎沒有權限存取任何『對話模型』。")
                st.write("可能原因：您的 Google Cloud 專案沒有啟用 Generative Language API。")
                
        except Exception as e:
            st.error("❌ 發生錯誤 (診斷失敗)")
            st.code(str(e))
            
            # 特別分析 400/403/404 錯誤
            err_msg = str(e)
            if "400" in err_msg:
                st.warning("提示：API Key 格式可能錯誤，或者 API Key 不適用於此專案。")
            elif "API not enabled" in err_msg:
                st.warning("提示：請去 Google AI Studio 或 Google Cloud Console 啟用 Generative Language API。")
            elif "404" in err_msg:
                st.warning("提示：找不到路徑。這通常是套件版本太舊，或者模型名稱已變更。")
