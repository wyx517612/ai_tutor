import streamlit as st
import requests
import os
import json

# ===== 页面配置 =====
st.set_page_config(
    page_title="考研AI助教",
    page_icon="📚",
    layout="wide"
)
# 初始化对话历史
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("📚 考研AI助教")
st.caption("基于RAG（检索增强生成）技术，回答408考研相关问题")

# ===== 后端地址 =====
API_BASE = "http://localhost:8000"

# ===== 侧边栏 =====
with st.sidebar:
    st.header("📊 系统信息")

    # 从后端获取统计数据
    try:
        stats = requests.get(f"{API_BASE}/stats", timeout=5).json()
        st.metric("基础资料文档数", stats.get("base_chunks", 0))
        st.metric("用户上传文档数", stats.get("user_chunks", 0))
        st.metric("总计文档数", stats.get("total_chunks", 0))
    except Exception as e:
        st.warning(f"⚠️ 无法连接后端：{e}")

    st.metric("向量维度", 768)
    st.markdown("---")
    st.markdown("### 💡 示例问题")
    st.markdown("- 什么是栈？")
    st.markdown("- 解释一下二叉树")
    st.markdown("- 快速排序的时间复杂度")

    # ===== 上传资料区域 =====
    st.markdown("---")
    st.markdown("### 📤 上传你的资料")
    uploaded_files = st.file_uploader(
        "选择PDF或TXT文件（可多选）",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        if st.button("📚 加载到知识库", type="primary", use_container_width=True):
            with st.spinner(f"正在处理 {len(uploaded_files)} 个文件..."):
                files = []
                for uploaded_file in uploaded_files:
                    files.append(
                        ("files", (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type))
                    )
                try:
                    response = requests.post(
                        f"{API_BASE}/upload",
                        files=files,
                        timeout=120
                    )
                    result = response.json()
                    st.success(f"✅ {result['message']}，新增 {result['new_chunks']} 个文本块！")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 上传失败：{e}")

    # ===== 自动出题区域 =====
    st.markdown("---")
    st.markdown("### 📝 自动出题")
    st.caption("基于当前知识库生成选择题")

    num_questions = st.slider("题目数量", min_value=1, max_value=5, value=3)

    if st.button("🎯 生成题目", type="primary", use_container_width=True):
        with st.spinner("🧠 正在生成题目..."):
            try:
                response = requests.post(
                    f"{API_BASE}/quiz",
                    json={"num_questions": num_questions},
                    timeout=120
                )
                result = response.json()
                st.session_state.questions = result.get("questions", [])
            except Exception as e:
                st.error(f"❌ 生成题目失败：{e}")

    # 显示生成的题目
    if "questions" in st.session_state and st.session_state.questions:
        st.markdown("---")
        st.markdown("### 📝 题目列表")

        for i, q in enumerate(st.session_state.questions):
            if "error" in q:
                st.error(q["error"])
                continue

            with st.expander(f"题目 {i + 1}"):
                st.markdown(f"**{q.get('question', '')}**")
                for opt in q.get('options', []):
                    st.markdown(f"- {opt}")

                if st.button(f"🔍 查看答案 {i + 1}", key=f"show_answer_{i}"):
                    st.success(f"✅ 正确答案：**{q.get('answer', '')}**")
                    st.info(f"📖 解析：{q.get('explanation', '')}")

# ===== 主界面：提问区域 =====
query = st.text_input("✏️ 输入你的问题", placeholder="例如：什么是栈？")

if st.button("🚀 提问", type="primary") and query:
    with st.spinner("🔍 正在检索知识库..."):
        try:
            response = requests.post(
                f"{API_BASE}/chat",
                json={"query": query},
                timeout=60
            )
            result = response.json()
            answer = result["answer"]
            reranked = result.get("references", [])
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"❌ 调用后端失败：{e}")
            st.stop()

    st.markdown("### 🧑‍🏫 助教回答")
    st.markdown(
        f"<div style='background-color:black;padding:20px;border-radius:10px;'>{answer}</div>",
        unsafe_allow_html=True
    )

    if reranked:
        with st.expander("📖 查看参考资料"):
            for i, chunk in enumerate(reranked):
                st.markdown(f"**资料 {i + 1}**")
                st.write(chunk[:300] + "..." if len(chunk) > 300 else chunk)