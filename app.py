from dotenv import load_dotenv
load_dotenv()

import streamlit as st

st.title("生成AIに質問しよう！ ～Lesson21課題のWebアプリ～")

st.write("### ２種類の専門家AIが、あなたの質問に的確にアドバイスします！")
st.write("##### 【クリエイティブAI】")
st.write("###### 「文章、デザイン、アイデア発想」のアドバイスを専門家とするAIがあなたの質問にお答えします")
st.write("##### 【ロジカルAI】")
st.write("###### 「論理展開、リスク評価、計画立案」のアドバイスを専門家とするAIがあなたの質問にお答えします")
st.write("##### ※入力フォームに質問を入力し、「実行」ボタンを押すと専門家AIの回答が表示されます。")

selected_item = st.radio(
    "どちらの専門家AIに質問するか選択してください。",
    ["クリエイティブAI", "ロジカルAI"]
)

st.write("★クリエイティブAIの提案アイデアを、ロジカルAIで計画立案するのもおすすめです。")

st.divider()

# LangChain & OpenAIのインポート
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import openai
import os

# OpenAI APIキーは.envで管理
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.5, model="gpt-4o-mini")

# LLM応答関数
def get_llm_response(input_text: str, expert_type: str) -> str:
    if expert_type == "クリエイティブAI":
        system_prompt = "あなたはクリエイティブ領域（文章、デザイン、アイデア発想）の専門家AIです。ユーザーの質問に的確なアドバイスを日本語で返してください。"
    else:
        system_prompt = "あなたはロジック領域（論理展開、リスク評価、計画立案）の専門家AIです。ユーザーの質問に的確なアドバイスを日本語で返してください。"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=input_text)
    ]
    response = llm(messages)
    try:
        response = llm(messages)
        # 型チェック: responseがcontent属性を持つか確認
        if hasattr(response, "content"):
            return response.content
        else:
            raise ValueError("AIの応答形式が予期せぬものです。'content'属性がありません。")
    except Exception as e:
        # OpenAI APIエラーやその他例外をキャッチ
        raise RuntimeError(f"AI応答取得中にエラーが発生しました: {e}")
    return response.content

# 入力フォーム
input_message = st.text_area(label="質問内容を入力してください。")

if st.button("実行"):
    st.divider()
    if input_message:
        with st.spinner("AIが回答中..."):
            try:
                answer = get_llm_response(input_message, selected_item)
                st.write("### AIの回答")
                st.success(answer)
            except Exception as e:
                import logging
                logging.error(f"AI response error: {e}")
                st.error("AIの回答取得中に問題が発生しました。しばらくしてから再度お試しください。")
    else:
        st.error("質問内容を入力してから「実行」ボタンを押してください。")