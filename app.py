import os
import streamlit as st
import anthropic

# ------------------------------------------------------------------------------
# 1. 페이지 기본 설정 및 스타일 정의
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="SNU AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS 적용 (깔끔한 채팅 UI 스타일링)
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.25rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .stChatMessage {
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# 2. 세션 상태 초기화
# ------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------------------------------------------------------------
# 3. 사이드바 구성 (설정 및 옵션)
# ------------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ 설정 (Settings)")
    st.markdown("---")

    # API 키 입력받기 (Secrets에 설정된 키가 없거나 사용자가 직접 입력하고자 할 때)
    default_api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    
    user_api_key = st.text_input(
        "Anthropic API Key",
        value=default_api_key if default_api_key else "",
        type="password",
        help="Anthropic Console에서 발급받은 API 키를 입력하세요.",
    )

    st.markdown("---")
    
    # 모델 선택
    selected_model = st.selectbox(
        "사용할 AI 모델 선택",
        options=[
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229",
        ],
        index=0,
    )

    # 파라미터 조절 Slider
    temperature = st.slider("Temperature (창의성)", 0.0, 1.0, 0.7, 0.05)
    max_tokens = st.slider("최대 토큰 수", 256, 4096, 2048, 128)

    st.markdown("---")
    
    # 대화 기록 초기화 버튼
    if st.button("🗑️ 대화 내용 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------------------
# 4. 메인 화면 구성
# ------------------------------------------------------------------------------
st.markdown('<div class="main-header">🤖 서울대학교 AI Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Claude API 기반의 대화형 인공지능 서비스입니다. 질문을 입력하세요.</div>',
    unsafe_allow_html=True,
)

# 과거 대화 메시지 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ------------------------------------------------------------------------------
# 5. 사용자 입력 처리 및 API 호출
# ------------------------------------------------------------------------------
if prompt := st.chat_input("질문을 입력하세요..."):
    # API 키 유효성 기본 확인
    api_key_to_use = user_api_key.strip()
    if not api_key_to_use:
        st.error("⚠️ Anthropic API 키가 설정되지 않았습니다. 사이드바에 API 키를 입력해주세요.")
        st.stop()

    # 1) 사용자 입력 메시지 세션 저장 및 화면 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) AI 응답 생성 처리 (스트리밍 및 예외 처리 포함)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Anthropic 클라이언트 생성
            client = anthropic.Anthropic(api_key=api_key_to_use)

            # API 호출 전달용 메시지 구성 (role, content 규격 준수)
            api_messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]

            # 스트리밍 방식 호출
            with client.messages.stream(
                model=selected_model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=api_messages,
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    message_placeholder.markdown(full_response + "▌")

            # 최종 메시지 렌더링 (커서 제거)
            message_placeholder.markdown(full_response)

            # 3) AI 응답 세션 저장
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except anthropic.AuthenticationError:
            st.error("❌ API 키 인증에 실패했습니다. 올바른 Anthropic API Key인지 확인해주세요.")
        except anthropic.APIConnectionError:
            st.error("📡 네트워크 연결에 실패했습니다. 인터넷 연결을 확인해주세요.")
        except anthropic.RateLimitError:
            st.error("⏳ API 요청 한도(Rate Limit)를 초과했습니다. 잠시 후 다시 시도해주세요.")
        except anthropic.BadRequestError as e:
            st.error(f"⚠️ 요청 형식 오류: {e.message}")
        except Exception as e:
            st.error(f"🚨 예상치 못한 오류가 발생했습니다: {str(e)}")
