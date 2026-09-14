import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ==========================================
# 1. 페이지 기본 설정
# ==========================================
st.set_page_config(
    page_title="텍스트 & 데이터 분석 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. 세션 상태 초기화
# ==========================================
if "history" not in st.session_state:
    st.session_state.history = []

# ==========================================
# 3. 헬퍼 함수 정의 (예외 처리 포함)
# ==========================================
def analyze_text(text: str) -> dict:
    """입력된 텍스트를 분석하여 통계 수치를 반환하는 함수"""
    try:
        if not text or not text.strip():
            raise ValueError("입력된 텍스트가 비어 있습니다.")

        char_count = len(text)
        char_count_no_spaces = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
        words = text.split()
        word_count = len(words)
        lines = text.splitlines()
        line_count = len([line for line in lines if line.strip()])

        # 단어 빈도수 계산
        word_freq = {}
        for w in words:
            clean_w = w.strip(".,!?\"'()[]{}").lower()
            if clean_w:
                word_freq[clean_w] = word_freq.get(clean_w, 0) + 1

        freq_df = pd.DataFrame(list(word_freq.items()), columns=["단어", "빈도수"])
        freq_df = freq_df.sort_values(by="빈도수", ascending=False).reset_index(drop=True)

        return {
            "success": True,
            "char_count": char_count,
            "char_count_no_spaces": char_count_no_spaces,
            "word_count": word_count,
            "line_count": line_count,
            "freq_df": freq_df,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# ==========================================
# 4. 사이드바 영역
# ==========================================
with st.sidebar:
    st.title("⚙️ 설정 및 안내")
    st.markdown("---")
    
    st.subheader("📌 사용 안내")
    st.info(
        "1. 중앙 텍스트 상자에 분석할 문장을 입력하세요.\n"
        "2. **[텍스트 분석 실행]** 버튼을 클릭합니다.\n"
        "3. 실시간 분석 결과와 단어 빈도 차트를 확인하세요."
    )
    
    st.markdown("---")
    st.subheader("📜 작업 기록")
    if st.session_state.history:
        st.write(f"총 **{len(st.session_state.history)}개**의 기록이 저장됨")
        if st.button("기록 전체 삭제", type="secondary"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("저장된 작업 기록이 없습니다.")

# ==========================================
# 5. 메인 레이아웃 영역
# ==========================================
st.title("📊 실시간 텍스트 & 데이터 분석 대시보드")
st.caption("텍스트 데이터를 입력 받아 통계 지표와 시각화 차트를 즉시 생성합니다.")

st.markdown("---")

# 입력 섹션
input_text = st.text_area(
    label="분석할 텍스트를 입력하세요",
    height=200,
    placeholder="여기에 문단이나 텍스트 데이터를 붙여넣으세요...",
    help="한글, 영어, 숫자 등 모든 문자열 지원"
)

col_btn1, col_btn2, _ = st.columns([1, 1, 4])

with col_btn1:
    submit_button = st.button("🚀 텍스트 분석 실행", type="primary", use_container_width=True)

with col_btn2:
    clear_button = st.button("🗑️ 입력 초기화", use_container_width=True)

if clear_button:
    st.rerun()

# ==========================================
# 6. 실행 및 결과 처리
# ==========================================
if submit_button:
    if not input_text.strip():
        st.warning("⚠️ 분석할 텍스트를 입력한 후 실행 버튼을 눌러주세요.")
    else:
        with st.spinner("텍스트를 분석하는 중입니다..."):
            result = analyze_text(input_text)
            
            if not result["success"]:
                st.error(f"❌ 분석 중 오류가 발생했습니다: {result['error']}")
            else:
                st.success("✅ 분석이 완료되었습니다!")
                
                # 히스토리 기록 저장
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.history.append({
                    "timestamp": timestamp,
                    "char_count": result["char_count"],
                    "word_count": result["word_count"]
                })

                # 요약 지표 카드 (Metrics)
                st.markdown("### 📈 요약 지표")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(label="전체 글자 수", value=f"{result['char_count']:,} 자")
                m2.metric(label="공백 제외 글자 수", value=f"{result['char_count_no_spaces']:,} 자")
                m3.metric(label="단어 수", value=f"{result['word_count']:,} 개")
                m4.metric(label="문단/줄 수", value=f"{result['line_count']:,} 줄")

                st.markdown("---")

                # 시각화 및 데이터 테이블 영역
                col_chart, col_table = st.columns([3, 2])

                with col_chart:
                    st.markdown("### 📊 상위 단어 빈도수")
                    top_df = result["freq_df"].head(10)

                    if not top_df.empty:
                        fig = px.bar(
                            top_df,
                            x="빈도수",
                            y="단어",
                            orientation="h",
                            text="빈도수",
                            color="빈도수",
                            color_continuous_scale="Viridis"
                        )
                        fig.update_layout(
                            yaxis={"categoryorder": "total ascending"},
                            xaxis_title="출현 횟수",
                            yaxis_title="단어",
                            margin=dict(l=20, r=20, t=30, b=20),
                            height=380
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("시각화할 단어가 부족합니다.")

                with col_table:
                    st.markdown("### 📋 전체 단어 목록")
                    st.dataframe(
                        result["freq_df"],
                        use_container_width=True,
                        height=380,
                        column_config={
                            "단어": st.column_config.TextColumn("단어"),
                            "빈도수": st.column_config.NumberColumn("출현 횟수", format="%d 회")
                        }
                    )

# ==========================================
# 7. 세션 기록 보기 (하단 Expander)
# ==========================================
if st.session_state.history:
    st.markdown("---")
    with st.expander("📂 이전 분석 세션 기록 확인"):
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)