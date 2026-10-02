from __future__ import annotations

import streamlit as st

from job_ml.service import ArtifactNotFoundError, predict_job


st.set_page_config(
    page_title="招聘技能与薪资预测",
    layout="centered",
)

st.title("招聘技能与薪资预测")
st.caption("输入岗位描述，输出可能要求的技能和月薪预测基线。")

with st.form("job-form"):
    col1, col2 = st.columns(2)
    with col1:
        job_title = st.text_input("岗位名称", "机器学习算法工程师")
        city = st.selectbox("城市", ["广州", "深圳", "上海", "北京", "杭州"])
    with col2:
        experience = st.selectbox("经验", ["应届", "1年以下", "1-3年", "3-5年"])
        education = st.selectbox("学历", ["大专", "本科", "硕士"])

    job_description = st.text_area(
        "岗位描述",
        "负责中文文本分类、特征工程和模型评估，熟悉 Python、PyTorch、"
        "机器学习，能够使用 SQL 完成数据分析。",
        height=150,
    )
    submitted = st.form_submit_button("开始预测", use_container_width=True)

if submitted:
    try:
        result = predict_job(
            {
                "job_title": job_title,
                "city": city,
                "experience": experience,
                "education": education,
                "job_description": job_description,
            }
        )
    except ArtifactNotFoundError as error:
        st.error(str(error))
    else:
        st.metric("预测月薪", f"{result['predicted_salary_month_k']:.1f} K")
        st.subheader("可能要求的技能")
        if result["skills"]:
            for item in result["skills"]:
                st.progress(
                    min(item["probability"], 1.0),
                    text=f"{item['skill']}  {item['probability']:.1%}",
                )
        else:
            st.info("当前阈值下未识别到技能，可降低 service.py 中的 threshold 后重试。")
