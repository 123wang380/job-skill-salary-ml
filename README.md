# 招聘 JD 技能抽取与薪资预测

[![CI](https://github.com/123wang380/job-skill-salary-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/123wang380/job-skill-salary-ml/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

这是一个面向机器学习求职展示的端到端基线项目：输入招聘岗位描述，输出可能要求的技术技能与月薪预测。

项目已经实现可运行的 TF-IDF + 多标签分类、回归基线、FastAPI 推理接口、Streamlit 交互页面、Docker 和 GitHub Actions。BERT 微调、MLflow 和真实数据实验属于下一阶段路线，不会在当前版本中伪装成已完成。

## 功能

- 招聘数据清洗：字段标准化、薪资区间解析、技能标签切分和去重。
- 技能识别：TF-IDF 字符特征 + One-vs-Rest 逻辑回归多标签分类。
- 薪资预测：LightGBM 回归；未安装 LightGBM 时自动回退到随机森林。
- 模型评估：Macro-F1、Micro-F1、Precision、Recall、MAE、RMSE、R²。
- 模型服务：FastAPI 提供 `/predict` 与 `/health`。
- 交互页面：Streamlit 展示技能概率与薪资预测。
- 工程化：pytest、ruff、Docker Compose、GitHub Actions。

## 系统流程

```text
招聘数据 -> 清洗与标签 -> TF-IDF / 分类 -> 薪资回归 -> 指标报告
                                      \-> FastAPI / Streamlit
```

## 项目结构

```text
.
├── data/raw/sample_jobs.csv
├── src/job_ml/
│   ├── api.py
│   ├── dashboard.py
│   ├── data.py
│   ├── explain.py
│   ├── features.py
│   ├── modeling.py
│   ├── pipeline.py
│   └── service.py
├── tests/
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

## 本地运行

要求 Python 3.10 及以上版本。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,ml]"
python -m job_ml.pipeline --data data/raw/sample_jobs.csv
```

如果暂时不安装 LightGBM：

```powershell
python -m job_ml.pipeline --data data/raw/sample_jobs.csv --no-lightgbm
```

启动 API：

```powershell
uvicorn job_ml.api:app --reload
```

打开：

- Swagger 文档：http://127.0.0.1:8000/docs
- 健康检查：http://127.0.0.1:8000/health

启动 Streamlit：

```powershell
streamlit run src/job_ml/dashboard.py
```

## API 示例

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "job_title": "机器学习算法工程师",
    "city": "广州",
    "experience": "1-3年",
    "education": "本科",
    "job_description": "负责文本分类和模型评估，熟悉 Python、PyTorch、机器学习。"
  }'
```

## Docker

```bash
docker compose up --build
```

- API：http://127.0.0.1:8000/docs
- Dashboard：http://127.0.0.1:8501

## 测试

```bash
pytest
ruff check src tests
```

## 数据与结果说明

仓库自带的 `data/raw/sample_jobs.csv` 是人工构造的演示数据，只用于验证工程流程。所有指标都必须使用真实、合规且经过标注的数据重新计算后再写入简历或对外展示。

## 下一阶段路线

- [ ] 使用 1,000 条以上真实 JD 重新标注技能标签。
- [ ] 增加中文 BERT 多标签分类，并与 TF-IDF 基线比较。
- [ ] 增加 SHAP 解释和错误样本分析报告。
- [ ] 使用 MLflow 记录参数、指标和模型版本。
- [ ] 增加数据漂移监控与低置信度样本回流。
- [ ] 部署在线演示，并在 README 中添加截图和真实指标。

## 参考项目

- [Made-With-ML](https://github.com/GokuMohandas/Made-With-ML)：生产级机器学习工程结构。
- [Bert-Chinese-Text-Classification-Pytorch](https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch)：中文 BERT 文本分类参考。
- [LightGBM](https://github.com/lightgbm-org/LightGBM)：梯度提升模型。

参考项目仅用于学习设计和实现思路，本项目代码为独立实现。

## License

[MIT](LICENSE)
