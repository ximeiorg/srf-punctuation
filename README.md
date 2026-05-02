# 语音输入法标点预测模型

为语音输入法设计的轻量级标点预测模型，可在安卓手机上实时运行。

## 特性

- **超轻量模型**: 参数量约 583K，ONNX 模型约 2.7MB
- **字符级建模**: 词表小，部署稳定
- **多标点支持**: 逗号、句号、问号、感叹号
- **ONNX 导出**: 支持安卓端部署

## 安装

```bash
uv sync --quiet
```

## 使用

### 数据处理

```bash
# 自动处理（数据文件缺失时自动执行）
uv run python scripts/train.py

# 强制重新处理数据
uv run python scripts/train.py --process-data

# 限制样本数量（默认不限制）
uv run python scripts/train.py --process-data --max-samples 100000
```

### 训练模型

```bash
# 默认参数训练
uv run python scripts/train.py

# 自定义参数
uv run python scripts/train.py --batch-size 32 --max-epochs 30 --learning-rate 5e-4

# 强制重新处理数据并训练
uv run python scripts/train.py --process-data --max-epochs 20
```

### 查看训练日志

```bash
# TensorBoard 可视化
tensorboard --logdir=logs
```

### 模型分析

```bash
uv run python scripts/analyze_model.py
```

### 推理

```python
from srf_punctuation.inference import load_inference

inf = load_inference()
result = inf.predict("今天天气很好我们出去散步吧")
# 输出: 今天天气很好，我们出去散步吧。
```

## 模型结构

- **架构**: SmallTransformer (2 层 Transformer Encoder)
- **参数**: vocab_size=8000, embed_dim=64, hidden_dim=128, num_heads=4
- **总参数量**: 583,429 (~0.58M)
- **模型大小**: FP32 ~2.23MB, INT8量化后 ~0.56MB
- **输出**: 5 类标点标签 (O, COMMA, PERIOD, QUESTION, EXCLAMATION)

### 参数分布

| 模块 | 参数量 | 占比 |
|------|--------|------|
| Embedding | 512,000 | 87.8% |
| Transformer | 66,944 | 11.3% |
| Classifier | 4,485 | 0.8% |

## 许可

MIT