# 语音输入法标点预测模型

为语音输入法设计的轻量级标点预测模型，可在安卓手机上实时运行。

## 特性

- **轻量模型**: 参数量约 1.57M，ONNX 模型约 6.7MB (FP32) / 2.1MB (INT8)
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

### 导出ONNX

```bash
# 导出FP32模型
uv run python scripts/export_onnx.py

# 导出并量化为INT8
uv run python scripts/export_onnx.py --quantize

# 验证导出的模型
uv run python scripts/export_onnx.py --quantize --verify
```

### 推理

```bash
# 演示模式（使用预设文本）
uv run python scripts/inference.py

# 交互模式
uv run python scripts/inference.py -i

# 单个文本预测
uv run python scripts/inference.py -t "今天天气很好我们出去散步吧"

# 使用ONNX模型推理（更快）
uv run python scripts/inference.py --onnx onnx/punctuation_int8.onnx

# 指定检查点
uv run python scripts/inference.py -c path/to/checkpoint.ckpt
```

或在代码中使用：

```python
from srf_punctuation.inference import PunctuationInference, ONNXInference

# PyTorch模型推理
inf = PunctuationInference("checkpoints/best.ckpt", "data/vocab.json")
result = inf.predict("今天天气很好我们出去散步吧")

# ONNX模型推理（更轻量）
inf = ONNXInference("onnx/punctuation_int8.onnx", "data/vocab.json")
result = inf.predict("今天天气很好我们出去散步吧")
```

## 模型结构

- **架构**: SmallTransformer (4 层 Transformer Encoder)
- **参数**: vocab_size=8000, embed_dim=128, hidden_dim=256, num_heads=8
- **总参数量**: 1,571,077 (~1.57M)
- **模型大小**: FP32 ~6.7MB, INT8量化后 ~2.1MB
- **输出**: 5 类标点标签 (O, COMMA, PERIOD, QUESTION, EXCLAMATION)

### 参数分布

| 模块 | 参数量 | 占比 |
|------|--------|------|
| Embedding | 1,024,000 | 65.2% |
| Transformer | 529,920 | 33.7% |
| Classifier | 17,157 | 1.1% |

## 许可

MIT