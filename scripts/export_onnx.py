import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from srf_punctuation.config import Config
from srf_punctuation.models import PunctuationPredictor


def find_best_checkpoint():
    checkpoint_dir = Path("logs/punctuation")
    if not checkpoint_dir.exists():
        return None
    
    ckpt_files = list(checkpoint_dir.glob("**/*.ckpt"))
    if not ckpt_files:
        return None
    
    best_ckpt = None
    best_loss = float("inf")
    for ckpt in ckpt_files:
        if ckpt.name == "last.ckpt":
            continue
        name = ckpt.name
        if "val/loss=" in name:
            try:
                loss_str = name.split("val/loss=")[1].split(".ckpt")[0]
                loss = float(loss_str)
                if loss < best_loss:
                    best_loss = loss
                    best_ckpt = ckpt
            except (ValueError, IndexError):
                continue
    
    return best_ckpt or (ckpt_files[0] if ckpt_files else None)


def load_model(checkpoint_path: str, config: Config):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    model = PunctuationPredictor(
        vocab_size=config.model.vocab_size,
        embed_dim=config.model.embed_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        num_labels=config.model.num_labels,
        dropout=0.0,
        max_seq_len=config.model.max_seq_len,
    )
    
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        model_state = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                model_state[k[6:]] = v
        model.load_state_dict(model_state)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def export_fp32_onnx(model, output_path: str, seq_len: int = 64):
    dummy_input = torch.randint(0, model.encoder.embedding.num_embeddings, (1, seq_len), dtype=torch.long)
    dummy_mask = torch.ones(1, seq_len, dtype=torch.long)
    
    torch.onnx.export(
        model,
        (dummy_input, dummy_mask),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"FP32 ONNX模型已导出: {output_path}")
    return output_path


def quantize_onnx(input_path: str, output_path: str, quantization_mode: str = "dynamic"):
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except ImportError:
        print("错误: 需要安装 onnxruntime")
        print("运行: uv pip install onnxruntime")
        return None
    
    preprocessed_path = input_path.replace(".onnx", "_preprocessed.onnx")
    quant_pre_process(input_path, preprocessed_path, skip_symbolic_shape=True)
    
    if quantization_mode == "dynamic":
        quantize_dynamic(
            preprocessed_path,
            output_path,
            weight_type=QuantType.QInt8,
        )
        print(f"INT8 动态量化模型已导出: {output_path}")
    else:
        print("静态量化需要校准数据，暂时只支持动态量化")
        return None
    
    if os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)
    
    return output_path


def verify_onnx(model_path: str, vocab_path: str, test_text: str = "今天天气很好"):
    try:
        import onnxruntime as ort
    except ImportError:
        print("警告: onnxruntime 未安装，跳过验证")
        return False
    
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    
    char_ids = [vocab.get(char, vocab.get("<UNK>", 1)) for char in test_text]
    input_ids = torch.tensor([char_ids], dtype=torch.long).numpy()
    attention_mask = torch.ones(1, len(char_ids), dtype=torch.long).numpy()
    
    outputs = session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    )
    
    logits = outputs[0]
    preds = logits.argmax(axis=-1)[0].tolist()
    
    print(f"验证文本: {test_text}")
    print(f"预测标签: {preds}")
    print(f"输出形状: {logits.shape}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="导出ONNX模型")
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=None,
        help="模型检查点路径 (默认自动查找最佳检查点)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="onnx",
        help="输出目录 (默认: onnx)"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="导出时的序列长度 (默认: 64)"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="是否进行INT8量化"
    )
    parser.add_argument(
        "--quantize-mode",
        type=str,
        default="dynamic",
        choices=["dynamic", "static"],
        help="量化模式 (默认: dynamic)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="验证导出的模型"
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default="data/vocab.json",
        help="词表路径"
    )
    
    args = parser.parse_args()
    
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_best_checkpoint()
        if checkpoint_path is None:
            print("错误: 未找到检查点文件，请先训练模型")
            print("运行: uv run python scripts/train.py")
            sys.exit(1)
        print(f"使用检查点: {checkpoint_path}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = Config()
    
    print("加载模型...")
    model = load_model(str(checkpoint_path), config)
    print(f"模型参数量: {model.count_parameters():,}")
    
    fp32_path = str(output_dir / "punctuation_fp32.onnx")
    print("\n导出FP32 ONNX模型...")
    export_fp32_onnx(model, fp32_path, args.seq_len)
    
    fp32_size = os.path.getsize(fp32_path)
    data_file = fp32_path + ".data"
    if os.path.exists(data_file):
        fp32_size += os.path.getsize(data_file)
    fp32_size_mb = fp32_size / (1024 * 1024)
    print(f"FP32模型大小: {fp32_size_mb:.2f} MB")
    
    if args.quantize:
        print("\n进行INT8量化...")
        int8_path = str(output_dir / "punctuation_int8.onnx")
        quantize_onnx(fp32_path, int8_path, args.quantize_mode)
        
        if os.path.exists(int8_path):
            int8_size_mb = os.path.getsize(int8_path) / (1024 * 1024)
            print(f"INT8模型大小: {int8_size_mb:.2f} MB")
            print(f"压缩比: {fp32_size_mb / int8_size_mb:.2f}x")
    
    if args.verify:
        print("\n验证ONNX模型...")
        verify_path = int8_path if args.quantize and os.path.exists(int8_path) else fp32_path
        verify_onnx(verify_path, args.vocab)
    
    print("\n导出完成!")
    print(f"输出目录: {output_dir.absolute()}")


if __name__ == "__main__":
    main()