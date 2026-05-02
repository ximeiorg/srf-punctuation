import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from srf_punctuation.inference import PunctuationInference


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


def interactive_mode(inf: PunctuationInference):
    print("\n=== 标点预测交互模式 ===")
    print("输入不带标点的文本，按回车查看预测结果")
    print("输入 'quit' 或 'exit' 退出\n")
    
    while True:
        try:
            text = input("输入文本: ").strip()
            if text.lower() in ["quit", "exit", "q"]:
                break
            if not text:
                continue
            
            result = inf.predict(text)
            print(f"预测结果: {result}\n")
        except KeyboardInterrupt:
            break
    
    print("\n退出交互模式")


def batch_mode(inf: PunctuationInference, texts: list[str]):
    print("\n=== 批量预测结果 ===\n")
    for text in texts:
        result = inf.predict(text)
        print(f"输入: {text}")
        print(f"输出: {result}\n")


def demo_mode(inf: PunctuationInference):
    demo_texts = [
        "今天天气很好我们出去散步吧",
        "你吃饭了吗我还没吃呢",
        "请问这个多少钱能不能便宜点",
        "太棒了这个产品真的非常好用",
        "明天早上八点开会不要迟到",
        "我觉得这个方案可行但是需要一些调整",
        "你是谁为什么会在这里",
        "好的我知道了马上就去办",
    ]
    batch_mode(inf, demo_texts)


def main():
    parser = argparse.ArgumentParser(description="标点预测推理脚本")
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=None,
        help="模型检查点路径 (默认自动查找最佳检查点)"
    )
    parser.add_argument(
        "--vocab", "-v",
        type=str,
        default="data/vocab.json",
        help="词表路径"
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        default=None,
        help="要预测的单个文本"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="交互模式"
    )
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="演示模式，使用预设文本"
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
    
    vocab_path = Path(args.vocab)
    if not vocab_path.exists():
        print(f"错误: 词表文件不存在: {vocab_path}")
        print("请先运行数据处理: uv run python scripts/train.py --process-data")
        sys.exit(1)
    
    print("加载模型...")
    inf = PunctuationInference(str(checkpoint_path), str(vocab_path))
    print("模型加载完成\n")
    
    if args.text:
        result = inf.predict(args.text)
        print(f"输入: {args.text}")
        print(f"输出: {result}")
    elif args.interactive:
        interactive_mode(inf)
    elif args.demo:
        demo_mode(inf)
    else:
        demo_mode(inf)


if __name__ == "__main__":
    main()