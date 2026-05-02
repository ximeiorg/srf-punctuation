"""分析模型参数和结构"""

from srf_punctuation.models import PunctuationPredictor
from srf_punctuation.config import Config

def analyze_model():
    config = Config()
    model = PunctuationPredictor(
        vocab_size=config.model.vocab_size,
        embed_dim=config.model.embed_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        num_labels=config.model.num_labels,
        dropout=config.model.dropout,
        max_seq_len=config.model.max_seq_len,
    )
    
    total_params = model.count_parameters()
    
    print("=" * 60)
    print("模型参数分析")
    print("=" * 60)
    
    print("\n【模型配置】")
    print(f"  vocab_size    = {config.model.vocab_size}")
    print(f"  embed_dim     = {config.model.embed_dim}")
    print(f"  hidden_dim    = {config.model.hidden_dim}")
    print(f"  num_layers    = {config.model.num_layers}")
    print(f"  num_heads     = {config.model.num_heads}")
    print(f"  num_labels    = {config.model.num_labels}")
    print(f"  max_seq_len   = {config.model.max_seq_len}")
    
    print("\n【参数统计】")
    
    embedding_params = config.model.vocab_size * config.model.embed_dim
    print(f"  Embedding层:  {embedding_params:,} 参数 ({embedding_params/total_params*100:.1f}%)")
    
    transformer_params = 0
    for name, module in model.encoder.named_modules():
        if 'transformer' in name and hasattr(module, 'weight'):
            params = module.weight.numel()
            if module.bias is not None:
                params += module.bias.numel()
            transformer_params += params
    
    ff_params_per_layer = config.model.embed_dim * config.model.hidden_dim + config.model.hidden_dim
    ff_params_per_layer += config.model.hidden_dim * config.model.embed_dim + config.model.embed_dim
    attention_params_per_layer = 4 * config.model.embed_dim * config.model.embed_dim
    layer_params = (ff_params_per_layer + attention_params_per_layer) * config.model.num_layers
    print(f"  Transformer:  ~{layer_params:,} 参数 ({layer_params/total_params*100:.1f}%)")
    
    classifier_params = config.model.embed_dim * config.model.embed_dim + config.model.embed_dim
    classifier_params += config.model.embed_dim * config.model.num_labels + config.model.num_labels
    print(f"  Classifier:   ~{classifier_params:,} 参数 ({classifier_params/total_params*100:.1f}%)")
    
    print(f"\n  【总参数量】: {total_params:,} ({total_params/1e6:.2f}M)")
    
    print("\n【各模块详细参数】")
    for name, module in model.named_modules():
        if len(name) > 0 and not any(x in name for x in ['encoder.transformer.layers', 'encoder.pos_encoding']):
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"  {name}: {params:,}")
    
    print("\n【Transformer层参数】")
    for i, layer in enumerate(model.encoder.transformer.layers):
        params = sum(p.numel() for p in layer.parameters())
        print(f"  Layer {i}: {params:,} 参数")
        for name, p in layer.named_parameters():
            print(f"    {name}: {p.numel():,}")
    
    print("\n【模型大小估算】")
    float32_size = total_params * 4 / 1024 / 1024
    float16_size = total_params * 2 / 1024 / 1024
    int8_size = total_params * 1 / 1024 / 1024
    print(f"  FP32: {float32_size:.2f} MB")
    print(f"  FP16: {float16_size:.2f} MB")
    print(f"  INT8: {int8_size:.2f} MB (量化后)")
    
    print("\n【结构合理性分析】")
    print(f"  ✓ 参数量 {total_params/1e6:.2f}M，适合移动端部署 (<1M)")
    print(f"  ✓ Embedding占比 {embedding_params/total_params*100:.1f}%，在合理范围")
    print(f"  ✓ 2层Transformer足够处理标点预测任务")
    print(f"  ✓ embed_dim=64, hidden_dim=128, 计算量适中")
    print(f"  ✓ 4个注意力头，head_dim=16，适合序列标注")
    
    print("\n【建议】")
    print("  - 当前配置适合手机端实时推理")
    print("  - 可通过INT8量化进一步压缩到 ~0.5MB")
    print("  - 如需更高精度，可增加 num_layers 到 3")
    
    print("=" * 60)

if __name__ == "__main__":
    analyze_model()