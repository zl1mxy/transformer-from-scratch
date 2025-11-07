import torch
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from train import Trainer
from model import Transformer
from data import DataProcessor
from config import Config
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ComprehensiveExperiment:
    """综合实验类"""
    
    def __init__(self):
        self.base_config = Config.from_yaml('configs/base.yaml')
        self.results = {}
        
    def run_comprehensive_ablation(self):
        """运行综合消融实验"""
        print("运行综合消融实验...")
        
        # 准备数据（一次性加载）
        print("准备实验数据...")
        processor = DataProcessor(
            src_lang=self.base_config.data.src_lang,
            tgt_lang=self.base_config.data.tgt_lang,
            vocab_size=1000,
            max_seq_len=50
        )
        
        src_sentences, tgt_sentences = processor.load_iwslt2017_dataset()
        print(f"总数据量: {len(src_sentences)} 条句子")
        
        if len(src_sentences) < 30:
            print("数据量不足，使用全部数据")
            # 使用所有数据，重复一部分作为验证集
            train_src, train_tgt = src_sentences, tgt_sentences
            val_src, val_tgt = src_sentences[:5], tgt_sentences[:5]
        else:
            # 数据分割
            train_size = min(40, len(src_sentences) - 10)
            train_src, train_tgt = src_sentences[:train_size], tgt_sentences[:train_size]
            val_src, val_tgt = src_sentences[train_size:train_size+10], tgt_sentences[train_size:train_size+10]
        
        print(f"训练数据: {len(train_src)} 条")
        print(f"验证数据: {len(val_src)} 条")
        
        # 检查数据是否为空
        if len(train_src) == 0:
            print("训练数据为空，无法继续实验")
            return self.results
            
        if len(val_src) == 0:
            print("验证数据为空，将使用训练数据进行验证")
            val_src, val_tgt = train_src[:5], train_tgt[:5]
        
        # 训练分词器
        processor.train_tokenizers(src_sentences, tgt_sentences)
        src_vocab_size = processor.src_tokenizer.get_vocab_size()
        tgt_vocab_size = processor.tgt_tokenizer.get_vocab_size()
        
        # 消融实验配置
        experiments = [
            # 基线模型
            {
                'name': 'baseline',
                'description': '基线模型 (d_model=128, heads=4, layers=2)',
                'model_params': {
                    'd_model': 128,
                    'num_heads': 4,
                    'd_ff': 512,
                    'num_encoder_layers': 2,
                    'num_decoder_layers': 2,
                    'dropout': 0.1
                }
            },
            # 模型架构实验
            {
                'name': 'no_positional_encoding',
                'description': '无位置编码',
                'model_params': {
                    'd_model': 128,
                    'num_heads': 4,
                    'd_ff': 512,
                    'num_encoder_layers': 2,
                    'num_decoder_layers': 2,
                    'dropout': 0.1
                },
                'remove_pe': True
            },
            {
                'name': 'single_head',
                'description': '单头注意力',
                'model_params': {
                    'd_model': 128,
                    'num_heads': 1,
                    'd_ff': 512,
                    'num_encoder_layers': 2,
                    'num_decoder_layers': 2,
                    'dropout': 0.1
                }
            },
            # 模型容量实验
            {
                'name': 'tiny_model',
                'description': '极小模型 (d_model=64)',
                'model_params': {
                    'd_model': 64,
                    'num_heads': 4,
                    'd_ff': 256,
                    'num_encoder_layers': 2,
                    'num_decoder_layers': 2,
                    'dropout': 0.1
                }
            },
            {
                'name': 'larger_model',
                'description': '较大模型 (d_model=256)',
                'model_params': {
                    'd_model': 256,
                    'num_heads': 4,
                    'd_ff': 1024,
                    'num_encoder_layers': 2,
                    'num_decoder_layers': 2,
                    'dropout': 0.1
                }
            }
        ]
        
        # 运行所有实验
        for exp in experiments:
            success = self._run_single_experiment(exp, processor, train_src, train_tgt, val_src, val_tgt, 
                                                src_vocab_size, tgt_vocab_size)
            if not success:
                print(f"实验 {exp['name']} 失败，跳过")
        
        # 分析结果
        if self.results:
            self._analyze_results()
        else:
            print("所有实验都失败了，无法分析结果")
        
        return self.results
    
    def _run_single_experiment(self, exp_config, processor, train_src, train_tgt, val_src, val_tgt,
                             src_vocab_size, tgt_vocab_size):
        """运行单个实验"""
        print(f"\n运行实验: {exp_config['description']}")
        
        try:
            # 创建数据加载器（添加安全检查）
            if len(train_src) == 0:
                print("训练数据为空，跳过实验")
                return False
                
            train_dataloader = processor.create_data_loader(
                train_src, train_tgt, batch_size=min(4, len(train_src)), shuffle=True
            )
            
            if len(val_src) > 0:
                val_dataloader = processor.create_data_loader(
                    val_src, val_tgt, batch_size=min(4, len(val_src)), shuffle=False
                )
            else:
                val_dataloader = None
                print("无验证数据，将跳过验证")
            
            # 检查数据加载器是否创建成功
            if len(train_dataloader) == 0:
                print("训练数据加载器为空，跳过实验")
                return False
            
            # 创建模型
            if exp_config.get('remove_pe', False):
                model = self._create_transformer_no_pe(
                    src_vocab_size, tgt_vocab_size, **exp_config['model_params']
                )
            else:
                model = Transformer(
                    src_vocab_size=src_vocab_size,
                    tgt_vocab_size=tgt_vocab_size,
                    **exp_config['model_params']
                )
            
            model_params = sum(p.numel() for p in model.parameters())
            print(f"模型参数: {model_params:,}")
            
            # 配置训练参数
            self.base_config.training.max_epochs = 3  # 减少epoch数以加快实验
            self.base_config.training.batch_size = min(4, len(train_src))
            
            # 创建训练器
            trainer = Trainer(self.base_config, model, train_dataloader, val_dataloader)
            
            # 训练
            start_time = time.time()
            for epoch in range(self.base_config.training.max_epochs):
                trainer.epoch = epoch
                train_loss = trainer.train_epoch()
                if val_dataloader is not None:
                    val_loss = trainer.validate()
                else:
                    val_loss = None
            
            training_time = time.time() - start_time
            
            # 记录结果
            self.results[exp_config['name']] = {
                'description': exp_config['description'],
                'final_train_loss': trainer.train_losses[-1],
                'final_val_loss': trainer.val_losses[-1] if trainer.val_losses and len(trainer.val_losses) > 0 else None,
                'best_val_loss': trainer.best_val_loss,
                'training_time': training_time,
                'model_params': model_params,
                'train_losses': trainer.train_losses.copy(),
                'val_losses': trainer.val_losses.copy() if trainer.val_losses else []
            }
            
            print(f"{exp_config['description']} 完成!")
            print(f"   最终训练损失: {trainer.train_losses[-1]:.4f}")
            if trainer.val_losses and len(trainer.val_losses) > 0:
                print(f"   最终验证损失: {trainer.val_losses[-1]:.4f}")
            print(f"   最佳验证损失: {trainer.best_val_loss:.4f}")
            print(f"   训练时间: {training_time:.1f}s")
            
            return True
            
        except Exception as e:
            print(f"实验 {exp_config['name']} 失败: {e}")
            return False
    
    def _create_transformer_no_pe(self, src_vocab_size, tgt_vocab_size, **kwargs):
        """创建无位置编码的Transformer"""
        from model import Transformer
        
        class TransformerNoPE(Transformer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.encoder.positional_encoding = torch.nn.Identity()
                self.decoder.positional_encoding = torch.nn.Identity()
        
        return TransformerNoPE(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            **kwargs
        )
    
    def _analyze_results(self):
        """分析实验结果"""
        print("\n综合实验结果分析")
        print("=" * 120)
        print(f"{'实验':<20} {'描述':<35} {'最终训练损失':<12} {'最终验证损失':<12} {'最佳验证损失':<12} {'参数数量':<12} {'时间':<10}")
        print("=" * 120)
        
        for exp_name, result in self.results.items():
            val_loss_str = f"{result['final_val_loss']:.4f}" if result['final_val_loss'] else 'N/A'
            print(f"{exp_name:<20} {result['description']:<35} "
                  f"{result['final_train_loss']:<12.4f} "
                  f"{val_loss_str:<12} "
                  f"{result['best_val_loss']:<12.4f} "
                  f"{result['model_params']:<12,} "
                  f"{result['training_time']:<10.1f}s")
        
        # 绘制比较图
        self._plot_comprehensive_comparison()
        
        # 保存详细结果
        self._save_detailed_results()
    
    def _plot_comprehensive_comparison(self):
        """绘制综合比较图"""
        os.makedirs('results/analysis', exist_ok=True)
        
        plt.figure(figsize=(20, 12))
        
        # 1. 训练损失比较
        plt.subplot(2, 3, 1)
        for exp_name, result in self.results.items():
            if 'train_losses' in result and result['train_losses']:
                plt.plot(result['train_losses'], label=exp_name, alpha=0.7, linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('训练损失比较')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 2. 验证损失比较
        plt.subplot(2, 3, 2)
        for exp_name, result in self.results.items():
            if 'val_losses' in result and result['val_losses']:
                plt.plot(result['val_losses'], label=exp_name, alpha=0.7, linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('验证损失比较')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 3. 最终验证损失比较
        plt.subplot(2, 3, 3)
        exp_names = []
        final_val_losses = []
        for exp_name, result in self.results.items():
            if result['final_val_loss']:
                exp_names.append(exp_name)
                final_val_losses.append(result['final_val_loss'])
        
        if exp_names:
            bars = plt.bar(exp_names, final_val_losses, alpha=0.7)
            plt.ylabel('Final Validation Loss')
            plt.title('最终验证损失比较')
            plt.xticks(rotation=45, ha='right')
            
            # 添加数值标签
            for bar, loss in zip(bars, final_val_losses):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{loss:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 4. 模型参数数量比较
        plt.subplot(2, 3, 4)
        exp_names = []
        param_counts = []
        for exp_name, result in self.results.items():
            exp_names.append(exp_name)
            param_counts.append(result['model_params'])
        
        bars = plt.bar(exp_names, param_counts, alpha=0.7, color='green')
        plt.ylabel('Parameter Count')
        plt.title('模型参数数量比较')
        plt.xticks(rotation=45, ha='right')
        
        # 5. 训练时间比较
        plt.subplot(2, 3, 5)
        exp_names = []
        training_times = []
        for exp_name, result in self.results.items():
            exp_names.append(exp_name)
            training_times.append(result['training_time'])
        
        bars = plt.bar(exp_names, training_times, alpha=0.7, color='orange')
        plt.ylabel('Training Time (seconds)')
        plt.title('训练时间比较')
        plt.xticks(rotation=45, ha='right')
        
        # 6. 损失下降幅度比较
        plt.subplot(2, 3, 6)
        exp_names = []
        loss_reductions = []
        for exp_name, result in self.results.items():
            if 'train_losses' in result and len(result['train_losses']) >= 2:
                initial_loss = result['train_losses'][0]
                final_loss = result['train_losses'][-1]
                reduction = initial_loss - final_loss
                exp_names.append(exp_name)
                loss_reductions.append(reduction)
        
        if exp_names:
            bars = plt.bar(exp_names, loss_reductions, alpha=0.7, color='red')
            plt.ylabel('Loss Reduction')
            plt.title('训练损失下降幅度比较')
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('results/analysis/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("综合比较图已保存到 results/analysis/comprehensive_comparison.png")
    
    def _save_detailed_results(self):
        """保存详细结果"""
        import json
        
        # 转换为可序列化格式
        serializable_results = {}
        for exp_name, result in self.results.items():
            serializable_results[exp_name] = {
                'description': result['description'],
                'final_train_loss': float(result['final_train_loss']),
                'final_val_loss': float(result['final_val_loss']) if result['final_val_loss'] else None,
                'best_val_loss': float(result['best_val_loss']),
                'training_time': float(result['training_time']),
                'model_params': int(result['model_params']),
                'train_losses': [float(x) for x in result['train_losses']],
                'val_losses': [float(x) for x in result['val_losses']] if result['val_losses'] else []
            }
        
        with open('results/analysis/comprehensive_results.json', 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print("详细结果已保存到 results/analysis/comprehensive_results.json")

def debug_data_loading():
    """调试数据加载"""
    print("调试数据加载...")
    
    processor = DataProcessor(
        src_lang='en',
        tgt_lang='de', 
        vocab_size=500,
        max_seq_len=30
    )
    
    src_sentences, tgt_sentences = processor.load_iwslt2017_dataset()
    print(f"加载的数据量: {len(src_sentences)}")
    
    if len(src_sentences) == 0:
        print("没有加载到任何数据")
        return False
        
    # 显示前几条数据
    for i in range(min(3, len(src_sentences))):
        print(f"  {i+1}. 源: {src_sentences[i]}")
        print(f"     目标: {tgt_sentences[i]}")
    
    # 训练分词器
    processor.train_tokenizers(src_sentences, tgt_sentences)
    
    # 测试数据加载器
    if len(src_sentences) >= 10:
        train_dataloader = processor.create_data_loader(
            src_sentences[:8], tgt_sentences[:8], batch_size=2
        )
        val_dataloader = processor.create_data_loader(
            src_sentences[8:10], tgt_sentences[8:10], batch_size=2
        )
        
        print(f"训练批次: {len(train_dataloader)}")
        print(f"验证批次: {len(val_dataloader)}")
        
        # 测试一个批次
        for i, batch in enumerate(train_dataloader):
            print(f"批次 {i}: src_shape={batch['src_tokens'].shape}, tgt_shape={batch['tgt_tokens'].shape}")
            if i >= 1:
                break
        return True
    else:
        print(f"数据量不足 ({len(src_sentences)})，无法创建验证集")
        return len(src_sentences) > 0

def main():
    print("=" * 80)
    print("综合消融实验")
    print("=" * 80)
    
    try:
        # 先调试数据加载
        data_ok = debug_data_loading()
        
        if not data_ok:
            print("数据加载调试失败，终止实验")
            return
        
        print("\n" + "="*50)
        experiment = ComprehensiveExperiment()
        results = experiment.run_comprehensive_ablation()
        
        print("\n" + "=" * 80)
        if results:
            print("综合消融实验完成！")
            print(f"成功完成 {len(results)} 个实验")
            print("结果已保存到 results/analysis/")
        else:
            print("实验失败，没有成功完成任何实验")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n 实验失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()