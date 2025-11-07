import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import time
import os
import math
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from model import Transformer
from data import DataProcessor
from config import Config

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class FullTrainer:
    """完整训练器类"""
    
    def __init__(self, config: Config, model: Transformer, 
                 train_dataloader, val_dataloader=None, vocab_size=None):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # 设置设备
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 获取词汇表大小
        if vocab_size is None:
            vocab_size = self._get_vocab_size()
        
        # 损失函数（带标签平滑）
        self.criterion = LabelSmoothingLoss(
            vocab_size=vocab_size,
            smoothing=config.training.label_smoothing
        )
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        
        # 记录训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []

    def _get_vocab_size(self):
        """从模型推断词汇表大小"""
        # 检查输出投影层
        if hasattr(self.model, 'output_projection'):
            return self.model.output_projection.out_features
        
        # 检查其他可能的输出层名称
        possible_names = ['output_layer', 'projection', 'classifier', 'lm_head', 'fc']
        for name in possible_names:
            if hasattr(self.model, name):
                layer = getattr(self.model, name)
                if hasattr(layer, 'out_features'):
                    vocab_size = layer.out_features
                    print(f"从 {name} 获取词汇表大小: {vocab_size}")
                    return vocab_size
        
        # 从目标嵌入层获取
        if hasattr(self.model, 'tgt_embedding'):
            vocab_size = self.model.tgt_embedding.num_embeddings
            print(f"从 tgt_embedding 获取词汇表大小: {vocab_size}")
            return vocab_size
        
        # 使用默认值
        print("无法推断词汇表大小，使用默认值 1000")
        return 1000

    def _create_optimizer(self):
        """创建优化器"""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.base_learning_rate,
            betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
            eps=self.config.training.adam_epsilon,
            weight_decay=0.01
        )
    
    def _create_scheduler(self):
        """创建学习率调度器（带线性热身）"""
        def lr_lambda(step):
            # 线性热身：从0线性增加到base_learning_rate
            if step < self.config.training.warmup_steps:
                return float(step) / float(max(1, self.config.training.warmup_steps))
            
            # 余弦衰减
            total_steps = self.config.training.max_epochs * len(self.train_dataloader)
            progress = float(step - self.config.training.warmup_steps) / float(
                max(1, total_steps - self.config.training.warmup_steps)
            )
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))  # 最低到10%的基础学习率
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_tokens = 0
        total_accuracy = 0
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # 移动数据到设备
            src_tokens = batch['src_tokens'].to(self.device)
            tgt_tokens = batch['tgt_tokens'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(src_tokens, tgt_tokens[:, :-1])
            
            # 计算损失
            loss = self.criterion(
                output.contiguous().view(-1, output.size(-1)),
                tgt_tokens[:, 1:].contiguous().view(-1)
            )
            
            # 检查损失是否合理
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"异常损失值: {loss.item()}, 跳过该批次")
                continue
            
            # 计算准确率
            accuracy = calculate_accuracy(
                output.contiguous().view(-1, output.size(-1)),
                tgt_tokens[:, 1:].contiguous().view(-1)
            )
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.training.grad_clip
            )
            
            # 优化器步进
            self.optimizer.step()
            self.scheduler.step()
            
            # 统计
            batch_tokens = (tgt_tokens[:, 1:] != 0).sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            total_accuracy += accuracy * batch_tokens
            num_batches += 1
            self.global_step += 1
            
            # 记录学习率
            current_lr = self.scheduler.get_last_lr()[0]
            self.learning_rates.append(current_lr)
            
            # 每25%进度打印一次
            if batch_idx % max(1, len(self.train_dataloader) // 4) == 0:
                avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
                avg_accuracy = total_accuracy / total_tokens if total_tokens > 0 else 0
                print(f'Epoch: {self.epoch} [{batch_idx}/{len(self.train_dataloader)}] '
                      f'Loss: {avg_loss:.4f} | Acc: {avg_accuracy:.4f} | LR: {current_lr:.2e}')
        
        if total_tokens == 0:
            print("本epoch没有有效的token")
            return float('inf'), 0.0
            
        epoch_loss = total_loss / total_tokens
        epoch_accuracy = total_accuracy / total_tokens
        epoch_time = time.time() - start_time
        
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_accuracy)
        
        print(f'====> Epoch: {self.epoch} 训练完成, 平均损失: {epoch_loss:.4f}, '
              f'准确率: {epoch_accuracy:.4f}, 时间: {epoch_time:.2f}秒, 总token数: {total_tokens}')
        
        return epoch_loss, epoch_accuracy
    
    def validate(self):
        """验证模型"""
        if self.val_dataloader is None:
            print("无验证数据，跳过验证")
            return None, None
            
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        total_accuracy = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # 检查批次是否为空
                if batch['src_tokens'].numel() == 0 or batch['tgt_tokens'].numel() == 0:
                    continue
                    
                src_tokens = batch['src_tokens'].to(self.device)
                tgt_tokens = batch['tgt_tokens'].to(self.device)
                
                output = self.model(src_tokens, tgt_tokens[:, :-1])
                loss = self.criterion(
                    output.contiguous().view(-1, output.size(-1)),
                    tgt_tokens[:, 1:].contiguous().view(-1)
                )
                
                # 计算准确率
                accuracy = calculate_accuracy(
                    output.contiguous().view(-1, output.size(-1)),
                    tgt_tokens[:, 1:].contiguous().view(-1)
                )
                
                batch_tokens = (tgt_tokens[:, 1:] != 0).sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens
                total_accuracy += accuracy * batch_tokens
                num_batches += 1
        
        # 安全检查
        if num_batches == 0 or total_tokens == 0:
            print("验证批次为空，跳过验证")
            return None, None
            
        val_loss = total_loss / total_tokens
        val_accuracy = total_accuracy / total_tokens
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy)
        
        print(f'====> 验证集损失: {val_loss:.4f}, 准确率: {val_accuracy:.4f} (基于 {total_tokens} tokens)')
    
        # 保存最佳模型
        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            self.best_val_loss = val_loss
            self.save_checkpoint(is_best=True)
            print(f"新的最佳模型! 准确率: {val_accuracy:.4f}")
    
        return val_loss, val_accuracy
    
    def train(self):
        """完整训练循环"""
        print(f"训练数据: {len(self.train_dataloader)} 批次")
        if self.val_dataloader:
            print(f"验证数据: {len(self.val_dataloader)} 批次")
        
        for epoch in range(self.config.training.max_epochs):
            self.epoch = epoch
            
            # 训练
            train_loss, train_accuracy = self.train_epoch()
            
            # 验证
            val_loss, val_accuracy = self.validate()
            
            # 保存检查点
            if epoch % self.config.training.save_interval == 0:
                self.save_checkpoint()
            
            # 绘制训练曲线
            if epoch % 5 == 0:  # 每5个epoch绘制一次
                self.plot_training_progress()
        print(f"最佳验证准确率: {self.best_val_accuracy:.4f}")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        self.plot_training_progress()
    
    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config.to_dict()
        }
        
        # 创建检查点目录
        os.makedirs('results/checkpoints', exist_ok=True)
        
        # 保存常规检查点
        checkpoint_path = f'results/checkpoints/full_checkpoint_epoch_{self.epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = 'results/checkpoints/full_best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型: {best_path}")
    
    def plot_training_progress(self):
        """绘制训练进度图"""
        os.makedirs('results/figures', exist_ok=True)
        
        plt.figure(figsize=(15, 10))
        
        # 损失曲线
        plt.subplot(2, 2, 1)
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        if self.val_losses:
            plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 准确率曲线
        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        if self.val_accuracies:
            plt.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 学习率曲线
        plt.subplot(2, 2, 3)
        steps = range(len(self.learning_rates))
        display_steps = min(1000, len(self.learning_rates))
        plt.plot(steps[:display_steps], self.learning_rates[:display_steps])
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        
        # 最近几个epoch的详细损失
        plt.subplot(2, 2, 4)
        recent_epochs = min(10, len(self.train_losses))
        if recent_epochs > 0:
            start_idx = max(0, len(self.train_losses) - recent_epochs)
            epochs_range = range(start_idx + 1, len(self.train_losses) + 1)
            plt.plot(epochs_range, self.train_losses[start_idx:], 'b-', label='Training Loss', linewidth=2)
            if self.val_losses and len(self.val_losses) > start_idx:
                plt.plot(epochs_range, self.val_losses[start_idx:], 'r-', label='Validation Loss', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Last {recent_epochs} Epochs')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/figures/full_training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("完整训练进度图已保存")

class LabelSmoothingLoss(nn.Module):
    """带标签平滑的交叉熵损失"""
    
    def __init__(self, vocab_size, smoothing=0.1, ignore_index=0):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        
    def forward(self, output, target):
        log_probs = torch.nn.functional.log_softmax(output, dim=-1)
        
        # 创建平滑的目标分布
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # 计算损失，忽略padding位置
        mask = (target != self.ignore_index)
        loss = -torch.sum(true_dist * log_probs, dim=-1)
        loss = loss * mask.float()
        
        return loss.sum() / mask.sum()

def calculate_accuracy(output, target, ignore_index=0):
    """计算准确率（忽略padding）"""
    with torch.no_grad():
        pred = output.argmax(dim=-1)
        correct = (pred == target)
        
        # 忽略padding位置
        mask = (target != ignore_index)
        correct = correct & mask
        
        accuracy = correct.float().sum() / mask.float().sum()
        return accuracy.item()

def train_full_model():
    """完整模型训练"""
    
    # 加载配置
    config = Config.from_yaml('configs/base.yaml')
    config.training.max_epochs = 50  # 增加训练轮数
    config.training.base_learning_rate = 5.0e-4  # 提高学习率
    
    # 创建数据处理器
    processor = DataProcessor(
        src_lang=config.data.src_lang,
        tgt_lang=config.data.tgt_lang,
        vocab_size=config.data.vocab_size,
        max_seq_len=config.data.max_seq_len
    )
    
    # 加载完整数据
    src_sentences, tgt_sentences = processor.load_iwslt2017_dataset()
    print(f"完整数据集大小: {len(src_sentences)} 条句子")
    
    # 分割训练集和验证集
    split_idx = int(0.9 * len(src_sentences))  # 90% 训练，10% 验证
    src_train = src_sentences[:split_idx]
    tgt_train = tgt_sentences[:split_idx]
    src_val = src_sentences[split_idx:]
    tgt_val = tgt_sentences[split_idx:]
    
    print(f"训练集: {len(src_train)} 条, 验证集: {len(src_val)} 条")
    
    # 训练分词器
    processor.train_tokenizers(src_train, tgt_train)
    
    # 创建数据加载器
    train_dataloader = processor.create_data_loader(
        src_train, tgt_train, 
        batch_size=config.data.batch_size, shuffle=True
    )
    
    val_dataloader = processor.create_data_loader(
        src_val, tgt_val,
        batch_size=config.data.batch_size, shuffle=False
    )
    
    print(f"训练批次: {len(train_dataloader)}, 验证批次: {len(val_dataloader)}")
    
    # 创建模型
    src_vocab_size = processor.src_tokenizer.get_vocab_size()
    tgt_vocab_size = processor.tgt_tokenizer.get_vocab_size()
    
    print(f"源语言词汇表大小: {src_vocab_size}")
    print(f"目标语言词汇表大小: {tgt_vocab_size}")
    
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config.model.d_model,
        num_heads=config.model.num_heads,
        d_ff=config.model.d_ff,
        num_encoder_layers=config.model.num_encoder_layers,
        num_decoder_layers=config.model.num_decoder_layers,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout
    )
    
    # 计算模型参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数: {total_params:,}")
    
    # 创建训练器
    trainer = FullTrainer(config, model, train_dataloader, val_dataloader, vocab_size=tgt_vocab_size)
    
    # 开始训练
    trainer.train()
    
    return trainer, processor

if __name__ == "__main__":
    trainer, processor = train_full_model()