import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size, seq_len = q.size(0), q.size(1)
        
        # 线性变换并分头
        Q = self.w_q(q).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重到V
        attn_output = torch.matmul(attn_weights, V)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 输出线性变换
        output = self.w_o(attn_output)
        
        return output

class PositionalEncoding(nn.Module):
    """正弦余弦位置编码"""
    
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # 正弦函数用于偶数位置，余弦函数用于奇数位置
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加批次维度: (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区（不参与训练）
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x形状: (batch_size, seq_len, d_model)
        # 添加位置编码到输入
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class PositionwiseFFN(nn.Module):
    """逐位置前馈网络"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 使用GELU，效果比ReLU更好
        
    def forward(self, x):
        # x形状: (batch_size, seq_len, d_model)
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
    
class ResidualConnection(nn.Module):
    """残差连接 + LayerNorm"""
    
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        """x: 输入, sublayer: 子层函数"""
        # 先LayerNorm，再通过子层，最后残差连接
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFFN(d_model, d_ff, dropout)
        
        # 两个残差连接
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
        
    def forward(self, x, mask=None):
        """
        参数:
            x: 输入张量 (batch_size, seq_len, d_model)
            mask: 注意力掩码 (batch_size, seq_len, seq_len)
        """
        # 第一个子层：自注意力 + 残差连接
        def self_attn_sublayer(x):
            return self.self_attention(x, x, x, mask)
        
        x = self.residual1(x, self_attn_sublayer)
        
        # 第二个子层：前馈网络 + 残差连接
        def ff_sublayer(x):
            return self.feed_forward(x)
        
        x = self.residual2(x, ff_sublayer)
        
        return x   

class DecoderLayer(nn.Module):
    """Transformer解码器层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # 三个子层
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFFN(d_model, d_ff, dropout)
        
        # 三个残差连接
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
        self.residual3 = ResidualConnection(d_model, dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        参数:
            x: 解码器输入 (batch_size, tgt_seq_len, d_model)
            encoder_output: 编码器输出 (batch_size, src_seq_len, d_model)
            src_mask: 源序列掩码 (batch_size, tgt_seq_len, src_seq_len)
            tgt_mask: 目标序列掩码 (batch_size, tgt_seq_len, tgt_seq_len)
        """
        # 第一个子层：掩码自注意力 + 残差连接
        def self_attn_sublayer(x):
            return self.self_attention(x, x, x, tgt_mask)
        
        x = self.residual1(x, self_attn_sublayer)
        
        # 第二个子层：编码器-解码器注意力 + 残差连接
        def cross_attn_sublayer(x):
            return self.cross_attention(x, encoder_output, encoder_output, src_mask)
        
        x = self.residual2(x, cross_attn_sublayer)
        
        # 第三个子层：前馈网络 + 残差连接
        def ff_sublayer(x):
            return self.feed_forward(x)
        
        x = self.residual3(x, ff_sublayer)
        
        return x

def create_padding_mask(seq, pad_token_id=0):
    """创建填充掩码"""
    # seq: (batch_size, seq_len)
    mask = (seq != pad_token_id).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
    return mask

def create_look_ahead_mask(seq_len):
    """创建因果掩码（防止看到未来信息）"""
    # 创建上三角矩阵（包含对角线）
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    # 转换为布尔掩码：1表示要屏蔽的位置
    mask = mask == 1
    return mask.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, seq_len)

def create_decoder_mask(tgt_seq, pad_token_id=0):
    """创建解码器掩码（结合填充掩码和因果掩码）"""
    # 填充掩码
    padding_mask = create_padding_mask(tgt_seq, pad_token_id)
    # 因果掩码
    seq_len = tgt_seq.size(1)
    look_ahead_mask = create_look_ahead_mask(seq_len)
    
    # 合并掩码：任何为True的位置都要屏蔽
    combined_mask = torch.logical_or(padding_mask, look_ahead_mask)
    return combined_mask

class Encoder(nn.Module):
    """Transformer编码器"""
    
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 词嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # 编码器层堆栈
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        # 输出层归一化
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, src_tokens, src_mask=None):
        """
        参数:
            src_tokens: 源序列token IDs (batch_size, src_seq_len)
            src_mask: 源序列掩码 (batch_size, 1, 1, src_seq_len)
        """
        # 词嵌入 + 缩放
        x = self.token_embedding(src_tokens) * math.sqrt(self.d_model)
        
        # 添加位置编码
        x = self.positional_encoding(x)
        
        # 通过所有编码器层
        for layer in self.layers:
            x = layer(x, src_mask)
        
        # 最终层归一化
        x = self.norm(x)
        
        return x
    
class Decoder(nn.Module):
    """Transformer解码器"""
    
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 词嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # 解码器层堆栈
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        # 输出层归一化
        self.norm = nn.LayerNorm(d_model)
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, tgt_tokens, encoder_output, src_mask=None, tgt_mask=None):
        """
        参数:
            tgt_tokens: 目标序列token IDs (batch_size, tgt_seq_len)
            encoder_output: 编码器输出 (batch_size, src_seq_len, d_model)
            src_mask: 源序列掩码 (batch_size, 1, 1, src_seq_len)
            tgt_mask: 目标序列掩码 (batch_size, 1, tgt_seq_len, tgt_seq_len)
        """
        # 词嵌入 + 缩放
        x = self.token_embedding(tgt_tokens) * math.sqrt(self.d_model)
        
        # 添加位置编码
        x = self.positional_encoding(x)
        
        # 通过所有解码器层
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        # 最终层归一化
        x = self.norm(x)
        
        # 输出投影到词汇表
        logits = self.output_projection(x)
        
        return logits
    
class Transformer(nn.Module):
    """完整的Transformer模型"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 d_ff=2048, num_encoder_layers=6, num_decoder_layers=6, 
                 max_seq_len=5000, dropout=0.1, pad_token_id=0):
        super().__init__()
        self.pad_token_id = pad_token_id
        
        # 编码器
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_encoder_layers,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # 解码器
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_decoder_layers,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def encode(self, src_tokens, src_mask=None):
        """编码源序列"""
        if src_mask is None:
            src_mask = create_padding_mask(src_tokens, self.pad_token_id)
        return self.encoder(src_tokens, src_mask)
    
    def decode(self, tgt_tokens, encoder_output, src_mask=None, tgt_mask=None):
        """解码目标序列"""
        if tgt_mask is None:
            tgt_mask = create_decoder_mask(tgt_tokens, self.pad_token_id)
        return self.decoder(tgt_tokens, encoder_output, src_mask, tgt_mask)
    
    def forward(self, src_tokens, tgt_tokens):
        """
        完整的前向传播
        
        参数:
            src_tokens: 源序列token IDs (batch_size, src_seq_len)
            tgt_tokens: 目标序列token IDs (batch_size, tgt_seq_len)
        """
        # 创建掩码
        src_mask = create_padding_mask(src_tokens, self.pad_token_id)
        tgt_mask = create_decoder_mask(tgt_tokens, self.pad_token_id)
        
        # 编码
        encoder_output = self.encode(src_tokens, src_mask)
        
        # 解码
        decoder_output = self.decode(tgt_tokens, encoder_output, src_mask, tgt_mask)
        
        return decoder_output
    
    def generate(self, src_tokens, max_length=50, start_token_id=1, end_token_id=2):
        """生成序列（简单的贪婪解码）"""
        self.eval()
        batch_size = src_tokens.size(0)
        
        # 编码源序列
        encoder_output = self.encode(src_tokens)
        
        # 初始化目标序列（开始token）
        generated = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=src_tokens.device)
        
        for _ in range(max_length - 1):
            # 解码
            logits = self.decode(generated, encoder_output)
            
            # 获取最后一个token的预测
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 添加到生成序列
            generated = torch.cat([generated, next_token], dim=1)
            
            # 如果所有序列都生成了结束token，则停止
            if (next_token == end_token_id).all():
                break
                
        return generated
