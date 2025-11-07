import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import sentencepiece as spm
import os
import tempfile
from typing import List, Tuple, Dict
import time

class TranslationDataset(Dataset):
    """机器翻译数据集"""
    
    def __init__(self, src_sentences: List[str], tgt_sentences: List[str], 
                 src_tokenizer, tgt_tokenizer, max_seq_len: int = 100):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_seq_len = max_seq_len
        
        assert len(src_sentences) == len(tgt_sentences), "源语言和目标语言句子数量不匹配"
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src_text = self.src_sentences[idx]
        tgt_text = self.tgt_sentences[idx]
        
        # 分词
        src_tokens = self.src_tokenizer.encode(src_text)
        tgt_tokens = self.tgt_tokenizer.encode(tgt_text)
        
        # 截断到最大长度
        src_tokens = src_tokens[:self.max_seq_len]
        tgt_tokens = tgt_tokens[:self.max_seq_len]
        
        return {
            'src_tokens': src_tokens,
            'tgt_tokens': tgt_tokens,
            'src_text': src_text,
            'tgt_text': tgt_text
        }

class SentencePieceTokenizer:
    """SentencePiece分词器封装"""
    
    def __init__(self, model_path=None):
        self.sp = spm.SentencePieceProcessor()
        if model_path and os.path.exists(model_path):
            self.sp.load(model_path)
    
    def encode(self, text: str) -> List[int]:
        """将文本编码为token IDs"""
        return self.sp.encode_as_ids(text)
    
    def decode(self, tokens: List[int]) -> str:
        """将token IDs解码为文本"""
        return self.sp.decode_ids(tokens)
    
    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        return self.sp.get_piece_size()

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, src_lang: str = 'en', tgt_lang: str = 'de', vocab_size: int = 8000, max_seq_len: int = 100):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.src_tokenizer = None
        self.tgt_tokenizer = None
    
    def load_iwslt2017_dataset(self, split: str = 'train'):
        """加载IWSLT2017数据集"""
        print(f"加载IWSLT2017 {split}数据集...")
        
        try:
            # 设置超时时间
            import socket
            socket.setdefaulttimeout(30)  # 30秒超时
            
            dataset = load_dataset('iwslt2017', 'iwslt2017-de-en', split=split)
            print(f"成功加载 {len(dataset)} 条数据")
            
            # 提取源语言和目标语言句子
            src_sentences = [item['translation'][self.src_lang] for item in dataset]
            tgt_sentences = [item['translation'][self.tgt_lang] for item in dataset]
            
            return src_sentences, tgt_sentences
            
        except Exception as e:
            print(f"加载数据集失败: {e}")
            print("使用模拟数据进行开发...")
            return self._create_dummy_data(1000)  # 创建1000条模拟数据
    
    def _create_dummy_data(self, num_samples=1000):
        """创建模拟数据用于测试"""
        base_sentences = [
            ("Hello world, how are you today?", "Hallo Welt, wie geht es dir heute?"),
            ("This is a test sentence for machine translation.", "Dies ist ein Testsatz für maschinelle Übersetzung."),
            ("The weather is very nice outside.", "Das Wetter ist sehr schön draußen."),
            ("I love learning about artificial intelligence.", "Ich liebe es, etwas über künstliche Intelligenz zu lernen."),
            ("Deep learning models require large amounts of data.", "Deep-Learning-Modelle benötigen große Datenmengen."),
            ("Natural language processing is a fascinating field.", "Natürliche Sprachverarbeitung ist ein faszinierendes Feld."),
            ("Transformers have revolutionized machine learning.", "Transformer haben maschinelles Lernen revolutioniert."),
            ("Attention mechanisms allow models to focus on important parts.", "Aufmerksamkeitsmechanismen ermöglichen es Modellen, sich auf wichtige Teile zu konzentrieren."),
            ("We are building a transformer from scratch.", "Wir bauen einen Transformer von Grund auf."),
            ("This project will help understand the architecture better.", "Dieses Projekt wird helfen, die Architektur besser zu verstehen."),
            ("The cat is sleeping on the sofa.", "Die Katze schläft auf dem Sofa."),
            ("I have two brothers and one sister.", "Ich habe zwei Brüder und eine Schwester."),
            ("She works in a big company.", "Sie arbeitet in einer großen Firma."),
            ("We are going to the cinema tonight.", "Wir gehen heute Abend ins Kino."),
            ("My favorite color is blue.", "Meine Lieblingsfarbe ist blau."),
            ("The food in this restaurant is delicious.", "Das Essen in diesem Restaurant ist köstlich."),
            ("He is reading an interesting book.", "Er liest ein interessantes Buch."),
            ("They are playing football in the park.", "Sie spielen Fußball im Park."),
            ("I need to buy some groceries.", "Ich muss einige Lebensmittel einkaufen."),
            ("The meeting starts at nine o'clock.", "Die Besprechung beginnt um neun Uhr."),
            ("Can you help me with this problem?", "Kannst du mir bei diesem Problem helfen?"),
            ("I like to listen to music in my free time.", "In meiner Freizeit höre ich gerne Musik."),
            ("The children are playing in the garden.", "Die Kinder spielen im Garten."),
            ("This computer is very fast and efficient.", "Dieser Computer ist sehr schnell und effizient."),
            ("We should protect the environment for future generations.", "Wir sollten die Umwelt für zukünftige Generationen schützen."),
            ("Learning a new language is challenging but rewarding.", "Eine neue Sprache zu lernen ist herausfordernd aber lohnend."),
            ("The sun is shining brightly today.", "Die Sonne scheint heute hell."),
            ("I enjoy reading books about history and science.", "Ich lese gerne Bücher über Geschichte und Wissenschaft."),
            ("The train arrives at the station in five minutes.", "Der Zug kommt in fünf Minuten am Bahnhof an."),
            ("She speaks three languages fluently.", "Sie spricht drei Sprachen fließend.")
        ]
        
        # 重复数据以达到所需数量
        src_sentences = []
        tgt_sentences = []
        
        for i in range(num_samples):
            pair = base_sentences[i % len(base_sentences)]
            src_sentences.append(pair[0])
            tgt_sentences.append(pair[1])
        
        print(f"创建模拟数据: {len(src_sentences)} 条句子")
        return src_sentences, tgt_sentences
    
    def train_tokenizers(self, src_sentences: List[str], tgt_sentences: List[str], 
                        model_prefix: str = "tokenizer"):
        """训练SentencePiece分词器"""
        print("训练分词器...")
        
        # 根据数据量动态调整词汇表大小
        estimated_vocab_size = min(self.vocab_size, len(src_sentences) * 2)
        estimated_vocab_size = max(500, estimated_vocab_size)  # 最小500
        print(f"设置词汇表大小: {estimated_vocab_size}")
        
        try:
            # 为源语言和目标语言分别创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as src_f:
                src_f.write('\n'.join(src_sentences))
                src_temp_file = src_f.name
                
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as tgt_f:
                tgt_f.write('\n'.join(tgt_sentences))
                tgt_temp_file = tgt_f.name
            
            # 训练源语言分词器
            src_model_path = f"{model_prefix}_{self.src_lang}"
            spm.SentencePieceTrainer.train(
                input=src_temp_file,
                model_prefix=src_model_path,
                vocab_size=estimated_vocab_size,
                character_coverage=1.0,
                model_type='bpe',
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3
            )
            
            # 训练目标语言分词器
            tgt_model_path = f"{model_prefix}_{self.tgt_lang}"
            spm.SentencePieceTrainer.train(
                input=tgt_temp_file,
                model_prefix=tgt_model_path,
                vocab_size=estimated_vocab_size,
                character_coverage=1.0,
                model_type='bpe',
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3
            )
            
            # 加载训练好的分词器
            self.src_tokenizer = SentencePieceTokenizer(src_model_path + ".model")
            self.tgt_tokenizer = SentencePieceTokenizer(tgt_model_path + ".model")
            
            print(f"分词器训练完成")
            print(f"   源语言词汇表大小: {self.src_tokenizer.get_vocab_size()}")
            print(f"   目标语言词汇表大小: {self.tgt_tokenizer.get_vocab_size()}")
            
        except Exception as e:
            print(f"分词器训练失败: {e}")
            # 创建简单的分词器回退方案
            self._create_fallback_tokenizers(src_sentences, tgt_sentences)
        finally:
            # 清理临时文件
            for temp_file in [src_temp_file, tgt_temp_file]:
                if 'temp_file' in locals() and os.path.exists(temp_file):
                    os.unlink(temp_file)
    
    def _create_fallback_tokenizers(self, src_sentences, tgt_sentences):
        """创建简单的回退分词器"""
        print("使用简单分词器作为回退...")
        
        class SimpleTokenizer:
            def __init__(self, sentences, lang):
                # 构建词汇表
                self.vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
                self.reverse_vocab = {v: k for k, v in self.vocab.items()}
                
                # 从句子中提取单词
                word_count = {}
                for sentence in sentences:
                    words = sentence.lower().replace(',', '').replace('.', '').replace('?', '').split()
                    for word in words:
                        word_count[word] = word_count.get(word, 0) + 1
                
                # 选择最常见的单词
                vocab_size = min(1000, len(word_count))
                common_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:vocab_size]
                
                for idx, (word, count) in enumerate(common_words):
                    self.vocab[word] = idx + 4  # 从4开始，前4个是特殊token
                    self.reverse_vocab[idx + 4] = word
                
                self.vocab_size = len(self.vocab)
                print(f"  简单分词器 ({lang}): 词汇表大小 {self.vocab_size}")
                
            def encode(self, text):
                # 简单的空格分词，清理标点
                cleaned_text = text.lower().replace(',', '').replace('.', '').replace('?', '')
                tokens = cleaned_text.split()
                encoded = [self.vocab.get(token, 1) for token in tokens]  # 1是unk token
                # 添加开始和结束标记
                return [2] + encoded + [3]  # <bos> + tokens + <eos>
                
            def decode(self, tokens):
                # 过滤特殊token
                filtered_tokens = [t for t in tokens if t not in [0, 1, 2, 3]]
                words = [self.reverse_vocab.get(t, "<unk>") for t in filtered_tokens]
                return " ".join(words)
                
            def get_vocab_size(self):
                return self.vocab_size
        
        self.src_tokenizer = SimpleTokenizer(src_sentences, self.src_lang)
        self.tgt_tokenizer = SimpleTokenizer(tgt_sentences, self.tgt_lang)
    
    def create_data_loader(self, src_sentences: List[str], tgt_sentences: List[str], 
                          batch_size: int = 32, shuffle: bool = True):
        """创建数据加载器"""
        
        if self.src_tokenizer is None or self.tgt_tokenizer is None:
            raise ValueError("请先训练分词器")
        
        dataset = TranslationDataset(
            src_sentences=src_sentences,
            tgt_sentences=tgt_sentences,
            src_tokenizer=self.src_tokenizer,
            tgt_tokenizer=self.tgt_tokenizer,
            max_seq_len=self.max_seq_len
        )
        
        def collate_fn(batch):
            """批处理函数"""
            src_tokens = [item['src_tokens'] for item in batch]
            tgt_tokens = [item['tgt_tokens'] for item in batch]
            src_texts = [item['src_text'] for item in batch]
            tgt_texts = [item['tgt_text'] for item in batch]
            
            # 动态padding
            src_padded = self._pad_sequences(src_tokens, pad_token_id=0)
            tgt_padded = self._pad_sequences(tgt_tokens, pad_token_id=0)
            
            return {
                'src_tokens': torch.tensor(src_padded, dtype=torch.long),
                'tgt_tokens': torch.tensor(tgt_padded, dtype=torch.long),
                'src_text': src_texts,  
                'tgt_text': tgt_texts   
            }
        
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=0  # Windows上设为0避免出现问题
        )
    
    def _pad_sequences(self, sequences: List[List[int]], pad_token_id: int = 0) -> List[List[int]]:
        """对序列进行padding"""
        max_len = max(len(seq) for seq in sequences)
        padded_sequences = []
        for seq in sequences:
            padded_seq = seq + [pad_token_id] * (max_len - len(seq))
            padded_sequences.append(padded_seq)
        return padded_sequences

def test_data_loading():
    """测试数据加载功能"""
    
    # 创建数据处理器（使用默认参数）
    processor = DataProcessor(src_lang='en', tgt_lang='de', vocab_size=2000, max_seq_len=50)
    
    # 加载数据（会自动使用模拟数据）
    src_sentences, tgt_sentences = processor.load_iwslt2017_dataset('train')
    
    print(f"源语言示例: {src_sentences[0]}")
    print(f"目标语言示例: {tgt_sentences[0]}")
    print(f"数据量: {len(src_sentences)} 条句子")
    
    # 训练分词器
    processor.train_tokenizers(src_sentences, tgt_sentences)
    
    # 创建数据加载器
    dataloader = processor.create_data_loader(
        src_sentences, tgt_sentences, 
        batch_size=2, shuffle=False
    )
    
    # 测试一个批次
    for i, batch in enumerate(dataloader):
        print(f"\n批次 {i + 1}:")
        print(f"  源序列形状: {batch['src_tokens'].shape}")
        print(f"  目标序列形状: {batch['tgt_tokens'].shape}")
        print(f"  源文本: {batch['src_text'][0]}")
        print(f"  目标文本: {batch['tgt_text'][0]}")
        print(f"  源token IDs: {batch['src_tokens'][0][:10]}...")  # 显示前10个token
        print(f"  目标token IDs: {batch['tgt_tokens'][0][:10]}...")
        
        # 测试分词器解码
        src_decoded = processor.src_tokenizer.decode(batch['src_tokens'][0].tolist())
        tgt_decoded = processor.tgt_tokenizer.decode(batch['tgt_tokens'][0].tolist())
        print(f"  源序列解码: {src_decoded[:50]}...")
        print(f"  目标序列解码: {tgt_decoded[:50]}...")
        
        if i >= 1:  # 只显示前2个批次
            break
    
    print("\n 数据加载测试通过！")
    return processor

if __name__ == "__main__":
    print("=" * 60)
    print("数据加载和预处理系统")
    print("=" * 60)
    
    try:
        processor = test_data_loading()
        
        print("\n" + "=" * 60)
        print("数据加载系统实现成功")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n 测试失败: {e}")
        import traceback
        traceback.print_exc()