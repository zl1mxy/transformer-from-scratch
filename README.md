具体开发步骤建议
第一天：搭建项目结构，实现MultiHeadAttention并测试

第二天：实现PositionalEncoding和FFN，集成测试

第三天：构建EncoderLayer和DecoderLayer

第四天：完成完整的Transformer类，进行前向传播测试

第五天：实现数据加载和预处理

第六天：编写训练循环，在小批量数据上测试

第七天及以后：完整训练，调试，运行消融实验

 
# 1.环境信息:
Python: 3.8.18
PyTorch: 2.4.1+cpu
设备: CPU

# 2.创建虚拟环境
python -m venv transformer_env
激活环境命令：.\transformer_env\Scripts\Activate.ps1
关闭虚拟环境：deactivate

# 3.安装依赖
pip install -r requirements.txt

# 4.运行训练
python src/train.py

# 5.运行实验
python src/comprehensive_experiment.py