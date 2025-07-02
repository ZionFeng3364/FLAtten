# models/fedformor_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightFusionLayer(nn.Module):
    """
    一个轻量级的融合模块，直接在原始参数维度上操作，避免内存爆炸。
    它学习生成注意力权重，而不是进行高维投影。
    
    改进版：引入了客户端间的交互 (Self-Attention)。
    """
    def __init__(self, num_clients, temperature=1.0, beta=0.5):
        """
        Args:
            num_clients (int): 参与聚合的客户端数量 (保留以备后用)
            temperature (float): Softmax的温度系数
            beta (float): 融合系数
        """
        super().__init__()
        self.d_input_sample = 256
        self.temperature = temperature
        self.beta = beta
        
        # ======================= 核心结构修改 =======================
        # 定义自注意力交互模块
        self.d_model = self.d_input_sample * 2 # 交互模型的维度
        
        # 1. 创建一个Transformer编码器层作为基础构建块
        #    nhead可以调整，4或8是常用值
        #    batch_first=True 是一个非常重要的参数，它让输入形状更直观
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=8, 
            dim_feedforward=self.d_model * 2, # 通常设为d_model的2到4倍
            dropout=0.1,
            activation='relu', # 明确指定激活函数
            batch_first=True
        )
        
        # 2. 用上面创建的层构建一个完整的Transformer编码器
        #    我们只用一层就足够了，避免模型过于复杂
        self.client_interactor = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # 3. 最终的打分器：一个简单的线性层，从交互后的特征中得出分数
        self.scorer = nn.Linear(self.d_model, 1)
        # ==========================================================

    def _get_fixed_size_sample(self, token_tensor):
        """ 从一个任意长度的token中获取一个固定大小的、有代表性的样本 (此函数保持不变) """
        n, l = token_tensor.shape
        indices = torch.arange(self.d_input_sample, dtype=torch.long, device=token_tensor.device) % l
        indices_expanded = indices.unsqueeze(0).expand(n, -1)
        sample = torch.gather(token_tensor, 1, indices_expanded)
        return sample

    def forward(self, client_tokens, global_token):
        """
        前向传播。
        """
        K, D = client_tokens.shape
        
        # 1. 准备输入 (和之前一样)
        expanded_global = global_token.repeat(K, 1)
        client_samples = self._get_fixed_size_sample(client_tokens)
        global_samples = self._get_fixed_size_sample(expanded_global)
        attention_input = torch.cat([client_samples, global_samples], dim=1) # 形状: [K, d_model]
        
        # ======================= 关键修复点 =======================
        # 在将数据送入任何需要浮点运算的神经网络层（尤其是Transformer）之前，
        # 强制确保其数据类型为 float。
        attention_input_float = attention_input.float()
        # ========================================================
        
        # 2. 客户端间信息交互 (Self-Attention)
        #    Transformer需要 [batch_size, seq_len, feature_dim] 的输入
        interacted_features = self.client_interactor(attention_input_float.unsqueeze(0)) # 输入形状: [1, K, d_model]
        
        # 3. 基于交互后的特征进行打分
        scores = self.scorer(interacted_features.squeeze(0)) # 输出形状: [K, 1]
        
        # 4. 计算权重 (和之前一样)
        attention_weights = F.softmax(scores / self.temperature, dim=0).view(K, 1)

        # 5. 融合 (和之前一样)
        fused_by_clients = torch.sum(client_tokens * attention_weights, dim=0, keepdim=True)
        fused_token = (1 - self.beta) * global_token + self.beta * fused_by_clients
        
        return fused_token