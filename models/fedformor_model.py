# models/fedformor_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightFusionLayer(nn.Module):
    """
    一个轻量级的融合模块，直接在原始参数维度上操作，避免内存爆炸。
    它学习生成注意力权重，而不是进行高维投影。
    """
    def __init__(self, num_clients, temperature=1.0, beta=0.5):
        """
        Args:
            num_clients (int): 参与聚合的客户端数量 (这个参数目前没用到，但保留以备后用)
            temperature (float): Softmax的温度系数，控制权重分布的尖锐程度
            beta (float): 融合系数，控制客户端贡献与全局模型继承的平衡
        """
        super().__init__()
        self.d_input_sample = 256
        
        # ======================= 关键修复点 1 =======================
        # 将超参数保存为类的属性
        self.temperature = temperature
        self.beta = beta
        # ========================================================
        
        self.attention_net = nn.Sequential(
            nn.Linear(self.d_input_sample * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def _get_fixed_size_sample(self, token_tensor):
        """ 从一个任意长度的token中获取一个固定大小的、有代表性的样本 """
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
        
        expanded_global = global_token.repeat(K, 1)
        
        client_samples = self._get_fixed_size_sample(client_tokens)
        global_samples = self._get_fixed_size_sample(expanded_global)
        
        attention_input = torch.cat([client_samples, global_samples], dim=1)
        
        scores = self.attention_net(attention_input.float()) 
        
        # ======================= 关键修复点 2 =======================
        # 使用 self.temperature
        attention_weights = F.softmax(scores / self.temperature, dim=0).view(K, 1)
        # ========================================================

        # 计算客户端模型的加权融合结果
        fused_by_clients = torch.sum(client_tokens * attention_weights, dim=0, keepdim=True)
        
        # ======================= 关键修复点 3 =======================
        # 使用 self.beta
        fused_token = (1 - self.beta) * global_token + self.beta * fused_by_clients
        # ========================================================
        
        return fused_token