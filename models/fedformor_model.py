# models/fedformor_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightFusionLayer(nn.Module):
    """
    一个轻量级的融合模块，直接在原始参数维度上操作，避免内存爆炸。
    它学习生成注意力权重，而不是进行高维投影。
    """
    def __init__(self, num_clients):
        super().__init__()
        self.d_input_sample = 256
        
        self.attention_net = nn.Sequential(
            nn.Linear(self.d_input_sample * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def _get_fixed_size_sample(self, token_tensor):
        """ 从一个任意长度的token中获取一个固定大小的、有代表性的样本 """
        n, l = token_tensor.shape
        
        # 明确指定dtype为long，虽然这是默认的，但更清晰
        indices = torch.arange(self.d_input_sample, dtype=torch.long, device=token_tensor.device) % l
        
        # 使用gather而不是直接索引，有时更稳定
        # .unsqueeze(0).expand(n, -1) 是为了将一维的indices扩展成和batch匹配的二维
        indices_expanded = indices.unsqueeze(0).expand(n, -1)
        sample = torch.gather(token_tensor, 1, indices_expanded)
        
        return sample

    def forward(self, client_tokens, global_token):
        """
        前向传播。
        Args:
            client_tokens (Tensor): 形状为 (K, D) 的客户端层参数。
            global_token (Tensor): 形状为 (1, D) 的全局层参数。
        
        Returns:
            Tensor: 形状为 (1, D) 的融合后的新参数。
        """
        K, D = client_tokens.shape
        
        expanded_global = global_token.repeat(K, 1)
        
        client_samples = self._get_fixed_size_sample(client_tokens)
        global_samples = self._get_fixed_size_sample(expanded_global)
        
        attention_input = torch.cat([client_samples, global_samples], dim=1)
        
        # ======================= 关键修复点 =======================
        # 在送入线性层之前，确保输入是浮点类型。
        # attention_net的权重是float，所以输入也必须是float。
        # ========================================================
        scores = self.attention_net(attention_input.float()) 
        
        attention_weights = F.softmax(scores, dim=0).view(K, 1)
        
        # 确保这里的乘法也是在float类型上进行
        fused_token = torch.sum(client_tokens * attention_weights, dim=0, keepdim=True)
        
        return fused_token