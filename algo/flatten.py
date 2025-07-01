import copy, torch
import torch.nn as nn 
from flgo.algorithm import fedbase
from flgo.utils import fmodule
from models.fedformor_model import LightweightFusionLayer

class Server(fedbase.BasicServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # K是参与聚合的客户端数量，我们需要提前知道或设定
        self.num_clients_per_round = int(len(self.clients)*self.proportion)
        
        # 初始化轻量级融合层
        self.fusion_layer = LightweightFusionLayer(self.num_clients_per_round).to(self.device)
        
        # 优化器现在只管理这个小小的fusion_layer的参数
        self.fusion_optimizer = torch.optim.Adam(self.fusion_layer.parameters(), lr=1e-4)


    def iterate(self):
        self.selected_clients = self.sample()
        models = self.communicate(self.selected_clients)['model']
        if not models: return False

        # 序列化
        client_sequences, global_sequences = self.models_to_sequences(models)

        new_global_state_dict = self.model.state_dict()

        for name in client_sequences.keys():
            client_tensors = torch.stack(client_sequences[name]).to(self.device) # (K, D)
            global_tensor = global_sequences[name].unsqueeze(0).to(self.device) # (1, D)
            
            # 使用新的融合层
            fused_token = self.fusion_layer(client_tensors, global_tensor) # (1, D)
            
            original_shape = new_global_state_dict[name].shape
            new_global_state_dict[name] = fused_token.view(original_shape)

        w_candidate = copy.deepcopy(self.model)
        w_candidate.load_state_dict(new_global_state_dict)
        w_candidate = w_candidate.to(self.device)

        # 元学习部分
        # 手动创建DataLoader来获取代理数据
        proxy_data_loader = torch.utils.data.DataLoader(
            self.test_data,
            batch_size=64,
            shuffle=True, 
            num_workers=self.option.get('num_workers', 0)
        )

        try:
            # 从迭代器中获取一个批次
            proxy_batch = next(iter(proxy_data_loader))
            # 将数据移动到正确的设备上
            proxy_batch = self.calculator.to_device(proxy_batch)
        except StopIteration:
            self.gv.logger.warning("代理数据集(测试集)为空，本轮跳过融合网络训练。")
            proxy_batch = None
            
        if proxy_batch:
            self.fusion_optimizer.zero_grad()
            # 使用calculator计算损失
            loss_meta = self.calculator.compute_loss(w_candidate, proxy_batch)['loss']
            loss_meta.backward()
            self.fusion_optimizer.step()
            self.gv.logger.info(f"Fusion Meta Loss: {loss_meta.item():.4f}")

        # 更新全局模型
        self.model.load_state_dict(w_candidate.state_dict(keep_vars=False))
        self.model = self.model.to(self.device)
        return True

    def pack(self, client_id, mtype=0, *args, **kwargs):
        r"""
        Pack the necessary information for the client's local training.
        Any operations of compression or encryption should be done here.

        Args:
            client_id (int): the id of the client to communicate with
            mtype: the message type

        Returns:
            a dict contains necessary information (e.g. a copy of the global model as default)
        """
        return {
            "model": copy.deepcopy(self.model),
            "current_round": self.current_round
        }
    
    def unpack(self, packages_received_from_clients):
        r"""
        Unpack the information from the received packages. Return models and losses as default.

        Args:
            packages_received_from_clients (list): a list of packages

        Returns:
            res (dict): collections.defaultdict that contains several lists of the clients' reply
        """
        if len(packages_received_from_clients) == 0: return collections.defaultdict(list)
        res = {pname: [] for pname in packages_received_from_clients[0]}
        for cpkg in packages_received_from_clients:
            for pname, pval in cpkg.items():
                res[pname].append(pval)
        return res
        r"""
        Aggregate the locally trained models into the new one. The aggregation
        will be according to self.aggregate_option where

        pk = nk/n where n=self.data_vol
        K = |S_t|
        N = |S|
        -------------------------------------------------------------------------------------------------------------------------
         weighted_scale                 |uniform (default)          |weighted_com (original fedavg)   |other
        ==========================================================================================================================
        N/K * Σpk * model_k             |1/K * Σmodel_k             |(1-Σpk) * w_old + Σpk * model_k  |Σ(pk/Σpk) * model_k


        Args:
            models (list): a list of local models

        Returns:
            the aggregated model

        Example:
        ```python
            >>> models = [m1, m2] # m1, m2 are models with the same architecture
            >>> m_new = self.aggregate(models)
        ```
        """
        if len(models) == 0: return self.model
        nan_exists = [m.has_nan() for m in models]
        if any(nan_exists):
            if all(nan_exists): raise ValueError("All the received local models have parameters of nan value.")
            self.gv.logger.info('Warning("There exists nan-value in local models, which will be automatically removed from the aggregatino list.")')
            new_models = []
            received_clients = []
            for ni, mi, cid in zip(nan_exists, models, self.received_clients):
                if ni: continue
                new_models.append(mi)
                received_clients.append(cid)
            self.received_clients = received_clients
            models = new_models
        local_data_vols = [c.datavol for c in self.clients]
        total_data_vol = sum(local_data_vols)
        if self.aggregation_option == 'weighted_scale':
            p = [1.0 * local_data_vols[cid] / total_data_vol for cid in self.received_clients]
            K = len(models)
            N = self.num_clients
            return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)]) * N / K
        elif self.aggregation_option == 'uniform':
            return fmodule._model_average(models)
        elif self.aggregation_option == 'weighted_com':
            p = [1.0 * local_data_vols[cid] / total_data_vol for cid in self.received_clients]
            w = fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])
            return (1.0 - sum(p)) * self.model + w
        else:
            p = [1.0 * local_data_vols[cid] / total_data_vol for cid in self.received_clients]
            sump = sum(p)
            p = [pk / sump for pk in p]
            return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])
        
    def models_to_sequences(self, models: list):
        if not models: return {}, None
        cpu_device = torch.device('cpu')
        
        # 将上一轮的全局模型也移动到cpu
        global_model_cpu = self.model.to(cpu_device)
        client_models_cpu = [m.to(cpu_device) for m in models]

        # 分别处理客户端和全局模型
        client_state_dicts = [m.state_dict() for m in client_models_cpu]
        global_state_dict = global_model_cpu.state_dict()

        sequences = {name: [] for name, _ in global_state_dict.items()}
        global_sequences = {name: None for name, _ in global_state_dict.items()}

        for state_dict in client_state_dicts:
            for name, params in state_dict.items():
                sequences[name].append(params.view(-1))
        
        for name, params in global_state_dict.items():
            global_sequences[name] = params.view(-1)
        
        return sequences, global_sequences


class Client(fedbase.BasicClient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_round = 0
        self.task = self.option['task']

    @fmodule.with_multi_gpus
    def train(self, model):
        r"""
        Standard local training procedure. Train the transmitted model with
        local training dataset.

        Args:
            model (FModule): the global model
        """
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay,
                                                  momentum=self.momentum)
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss.backward()
            if self.clip_grad > 0: torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                                                  max_norm=self.clip_grad)
            optimizer.step()
        return
    
    def pack(self, model, *args, **kwargs):
        r"""
        Packing the package to be send to the server. The operations of compression
        of encryption of the package should be done here.

        Args:
            model: the locally trained model

        Returns:
            package: a dict that contains the necessary information for the server
        """
        return {
            "model": model,
        }

    def unpack(self, received_pkg):
        self.current_round = received_pkg['current_round']
        return received_pkg['model']
