import copy, torch
from flgo.algorithm import fedbase
from flgo.utils import fmodule


class Server(fedbase.BasicServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def iterate(self):
        """
        The standard iteration of each federated communication round that contains three
        necessary procedure in FL: client selection, communication and model aggregation.

        Returns:
            False if the global model is not updated in this iteration
        """
        # sample clients: MD sampling as default
        self.selected_clients = self.sample()
        # training
        models = self.communicate(self.selected_clients)['model']
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        self.model = self.aggregate(models)
        return len(models) > 0

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
    
    def aggregate(self, models: list, *args, **kwargs):
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
