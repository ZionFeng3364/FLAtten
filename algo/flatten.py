import copy, torch
from flgo.algorithm import fedbase
from flgo.utils import fmodule


class Server(fedbase.BasicServer):

    def pack(self, client_id, mtype=0, *args, **kwargs):
        return {
            "model": copy.deepcopy(self.model),
            "current_round": self.current_round
        }


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

    def unpack(self, received_pkg):
        self.current_round = received_pkg['current_round']
        return received_pkg['model']
