from pathlib import Path

import numpy as np
import torch
import yaml, os

from src.surrogate.surrogate_vae import SurrogateVAE
from types_ import *


class SurrogateInstance():
    def __init__(self, log_root_dir: str = "../../logs", domain: str = None, train_dim: int = 80, gpu_index: int = -1):
        self.gpu_index = gpu_index
        self.device = torch.device("cuda")
        self.domain = domain
        self.train_dim = train_dim
        self.log_path = Path(log_root_dir, "surrogate_logs", "{}-{}".format(self.domain, self.train_dim))
        config = yaml.safe_load(open(Path(self.log_path, "HyperParam.yaml"), 'r'))
        self.vae = SurrogateVAE(**config['vae_params']).to(self.device)
        model_dict = torch.load(str(Path(self.log_path, "best_model.pt")), map_location=self.device)
        self.vae.load_state_dict(model_dict["vae_model"])
        self.vae.to(self.device)
        self.reference_score = 1
        self.ins_emb: Tensor = model_dict["instance_embedding_0"].to(self.device)
        # self.vae.eval()

    def forward(self, X: Union[NpArray, Tensor]):
        if not isinstance(X, Tensor):
            X = torch.tensor(X)
        X = X.to(self.device)
        X = (X * 2 - 1)
        return self.vae.forward(X, self.ins_emb, train=False)[3].cpu().detach().numpy()

    def load_ins_emb(self, ins_emb: Tensor):
        self.ins_emb = ins_emb.to(self.device)

    def get_all_init_ins_emb(self):
        ins_embs = {}
        model_dict = torch.load(str(Path(self.log_path, "best_model.pt")), map_location=self.device)
        for key in model_dict:
            if 'instance_embedding' in key:
                ins_embs[key] = model_dict[key]
        return ins_embs

    def sample_and_eval(self, batch_num=100, batch_size=16384):
        all_max = []
        all_min = []
        for _ in range(batch_num):
            solution = torch.randint(0, 2, (batch_size, self.train_dim), dtype=torch.float32, device=self.device)
            solution = (solution * 2 - 1)
            score = self.vae.forward(solution, self.ins_emb, train=False)[3]
            all_max.append(torch.max(score).cpu().detach().numpy())
            all_min.append(torch.min(score).cpu().detach().numpy())
        self.reference_score = (np.min(all_min), np.max(all_max))


def _test():
    # domain = "com_influence_max_problem"
    domain = "compiler_args_selection_problem"
    # domain = "contamination_problem"
    train_dim = 80
    ins = SurrogateInstance(domain=domain, train_dim=train_dim)
    ins_embs = ins.get_all_init_ins_emb()
    ins_embs = [ins_embs["instance_embedding_{}".format(index)].cpu().detach().numpy() for index in range(5)]
    up, low = np.max(np.array(ins_embs)), np.min(np.array(ins_embs))

    for index in range(100):
        ins_emb = torch.tensor(np.random.random(size=[len(ins_embs[0])]) * 2 * (up - low) + low, dtype=torch.float32)
        ins.load_ins_emb(ins_emb)
        # y_prime_1 = ins.forward(x)
        # print(np.mean(np.abs(y - y_prime_1)))
        ins.sample_and_eval(batch_num=500)
        print(ins.reference_score)
        # ins.sample_and_eval(batch_num=500)
        # print(ins.reference_score)
        # ins.sample_and_eval(batch_num=500)
        # print(ins.reference_score)
        print()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    _test()
