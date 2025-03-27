import os
import pickle
from pathlib import Path

import yaml
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.experiments.experiment_problem import load_problem_data
from src.problem_domain import BaseProblem
from src.surrogate import SurrogateVAE, ZeroOneProblemData
from src.types_ import *


def fit_surrogate(problem_domain: str, ins_dir: str = '../data/problem_instance', ins_type: str = "train",
                  dim: int = 30,
                  index: int = 0, gpu_index: int = -1):
    device = torch.device("cuda:{}".format(gpu_index)) if gpu_index >= 0 else torch.device("cpu")
    with open("../configs/surrogate.yaml", 'r') as file:
        config = yaml.safe_load(file)
    problem_path = Path(ins_dir, ins_type, f"{problem_domain}_{dim}_{index}")
    problem_instance: BaseProblem = pickle.load(open(Path(problem_path, "problem.pkl"), 'rb'))

    config["model_params"]["in_dim"] = problem_instance.dimension
    config["model_params"]["latent_dim"] = problem_instance.dimension * config["model_params"]["latent_dim_coefficient"]
    config["trainer_params"]["gpus"] = [gpu_index]
    config["logging_params"]["name"] = "{}-{}".format(problem_domain, index)
    seed_everything(config['exp_params']['manual_seed'], True)
    model = SurrogateVAE(**config["vae_params"]).to(device)

    x, y = load_problem_data(problem_path)
    train_data = ZeroOneProblemData(x, y, 'train')
    valid_data = ZeroOneProblemData(x, y, 'valid')
    train_dataloader = DataLoader(train_data, batch_size=config['data_params']['train_batch_size'], shuffle=True,
                                  num_workers=config['data_params']['num_workers'])
    valid_dataloader = DataLoader(valid_data, batch_size=config['data_params']['val_batch_size'], shuffle=True,
                                  num_workers=config['data_params']['num_workers'])
    log_path = Path(config['logging_params']['save_dir'], config["logging_params"]["name"])
    if not os.path.exists(Path(config['logging_params']['save_dir'])):
        os.makedirs(Path(config['logging_params']['save_dir']))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(str(log_path))
    yaml.dump(config, open(Path(log_path, "HyperParam.yaml"), "w"))
    optimizer = optim.Adam(model.parameters(),
                           lr=config['exp_params']['LR'],
                           weight_decay=config['exp_params']['weight_decay'])
    # writer.add_graph(model)
    # epoch_bar = tqdm(range(int(config['trainer_params']['max_epochs'])))
    best_val_loss = np.inf
    for epoch in range(int(config['trainer_params']['max_epochs'])):
        loss_records = {}
        for solution, quality in train_dataloader:
            optimizer.zero_grad()
            train_loss = model.loss_function(solution.to(device), quality.to(device))
            train_loss['loss'].backward()
            optimizer.step()
            for key in train_loss.keys():
                if key not in loss_records:
                    loss_records[key] = []
                loss_records[key].append(train_loss[key] if key != "loss" else train_loss[key].cpu().detach().numpy())
        for solution, quality in valid_dataloader:
            valid_loss = model.loss_function(solution.to(device), quality.to(device))
            for key in valid_loss.keys():
                if "val_{}".format(key) not in loss_records:
                    loss_records["val_{}".format(key)] = []
                loss_records["val_{}".format(key)].append(
                    valid_loss[key] if key != "loss" else valid_loss[key].cpu().detach().numpy())
        if np.mean(loss_records['val_loss']) < best_val_loss:
            best_val_loss = np.mean(loss_records['val_loss'])
            torch.save(model.state_dict(), Path(log_path, "best_model.pt"))

        for key in loss_records.keys():
            writer.add_scalar(key, np.mean(loss_records[key]), epoch)

        # epoch_bar.set_description("Epoch {}".format(epoch))
        # epoch_bar.set_postfix_str("MSE {:.5f}".format(np.mean(loss_records['loss'])))
    print("Finish the surrogate Task of {}_{}".format(problem_domain, index))


