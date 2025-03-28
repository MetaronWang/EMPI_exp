from multiprocessing import Process

import requests
import yaml

from src.distribution.util import random_str
from src.experiments.experiment_problem import train_problem_types, problem_settings
from src.types_ import *

master_host = "10.16.104.19:1088"

def send_surrogate_fit_task(domain, model_dir: Union[Path, str], ins_dir: Union[Path, str], dim: int,
                            index: int):
    task_id = f"{str(int(time.time() * 1000))}_{random_str(32)}"
    task = {
        "task_id": task_id,
        "task_func": "eval_surrogate_training",
        "task_args": (domain, ins_dir, "train", dim, index),
        "task_cost": 1,
        "task_type": "gpu"
    }
    task_data = pickle.dumps(task)
    response_code = 500
    while response_code > 299 or response_code < 200:
        try:
            response = requests.post(url="http://{}/create_task".format(master_host),
                                     data={"check_val": "81600a92e8416bba7d9fada48e9402a4",
                                           "task_type": "gpu", "task_cost": 1, "task_id": task_id},
                                     files={"task_data": task_data},
                                     verify=False)
            response_code = response.status_code
        except Exception:
            pass
    response_code = 500
    while response_code > 299 or response_code < 200:
        try:
            time.sleep(1)
            response = requests.post(url="http://{}/get_result".format(master_host),
                                     data={"check_val": "81600a92e8416bba7d9fada48e9402a4",
                                           "task_id": task_id},
                                     verify=False)
            response_code = response.status_code
        except Exception:
            pass
    result = pickle.loads(response.content)
    yaml.dump(result[0], open(Path(model_dir, "HyperParam.yaml"), "w"))
    open(Path(model_dir, "best_model.pt"), "wb").write(result[1])
    while response_code > 299 or response_code < 200:
        try:
            time.sleep(1)
            response = requests.post(url="http://{}/clear_record".format(master_host),
                                     data={"check_val": "81600a92e8416bba7d9fada48e9402a4",
                                           "task_id": task_id},
                                     verify=False)
            response_code = response.status_code
        except Exception:
            pass



def generate_all_surrogate(save_dir: Union[Path, str] = "../../out/surrogate_models",
                           ins_dir: Union[Path, str] = "../../data/problem_instance"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    p_list = []
    for domain in train_problem_types.keys():
        for dim in problem_settings["training_dims"]:
            for index in range(problem_settings["training_ins_num"]):
                model_dir = Path(save_dir, f"{domain}_{dim}_{index}")
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                if os.path.exists(Path(model_dir, "best_model.pt")) and os.path.exists(
                        Path(model_dir, "HyperParam.yaml")):
                    continue
                p = Process(target=send_surrogate_fit_task,
                           args=(domain, model_dir, ins_dir, dim, index))
                p_list.append(p)
                p.start()
                time.sleep(0.1)

if __name__ == '__main__':
    work_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../data/problem_instance")
    surrogate_model_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../out/surrogate_models")
    generate_all_surrogate(save_dir=surrogate_model_dir, ins_dir=work_dir)