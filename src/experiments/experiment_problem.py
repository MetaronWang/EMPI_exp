import os
import pickle
import random
import time
from multiprocessing import Manager, Process
from pathlib import Path
from threading import Thread

import requests
import urllib3

from src.distribution.util import random_str
from src.problem_domain import BaseProblem, ContaminationProblem, ComInfluenceMaxProblem, CompilerArgsSelectionProblem
from src.problem_domain import MatchMaxProblem, ZeroOneKnapsackProblem, MaxCutProblem
from src.types_ import *

urllib3.disable_warnings(category=urllib3.exceptions.InsecureRequestWarning)

train_problem_types: Dict[str, Type[BaseProblem]] = {
    "match_max_problem": MatchMaxProblem,
    "max_cut_problem": MaxCutProblem,
    "zero_one_knapsack_problem": ZeroOneKnapsackProblem,
}

valid_problem_types: Dict[str, Type[BaseProblem]] = {
    "match_max_problem": MatchMaxProblem,
    "max_cut_problem": MaxCutProblem,
    "zero_one_knapsack_problem": ZeroOneKnapsackProblem,
    "contamination_problem": ContaminationProblem,
    "com_influence_max_problem": ComInfluenceMaxProblem,
    "compiler_args_selection_problem": CompilerArgsSelectionProblem,
}

problem_cost: Dict[str, int] = {
    "match_max_problem": 1,
    "max_cut_problem": 1,
    "zero_one_knapsack_problem": 1,
    "contamination_problem": 1,
    "com_influence_max_problem": 6,
    "compiler_args_selection_problem": 1,
}

problem_settings: Dict[str, Union[int, List[int]]] = {
    "training_dims": [30, 35, 40],
    "valid_dims": [40, 60, 80, 100],
    "training_ins_num": 3,
    "training_gate_ins_num": 3,
    "valid_ins_num": 3,
}

master_host = "10.16.104.19:1088"


def send_problem_eval_task(domain, problem_path: Union[Path, str], solutions: NpArray, solution_index: int,
                           result_dict: dict):
    task_id = f"{str(int(time.time() * 1000))}_{random_str(32)}"
    task = {
        "task_id": task_id,
        "task_func": "eval_problem_solutions",
        "task_args": (problem_path, solutions),
        "task_cost": problem_cost[domain],
        "task_type": "cpu"
    }
    task_data = pickle.dumps(task)
    response_code = 500
    while response_code > 299 or response_code < 200:
        try:
            # start = time.time()
            response = requests.post(url="http://{}/create_task".format(master_host),
                                     data={"check_val": "81600a92e8416bba7d9fada48e9402a4",
                                           "task_type": "cpu", "task_cost": problem_cost[domain], "task_id": task_id},
                                     files={"task_data": task_data},
                                     verify=False)
            response_code = response.status_code
            # print(time.time() - start, task_id)
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
    result_dict[solution_index] = result
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


def load_problem_data(problem_dir):
    x = np.load(str(Path(problem_dir, "x.npy")))
    y = np.load(str(Path(problem_dir, "y.npy")))
    return x, y


def generate_instance(problem_class, dim, path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("mkdir -p ", str(path))
        if not os.path.exists(Path(path, "problem.pkl")):
            problem_instance = problem_class(dimension=dim)
            pickle.dump(problem_instance, open(Path(path, "problem.pkl"), "wb"))


def generate_problem_solution(domain: str, problem_path: Union[Path, str], dim: int, sample_num: int):
    if os.path.exists(Path(problem_path, "x.npy")) and os.path.exists(Path(problem_path, "y.npy")):
        return
    solutions = set()
    for _ in range(sample_num):
        temp = tuple(random.randint(0, 1) for _ in range(dim))
        while temp in solutions:
            temp = tuple(random.randint(0, 1) for _ in range(dim))
        solutions.add(temp)
    X: NpArray = np.array(list(solutions), dtype=np.float32)
    x_batches = np.array_split(X, 100)
    manager = Manager()
    y_batch_dict = manager.dict()
    p_list = []
    for batch_index, x_batch in enumerate(x_batches):
        p = Thread(target=send_problem_eval_task, args=(domain, problem_path, x_batch, batch_index, y_batch_dict))
        p_list.append(p)
        p.start()
        time.sleep(0.1)
    [p.join() for p in p_list]
    x = np.concatenate([x_batches[batch_index] for batch_index in range(len(x_batches))])
    y = np.concatenate([y_batch_dict[batch_index] for batch_index in range(len(x_batches))])
    np.save(str(Path(problem_path, "x.npy")), x)
    np.save(str(Path(problem_path, "y.npy")), y)
    print("Generate New DATA for", str(problem_path))


def generate_all_problem_instance(ins_dir: Union[Path, str] = "../../data/problem_instance"):
    if not os.path.exists(Path(ins_dir, "train")):
        os.makedirs(Path(ins_dir, "train"))
    if not os.path.exists(Path(ins_dir, "test")):
        os.makedirs(Path(ins_dir, "test"))
    if not os.path.exists(Path(ins_dir, "gate_train")):
        os.makedirs(Path(ins_dir, "gate_train"))
    for problem_domain in train_problem_types.keys():
        for index in range(problem_settings["training_ins_num"]):
            for dim in problem_settings["training_dims"]:
                path = Path(ins_dir, "train", f"{problem_domain}_{dim}_{index}")
                generate_instance(train_problem_types[problem_domain], dim, path)
            for dim in problem_settings["valid_dims"]:
                path = Path(ins_dir, "gate_train", f"{problem_domain}_{dim}_{index}")
                generate_instance(train_problem_types[problem_domain], dim, path)
    for problem_domain in valid_problem_types.keys():
        for index in range(problem_settings["valid_ins_num"]):
            for dim in problem_settings["valid_dims"]:
                path = Path(ins_dir, "test", f"{problem_domain}_{dim}_{index}")
                generate_instance(valid_problem_types[problem_domain], dim, path)


def generate_all_problem_solution_data(ins_dir: Union[Path, str] = "../../data/problem_instance"):
    p_list = []
    for problem_domain in train_problem_types.keys():
        for index in range(problem_settings["training_ins_num"]):
            for dim in problem_settings["training_dims"]:
                path = Path(ins_dir, "train", f"{problem_domain}_{dim}_{index}")
                p = Process(target=generate_problem_solution, args=(problem_domain, path, dim, 100000))
                p_list.append(p)
                p.start()
                # p.join()
            for dim in problem_settings["valid_dims"]:
                path = Path(ins_dir, "gate_train", f"{problem_domain}_{dim}_{index}")
                p = Process(target=generate_problem_solution, args=(problem_domain, path, dim, 100000))
                p_list.append(p)
                p.start()
    for problem_domain in valid_problem_types.keys():
        for index in range(problem_settings["valid_ins_num"]):
            for dim in problem_settings["valid_dims"]:
                path = Path(ins_dir, "test", f"{problem_domain}_{dim}_{index}")
                p = Process(target=generate_problem_solution, args=(problem_domain, path, dim, 100000))
                p_list.append(p)
                p.start()
    print("WAIT EVAL")
    [p.join() for p in p_list]


if __name__ == "__main__":
    work_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../data/problem_instance")
    # generate_all_problem_instance(ins_dir=work_dir)
    generate_all_problem_solution_data(work_dir)
