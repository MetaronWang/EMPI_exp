import os
import pickle
import time
from multiprocessing import Manager, Process, Queue, Lock
from pathlib import Path

import logzero
import requests
import urllib3

from src.types_ import *

urllib3.disable_warnings(category=urllib3.exceptions.InsecureRequestWarning)
from src.eval_func import eval_problem_solutions

eval_funcs = {
    "eval_problem_solutions": eval_problem_solutions
}


def eval_worker(up_queue, result_queue):
    while True:
        task_info = up_queue.get()
        task_id = task_info["task_id"]
        task_func = task_info["task_func"]
        task_args = task_info["task_args"]
        if task_func in eval_funcs:
            result = eval_funcs[task_func](*task_args)
            result_queue.put({
                "task_id": task_id,
                "result": result
            })


class DistributedEvaluator:
    def __init__(self, master_host="10.16.104.19:1088", task_capacity=512, task_type="cpu"):
        if not os.path.exists(Path(os.path.dirname(os.path.abspath(__file__)), "../../logs/Distribution_logs")):
            os.makedirs(Path(os.path.dirname(os.path.abspath(__file__)), "../../logs/Distribution_logs"))
        self.logger = logzero.setup_logger(
            logfile=str(Path(os.path.dirname(os.path.abspath(__file__)), "../../logs/Distribution_logs/evaluator.log")),
            name="DistributedEvaluator Log", level=logzero.ERROR, fileLoglevel=logzero.INFO, backupCount=100,
            maxBytes=int(1e7))
        self.master_host = master_host
        self.task_capacity = task_capacity
        self.task_lock = Lock()
        self.task_type = task_type
        self.manager = Manager()
        self.up_queue = Queue()
        self.result_queue = Queue()
        self.eval_processes = []
        self.task_ongoing: Dict = self.manager.dict()
        for p_num in range(self.task_capacity):
            p = Process(target=eval_worker, args=(self.up_queue, self.result_queue))
            p.start()
            self.eval_processes.append(p)
        self.task_process = Process(target=self.eval_new_task, args=())
        self.result_process = Process(target=self.report_task_result, args=())
        self.task_process.start()
        self.result_process.start()
        print("EVAL START")

    def eval_new_task(self):
        fail_num = 0
        while True:
            try:
                response_code = 500
                while response_code > 299 or response_code < 200:
                    try:
                        fail_num += 1
                        if fail_num >= 10:
                            time.sleep(2)
                        elif fail_num > 100:
                            time.sleep(10)
                            fail_num = 10
                        with self.task_lock:
                            current_tasks = dict(self.task_ongoing)
                            rest_capacity = self.task_capacity - sum([value for value in current_tasks.values()])
                        if rest_capacity <= 0:
                            time.sleep(1)
                            continue
                        response = requests.post(url="http://{}/get_task".format(self.master_host),
                                                 data={
                                                     "check_val": "81600a92e8416bba7d9fada48e9402a4",
                                                     "task_type": self.task_type,
                                                     "max_cost": rest_capacity,
                                                 }, verify=False)
                        response_code = response.status_code
                    except requests.exceptions.ConnectionError as e:
                        pass
                task_info: Dict = pickle.loads(response.content)
                self.logger.info(f"Get New Task {task_info['task_id']} with cost {task_info['task_cost']}")
                self.up_queue.put(task_info)
                with self.task_lock:
                    self.task_ongoing[task_info["task_id"]] = task_info["task_cost"]
                fail_num = 0
            except Exception as e:
                self.logger.error(e)

    def report_task_result(self):
        while True:
            try:
                result_data = self.result_queue.get()
                response_code = 500
                while response_code > 299 or response_code < 200:
                    try:
                        response = requests.post(url="http://{}/report_result".format(self.master_host),
                                                 data={"check_val": "81600a92e8416bba7d9fada48e9402a4",
                                                       "task_id": result_data["task_id"]},
                                                 files={"result": pickle.dumps(result_data["result"])}, verify=False)
                        response_code = response.status_code
                    except requests.exceptions.ConnectionError as e:
                        pass
                self.logger.info(f"Report Task Result of {result_data['task_id']}")
                with self.task_lock:
                    self.task_ongoing.pop(result_data["task_id"])
            except Exception as e:
                self.logger.error(str(e))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--master_host', type=str, default="10.16.104.19:1088", help='The Host of the Master')
    parser.add_argument('--task_capacity', type=int, default=128, help='The Host of the Master')
    parser.add_argument('--task_type', type=str, default="cpu", help='The Host of the Master')
    args = parser.parse_args()
    evaluator = DistributedEvaluator(master_host=args.master_host, task_capacity=args.task_capacity,
                                     task_type=args.task_type)
    while True:
        time.sleep(100)
