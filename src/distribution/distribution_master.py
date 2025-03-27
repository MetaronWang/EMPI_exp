import logging
import os
import time
from multiprocessing import Manager, Process, Lock
from pathlib import Path

import logzero
import redis
import urllib3
from flask import Flask, request, make_response


urllib3.disable_warnings(category=urllib3.exceptions.InsecureRequestWarning)


class DistributedMaster(Process):
    def __init__(self, port: int = 1088):
        super().__init__()
        if not os.path.exists(Path(os.path.dirname(os.path.abspath(__file__)), "../../logs/Distribution_logs")):
            os.makedirs(Path(os.path.dirname(os.path.abspath(__file__)), "../../logs/Distribution_logs"))
        self.logger = logzero.setup_logger(
            logfile=str(Path(os.path.dirname(os.path.abspath(__file__)), "../../logs/Distribution_logs/master.log")),
            name="DistributedMaster Log", level=logzero.ERROR, fileLoglevel=logzero.INFO, backupCount=100,
            maxBytes=int(1e7))
        self.port = port
        self.manager = Manager()
        self.redis_pool = redis.ConnectionPool(host='localhost', port=6379, db=0, max_connections=256)
        self.redis = redis.Redis(connection_pool=self.redis_pool)
        self.result_hash_key = "distributed_result"
        self.redis.delete(self.result_hash_key)
        self.redis.delete("cpu")
        self.redis.delete("gpu")
        self.task_lock = Lock()
        self.app = Flask(__name__)
        self.app.add_url_rule("/get_task", view_func=self.get_task, methods=['POST'])
        self.app.add_url_rule("/get_result", view_func=self.get_result, methods=['POST'])
        self.app.add_url_rule("/clear_record", view_func=self.clear_record, methods=['POST'])
        self.app.add_url_rule("/report_result", view_func=self.report_result, methods=['POST'])
        self.app.add_url_rule("/create_task", view_func=self.create_task, methods=['POST'])
        werkzeug_logger = logging.getLogger('werkzeug')
        werkzeug_logger.disabled = True
        self.set_middleware()
        self.task_num_process = Process(target=self.show_task_rest, args=())
        self.task_num_process.start()

    def run(self):
        self.app.run(debug=False, port=self.port, host='0.0.0.0', threaded=False, processes=64)

    def set_middleware(self):
        @self.app.before_request
        def check_verification():
            try:
                data = request.form.to_dict()
                check_val = data["check_val"]
                if check_val != "81600a92e8416bba7d9fada48e9402a4":
                    return {"success": False, "msg": "ERROR"}
            except Exception as e:
                self.logger.error(str(e))
                return {"success": False, "msg": str(e)}

    def get_task(self):
        try:
            data = request.form.to_dict()
            task_type = data["task_type"]
            if task_type not in ["cpu", "gpu"]:
                return make_response({"success": False, "msg": "TASK Type ERROR"}, 500)
            max_cost = int(data["max_cost"])
            rem_num = 0
            while rem_num <= 0:
                results = self.redis.zrangebyscore(
                    task_type, 0, max_cost,
                    start=0, num=1, withscores=False
                )
                if results:
                    rem_num = self.redis.zrem(task_type, results[0])
                else:
                    return make_response({"success": False, "msg": "There is no Task can RUN"}, 500)
            pickled_data = results[0]
            response = make_response(pickled_data, 200)
            response.headers['Content-Type'] = 'application/octet-stream'
            return response
        except Exception as e:
            self.logger.error("There is ERROR: " + str(e))
            return make_response({"success": False, "msg": str(e)}, 500)

    def clear_record(self):
        try:
            data = request.form.to_dict()
            task_id = data["task_id"]
            self.redis.hdel(self.result_hash_key, task_id)
        except Exception as e:
            self.logger.error(str(e))
            return make_response({"success": False, "msg": str(e)}, 500)
        return make_response({"success": True}, 200)

    def get_result(self):
        try:
            data = request.form.to_dict()
            task_id = data["task_id"]
            result = self.redis.hget(self.result_hash_key, task_id)
            # if result is None:
            #     return make_response({"success": False, "msg": "Task does not EXIST"}, 500)
            if result is None:
                return make_response({"success": False, "msg": "Task haven't FINISHED"}, 500)
            pickled_data = result
            response = make_response(pickled_data, 200)
            response.headers['Content-Type'] = 'application/octet-stream'
            return response
        except Exception as e:
            self.logger.error(str(e))
            return make_response({"success": False, "msg": str(e)}, 500)

    def report_result(self):
        try:
            data = request.form.to_dict()
            task_id = data["task_id"]
            task_result = request.files['result'].stream.read()
            self.redis.hset(self.result_hash_key, task_id, task_result)
            return make_response({"success": True, "msg": "SUCCESS"}, 200)
        except Exception as e:
            self.logger.error(str(e))
            return make_response({"success": False, "msg": str(e)}, 500)

    def create_task(self):
        try:
            task_data = request.files['task_data'].stream.read()
            data = request.form.to_dict()
            try:
                task_type = data["task_type"]
                if task_type not in ["cpu", "gpu"]:
                    return make_response({"success": False, "msg": "TASK Type ERROR"}, 500)
                task_cost = data["task_cost"]
                self.redis.zadd(task_type, {task_data: task_cost})
                return make_response({"success": True, "msg": "SUCCESS"}, 200)
            except KeyError:
                return make_response({"success": False, "msg": "TASK Param Missing"}, 500)
        except Exception as e:
            self.logger.error(str(e))
            return make_response({"success": False, "msg": str(e)}, 500)

    def show_task_rest(self):
        while True:
            time.sleep(5)
            task_rest = self.redis.zcard("cpu") + self.redis.zcard("gpu")
            self.logger.info(f"REST TASK NUM is {task_rest}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=1088, help='The port of the Master')
    args = parser.parse_args()
    master = DistributedMaster(port=args.port)
    master.start()
    master.join()
