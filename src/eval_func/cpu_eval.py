import pickle
from pathlib import Path

from src.problem_domain import BaseProblem
from src.types_ import *


def eval_problem_solutions(problem_path: str = '../../data/problem_instance', solutions: NpArray = None):
    problem_instance: BaseProblem = pickle.load(open(Path(problem_path, "problem.pkl"), 'rb'))
    result = [problem_instance.evaluate(solution) for solution in solutions]
    return np.array(result)


def eval_problem_EA():
    pass
