from typing import List, Callable, Union, Any, TypeVar, Tuple, Dict, Type, Set, Callable
from types import MethodType
import numpy as np
import torch
import networkx as nx
import time
import pickle
import os
from pathlib import Path
from tqdm import tqdm
from pytorch_lightning import seed_everything

Tensor = torch.Tensor
NpArray = np.ndarray
Graph = nx.Graph

