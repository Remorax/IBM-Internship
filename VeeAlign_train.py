import os, itertools, pickle, sys, requests, re
import tensorflow_hub as hub
import numpy as np
from xml.dom import minidom
from collections import OrderedDict
from scipy import spatial
from math import ceil
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from ontology import *

