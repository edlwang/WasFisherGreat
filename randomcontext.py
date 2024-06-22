from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain

import pandas as pd
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import RBFInterpolator, LinearNDInterpolator
from scipy.stats import pearsonr
import random
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from tqdm import tqdm

llama_model_path = 'C:/Users/Edward/Documents/llama2/llama.cpp/models/7B/llama-2-7b.Q4_0.gguf'
embedding_model = LlamaCppEmbeddings(
    model_path=llama_model_path, 
    #n_ctx=2048,
    n_gpu_layers=40,
    n_threads=8,
    n_batch=1000,
    verbose=False)