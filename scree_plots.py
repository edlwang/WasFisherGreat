import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
full_prompt_llama = np.load('1-embeddings-random-full-prompt-llama-100.npy')
full_prompt_nomic = np.load('1-embeddings-random-full-prompt-nomic-100.npy')

llama_pca = PCA().fit(full_prompt_llama)
print(llama_pca.explained_variance_)
print(len(llama_pca.explained_variance_))
