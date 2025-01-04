import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
full_prompt_llama = np.load('1-embeddings-random-full-prompt-llama-100.npy')
full_prompt_nomic = np.load('1-embeddings-random-full-prompt-nomic-100.npy')

llama_pca = PCA().fit(full_prompt_llama)
plt.plot(range(1,551), llama_pca.explained_variance_)
plt.title('Scree Plot for PCA on Llama Full Prompt Embeddings')
plt.xlabel('Dimension')
plt.ylabel('Explained Variance')
plt.savefig('llama-scree')


nomic_pca = PCA().fit(full_prompt_nomic)
plt.title('Scree Plot for PCA on Nomic Full Prompt Embeddings')
plt.plot(range(1,551), llama_pca.explained_variance_)
plt.xlabel('Dimension')
plt.ylabel('Explained Variance')
plt.savefig('nomic-scree')
