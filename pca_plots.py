import numpy as np
import matplotlib.pyplot as plt

# Llama + MSE 

full_embed = np.load('llama-full-mse-pca.npy')
context_only_embed = np.load('llama-only-mse-pca.npy')


plt.figure()
plt.plot(range(2,550), full_embed[1:])
plt.title('MSE, Full Prompt, Llama')
plt.xlabel('PCA Dimension')
plt.ylabel('MSE')
plt.savefig('pca-llama-full-prompt-mse')


plt.figure()
plt.plot(range(2,550), context_only_embed[1:])
plt.title('MSE, Context, Llama')
plt.xlabel('PCA Dimension')
plt.ylabel('MSE')
plt.savefig('pca-llama-context-mse')

# Llama + Rel MSE

full_embed = np.load('llama-full-rel-mse-pca.npy')
context_only_embed = np.load('llama-only-rel-mse-pca.npy')

plt.figure()
plt.plot(range(2,550), full_embed[1:])
plt.title('Relative MSE, Full Prompt, Llama')
plt.xlabel('PCA Dimension')
plt.ylabel('Relative MSE')
plt.savefig('pca-llama-full-prompt-rel-mse')


plt.figure()
plt.plot(range(2,550), context_only_embed[1:])
plt.title('Relative MSE, Context, Llama')
plt.xlabel('PCA Dimension')
plt.ylabel('Relative MSE')
plt.savefig('pca-llama-context-rel-mse')

# Nomic + MSE

full_embed = np.load('nomic-full-mse-pca.npy')
context_only_embed = np.load('nomic-only-mse-pca.npy')

plt.figure()
plt.plot(range(2,550), full_embed[1:])
plt.title('MSE, Full Prompt, Nomic')
plt.xlabel('PCA Dimension')
plt.ylabel('MSE')
plt.savefig('pca-nomic-full-prompt-mse')


plt.figure()
plt.plot(range(2,550), context_only_embed[1:])
plt.title('MSE, Context, Nomic')
plt.xlabel('PCA Dimension')
plt.ylabel('MSE')
plt.savefig('pca-nomic-context-mse')

# Nomic + Rel MSE

full_embed = np.load('nomic-full-rel-mse-pca.npy')
context_only_embed = np.load('nomic-only-rel-mse-pca.npy')

plt.figure()
plt.plot(range(2,550), full_embed[1:])
plt.title('Relative MSE, Full Prompt, Nomic')
plt.xlabel('PCA Dimension')
plt.ylabel('Relative MSE')
plt.savefig('pca-nomic-full-prompt-rel-mse')


plt.figure()
plt.plot(range(2,550), context_only_embed[1:])
plt.title('Relative MSE, Context, Nomic')
plt.xlabel('PCA Dimension')
plt.ylabel('Relative MSE')
plt.savefig('pca-nomic-context-rel-mse')
