import numpy as np
from graspologic.embed import ClassicalMDS as MDS

from pandas import DataFrame
from numpy.linalg import svd
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator, LinearNDInterpolator
from matplotlib import cm
from scipy.stats import pearsonr
import random
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

embeddings = np.load("1-embeddings-random-no-query-nomic10.npy")
outputs = np.load("1-outputs-random-10.npy")
binary_outputs = np.load("1-binary_outputs-random-10.npy")
labels = np.load("1-labels-random-10.npy")
p = np.load("1-probabilities-random-10.npy")

# scree plot

def plot_scree(svs, title="", d=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(6, 3))
    sv_dat = DataFrame({"Singular Value": svs, "Dimension": range(1, len(svs) + 1)})
    #sns.scatterplot(data=sv_dat, x="Dimension", y="Singular Value", ax=ax)
    ax.scatter(data=sv_dat, x="Dimension", y="Singular Value")
    ax.set_xlim([0, len(s)])
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Singular Value")
    ax.set_title(title)
    if d is not None:
        ax.axvline(x=d, color='r')
        ax.text(x=d+1, y=svs[d], s="Dimension {:d}".format(d), color='r', size=15)

#print(type(lowdim_embeddings))
mds2 = MDS(n_components=100)
df = pd.DataFrame(mds2.fit_transform(embeddings))
#df = pd.DataFrame(embeddings)
#s = df.std()
U, s, Vt = svd(df)

#print(s)
plot_scree(s, title="Scree plot of mds")

mds = MDS(n_components=2) # 123456 with X0, 12345 for VL
lowdim_embeddings = mds.fit_transform(embeddings)

sizes = np.tile(np.repeat([30, 1], repeats=[1, 10]), 25)
#print(len(sizes))

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(lowdim_embeddings[:, 0][labels == 1], lowdim_embeddings[:, 1][labels == 1], s=sizes, c='purple', alpha=0.7, label='statistics')
ax.scatter(lowdim_embeddings[:, 0][labels == 0], lowdim_embeddings[:, 1][labels == 0], s=sizes, c='orange', alpha=0.7, label='eugenics')
#ax.scatter(X0[:,0], X0[:,1], s=60, c='red', marker="X")
ax.set_xlabel("MDS dim 1")
ax.set_ylabel("MDS dim 2")
ax.legend(frameon=True, fontsize=16)
plt.show()

fig.savefig('./2d-scatter-random-X.pdf', bbox_inches='tight')

# surface plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

xg, yg = np.meshgrid(lowdim_embeddings[:, 0], lowdim_embeddings[:, 1], indexing='ij')
interp = RBFInterpolator(lowdim_embeddings, p, kernel='linear', smoothing=5)


xx = np.linspace(-0.40, 0.40, 100)
yy = np.linspace(-0.40, 0.40, 100)
X, Y = np.meshgrid(xx, yy, indexing='ij')

grid = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)
vals = interp(grid).reshape(100,100)
Z = vals.reshape(100,100)

C = ax.plot_surface(X, Y, vals, color='m', cmap=cm.coolwarm, antialiased=False, linewidth=0, vmin=0, vmax=1, alpha=0.8)

ax.scatter(lowdim_embeddings[:, 0][labels == 1], lowdim_embeddings[:, 1][labels == 1], 0, s=sizes, c='purple', label='statistics')
ax.scatter(lowdim_embeddings[:, 0][labels == 0], lowdim_embeddings[:, 1][labels == 0], 0, s=sizes, c='orange', label='eugenics')
#ax.scatter(X0[:,0], X0[:,1], s=60, c='red', marker="X")

ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')


fig.colorbar(C, ax=ax, fraction=0.02, pad=0, label=r'$\hat P \;[Z=1 \mid prompt, a_i]$')
ax.legend(frameon=True, fontsize=12, bbox_to_anchor=(0.3, 0.5))
ax.set_xlabel("MDS dim 1")
ax.set_ylabel("MDS dim 2")

#ax.view_init(elev=20, azim=25, roll=0)
#ax.view_init(elev=20, azim=185, roll=0)
ax.view_init(elev=20, azim=15, roll=0)
#ax.view_init(elev=20, azim=-105, roll=0)

plt.show()

fig.savefig('./3d-surface-random.pdf', bbox_inches='tight')

import plotly.graph_objects as go

# Blackbody,Bluered,Blues,C ividis,Earth,Electric,Greens,Greys,Hot,Jet,Picnic,Portland,Rainbow,RdBu,Reds,Viridis,YlGnBu,YlOrRd.
#fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Bluered')])
#fig.update_traces(contours_z=dict(show=True, usecolormap=True,
#                                 highlightcolor="limegreen", project_z=True), colorscale="portland")


surface = go.Surface(x=X, 
                     y=Y, 
                     z=Z, 
                     colorscale='RdBu_r', 
                     opacity=0.7,
                     colorbar=dict(
                         title="r'$\hat P \;[Z=1 \mid prompt, a_i]$'",
                         titleside="bottom",
                         len=1, 
                         orientation='h',
                         xanchor="right", 
                         x=1, 
                         yanchor='bottom', 
                         y=-0.1, 
                         thickness=15),
                     )
scatter1 = go.Scatter3d(x=lowdim_embeddings[:,0][labels == 1], 
                        y=lowdim_embeddings[:,1][labels == 1], 
                        z=np.zeros(275,),
                        hoverinfo='skip', 
                        mode="markers", 
                        marker=dict(size=5, color='orange', opacity=0.5), 
                        name = 'eugenics')
scatter2 = go.Scatter3d(x=lowdim_embeddings[:,0][labels == 0], 
                        y=lowdim_embeddings[:,1][labels == 0], 
                        z=np.zeros(275,),
                        hoverinfo='skip', 
                        mode="markers", 
                        marker=dict(size=5, color='purple', opacity=0.5), 
                        name = 'eugenics')
fig = go.Figure(data=[surface, scatter1, scatter2])

fig.update_layout(title='', autosize=False,
                  width=800, height=800,
                  legend=dict(
                      font=dict(size= 20),
                      yanchor="top",
                      y=0.99,
                      xanchor="left",
                      x=0.01),
                 margin=dict(l=65, r=50, b=65, t=90))
#fig.data[0].colorbar.title = r'$\hat P \;[Z=1 \mid prompt, a_i]$'
#fig.update_coloraxes(colorbar_title_side='right')

fig.show()

clf = LinearDiscriminantAnalysis()
pos = clf.fit_transform(lowdim_embeddings, labels).squeeze()

x0_pos = pos[labels==0]
y0_p = p[labels==0]
x1_pos = pos[labels==1]
y1_p = p[labels==1]
# jittered_x0 = x0_pos + 0.05 * np.random.rand(len(x0_pos)) -0.05
# jittered_y0 = y0_p + 0.05 * np.random.rand(len(y0_p)) -0.05

# compute correlation between FLD projection and P(Z=1 | eugenics)
r, p_val = pearsonr(x0_pos, y0_p)
print("Spearman correlation coefficient : ", r)
print("p-val : ", p_val)

# fit using only big yellow dots
big_x = x0_pos[sizes==30]
big_y = y0_p[sizes==30]
model = np.poly1d(np.polyfit(big_x, big_y, 2))
new_x = np.linspace(min(x0_pos), max(x0_pos), 50)
new_y = model(new_x)
# model = np.poly1d(np.polyfit(big_x, big_y, 2))
# new_x2 = np.linspace(min(x0_pos), max(x0_pos), 50)
# new_y2 = model(new_x)

fig, ax = plt.subplots(figsize=(6, 3))
ax.scatter(x1_pos, y1_p, s=sizes, color='purple', alpha=0.7, label='statistics')
ax.scatter(x0_pos, y0_p, s=sizes, color='orange', alpha=0.7, label='eugenics')
ax.plot(new_x, new_y)
# ax.plot(new_x2, new_y2)
ax.set_xlabel(r"FLD Projection of $MDS(embed(prompt, a_i))$")
ax.set_ylabel(r"$\hat P \; [Z = 1 \mid prompt, a_i] $")
ax.legend(loc=5)
plt.show()
