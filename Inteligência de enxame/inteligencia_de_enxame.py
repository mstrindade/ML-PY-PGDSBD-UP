#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def create_particles(n, m):
    bee = np.random.rand(n, m)
    return bee


# In[3]:


def eval_swarm(bee, a):
    group = abs(np.dot(bee, a))
    return group    


# In[4]:


def update_best_memory(bee, memory):
    mask = [eval_swarm(bee, a) < eval_swarm(memory, a)]
    new_memory = []
    for ver in range(len(bee)):
        if mask[0][ver] == True:
            new_memory.append(bee[ver])
        else:
            new_memory.append(memory[ver])
    memory = np.array(new_memory)
    return memory


# In[5]:


def find_new_goal(bee, a, goal):
    for ver in range(len(bee)):
        if eval_swarm(bee[ver], a) < eval_swarm(goal, a):
            goal = bee[ver]
            # print(bee[ver])
    return goal 


# In[6]:


m = 2; n = 200; c2 = 1.9; c1 = 0.9; d = 0
w = 0.9; wd = 0.9; a = [1, 1]

goal = np.random.rand(1, m)
bee = create_particles(n, m)

velocity = create_particles(n, m)
memory = create_particles(n, m)


# In[7]:


vet = {}
for fly in range(100):
    vet[fly] = []
    
    #r1 = np.random.rand(); r2 = np.random.rand()
    #velocity = w*velocity + r1*c1*(memory - bee) + r2*c2*(goal - bee)
    for j in range(len(bee)):
        r1 = np.random.rand(); r2 = np.random.rand()
        velocity[j] = w*velocity[j] + r1*c1*(memory[j] - bee[j]) + r2*c2*(goal - bee[j])
    
    bee = bee + velocity + d*create_particles(n, m)
    
    goal = find_new_goal(bee, a, goal)
    memory = update_best_memory(bee, memory)
    
    w = w*wd
    
    vet[fly].append(bee)
    
    # print(eval_swarm(goal, a))
    # print(eval_swarm(bee, a))
goal


# In[8]:


bee[88]


# In[9]:


eval_swarm(bee, a)


# In[10]:


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot;
from IPython.display import display, HTML; init_notebook_mode(connected=True);
import plotly.graph_objs as go
import numpy as np

# Generate curve data
t = np.linspace(-1, 1, 100)
x = t
y = -t
xm = -2
xM = 2
ym = -2
yM = 2
N = 25
s = np.linspace(-1, 1, N)
xx = s + s ** 2
yy = s - s ** 2


# Create figure
fig = go.Figure(
    data=[go.Scatter(x=x, y=y,
                     mode="lines",
                     line=dict(width=2, color="blue"),
                     name='goal'),
          go.Scatter(x=x, y=y,
                     mode="lines",
                     line=dict(width=2, color="blue"),
                     name='goal')],
    layout=go.Layout(
        xaxis=dict(range=[xm, xM], autorange=False, zeroline=False),
        yaxis=dict(range=[ym, yM], autorange=False, zeroline=False),
        title_text="Swarm Intelligence Algorithm", hovermode="closest",
        updatemenus=[dict(type="buttons",
                          buttons=[dict(label="Play",
                                        method="update",
                                        args=[None])])]),
    frames=[go.Frame(
            data=[go.Scatter(
            x=list(vet[k][0].T[:][0]),
            y=list(vet[k][0].T[:][1]),
            mode="markers",
            name = 'bee',
            marker=dict(color = "yellow", size=10, 
                         line = dict( color='black', width=2)))])
                         for k in range(N)]
)

iplot(fig)


# In[ ]:




