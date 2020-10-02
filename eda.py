import pandas as pd

import matplotlib.pyplot as plt

import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as offline
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

train_x = pd.read_csv('./datasets/train.csv')
test_x  = pd.read_csv('./datasets/test.csv')
print(train_x.head())

traces = []
traces.append(go.Scatter(x=train_x['education'], y=train_x['y'], mode='markers'))
# traces.append(go.Scatter(x=nn_lower_data[:,3],  y=nn_lower_data[:,2], text=text,  mode='markers', name='下層近傍', marker_color='rgba(0,0,255,.8)', marker_size=5))

# レイアウトの指定
layout = go.Layout(width=1000, height=750)

fig = dict(data=traces, layout=layout)
plotly.offline.plot(fig, filename='./test.html', auto_open=True)