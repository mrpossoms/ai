import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import glob


# Load all CSV files from the directory
csv_files = [f"/tmp/plot{i}.csv" for i in range(100)]  # Adjust the path to your files


T = {}
for file in csv_files:
    tbl = pd.read_csv(file)
    for col_name in tbl.columns:
        if 'weight' in col_name or 'bias' in col_name:
            if col_name not in T:
                T[col_name] = []
            T[col_name].append(tbl[col_name].iloc[0])

print(csv_files)

def M(params, name):
    return np.array([ params[col_name][0] for col_name in params.columns if name in col_name ])

def recompute(x, a, params):
    # w = np.array([params['l0.weight1'].iloc[0],params['l0.weight0'].iloc[0]])
    # make a weight matrix out of all l0.weight# parameters
    
    W = M(params, 'weight').reshape(4, 4)
    b = M(params, 'bias').reshape(4,1)

    y = (W @ x.T + b).T

    sig = np.log(np.exp(y[:,1]) + 1)
    var = sig ** 2
    #import pdb; pdb.set_trace()
    g = np.exp(-(a[:,0] - y[:,0])**2 / (2*var))

    return g

# Setup the figure and axis for the animation
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)

# Initialize empty line objects for each parameter
lines = {}
param_lines = {}
# line_pr, = ax.plot([], [], label='pr')
# line_gw0, = ax.plot([], [], label='gw0')
# line_gw1, = ax.plot([], [], label='gw1')
# line_gb0, = ax.plot([], [], label='gb0')
# line_gb1, = ax.plot([], [], label='gb1')

# Set up the plot limits and labels
ax.set_xlim(-6, 6)  # Adjust x-limits based on your data
ax.set_ylim(-2, 4)   # Adjust y-limits based on your data
ax.axvline(x=1, color='gray', linestyle='--', linewidth=1)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax.axhline(y=1, color='gray', linestyle='--', linewidth=1)

# Update function for the animation
def update(i, lines):
    # Load the i-th CSV file
    tbl = pd.read_csv(csv_files[i])
    if i == 0:
        print('update')
    # Extract the x-values and parameters
    # TODO: all the parameters for each CSV are the same. Sample a single parameter
    # value from each CSV and use that as the sequence we are animating/plotting. Another interesting
    # thing possibly worth visualizing are the gradients and probability that are otherwise visualized
    # in the static plot
    x = tbl['x']
    for col_name in tbl.columns:
        if col_name == 'x' or 'weight' in col_name or 'bias' in col_name:
            continue
        if col_name not in lines:
            lines[col_name] = ax.plot(x, tbl[col_name], label=col_name)
        else:
            lines[col_name][0].set_data(x, tbl[col_name])

    g_check = recompute(np.ones((x.size,4)), np.column_stack((x, x)), tbl)

    if 'check0' not in lines:
        lines['check0'] = ax.plot(x, g_check, linestyle=':', label='check0')
        # lines['check1'] = ax.plot(x, g_check[:,1], label='check1')
    else:
        lines['check0'][0].set_data(x, g_check)
        # lines['check1'][0].set_data(x, g_check[:,1])


    return lines #_pr, line_gw0, line_gw1, line_gb0, line_gb1

# Create the animation object
# ani = animation.FuncAnimation(fig, update, len(csv_files), fargs=(line_pr, line_gw1, line_gb0, line_gb1), init_func=init, blit=True, repeat=True)

ax_slider = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')  # x, y, width, height
slider = Slider(ax_slider, 'Frame', 0, len(csv_files)-1, valinit=0, valfmt='%0.0f')

# Update the animation when the slider is changed
def on_slider_change(val):
    frame = int(slider.val)
    update(frame, lines)

slider.on_changed(on_slider_change)
on_slider_change(0)
ax.legend()
ax = fig.add_subplot(1, 2, 2)
# '''
for col_name in T:
    T[col_name] = np.array(T[col_name])
    ax.plot(T[col_name], label=col_name)
ax.legend()
# '''

# Display the plot
plt.show()
