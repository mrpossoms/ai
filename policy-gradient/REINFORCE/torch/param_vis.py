import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import glob


# Load all CSV files from the directory
csv_files = [f"/tmp/plot{i}.csv" for i in range(100)]  # Adjust the path to your files

T = {
    'w0': [],
    'w1': [],
    'b0': [],
    'b1': [],
}
for file in csv_files:
    tbl = pd.read_csv(file)
    T['w0'].append(tbl.w0[0])
    T['w1'].append(tbl.w1[0])
    T['b0'].append(tbl.b0[0])
    T['b1'].append(tbl.b1[0])

print(csv_files)

# Setup the figure and axis for the animation
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)

# Initialize empty line objects for each parameter
line_pr, = ax.plot([], [], label='pr')
line_gw0, = ax.plot([], [], label='gw0')
line_gw1, = ax.plot([], [], label='gw1')
line_gb0, = ax.plot([], [], label='gb0')
line_gb1, = ax.plot([], [], label='gb1')

# Set up the plot limits and labels
ax.set_xlim(-6, 6)  # Adjust x-limits based on your data
ax.set_ylim(-2, 4)   # Adjust y-limits based on your data
ax.axvline(x=1, color='gray', linestyle='--', linewidth=1)
ax.legend()

# Initialize the plot
def init():
    print('init')
    line_pr.set_data([], [])
    line_gw0.set_data([], [])
    line_gw1.set_data([], [])
    line_gb0.set_data([], [])
    line_gb1.set_data([], [])
    return line_pr, line_gw1, line_gb0, line_gb1

# Update function for the animation
def update(i, line_pr, line_w1, line_b0, line_b1):
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

    # Update the lines with new data
    line_pr.set_data(x, tbl['pr'])
    line_gw0.set_data(x, tbl['gw0'])
    line_gw1.set_data(x, tbl['gw1'])
    line_gb0.set_data(x, tbl['gb0'])
    line_gb1.set_data(x, tbl['gb1'])

    return line_pr, line_gw0, line_gw1, line_gb0, line_gb1

# Create the animation object
# ani = animation.FuncAnimation(fig, update, len(csv_files), fargs=(line_pr, line_gw1, line_gb0, line_gb1), init_func=init, blit=True, repeat=True)

ax_slider = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')  # x, y, width, height
slider = Slider(ax_slider, 'Frame', 0, len(csv_files)-1, valinit=0, valfmt='%0.0f')

# Update the animation when the slider is changed
def on_slider_change(val):
    frame = int(slider.val)
    update(frame, line_pr, line_gw1, line_gb0, line_gb1)

slider.on_changed(on_slider_change)

ax = fig.add_subplot(1, 2, 2)
ax.plot(T['w0'], label='w0')
ax.plot(T['w1'], label='w1')
ax.plot(T['b0'], label='b0')
ax.plot(T['b1'], label='b1')
ax.legend()

# Display the plot
plt.show()