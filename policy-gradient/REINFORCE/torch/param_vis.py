import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
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
fig, ax = plt.subplots()

# Initialize empty line objects for each parameter
line_w0, = ax.plot([], [], label='w0')
line_w1, = ax.plot([], [], label='w1')
line_b0, = ax.plot([], [], label='b0')
line_b1, = ax.plot([], [], label='b1')

# Set up the plot limits and labels
# ax.set_xlim(0, 100)  # Adjust x-limits based on your data
ax.set_ylim(-5, 5)   # Adjust y-limits based on your data
ax.legend()

# Initialize the plot
def init():
    print('init')
    line_w0.set_data([], [])
    line_w1.set_data([], [])
    line_b0.set_data([], [])
    line_b1.set_data([], [])
    return line_w0, line_w1, line_b0, line_b1

# Update function for the animation
def update(i, line_w0, line_w1, line_b0, line_b1):
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
    w0 = tbl['pr']
    print(w0)
    w1 = tbl['w1']
    b0 = tbl['b0']
    b1 = tbl['b1']

    # Update the lines with new data
    line_w0.set_data(x, w0)
    line_w1.set_data(x, w1)
    line_b0.set_data(x, b0)
    line_b1.set_data(x, b1)

    return line_w0, line_w1, line_b0, line_b1

# Create the animation object
ani = animation.FuncAnimation(fig, update, len(csv_files), fargs=(line_w0, line_w1, line_b0, line_b1), init_func=init, blit=True, repeat=True)

# Display the plot
plt.show()

plt.plot(T['w0'], label='w0')
plt.plot(T['w1'], label='w1')
plt.plot(T['b0'], label='b0')
plt.plot(T['b1'], label='b1')
plt.legend()
plt.show()
