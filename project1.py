import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading  data from a csv file and convert that into a dataframe, which will allow for all further
# analysis and data manipulation.
df = pd.read_csv("Project_1_Data.csv")

print(df.info())
print(df.head())

# Load the CSV file
file_path = 'Project_1_Data.csv'
data = pd.read_csv(file_path)

# Extracting the relevant columns for plotting
x = data['X']
y = data['Y']
z = data['Z']
steps = data['Step']

# Creating a 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection='3d')  # No need to import Axes3D separately

# Plotting the coordinates
scatter = ax.scatter(x, y, z, c=steps, cmap='viridis', marker='o')

# Adding color bar to show steps mapping
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('Step')

# Labels and title
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('3D Plot of Coordinates (X, Y, Z) versus Steps')

# Show the plot for 2.2
plt.show()



    