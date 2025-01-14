import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Data
data = {'Epreuve A': [3, 4, 6, 7, 9, 10, 9, 11, 12, 13, 15, 4], 
        'Epreuve B': [8, 9, 10, 13, 15, 14, 13, 16, 13, 19, 6, 19]}
df = pd.DataFrame(data)

# 1. Regression with all points
slope, intercept, r_value, p_value, std_err = linregress(df['Epreuve A'], df['Epreuve B'])

r_squared = r_value**2

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Epreuve A'], df['Epreuve B'])
plt.plot(df['Epreuve A'], slope * df['Epreuve A'] + intercept, color='red')
plt.xlabel('Epreuve A')
plt.ylabel('Epreuve B')
plt.title('Regression with all points')
plt.savefig('regression_all_points.png')

# Print results
print("--- Regression with all points ---")
print(f"Slope: {slope:.2f}")
print(f"Intercept: {intercept:.2f}")
print(f"R-squared: {r_squared:.2f}")