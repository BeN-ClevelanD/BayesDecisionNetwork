import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the processed DataFrame
port_final_df = pd.read_csv('processed_port_data.csv')

# Crosstab for Free Time vs Study Time
crosstab = pd.crosstab(port_final_df['freetime'], port_final_df['studytime'])
print(crosstab)

# Visualize the crosstab with a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(crosstab, annot=True, cmap='Blues', fmt='g')
plt.title('Heatmap of Free Time vs Study Time')
plt.ylabel('Free Time')
plt.xlabel('Study Time')
plt.show()
