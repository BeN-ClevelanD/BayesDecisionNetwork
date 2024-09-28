import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a directory for saving graphs if it doesn't exist
os.makedirs('graphs', exist_ok=True)

# Load the processed DataFrames

port_final_df = pd.read_csv('processed_port_data.csv')

# Function to plot and save histograms and bar charts for each column
def visualize_data(df, title):
    print(f"\nSummary statistics for {title}:")
    print(df.describe(include='all'))
    
    for column in df.columns:
        plt.figure(figsize=(8, 5))
        
        if df[column].dtype == 'object' or df[column].nunique() < 10:  # Categorical data
            sns.countplot(x=column, data=df, palette='viridis')
            plt.title(f'{title} - {column} (Categorical)')
        else:  # Numerical data
            sns.histplot(df[column], kde=True, bins=10, palette='viridis')
            plt.title(f'{title} - {column} (Numerical)')
        
        plt.ylabel('Count')
        plt.xlabel(column)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save each figure
        filename = f"graphs/{title}_{column}.png"
        plt.savefig(filename)
        plt.close()  # Close the figure to free up memory
        print(f"Saved: {filename}")

# Visualize the math_final_df dataset


# Visualize the port_final_df dataset
visualize_data(port_final_df, 'Portuguese_Dataset')
