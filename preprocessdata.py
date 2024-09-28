import pandas as pd

# Load the datasets

port_data = pd.read_csv('student-por.csv', delimiter=';')

# Function to categorize Medu and Fedu
def categorize_education(x):
    return 1 if x == 4 else 0  # 1 for higher education, 0 otherwise

# Apply the education categories to both datasets

port_data['Medu'] = port_data['Medu'].apply(categorize_education)
port_data['Fedu'] = port_data['Fedu'].apply(categorize_education)



# Function to categorize Study Time
def categorize_studytime(x):
    if x ==1:

        return 'Very Low'
    elif x == 2:
        return 'Low'
    elif x == 3:
        return 'Medium'
    else:
        return 'High'

# Apply Study Time categories

port_data['studytime'] = port_data['studytime'].apply(categorize_studytime)

# Function to categorize Absences
def categorize_absences(x):
    if x < 20:
        return 'Low'
    elif 15 <= x <= 40:
        return 'Medium'
    else:
        return 'High'

# Apply Absences categories

port_data['absences'] = port_data['absences'].apply(categorize_absences)

# Function to categorize Failures
def categorize_failures(x):
    if x == 0:
        return '0'
    elif x == 1:
        return '1'
    else:
        return 'More than 1'

# Apply Failures categories

port_data['failures'] = port_data['failures'].apply(categorize_failures)


def passorfail(x):
    if x >= 10:
        return 'Pass'
    else:
        return 'Fail'
    
# Apply the pass/fail categories to both datasets
port_data['Final_Grade'] = port_data['G3'].apply(passorfail)



port_selected_columns = ['Medu', 'Fedu', 'higher', 'studytime', 'absences', 'failures', 'Final_Grade']

# Creating the final DataFrames

port_final_df = port_data[port_selected_columns].copy()
# Rename the G3 column to Final_Grade or any other name you prefer



# Print the first few rows of each DataFrame to check

print(port_final_df.head())

# Save the processed DataFrames for future use

port_final_df.to_csv('processed_port_data.csv', index=False)
