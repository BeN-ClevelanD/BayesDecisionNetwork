import pandas as pd

# Load the dataset
port_final_df = pd.read_csv('processed_port_data.csv')

# Function to calculate CPT for Medu (Root Node)
def calculate_cpt_medu():
    total = len(port_final_df)
    counts = port_final_df['Medu'].value_counts()
    cpt = {value: counts[value] / total for value in counts.index}
    print("CPT for Medu:", cpt)
    return cpt

# Function to calculate CPT for Fedu (Root Node)
def calculate_cpt_fedu():
    total = len(port_final_df)
    counts = port_final_df['Fedu'].value_counts()
    cpt = {value: counts[value] / total for value in counts.index}
    print("CPT for Fedu:", cpt)
    return cpt

# Function to calculate CPT for Absences (Root Node)
def calculate_cpt_absences():
    total = len(port_final_df)
    counts = port_final_df['absences'].value_counts()
    cpt = {value: counts[value] / total for value in counts.index}
    print("CPT for Absences:", cpt)
    return cpt

# Function to calculate CPT for Higher (Dependent on Medu and Fedu)
def calculate_cpt_higher():
    cpt = {}
    for medu_fedu_vals, group_df in port_final_df.groupby(['Medu', 'Fedu']):
        medu_fedu_total = len(group_df)
        counts = group_df['higher'].value_counts()
        cpt[medu_fedu_vals] = {value: counts[value] / medu_fedu_total for value in counts.index}
    print("CPT for Higher:", cpt)
    return cpt

# Function to calculate CPT for Failures (Dependent on Absences)
def calculate_cpt_failures():
    cpt = {}
    for absences_val, group_df in port_final_df.groupby('absences'):
        absences_total = len(group_df)
        counts = group_df['failures'].value_counts()
        cpt[absences_val] = {value: counts[value] / absences_total for value in counts.index}
    print("CPT for Failures:", cpt)
    return cpt

# Function to calculate CPT for Study Time (Dependent on Higher)
def calculate_cpt_studytime():
    cpt = {}
    for higher_val, group_df in port_final_df.groupby('higher'):
        higher_total = len(group_df)
        counts = group_df['studytime'].value_counts()
        cpt[higher_val] = {value: counts[value] / higher_total for value in counts.index}
    print("CPT for Study Time:", cpt)
    return cpt

# Function to calculate CPT for Final Grade (Dependent on Failures and Studytime)
def calculate_cpt_final_grade():
    cpt = {}
    for failures_studytime_vals, group_df in port_final_df.groupby(['failures', 'studytime']):
        failures_studytime_total = len(group_df)
        counts = group_df['Final_Grade'].value_counts()
        cpt[failures_studytime_vals] = {value: counts[value] / failures_studytime_total for value in counts.index}
    print("CPT for Final Grade:", cpt)
    return cpt

# Now let's calculate and store all the CPTs

cpts = {}

cpts['Medu'] = calculate_cpt_medu()
cpts['Fedu'] = calculate_cpt_fedu()
cpts['Absences'] = calculate_cpt_absences()
cpts['Higher'] = calculate_cpt_higher()
cpts['Failures'] = calculate_cpt_failures()
cpts['Studytime'] = calculate_cpt_studytime()
cpts['FinalGrade'] = calculate_cpt_final_grade()

# Write all CPTs to a text file in a readable format
with open('cpts_bayesian_network.txt', 'w') as file:
    for key, value in cpts.items():
        file.write(f'CPT for {key}:\n')
        file.write(f'{value}\n\n')

print("CPTs have been written to 'cpts_bayesian_network.txt'.")
