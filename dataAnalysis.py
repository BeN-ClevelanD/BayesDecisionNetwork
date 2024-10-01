import csv
from collections import Counter

# Open the CSV file
with open('student-por.csv', mode='r') as file:
    csv_reader = csv.DictReader(file, delimiter=';')
    
    # Grab the "studytime" column values
    studytime_values = [row['studytime'] for row in csv_reader]

# Count the occurrences of each unique value
studytime_counts = Counter(studytime_values)

# Display the counts
for value, count in studytime_counts.items():
    print(f'Studytime: {value}, Count: {count}')
