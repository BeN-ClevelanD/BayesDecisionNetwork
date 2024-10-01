import pyAgrum as gum
import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load the CSV data into a pandas DataFrame
data = pd.read_csv('processed_port_data.csv')

# Step 2: Preprocess the categorical columns
data['higher'] = data['higher'].map({'yes': 1, 'no': 0})
data['Final_Grade'] = data['Final_Grade'].map({'Pass': 1, 'Fail': 0})

# Map the 'failures' column: "0" -> 0, "1 or more" -> 1
data['failures'] = data['failures'].map({'0': 0, '1 or more': 1})

# Map 'absences' column: "Low" -> 0, "Medium" -> 1, "High" -> 2
data['absences'] = data['absences'].map({'Low': 0, 'Medium': 1, 'High': 2})

# Map 'studytime' column: "Very Low" -> 0, "Low" -> 1, "Medium" -> 2, "High" -> 3
data['studytime'] = data['studytime'].map({'Very Low': 0, 'Low': 1, 'Medium': 2, 'High': 3})

# Display the first few rows of your data for verification
print(data.head())

# Step 3: Define your Bayesian Network Structure
bn = gum.BayesNet('StudentPerformance')

# Add variables (nodes) based on the CSV columns
medu = bn.add(gum.LabelizedVariable("Medu", "Mother's Education", 2))
fedu = bn.add(gum.LabelizedVariable("Fedu", "Father's Education", 2))
higher = bn.add(gum.LabelizedVariable("higher", "Higher Education", 2))
absences = bn.add(gum.LabelizedVariable("absences", "Absences", 3))
studytime = bn.add(gum.LabelizedVariable("studytime", "Study Time", 4))
failures = bn.add(gum.LabelizedVariable("failures", "Failures", 2))  # Adjusted for the binary mapping
final_grade = bn.add(gum.LabelizedVariable("Final_Grade", "Final Grade", 2))

# Step 4: Add arcs (dependencies) as per the Bayesian network structure
bn.addArc(medu, higher)
bn.addArc(fedu, higher)
bn.addArc(higher, studytime)
bn.addArc(absences, failures)
bn.addArc(studytime, final_grade)
bn.addArc(failures, final_grade)

# Step 5: Compute CPTs from the data using Maximum Likelihood Estimation (MLE)
learner = gum.BNLearner(data, bn)

# Step 6: Learn the parameters
learner.useAprioriSmoothing(1)  # Optional: add Laplace smoothing
learner.learnParameters(bn)

# Step 7: Display the learned CPTs
for node in bn.names():
    print(f"\nCPT of {node}:")
    print(bn.cpt(bn.idFromName(node)))

# Step 8: Save the Bayesian network to a file (optional)
gum.saveBN(bn, 'learned_student_performance.bif')

# Step 9: Perform inference on the network
ie = gum.LazyPropagation(bn)

# Example of inference with evidence
ie.setEvidence({
    'Medu': 1,  # Higher Education for Mother
    'Fedu': 1,  # Higher Education for Father
    'absences': 0,  # Low absences
})

ie.makeInference()

# Get posterior distribution for a target node
posterior = ie.posterior(bn.idFromName('Final_Grade'))
print("\nPosterior distribution of Final_Grade with evidence:")
print(posterior)
