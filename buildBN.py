import pyAgrum as gum

# Initialize the Bayesian Network
bn = gum.BayesNet('StudentPerformance')

# Add Nodes to the Network
medu = bn.add(gum.LabelizedVariable("Medu", "Mother's Education", 2))  # 0: No Higher Education, 1: Higher Education
fedu = bn.add(gum.LabelizedVariable("Fedu", "Father's Education", 2))  # 0: No Higher Education, 1: Higher Education
higher = bn.add(gum.LabelizedVariable("Higher", "Higher Education", 2))  # 0: No, 1: Yes
absences = bn.add(gum.LabelizedVariable("Absences", "Absences", 3))  # 0: Low, 1: Medium, 2: High
studytime = bn.add(gum.LabelizedVariable("StudyTime", "Study Time", 4))  # 0: Very Low, 1: Low, 2: Medium, 3: High
failures = bn.add(gum.LabelizedVariable("Failures", "Failures", 3))  # 0: 0, 1: 1, 2: More than 1
final_grade = bn.add(gum.LabelizedVariable("FinalGrade", "Final Grade", 2))  # 0: Fail, 1: Pass

# Add arcs (dependencies) as per the Bayesian network structure
bn.addArc(medu, higher)
bn.addArc(fedu, higher)
bn.addArc(higher, studytime)
bn.addArc(absences, failures)
bn.addArc(studytime, final_grade)
bn.addArc(failures, final_grade)

# Set CPT for Medu (Root node)
bn.cpt(medu)[0] = 0.7303543913713405
bn.cpt(medu)[1] = 0.2696456086286595

# Set CPT for Fedu (Root node)
bn.cpt(fedu)[0] = 0.802773497688752
bn.cpt(fedu)[1] = 0.19722650231124808

# Set CPT for Absences (Root node)
bn.cpt(absences)[0] = 0.987673343605547  # Low
bn.cpt(absences)[1] = 0.012326656394453005  # Medium

# Set CPT for Higher (Dependent on Medu and Fedu)
bn.cpt(higher)[0, 0] = [0.8480725623582767, 0.15192743764172337]  # Medu=0, Fedu=0
bn.cpt(higher)[0, 1] = [1.0, 0.0]  # Medu=0, Fedu=1
bn.cpt(higher)[1, 0] = [1.0, 0.0]  # Medu=1, Fedu=0
bn.cpt(higher)[1, 1] = [0.9789473684210527, 0.021052631578947368]  # Medu=1, Fedu=1

# Set CPT for Failures (Dependent on Absences)
bn.cpt(failures)[0] = [0.8486739469578783, 0.1060842433697348, 0.0452418096723869]  # Absences=Low
bn.cpt(failures)[1] = [0.625, 0.25, 0.125]  # Absences=Medium

# Set CPT for Study Time (Dependent on Higher)
bn.cpt(studytime)[0] = [0.6376811594202898, 0.2753623188405797, 0.057971014492753624, 0.028985507246376812]  # Higher=No
bn.cpt(studytime)[1] = [0.2896551724137931, 0.49310344827586206, 0.16034482758620688, 0.056896551724137934]  # Higher=Yes

# Set CPT for FinalGrade (Dependent on Failures and StudyTime)
bn.cpt(final_grade)[0, 0] = [0.147239263803681, 0.852760736196319]  # Failures=0, StudyTime=Very Low
bn.cpt(final_grade)[0, 1] = [0.08712121212121213, 0.9128787878787878]  # Failures=0, StudyTime=Low
bn.cpt(final_grade)[0, 2] = [0.033707865168539325, 0.9662921348314607]  # Failures=0, StudyTime=Medium
bn.cpt(final_grade)[0, 3] = [0.030303030303030304, 0.9696969696969697]  # Failures=0, StudyTime=High
bn.cpt(final_grade)[1, 0] = [0.5, 0.5]  # Failures=1, StudyTime=Very Low
bn.cpt(final_grade)[1, 1] = [0.34615384615384615, 0.6538461538461539]  # Failures=1, StudyTime=Low
bn.cpt(final_grade)[1, 2] = [0.6666666666666666, 0.3333333333333333]  # Failures=1, StudyTime=Medium
bn.cpt(final_grade)[1, 3] = [0.5, 0.5]  # Failures=1, StudyTime=High
bn.cpt(final_grade)[2, 0] = [0.6153846153846154, 0.38461538461538464]  # Failures=More than 1, StudyTime=Very Low
bn.cpt(final_grade)[2, 1] = [0.6, 0.4]  # Failures=More than 1, StudyTime=Low
bn.cpt(final_grade)[2, 2] = [0.0, 1.0]  # Failures=More than 1, StudyTime=Medium

# Print the structure of the network
print("Bayesian Network Structure:")
print(bn)

# Print the CPTs for each node
nodes = ['Medu', 'Fedu', 'Absences', 'Higher', 'Failures', 'StudyTime', 'FinalGrade']
for node in nodes:
    print(f"\nCPT of {node}:")
    print(bn.cpt(bn.idFromName(node)))

# Save the Bayesian network to a file
gum.saveBN(bn, 'student_performance.bif')

# Inference Example: Predict Final Grade
ie = gum.LazyPropagation(bn)
# Example evidence with correct mappings
ie.setEvidence({
    'Medu': 1,  # Higher Education for Mother
    'Fedu': 1,  # Higher Education for Father
    'Higher': 1,  # Yes to Higher Education
    'StudyTime': 2,  # Medium Study Time
    'Failures': 0  # No Failures
})
ie.makeInference()
print("\nPosterior distribution of FinalGrade:")
print(ie.posterior(bn.idFromName('FinalGrade')))

