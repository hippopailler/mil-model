import pandas as pd
import numpy as np
import os

# Parameters for dataset
num_patients = 300
mod1_dim = 1
mod2_dim = 2
mod3_dim = 3

# Create patient IDs (no multiple slides per patient)
patients = list(range(1, num_patients+1))
slides = [str(i) for i in patients]

# Create survival relationship based on patient number
# Patients 1-60: survival 80-100, very low risk (activation 0.0-0.2)
# Patients 61-120: survival 60-80, low risk (activation 0.2-0.4)
# Patients 121-180: survival 40-60, medium risk (activation 0.4-0.6)
# Patients 181-240: survival 20-40, high risk (activation 0.6-0.8)
# Patients 241-300: survival 0-20, very high risk (activation 0.8-1.0)

patient_outcomes = {}
for i in range(1, num_patients+1):
    if i <= 60:
        # Very low risk group
        activation = np.random.uniform(0.0, 0.2)
        survival = np.random.uniform(80, 100)
        death = np.random.binomial(1, 0.1)  # 10% chance of event
    elif i <= 120:
        # Low risk group
        activation = np.random.uniform(0.2, 0.4)
        survival = np.random.uniform(60, 80)
        death = np.random.binomial(1, 0.3)  # 30% chance of event
    elif i <= 180:
        # Medium risk group
        activation = np.random.uniform(0.4, 0.6)
        survival = np.random.uniform(40, 60)
        death = np.random.binomial(1, 0.5)  # 50% chance of event
    elif i <= 240:
        # High risk group
        activation = np.random.uniform(0.6, 0.8)
        survival = np.random.uniform(20, 40)
        death = np.random.binomial(1, 0.7)  # 70% chance of event
    else:
        # Very high risk group
        activation = np.random.uniform(0.8, 1.0)
        survival = np.random.uniform(0, 20)
        death = np.random.binomial(1, 0.9)  # 90% chance of event
        
    patient_outcomes[i] = {
        'activation': activation,
        'survival': survival,
        'death': death
    }

# Apply outcomes to slides and create modalities
mod1 = []
mod2 = []
mod3 = []
missing_pattern = []

# Create a pattern of missing modalities (but never all three)
for i in patients:
    # Choose which modality will be missing (if any)
    # 0: none missing, 1: mod1 missing, 2: mod2 missing, 3: mod3 missing
    missing = np.random.choice([0, 1, 2, 3])
    missing_pattern.append(missing)
    
    # Get the activation value for this patient
    activation_value = patient_outcomes[i]['activation']
    
    # Add tiny bit of noise to activation, within range [0,1]
    if missing != 1:  # If mod1 is not missing
        noise = np.random.normal(0, 0.05, mod1_dim)
        mod1_value = np.clip(activation_value + noise, 0, 1)
        mod1.append(mod1_value)
    else:
        mod1.append(pd.NA)
        
    if missing != 2:  # If mod2 is not missing
        noise = np.random.normal(0, 0.05, mod2_dim)
        mod2_value = np.clip(activation_value + noise, 0, 1)
        mod2.append(mod2_value)
    else:
        mod2.append(pd.NA)
        
    if missing != 3:  # If mod3 is not missing
        noise = np.random.normal(0, 0.05, mod3_dim)
        mod3_value = np.clip(activation_value + noise, 0, 1)
        mod3.append(mod3_value)
    else:
        mod3.append(pd.NA)

# Create the DataFrame
df = pd.DataFrame({
    'mod1': mod1,
    'mod2': mod2,
    'mod3': mod3,
    'slide': slides
}, index=slides)

# Create directories if needed
if not os.path.exists('features/fake_mm_surv'):
    os.makedirs('features/fake_mm_surv')

# Save feature data
df.to_parquet('features/fake_mm_surv/df.parquet')

# Create patient risk groups for stratification
patient_risk_groups = {}
for i in range(1, num_patients+1):
    if i <= 60:
        patient_risk_groups[i] = 'very_low'
    elif i <= 120:
        patient_risk_groups[i] = 'low'
    elif i <= 180:
        patient_risk_groups[i] = 'medium'
    elif i <= 240:
        patient_risk_groups[i] = 'high'
    else:
        patient_risk_groups[i] = 'very_high'

# Assign train/val for each patient, stratified by risk group
np.random.seed(42)  # For reproducibility
patient_dataset = {}
for risk_group in ['very_low', 'low', 'medium', 'high', 'very_high']:
    patients_in_group = [p for p in range(1, num_patients+1) if patient_risk_groups[p] == risk_group]
    np.random.shuffle(patients_in_group)
    split_idx = int(len(patients_in_group) * 0.8)
    
    for p in patients_in_group[:split_idx]:
        patient_dataset[p] = 'train'
    for p in patients_in_group[split_idx:]:
        patient_dataset[p] = 'val'

# Extract all needed data for annotations
activation_values = [patient_outcomes[p]['activation'] for p in patients]
survival_times = [patient_outcomes[p]['survival'] for p in patients]
death_events = [patient_outcomes[p]['death'] for p in patients]
datasets = [patient_dataset[p] for p in patients]
risk_groups = [patient_risk_groups[p] for p in patients]

# Round values to 4 significant digits
activation_values = [round(x, 4) for x in activation_values]
survival_times = [round(x, 4) for x in survival_times]

# Add information about which modality is missing
missing_mod1 = [1 if pattern == 1 else 0 for pattern in missing_pattern]
missing_mod2 = [1 if pattern == 2 else 0 for pattern in missing_pattern]
missing_mod3 = [1 if pattern == 3 else 0 for pattern in missing_pattern]

# Create annotation dataframe
ann_df = pd.DataFrame({
    'slide': slides,
    'patient': patients,
    'death': death_events,
    'os': survival_times,
    'activation': activation_values,
    'risk_group': risk_groups,
    'dataset': datasets,
    'missing_mod1': missing_mod1,
    'missing_mod2': missing_mod2,
    'missing_mod3': missing_mod3
})

# Create directory if needed
if not os.path.exists('annotations'):
    os.makedirs('annotations')

# Save annotations
ann_df.to_csv('annotations/ann_mm_surv.csv', index=False)