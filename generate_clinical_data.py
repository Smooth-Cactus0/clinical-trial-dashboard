# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 14:37:56 2025

@author: alexy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# ========== TRIAL CONFIGURATION ==========
trial_start = datetime(2022, 1, 15)
trial_end = datetime(2024, 6, 30)
enrollment_end = datetime(2023, 3, 1)

# Drug arms
treatment_arms = {
    'Placebo': {'dose': 0, 'efficacy_mean': 0.15, 'ae_rate': 0.20},
    'Low Dose': {'dose': 50, 'efficacy_mean': 0.45, 'ae_rate': 0.35},
    'Medium Dose': {'dose': 100, 'efficacy_mean': 0.65, 'ae_rate': 0.45},
    'High Dose': {'dose': 200, 'efficacy_mean': 0.75, 'ae_rate': 0.60}
}

# ========== RESEARCH SITES ==========
sites_data = {
    'site_id': ['SITE001', 'SITE002', 'SITE003', 'SITE004', 'SITE005', 'SITE006'],
    'site_name': ['Mass General Hospital', 'Mayo Clinic', 'Johns Hopkins', 
                  'UCSF Medical Center', 'Cleveland Clinic', 'Stanford Medicine'],
    'city': ['Boston', 'Rochester', 'Baltimore', 'San Francisco', 'Cleveland', 'Palo Alto'],
    'state': ['MA', 'MN', 'MD', 'CA', 'OH', 'CA'],
    'principal_investigator': ['Dr. Sarah Chen', 'Dr. Michael Roberts', 'Dr. Emily Johnson',
                               'Dr. James Park', 'Dr. Lisa Anderson', 'Dr. David Kim'],
    'activation_date': ['2022-01-15', '2022-01-20', '2022-02-01', 
                       '2022-02-15', '2022-03-01', '2022-03-15']
}
sites_df = pd.DataFrame(sites_data)

# ========== GENERATE PATIENTS ==========
num_patients = 450
patients_list = []

for i in range(1, num_patients + 1):
    patient_id = f'PT{i:04d}'
    
    # Demographics
    age = int(np.random.normal(55, 12))
    age = max(18, min(85, age))  # Constrain to 18-85
    
    gender = np.random.choice(['Male', 'Female'], p=[0.48, 0.52])
    
    ethnicity = np.random.choice(
        ['White', 'Black', 'Asian', 'Hispanic', 'Other'],
        p=[0.60, 0.15, 0.12, 0.10, 0.03]
    )
    
    # Medical history
    bmi = round(np.random.normal(28, 5), 1)
    bmi = max(18, min(45, bmi))
    
    baseline_severity = np.random.choice(['Mild', 'Moderate', 'Severe'], p=[0.25, 0.50, 0.25])
    
    comorbidities = []
    if np.random.random() < 0.3:
        comorbidities.append('Hypertension')
    if np.random.random() < 0.2:
        comorbidities.append('Diabetes')
    if np.random.random() < 0.15:
        comorbidities.append('Heart Disease')
    comorbidities_str = ', '.join(comorbidities) if comorbidities else 'None'
    
    # Enrollment details
    site = np.random.choice(sites_df['site_id'].tolist())
    site_activation = sites_df[sites_df['site_id'] == site]['activation_date'].iloc[0]
    site_activation_date = datetime.strptime(site_activation, '%Y-%m-%d')
    
    # Enrollment date between site activation and enrollment_end
    days_range = (enrollment_end - site_activation_date).days
    if days_range > 0:
        enrollment_date = site_activation_date + timedelta(days=np.random.randint(0, days_range))
    else:
        enrollment_date = site_activation_date
    
    # Treatment assignment (randomized, stratified by severity)
    treatment_arm = np.random.choice(list(treatment_arms.keys()))
    
    # Patient status
    # Dropout probability increases with adverse events
    dropout_prob = 0.12 + (0.08 if treatment_arm in ['Medium Dose', 'High Dose'] else 0)
    completed_trial = np.random.random() > dropout_prob
    
    if not completed_trial:
        dropout_date = enrollment_date + timedelta(days=np.random.randint(30, 300))
        dropout_reason = np.random.choice(
            ['Adverse Event', 'Lost to Follow-up', 'Withdrew Consent', 'Protocol Violation'],
            p=[0.4, 0.3, 0.2, 0.1]
        )
    else:
        dropout_date = None
        dropout_reason = None
    
    patients_list.append({
        'patient_id': patient_id,
        'site_id': site,
        'enrollment_date': enrollment_date.strftime('%Y-%m-%d'),
        'age': age,
        'gender': gender,
        'ethnicity': ethnicity,
        'bmi': bmi,
        'baseline_severity': baseline_severity,
        'comorbidities': comorbidities_str,
        'treatment_arm': treatment_arm,
        'dose_mg': treatment_arms[treatment_arm]['dose'],
        'completed_trial': completed_trial,
        'dropout_date': dropout_date.strftime('%Y-%m-%d') if dropout_date else None,
        'dropout_reason': dropout_reason
    })

patients_df = pd.DataFrame(patients_list)

# ========== EFFICACY MEASUREMENTS ==========
# Multiple timepoints: Baseline, Week 4, Week 8, Week 12, Week 24, Week 36, Week 52
timepoints = [0, 4, 8, 12, 24, 36, 52]
efficacy_records = []

for _, patient in patients_df.iterrows():
    enrollment_date = datetime.strptime(patient['enrollment_date'], '%Y-%m-%d')
    treatment = patient['treatment_arm']
    
    # Baseline efficacy score (disease severity: 0-100, higher = worse)
    if patient['baseline_severity'] == 'Mild':
        baseline_score = np.random.uniform(30, 50)
    elif patient['baseline_severity'] == 'Moderate':
        baseline_score = np.random.uniform(50, 70)
    else:  # Severe
        baseline_score = np.random.uniform(70, 90)
    
    # Treatment effect parameters
    efficacy_mean = treatment_arms[treatment]['efficacy_mean']
    
    for week in timepoints:
        visit_date = enrollment_date + timedelta(weeks=week)
        
        # Check if patient was still in trial
        if patient['dropout_date']:
            dropout_date = datetime.strptime(patient['dropout_date'], '%Y-%m-%d')
            if visit_date > dropout_date:
                continue
        
        if week == 0:
            score = baseline_score
            change_from_baseline = 0
        else:
            # Treatment effect increases over time (diminishing returns)
            time_factor = 1 - np.exp(-week / 20)
            treatment_effect = efficacy_mean * time_factor * baseline_score
            noise = np.random.normal(0, 5)
            
            score = baseline_score - treatment_effect + noise
            score = max(0, min(100, score))
            change_from_baseline = baseline_score - score
        
        # Response classification
        if change_from_baseline >= 50:
            response = 'Complete Response'
        elif change_from_baseline >= 30:
            response = 'Partial Response'
        elif change_from_baseline >= 10:
            response = 'Stable Disease'
        else:
            response = 'Progressive Disease'
        
        efficacy_records.append({
            'patient_id': patient['patient_id'],
            'visit_week': week,
            'visit_date': visit_date.strftime('%Y-%m-%d'),
            'efficacy_score': round(score, 1),
            'change_from_baseline': round(change_from_baseline, 1),
            'response_category': response
        })

efficacy_df = pd.DataFrame(efficacy_records)

# ========== ADVERSE EVENTS ==========
ae_types = {
    'Headache': {'severity_dist': [0.6, 0.3, 0.1]},
    'Nausea': {'severity_dist': [0.5, 0.35, 0.15]},
    'Fatigue': {'severity_dist': [0.55, 0.35, 0.1]},
    'Dizziness': {'severity_dist': [0.6, 0.3, 0.1]},
    'Insomnia': {'severity_dist': [0.65, 0.25, 0.1]},
    'Rash': {'severity_dist': [0.7, 0.25, 0.05]},
    'Elevated Liver Enzymes': {'severity_dist': [0.4, 0.4, 0.2]},
    'Hypertension': {'severity_dist': [0.3, 0.5, 0.2]},
    'GI Upset': {'severity_dist': [0.6, 0.3, 0.1]},
    'Muscle Pain': {'severity_dist': [0.65, 0.3, 0.05]}
}

severity_levels = ['Mild', 'Moderate', 'Severe']

ae_records = []
ae_id = 1000

for _, patient in patients_df.iterrows():
    enrollment_date = datetime.strptime(patient['enrollment_date'], '%Y-%m-%d')
    treatment = patient['treatment_arm']
    ae_rate = treatment_arms[treatment]['ae_rate']
    
    # Determine trial duration for this patient
    if patient['dropout_date']:
        end_date = datetime.strptime(patient['dropout_date'], '%Y-%m-%d')
    else:
        end_date = enrollment_date + timedelta(weeks=52)
    
    trial_duration_days = (end_date - enrollment_date).days
    
    # Expected number of AEs based on treatment
    expected_aes = ae_rate * (trial_duration_days / 365) * 3
    num_aes = np.random.poisson(expected_aes)
    
    for _ in range(num_aes):
        ae_type = np.random.choice(list(ae_types.keys()))
        severity_dist = ae_types[ae_type]['severity_dist']
        severity = np.random.choice(severity_levels, p=severity_dist)
        
        onset_date = enrollment_date + timedelta(days=np.random.randint(1, trial_duration_days))
        
        # Duration depends on severity
        if severity == 'Mild':
            duration_days = np.random.randint(1, 7)
        elif severity == 'Moderate':
            duration_days = np.random.randint(3, 14)
        else:  # Severe
            duration_days = np.random.randint(7, 30)
        
        resolution_date = onset_date + timedelta(days=duration_days)
        
        # Serious AE (requiring hospitalization or life-threatening)
        is_serious = (severity == 'Severe' and np.random.random() < 0.15)
        
        # Related to study drug
        related_to_drug = np.random.choice([True, False], p=[0.7, 0.3])
        
        ae_records.append({
            'ae_id': f'AE{ae_id:05d}',
            'patient_id': patient['patient_id'],
            'ae_type': ae_type,
            'severity': severity,
            'onset_date': onset_date.strftime('%Y-%m-%d'),
            'resolution_date': resolution_date.strftime('%Y-%m-%d'),
            'duration_days': duration_days,
            'is_serious': is_serious,
            'related_to_drug': related_to_drug
        })
        ae_id += 1

ae_df = pd.DataFrame(ae_records)

# ========== LABORATORY VALUES ==========
# Key biomarkers measured at each visit
lab_records = []

for _, patient in patients_df.iterrows():
    for _, efficacy_row in efficacy_df[efficacy_df['patient_id'] == patient['patient_id']].iterrows():
        visit_week = efficacy_row['visit_week']
        visit_date = efficacy_row['visit_date']
        
        # Normal ranges with treatment effects
        treatment = patient['treatment_arm']
        
        # ALT (Liver enzyme) - can be elevated with treatment
        alt_baseline = np.random.uniform(20, 40)
        if treatment in ['Medium Dose', 'High Dose']:
            alt = alt_baseline * np.random.uniform(1.0, 2.5)
        else:
            alt = alt_baseline * np.random.uniform(0.9, 1.3)
        
        # Creatinine (kidney function)
        creat = np.random.uniform(0.7, 1.3)
        
        # White blood cell count
        wbc = np.random.uniform(4.0, 11.0)
        
        # Hemoglobin
        hgb = np.random.uniform(12.0, 17.0)
        
        lab_records.append({
            'patient_id': patient['patient_id'],
            'visit_week': visit_week,
            'visit_date': visit_date,
            'alt_ul': round(alt, 1),
            'creatinine_mg_dl': round(creat, 2),
            'wbc_k_ul': round(wbc, 1),
            'hemoglobin_g_dl': round(hgb, 1)
        })

labs_df = pd.DataFrame(lab_records)

# ========== SAVE ALL FILES ==========
sites_df.to_csv('C:/Users/alexy/Documents/Claude_projects/Portfolio creation/Clinical_Trial_Dashboard/trial_sites.csv', index=False)
patients_df.to_csv('C:/Users/alexy/Documents/Claude_projects/Portfolio creation/Clinical_Trial_Dashboard/trial_patients.csv', index=False)
efficacy_df.to_csv('C:/Users/alexy/Documents/Claude_projects/Portfolio creation/Clinical_Trial_Dashboard/trial_efficacy.csv', index=False)
ae_df.to_csv('C:/Users/alexy/Documents/Claude_projects/Portfolio creation/Clinical_Trial_Dashboard/trial_adverse_events.csv', index=False)
labs_df.to_csv('C:/Users/alexy/Documents/Claude_projects/Portfolio creation/Clinical_Trial_Dashboard/trial_laboratory.csv', index=False)

# ========== SUMMARY ==========
print("=" * 70)
print("CLINICAL TRIAL DATA GENERATED SUCCESSFULLY")
print("=" * 70)

print(f"\n🏥 Sites: {len(sites_df)} research centers")
print(f"👥 Patients: {len(patients_df)} enrolled")
print(f"   - Completed: {patients_df['completed_trial'].sum()} ({patients_df['completed_trial'].sum()/len(patients_df)*100:.1f}%)")
print(f"   - Dropped out: {(~patients_df['completed_trial']).sum()}")

print(f"\n💊 Treatment Arms:")
for arm in treatment_arms.keys():
    count = len(patients_df[patients_df['treatment_arm'] == arm])
    print(f"   - {arm}: {count} patients ({count/len(patients_df)*100:.1f}%)")

print(f"\n📊 Efficacy Measurements: {len(efficacy_df):,} assessments")
print(f"   - Timepoints: {len(timepoints)} visits per patient")

print(f"\n⚠️  Adverse Events: {len(ae_df):,} events")
print(f"   - Serious AEs: {ae_df['is_serious'].sum()}")
print(f"   - Drug-related: {ae_df['related_to_drug'].sum()} ({ae_df['related_to_drug'].sum()/len(ae_df)*100:.1f}%)")

print(f"\n🔬 Laboratory Tests: {len(labs_df):,} measurements")

print(f"\n📈 Response Rates (Latest Assessment):")
latest_efficacy = efficacy_df.sort_values('visit_week').groupby('patient_id').tail(1)
latest_efficacy = latest_efficacy.merge(patients_df[['patient_id', 'treatment_arm']], on='patient_id')
for arm in treatment_arms.keys():
    arm_data = latest_efficacy[latest_efficacy['treatment_arm'] == arm]
    complete_response = (arm_data['response_category'] == 'Complete Response').sum()
    partial_response = (arm_data['response_category'] == 'Partial Response').sum()
    total_response = complete_response + partial_response
    response_rate = (total_response / len(arm_data) * 100) if len(arm_data) > 0 else 0
    print(f"   - {arm}: {response_rate:.1f}% overall response rate")

print(f"\n📅 Trial Duration: {trial_start.strftime('%Y-%m-%d')} to {trial_end.strftime('%Y-%m-%d')}")
print(f"\n📁 Files saved:")
print("   - trial_sites.csv")
print("   - trial_patients.csv")
print("   - trial_efficacy.csv")
print("   - trial_adverse_events.csv")
print("   - trial_laboratory.csv")

print("\n✅ Ready to build the clinical trial dashboard!")
print("=" * 70)