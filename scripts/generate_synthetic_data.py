import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

def generate_synthetic_patients(n=10000):
    """Generate synthetic patient prescription data"""
    
    # Patient demographics
    ages = np.random.normal(58, 15, n).clip(18, 90).astype(int)
    
    # Zip codes (using real US median income patterns)
    zip_income_map = {
        '78701': 95000,  # Austin high-income
        '78702': 45000,  # Austin low-income
        '10001': 85000,  # NYC Manhattan
        '10456': 35000,  # NYC Bronx
        '90210': 120000, # Beverly Hills
        '90011': 40000,  # LA low-income
        '60614': 90000,  # Chicago high-income
        '60619': 38000,  # Chicago South Side
    }
    
    zip_codes = np.random.choice(list(zip_income_map.keys()), n)
    median_incomes = [zip_income_map[z] for z in zip_codes]
    
    # Insurance types
    insurance_types = np.random.choice(
        ['Medicare', 'Medicaid', 'Commercial', 'Uninsured'],
        n,
        p=[0.35, 0.15, 0.45, 0.05]
    )
    
    # Drug categories
    drugs = [
        {'name': 'Ocrevus', 'category': 'MS', 'base_cost': 65000/26, 'therapeutic_area': 'Neurology'},
        {'name': 'Keytruda', 'category': 'Oncology', 'base_cost': 12000/3, 'therapeutic_area': 'Oncology'},
        {'name': 'Humira', 'category': 'Immunology', 'base_cost': 7000/2, 'therapeutic_area': 'Immunology'},
        {'name': 'Xolair', 'category': 'Asthma', 'base_cost': 3000, 'therapeutic_area': 'Pulmonology'},
        {'name': 'Enbrel', 'category': 'RA', 'base_cost': 6000, 'therapeutic_area': 'Rheumatology'},
    ]
    
    drug_assignments = np.random.choice(len(drugs), n)
    
    # Calculate out-of-pocket costs
    def calculate_oop(insurance, drug_cost, income):
        if insurance == 'Medicare':
            return min(drug_cost * 0.25 + 480, drug_cost * 0.3)
        elif insurance == 'Medicaid':
            return np.random.uniform(5, 50)
        elif insurance == 'Commercial':
            if income > 80000:
                return drug_cost * 0.20
            else:
                return drug_cost * 0.35
        else:  # Uninsured
            return drug_cost * 0.9
    
    oop_costs = [
        calculate_oop(insurance_types[i], drugs[drug_assignments[i]]['base_cost'], median_incomes[i])
        for i in range(n)
    ]
    
    # Distance to specialty pharmacy
    urban_zips = ['78701', '10001', '90210', '60614']
    distances = [
        np.random.gamma(2, 3) if zip_codes[i] in urban_zips else np.random.gamma(5, 5)
        for i in range(n)
    ]
    
    # Prior authorization required
    pa_required = np.random.choice([True, False], n, p=[0.6, 0.4])
    
    # Prior abandonment history
    prior_abandonment = np.random.choice([0, 1, 2, 3], n, p=[0.6, 0.25, 0.1, 0.05])
    
    # Calculate abandonment probability
    abandonment_prob = np.zeros(n)
    
    for i in range(n):
        prob = 0.15
        
        # Cost factors
        oop_to_income_ratio = oop_costs[i] / (median_incomes[i] / 12)
        if oop_to_income_ratio > 0.5:
            prob += 0.40
        elif oop_to_income_ratio > 0.3:
            prob += 0.25
        elif oop_to_income_ratio > 0.1:
            prob += 0.10
        
        # Insurance factors
        if insurance_types[i] == 'Uninsured':
            prob += 0.35
        elif insurance_types[i] == 'Medicaid':
            prob -= 0.10
        
        # Access barriers
        if distances[i] > 30:
            prob += 0.15
        if pa_required[i]:
            prob += 0.10
        
        # Prior behavior
        prob += prior_abandonment[i] * 0.15
        
        # Age effects
        if ages[i] < 30:
            prob += 0.10
        elif ages[i] > 65:
            prob -= 0.05
        
        abandonment_prob[i] = np.clip(prob, 0, 0.95)
    
    # Generate actual abandonment
    abandoned = np.random.binomial(1, abandonment_prob)
    
    # Create DataFrame
    df = pd.DataFrame({
        'patient_id': [f'PAT{i:05d}' for i in range(n)],
        'age': ages,
        'zip_code': zip_codes,
        'median_income': median_incomes,
        'insurance_type': insurance_types,
        'drug_name': [drugs[drug_assignments[i]]['name'] for i in range(n)],
        'therapeutic_area': [drugs[drug_assignments[i]]['therapeutic_area'] for i in range(n)],
        'drug_cost': [drugs[drug_assignments[i]]['base_cost'] for i in range(n)],
        'oop_cost': np.round(oop_costs, 2),
        'distance_to_pharmacy': np.round(distances, 1),
        'pa_required': pa_required,
        'prior_abandonment_count': prior_abandonment,
        'abandoned': abandoned,
        'prescription_date': [
            (datetime.now() - timedelta(days=np.random.randint(1, 365))).strftime('%Y-%m-%d')
            for _ in range(n)
        ]
    })
    
    return df

# Generate and save
print("Generating synthetic patient data...")
df = generate_synthetic_patients(10000)

# Save to CSV
df.to_csv('backend/data/synthetic_patients.csv', index=False)

print(f"✅ Generated {len(df)} synthetic patient records")
print(f"Abandonment rate: {df['abandoned'].mean():.1%}")
print("\nFirst few rows:")
print(df.head())
print("\n=== Data Summary ===")
print(f"Total prescriptions: {len(df)}")
print(f"Abandoned: {df['abandoned'].sum()} ({df['abandoned'].mean():.1%})")
print(f"Filled: {(1-df['abandoned']).sum()} ({(1-df['abandoned']).mean():.1%})")
print(f"\nAbandonment by insurance type:")
print(df.groupby('insurance_type')['abandoned'].agg(['count', 'mean']))
