import pandas as pd
import numpy as np

class InterventionRecommender:
    """Recommends interventions to prevent prescription abandonment"""
    
    def __init__(self):
        # Intervention database
        self.interventions = {
            'pap_enrollment': {
                'name': 'Patient Assistance Program',
                'description': 'Enroll patient in manufacturer financial assistance program',
                'requirements': ['income_below_threshold', 'uninsured_or_underinsured'],
                'avg_cost_reduction': 0.90,
                'avg_success_rate': 0.85,
                'processing_time_days': 7
            },
            'copay_card': {
                'name': 'Manufacturer Co-pay Card',
                'description': 'Apply manufacturer co-pay assistance (up to $200/month)',
                'requirements': ['commercial_insurance', 'oop_over_100'],
                'avg_cost_reduction': 200,
                'avg_success_rate': 0.80,
                'processing_time_days': 1
            },
            'generic_switch': {
                'name': 'Generic/Biosimilar Alternative',
                'description': 'Switch to lower-cost therapeutic alternative',
                'requirements': ['generic_available'],
                'avg_cost_reduction': 0.70,
                'avg_success_rate': 0.65,
                'processing_time_days': 0
            },
            'hub_services': {
                'name': 'Hub Services Referral',
                'description': 'Connect patient with specialty pharmacy hub for navigation support',
                'requirements': ['specialty_medication'],
                'avg_cost_reduction': 0.40,
                'avg_success_rate': 0.75,
                'processing_time_days': 3
            },
            'foundation_grant': {
                'name': 'Independent Foundation Grant',
                'description': 'Apply for disease-specific foundation financial assistance',
                'requirements': ['diagnosis_specific', 'income_qualified'],
                'avg_cost_reduction': 0.60,
                'avg_success_rate': 0.50,
                'processing_time_days': 14
            },
            'payment_plan': {
                'name': 'Pharmacy Payment Plan',
                'description': 'Arrange installment payment plan with specialty pharmacy',
                'requirements': ['any'],
                'avg_cost_reduction': 0.0,
                'avg_success_rate': 0.60,
                'processing_time_days': 1
            }
        }
        
        # Generic availability by drug
        self.generic_available = {
            'Humira': True,
            'Enbrel': True,
            'Ocrevus': False,
            'Keytruda': False,
            'Xolair': False
        }
    
    def recommend(self, patient_data, abandonment_risk):
        """Generate ranked intervention recommendations"""
        recommendations = []
        
        # Extract patient features
        income = patient_data.get('median_income', 50000)
        oop_cost = patient_data.get('oop_cost', 0)
        insurance = patient_data.get('insurance_type', 'Commercial')
        drug_name = patient_data.get('drug_name', '')
        therapeutic_area = patient_data.get('therapeutic_area', '')
        
        # Only recommend if risk is meaningful
        if abandonment_risk < 0.3:
            return []
        
        # 1. PAP Enrollment
        if income < 100000 and (insurance in ['Uninsured', 'Medicaid'] or oop_cost > 1000):
            estimated_reduction = min(oop_cost * 0.90, oop_cost - 50)
            recommendations.append({
                'intervention_id': 'pap_enrollment',
                'name': self.interventions['pap_enrollment']['name'],
                'description': self.interventions['pap_enrollment']['description'],
                'priority': 1,
                'estimated_cost_reduction': round(estimated_reduction, 2),
                'new_oop_cost': round(oop_cost - estimated_reduction, 2),
                'success_probability': self.interventions['pap_enrollment']['avg_success_rate'],
                'processing_time_days': self.interventions['pap_enrollment']['processing_time_days'],
                'expected_impact': self._calculate_impact(
                    abandonment_risk, 
                    estimated_reduction, 
                    oop_cost,
                    self.interventions['pap_enrollment']['avg_success_rate']
                ),
                'action_items': [
                    'Verify patient income documentation',
                    'Complete PAP application form',
                    'Submit to manufacturer within 48 hours'
                ]
            })
        
        # 2. Co-pay Card
        if insurance == 'Commercial' and oop_cost > 100:
            estimated_reduction = min(200, oop_cost * 0.5)
            recommendations.append({
                'intervention_id': 'copay_card',
                'name': self.interventions['copay_card']['name'],
                'description': self.interventions['copay_card']['description'],
                'priority': 2,
                'estimated_cost_reduction': round(estimated_reduction, 2),
                'new_oop_cost': round(oop_cost - estimated_reduction, 2),
                'success_probability': self.interventions['copay_card']['avg_success_rate'],
                'processing_time_days': self.interventions['copay_card']['processing_time_days'],
                'expected_impact': self._calculate_impact(
                    abandonment_risk,
                    estimated_reduction,
                    oop_cost,
                    self.interventions['copay_card']['avg_success_rate']
                ),
                'action_items': [
                    'Provide patient with co-pay card activation link',
                    'Confirm pharmacy accepts manufacturer cards',
                    'Follow up in 24 hours'
                ]
            })
        
        # 3. Generic/Biosimilar Switch
        if self.generic_available.get(drug_name, False):
            estimated_reduction = oop_cost * 0.70
            recommendations.append({
                'intervention_id': 'generic_switch',
                'name': self.interventions['generic_switch']['name'],
                'description': f'Switch to biosimilar alternative for {drug_name}',
                'priority': 3,
                'estimated_cost_reduction': round(estimated_reduction, 2),
                'new_oop_cost': round(oop_cost - estimated_reduction, 2),
                'success_probability': self.interventions['generic_switch']['avg_success_rate'],
                'processing_time_days': self.interventions['generic_switch']['processing_time_days'],
                'expected_impact': self._calculate_impact(
                    abandonment_risk,
                    estimated_reduction,
                    oop_cost,
                    self.interventions['generic_switch']['avg_success_rate']
                ),
                'action_items': [
                    'Contact prescriber to discuss biosimilar option',
                    'Verify insurance coverage for alternative',
                    'Educate patient on biosimilar equivalence'
                ]
            })
        
        # 4. Hub Services
        if therapeutic_area in ['Oncology', 'Neurology', 'Immunology']:
            estimated_reduction = oop_cost * 0.40
            recommendations.append({
                'intervention_id': 'hub_services',
                'name': self.interventions['hub_services']['name'],
                'description': self.interventions['hub_services']['description'],
                'priority': 4,
                'estimated_cost_reduction': round(estimated_reduction, 2),
                'new_oop_cost': round(oop_cost - estimated_reduction, 2),
                'success_probability': self.interventions['hub_services']['avg_success_rate'],
                'processing_time_days': self.interventions['hub_services']['processing_time_days'],
                'expected_impact': self._calculate_impact(
                    abandonment_risk,
                    estimated_reduction,
                    oop_cost,
                    self.interventions['hub_services']['avg_success_rate']
                ),
                'action_items': [
                    'Refer patient to specialty pharmacy hub',
                    'Hub to assess all assistance program eligibility',
                    'Schedule follow-up call in 3 days'
                ]
            })
        
        # 5. Foundation Grant
        if income < 75000 and therapeutic_area in ['Oncology', 'Neurology', 'Immunology']:
            estimated_reduction = oop_cost * 0.60
            recommendations.append({
                'intervention_id': 'foundation_grant',
                'name': self.interventions['foundation_grant']['name'],
                'description': f'Apply to {therapeutic_area}-specific foundation for grant',
                'priority': 5,
                'estimated_cost_reduction': round(estimated_reduction, 2),
                'new_oop_cost': round(oop_cost - estimated_reduction, 2),
                'success_probability': self.interventions['foundation_grant']['avg_success_rate'],
                'processing_time_days': self.interventions['foundation_grant']['processing_time_days'],
                'expected_impact': self._calculate_impact(
                    abandonment_risk,
                    estimated_reduction,
                    oop_cost,
                    self.interventions['foundation_grant']['avg_success_rate']
                ),
                'action_items': [
                    'Identify relevant patient assistance foundations',
                    'Complete foundation application',
                    'Submit required documentation'
                ]
            })
        
        # 6. Payment Plan (fallback)
        if oop_cost > 200:
            recommendations.append({
                'intervention_id': 'payment_plan',
                'name': self.interventions['payment_plan']['name'],
                'description': f'Spread ${oop_cost:.0f} over 3-6 month payment plan',
                'priority': 6,
                'estimated_cost_reduction': 0,
                'new_oop_cost': oop_cost,
                'success_probability': self.interventions['payment_plan']['avg_success_rate'],
                'processing_time_days': self.interventions['payment_plan']['processing_time_days'],
                'expected_impact': self._calculate_impact(
                    abandonment_risk * 0.6,
                    0,
                    oop_cost,
                    self.interventions['payment_plan']['avg_success_rate']
                ),
                'action_items': [
                    'Contact specialty pharmacy billing department',
                    'Arrange monthly payment schedule',
                    'Confirm patient agreement'
                ]
            })
        
        # Sort by expected impact
        recommendations.sort(key=lambda x: x['expected_impact'], reverse=True)
        
        # Re-prioritize
        for i, rec in enumerate(recommendations, 1):
            rec['priority'] = i
        
        return recommendations
    
    def _calculate_impact(self, abandonment_risk, cost_reduction, original_cost, success_rate):
        """Calculate expected impact of intervention"""
        if original_cost == 0:
            return 0
        
        cost_reduction_pct = cost_reduction / original_cost if original_cost > 0 else 0
        risk_reduction = abandonment_risk * cost_reduction_pct * 0.8
        expected_impact = risk_reduction * success_rate * cost_reduction
        
        return round(expected_impact, 2)


if __name__ == '__main__':
    recommender = InterventionRecommender()
    
    # Test case
    test_patient = {
        'patient_id': 'TEST001',
        'age': 45,
        'median_income': 35000,
        'insurance_type': 'Medicaid',
        'drug_name': 'Keytruda',
        'therapeutic_area': 'Oncology',
        'oop_cost': 3500,
        'distance_to_pharmacy': 15
    }
    
    recommendations = recommender.recommend(test_patient, abandonment_risk=0.85)
    
    print("=== Test Patient: High Risk (85%) ===")
    print(f"Income: ${test_patient['median_income']:,}")
    print(f"OOP Cost: ${test_patient['oop_cost']:,}")
    print(f"Insurance: {test_patient['insurance_type']}")
    print(f"\nRecommended Interventions ({len(recommendations)}):\n")
    
    for rec in recommendations:
        print(f"{rec['priority']}. {rec['name']}")
        print(f"   Cost Reduction: ${rec['estimated_cost_reduction']:.2f}")
        print(f"   New OOP: ${rec['new_oop_cost']:.2f}")
        print(f"   Success Rate: {rec['success_probability']:.0%}")
        print(f"   Expected Impact: {rec['expected_impact']:.2f}")
        print(f"   Action: {rec['action_items'][0]}")
        print()
