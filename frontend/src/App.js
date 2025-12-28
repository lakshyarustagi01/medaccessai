import React, { useState } from 'react';
import axios from 'axios';
import { AlertCircle, TrendingUp, DollarSign, Clock } from 'lucide-react';
import './App.css';
import { fetchIncomeForZip } from './zipLookup';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const DRUG_DATABASE = {
  'Oncology': [
    { name: 'Keytruda', cost: 12000 },
    { name: 'Opdivo', cost: 13500 },
    { name: 'Imbruvica', cost: 14000 },
    { name: 'Revlimid', cost: 18000 },
    { name: 'Tecentriq', cost: 12500 },
    { name: 'Darzalex', cost: 15000 },
    { name: 'Zytiga', cost: 11000 }
  ],
  'Neurology': [
    { name: 'Ocrevus', cost: 16250 },
    { name: 'Tysabri', cost: 14500 },
    { name: 'Gilenya', cost: 8500 },
    { name: 'Tecfidera', cost: 7500 },
    { name: 'Copaxone', cost: 6800 },
    { name: 'Aubagio', cost: 7200 }
  ],
  'Immunology': [
    { name: 'Humira', cost: 7000 },
    { name: 'Stelara', cost: 12500 },
    { name: 'Cosentyx', cost: 6500 },
    { name: 'Skyrizi', cost: 13000 },
    { name: 'Taltz', cost: 6200 },
    { name: 'Dupixent', cost: 3700 }
  ],
  'Rheumatology': [
    { name: 'Enbrel', cost: 6000 },
    { name: 'Orencia', cost: 4500 },
    { name: 'Actemra', cost: 4800 },
    { name: 'Rinvoq', cost: 5900 },
    { name: 'Simponi', cost: 5200 }
  ],
  'Pulmonology': [
    { name: 'Xolair', cost: 3000 },
    { name: 'Nucala', cost: 3200 },
    { name: 'Fasenra', cost: 3500 },
    { name: 'Trikafta', cost: 26000 },
    { name: 'Spiriva', cost: 450 }
  ],
  'Cardiology': [
    { name: 'Entresto', cost: 550 },
    { name: 'Eliquis', cost: 520 },
    { name: 'Xarelto', cost: 500 },
    { name: 'Praluent', cost: 5800 },
    { name: 'Repatha', cost: 5850 }
  ],
  'Gastroenterology': [
    { name: 'Entyvio', cost: 8500 },
    { name: 'Stelara', cost: 12500 },
    { name: 'Remicade', cost: 5000 },
    { name: 'Rinvoq', cost: 5900 }
  ],
  'Dermatology': [
    { name: 'Otezla', cost: 4200 },
    { name: 'Dupixent', cost: 3700 },
    { name: 'Cosentyx', cost: 6500 },
    { name: 'Tremfya', cost: 11000 }
  ],
  'Hematology': [
    { name: 'Eliquis', cost: 520 },
    { name: 'Xarelto', cost: 500 },
    { name: 'Pomalyst', cost: 16500 },
    { name: 'Revlimid', cost: 18000 }
  ],
  'Endocrinology': [
    { name: 'Ozempic', cost: 935 },
    { name: 'Trulicity', cost: 890 },
    { name: 'Januvia', cost: 550 },
    { name: 'Victoza', cost: 900 }
  ]
};

function App() {
  const [formData, setFormData] = useState({
    patient_id: '',
    age: 55,
    zip_code: '78701',
    median_income: 65000,
    insurance_type: 'Commercial',
    drug_name: '',
    therapeutic_area: '',
    drug_cost: 0,
    oop_cost: 2400,
    distance_to_pharmacy: 8.5,
    pa_required: true,
    prior_abandonment_count: 0,
    
    // NEW FIELDS
    deductible_met: false,
    deductible_remaining: 3500,
    prescription_month: new Date().getMonth() + 1,
    is_refill: false,
    days_since_prescription: 0,
    administration_route: 'oral',
    specialty_pharmacy_required: true,
    lives_alone: false,
    primary_language: 'English',
    has_caregiver: false
  });

  const [availableDrugs, setAvailableDrugs] = useState([]);
  const [incomeEdited, setIncomeEdited] = useState(false);
  const [loadingIncome, setLoadingIncome] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_URL}/predict`, formData);
      setPrediction(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    let newValue = type === 'checkbox' ? checked : (type === 'number' ? parseFloat(value) : value);
    
    setFormData(prev => ({
      ...prev,
      [name]: newValue
    }));
    
    // If deductible is met, set remaining to 0
    if (name === 'deductible_met' && checked) {
      setFormData(prev => ({
        ...prev,
        deductible_remaining: 0
      }));
    }
  };

  const handleTherapeuticAreaChange = (e) => {
    const area = e.target.value;
    setFormData(prev => ({
      ...prev,
      therapeutic_area: area,
      drug_name: '',
      drug_cost: 0
    }));
    
    if (area && DRUG_DATABASE[area]) {
      setAvailableDrugs(DRUG_DATABASE[area]);
    } else {
      setAvailableDrugs([]);
    }
  };

  const handleDrugChange = (e) => {
    const drugName = e.target.value;
    const selectedDrug = availableDrugs.find(d => d.name === drugName);
    
    if (selectedDrug) {
      setFormData(prev => ({
        ...prev,
        drug_name: drugName,
        drug_cost: selectedDrug.cost
      }));
    }
  };

  const handleZipChange = async (e) => {
    const zip = e.target.value;
    setFormData(prev => ({
      ...prev,
      zip_code: zip
    }));

    if (/^\d{5}$/.test(zip) && !incomeEdited) {
      setLoadingIncome(true);
      const income = await fetchIncomeForZip(zip);
      if (income) {
        setFormData(prev => ({
          ...prev,
          median_income: income
        }));
      }
      setLoadingIncome(false);
    }
  };

  const handleIncomeChange = (e) => {
    setIncomeEdited(true);
    setFormData(prev => ({
      ...prev,
      median_income: parseFloat(e.target.value)
    }));
  };

  const getRiskColor = (risk) => {
    if (risk < 0.3) return '#22c55e';
    if (risk < 0.5) return '#eab308';
    if (risk < 0.75) return '#f97316';
    return '#ef4444';
  };

  const getRiskBadgeClass = (level) => {
    const classes = {
      'LOW': 'risk-badge-low',
      'MEDIUM': 'risk-badge-medium',
      'HIGH': 'risk-badge-high',
      'CRITICAL': 'risk-badge-critical'
    };
    return classes[level] || 'risk-badge-medium';
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>MedAccessAI</h1>
        <p>Medication Abandonment Prediction & Intervention Platform</p>
      </header>

      <div className="container">
        <div className="form-section">
          <h2>Patient & Prescription Information</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-grid">
              {/* Basic Info */}
              <div className="form-group">
                <label>Patient ID (optional)</label>
                <input
                  type="text"
                  name="patient_id"
                  value={formData.patient_id}
                  onChange={handleChange}
                  placeholder="e.g., PAT12345"
                />
              </div>

              <div className="form-group">
                <label>Age *</label>
                <input
                  type="number"
                  name="age"
                  value={formData.age}
                  onChange={handleChange}
                  required
                  min="18"
                  max="120"
                />
              </div>

              <div className="form-group">
                <label>Zip Code *</label>
                <input
                  type="text"
                  name="zip_code"
                  value={formData.zip_code}
                  onChange={handleZipChange}
                  required
                  placeholder="Enter 5-digit ZIP"
                  maxLength="5"
                />
              </div>

              <div className="form-group">
                <label>
                  Median Income (zip code) *
                  {loadingIncome && (
                    <span style={{fontSize: '12px', color: '#666', marginLeft: '8px'}}>
                      (looking up...)
                    </span>
                  )}
                  {!loadingIncome && !incomeEdited && (
                    <span style={{fontSize: '12px', color: '#666', marginLeft: '8px'}}>
                      (from ZIP, editable)
                    </span>
                  )}
                </label>
                <input
                  type="number"
                  name="median_income"
                  value={formData.median_income}
                  onChange={handleIncomeChange}
                  required
                  step="1000"
                  placeholder="Auto-fills from ZIP"
                />
              </div>

              <div className="form-group">
                <label>Insurance Type *</label>
                <select
                  name="insurance_type"
                  value={formData.insurance_type}
                  onChange={handleChange}
                  required
                >
                  <option value="Commercial">Commercial</option>
                  <option value="Medicare">Medicare</option>
                  <option value="Medicaid">Medicaid</option>
                  <option value="Uninsured">Uninsured</option>
                </select>
              </div>

              {/* NEW: Deductible Status */}
              <div className="form-group checkbox-group">
                <label>
                  <input
                    type="checkbox"
                    name="deductible_met"
                    checked={formData.deductible_met}
                    onChange={handleChange}
                  />
                  Patient Has Met Annual Deductible
                </label>
              </div>

              {!formData.deductible_met && (
                <div className="form-group">
                  <label>Deductible Remaining ($) *</label>
                  <input
                    type="number"
                    name="deductible_remaining"
                    value={formData.deductible_remaining}
                    onChange={handleChange}
                    required
                    step="100"
                    min="0"
                  />
                </div>
              )}

              {/* Medication Info */}
              <div className="form-group">
                <label>Therapeutic Area *</label>
                <select
                  name="therapeutic_area"
                  value={formData.therapeutic_area}
                  onChange={handleTherapeuticAreaChange}
                  required
                >
                  <option value="">-- Select Therapeutic Area --</option>
                  <option value="Oncology">Oncology</option>
                  <option value="Neurology">Neurology</option>
                  <option value="Immunology">Immunology</option>
                  <option value="Rheumatology">Rheumatology</option>
                  <option value="Pulmonology">Pulmonology</option>
                  <option value="Cardiology">Cardiology</option>
                  <option value="Gastroenterology">Gastroenterology</option>
                  <option value="Dermatology">Dermatology</option>
                  <option value="Hematology">Hematology</option>
                  <option value="Endocrinology">Endocrinology</option>
                </select>
              </div>

              <div className="form-group">
                <label>Drug Name *</label>
                <select
                  name="drug_name"
                  value={formData.drug_name}
                  onChange={handleDrugChange}
                  required
                  disabled={!formData.therapeutic_area}
                >
                  <option value="">
                    {formData.therapeutic_area ? '-- Select Drug --' : '-- Select Therapeutic Area First --'}
                  </option>
                  {availableDrugs.map(drug => (
                    <option key={drug.name} value={drug.name}>
                      {drug.name} (${drug.cost.toLocaleString()})
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label>Drug Cost (wholesale) *</label>
                <input
                  type="number"
                  name="drug_cost"
                  value={formData.drug_cost}
                  onChange={handleChange}
                  required
                  step="100"
                />
              </div>

              <div className="form-group">
                <label>Out-of-Pocket Cost *</label>
                <input
                  type="number"
                  name="oop_cost"
                  value={formData.oop_cost}
                  onChange={handleChange}
                  required
                  step="10"
                />
              </div>

              {/* NEW: Administration Route */}
              <div className="form-group">
                <label>Administration Route *</label>
                <select
                  name="administration_route"
                  value={formData.administration_route}
                  onChange={handleChange}
                  required
                >
                  <option value="oral">Oral (pill/capsule)</option>
                  <option value="injection">Self-injection</option>
                  <option value="infusion">Infusion (clinic/hospital)</option>
                </select>
              </div>

              {/* NEW: Prescription Timing */}
              <div className="form-group">
                <label>Prescription Month *</label>
                <select
                  name="prescription_month"
                  value={formData.prescription_month}
                  onChange={handleChange}
                  required
                >
                  <option value="1">January (deductible reset)</option>
                  <option value="2">February</option>
                  <option value="3">March</option>
                  <option value="4">April</option>
                  <option value="5">May</option>
                  <option value="6">June</option>
                  <option value="7">July</option>
                  <option value="8">August</option>
                  <option value="9">September</option>
                  <option value="10">October</option>
                  <option value="11">November</option>
                  <option value="12">December</option>
                </select>
              </div>

              <div className="form-group">
                <label>Days Since Prescription Written *</label>
                <input
                  type="number"
                  name="days_since_prescription"
                  value={formData.days_since_prescription}
                  onChange={handleChange}
                  required
                  min="0"
                  max="30"
                />
              </div>

              {/* NEW: Patient Type */}
              <div className="form-group checkbox-group">
                <label>
                  <input
                    type="checkbox"
                    name="is_refill"
                    checked={formData.is_refill}
                    onChange={handleChange}
                  />
                  This is a Refill (not new prescription)
                </label>
              </div>

              {/* Existing Fields */}
              <div className="form-group">
                <label>Distance to Pharmacy (miles) *</label>
                <input
                  type="number"
                  name="distance_to_pharmacy"
                  value={formData.distance_to_pharmacy}
                  onChange={handleChange}
                  required
                  step="0.1"
                />
              </div>

              <div className="form-group">
                <label>Prior Abandonments</label>
                <input
                  type="number"
                  name="prior_abandonment_count"
                  value={formData.prior_abandonment_count}
                  onChange={handleChange}
                  min="0"
                />
              </div>

              <div className="form-group checkbox-group">
                <label>
                  <input
                    type="checkbox"
                    name="pa_required"
                    checked={formData.pa_required}
                    onChange={handleChange}
                  />
                  Prior Authorization Required
                </label>
              </div>

              <div className="form-group checkbox-group">
                <label>
                  <input
                    type="checkbox"
                    name="specialty_pharmacy_required"
                    checked={formData.specialty_pharmacy_required}
                    onChange={handleChange}
                  />
                  Must Use Specialty Pharmacy
                </label>
              </div>

              {/* NEW: Social Support */}
              <div className="form-group">
                <label>Primary Language *</label>
                <select
                  name="primary_language"
                  value={formData.primary_language}
                  onChange={handleChange}
                  required
                >
                  <option value="English">English</option>
                  <option value="Spanish">Spanish</option>
                  <option value="Other">Other</option>
                </select>
              </div>

              <div className="form-group checkbox-group">
                <label>
                  <input
                    type="checkbox"
                    name="lives_alone"
                    checked={formData.lives_alone}
                    onChange={handleChange}
                  />
                  Patient Lives Alone
                </label>
              </div>

              <div className="form-group checkbox-group">
                <label>
                  <input
                    type="checkbox"
                    name="has_caregiver"
                    checked={formData.has_caregiver}
                    onChange={handleChange}
                  />
                  Has Caregiver/Family Support
                </label>
              </div>
            </div>

            <button type="submit" className="submit-btn" disabled={loading}>
              {loading ? 'Analyzing...' : 'Predict Abandonment Risk'}
            </button>
          </form>
        </div>

        {error && (
          <div className="error-message">
            <AlertCircle size={20} />
            <span>{error}</span>
          </div>
        )}

        {prediction && (
          <div className="results-section">
            <h2>Prediction Results</h2>
            
            <div className="risk-overview">
              <div className="risk-gauge">
                <div className="gauge-container">
                  <svg viewBox="0 0 200 120" className="gauge-svg">
                    <path
                      d="M 20 100 A 80 80 0 0 1 180 100"
                      fill="none"
                      stroke="#e5e7eb"
                      strokeWidth="20"
                    />
                    <path
                      d="M 20 100 A 80 80 0 0 1 180 100"
                      fill="none"
                      stroke={getRiskColor(prediction.abandonment_risk)}
                      strokeWidth="20"
                      strokeDasharray={`${prediction.abandonment_risk * 251.2} 251.2`}
                    />
                    <text x="100" y="80" textAnchor="middle" fontSize="32" fontWeight="bold">
                      {(prediction.abandonment_risk * 100).toFixed(0)}%
                    </text>
                    <text x="100" y="105" textAnchor="middle" fontSize="14" fill="#666">
                      Abandonment Risk
                    </text>
                  </svg>
                </div>
                <div className={`risk-badge ${getRiskBadgeClass(prediction.risk_level)}`}>
                  {prediction.risk_level} RISK
                </div>
              </div>

              <div className="risk-factors">
                <h3>Key Risk Factors</h3>
                <ul>
                  {prediction.risk_factors.map((factor, idx) => (
                    <li key={idx}>
                      <AlertCircle size={16} />
                      <span>{factor}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            {prediction.recommendations.length > 0 && (
              <div className="recommendations">
                <h3>Recommended Interventions</h3>
                <div className="intervention-cards">
                  {prediction.recommendations.map((rec, idx) => (
                    <div key={idx} className="intervention-card">
                      <div className="intervention-header">
                        <span className="priority-badge">#{rec.priority}</span>
                        <h4>{rec.name}</h4>
                      </div>
                      <p className="intervention-desc">{rec.description}</p>
                      
                      <div className="intervention-metrics">
                        <div className="metric">
                          <DollarSign size={16} />
                          <div>
                            <div className="metric-value">
                              ${rec.estimated_cost_reduction.toFixed(0)}
                            </div>
                            <div className="metric-label">Cost Reduction</div>
                          </div>
                        </div>
                        
                        <div className="metric">
                          <TrendingUp size={16} />
                          <div>
                            <div className="metric-value">
                              {(rec.success_probability * 100).toFixed(0)}%
                            </div>
                            <div className="metric-label">Success Rate</div>
                          </div>
                        </div>
                        
                        <div className="metric">
                          <Clock size={16} />
                          <div>
                            <div className="metric-value">
                              {rec.processing_time_days} days
                            </div>
                            <div className="metric-label">Processing Time</div>
                          </div>
                        </div>
                      </div>

                      <div className="new-cost">
                        New OOP Cost: <strong>${rec.new_oop_cost.toFixed(2)}</strong>
                      </div>

                      <div className="action-items">
                        <strong>Next Steps:</strong>
                        <ul>
                          {rec.action_items.map((action, i) => (
                            <li key={i}>{action}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      <footer className="app-footer">
        <p>MedAccessAI Demo v0.1 | For demonstration purposes only</p>
      </footer>
    </div>
  );
}

export default App;
