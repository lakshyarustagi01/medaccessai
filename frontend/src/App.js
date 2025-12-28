import React, { useState } from 'react';
import axios from 'axios';
import { AlertCircle, TrendingUp, DollarSign, Clock } from 'lucide-react';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [formData, setFormData] = useState({
    patient_id: '',
    age: 55,
    zip_code: '78701',
    median_income: 65000,
    insurance_type: 'Commercial',
    drug_name: 'Keytruda',
    therapeutic_area: 'Oncology',
    drug_cost: 12000,
    oop_cost: 2400,
    distance_to_pharmacy: 8.5,
    pa_required: true,
    prior_abandonment_count: 0,
    prescription_date: new Date().toISOString().split('T')[0]
  });

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
      console.error('API Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : 
              type === 'number' ? parseFloat(value) : value
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
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label>Median Income (zip code) *</label>
                <input
                  type="number"
                  name="median_income"
                  value={formData.median_income}
                  onChange={handleChange}
                  required
                  step="1000"
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

              <div className="form-group">
                <label>Drug Name *</label>
                <select
                  name="drug_name"
                  value={formData.drug_name}
                  onChange={handleChange}
                  required
                >
                  <option value="Keytruda">Keytruda (Oncology)</option>
                  <option value="Ocrevus">Ocrevus (MS)</option>
                  <option value="Humira">Humira (Immunology)</option>
                  <option value="Xolair">Xolair (Asthma)</option>
                  <option value="Enbrel">Enbrel (RA)</option>
                </select>
              </div>

              <div className="form-group">
                <label>Therapeutic Area *</label>
                <input
                  type="text"
                  name="therapeutic_area"
                  value={formData.therapeutic_area}
                  onChange={handleChange}
                  required
                />
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
