// Simple ZIP to income lookup using our backend
export const fetchIncomeForZip = async (zip) => {
  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  
  if (!/^\d{5}$/.test(zip)) return null;
  
  try {
    const response = await fetch(`${API_URL}/zip-income/${zip}`);
    const data = await response.json();
    return data.median_income;
  } catch (error) {
    console.error('ZIP lookup failed:', error);
    return null;
  }
};
