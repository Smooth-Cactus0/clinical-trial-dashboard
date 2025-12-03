# Clinical Trial Analytics Dashboard

Advanced Phase III multi-center clinical trial analysis platform with statistical efficacy analysis, safety monitoring, and regulatory-grade reporting.

![Dashboard Preview](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

## 🔬 Features

### Efficacy Analysis
- **Time-Series Analysis** - Efficacy scores over 52 weeks with 95% confidence intervals
- **Response Rate Tracking** - Complete/Partial/Stable/Progressive disease classification
- **Waterfall Plots** - Individual patient best response visualization
- **Dose-Response Analysis** - Statistical comparison across treatment arms

### Safety Monitoring
- **Adverse Event Dashboard** - Severity-stratified AE tracking by treatment arm
- **Incidence Heatmaps** - Visual safety profile across event types
- **Laboratory Safety** - Liver enzyme (ALT) monitoring over time
- **Serious Adverse Events** - Critical event flagging and analysis

### Statistical Rigor
- 95% confidence intervals on all efficacy measurements
- P-value calculations for treatment comparisons
- Standard error visualization
- Response threshold annotations

### Study Operations
- **Enrollment Timeline** - Patient accrual tracking with cumulative curves
- **Site Performance** - Enrollment and completion rates by research center
- **Dropout Analysis** - Patient retention and attrition tracking
- **Demographics** - Population distribution by age, gender, baseline severity

## 📊 Dataset

Simulated Phase III clinical trial with:
- **450 patients** across 6 research sites
- **4 treatment arms** (Placebo, Low/Medium/High dose)
- **7 timepoints** (Baseline through Week 52)
- **~3,000 efficacy assessments**
- **~500 adverse events** with severity grading
- **~3,000 laboratory measurements**

Clear dose-response relationship:
- Placebo: 0% response rate
- Low Dose: 27% response rate
- Medium Dose: 66% response rate  
- High Dose: 80% response rate

## 🛠️ Tech Stack

- **Streamlit** - Interactive dashboard framework
- **Plotly** - Advanced scientific visualizations
- **Pandas** - Clinical data manipulation
- **NumPy** - Statistical computations
- **SciPy** - Statistical testing and confidence intervals

## 🎨 Visualization Types

- Line charts with confidence bands
- Stacked bar charts for response distribution
- Waterfall plots for individual patient response
- Heatmaps for adverse event incidence
- Horizontal bar charts for site comparison
- Histograms for demographic distributions
- Time-series with reference lines

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/clinical-trial-dashboard.git
cd clinical-trial-dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Generate sample clinical data:
```bash
python generate_clinical_data.py
```

4. Launch the dashboard:
```bash
streamlit run clinical_dashboard.py
```

## 📈 Use Cases

- **Pharmaceutical Companies** - Drug development efficacy and safety analysis
- **Clinical Research Organizations (CROs)** - Multi-site trial monitoring
- **Regulatory Affairs** - Submission-ready visualizations
- **Biostatistics** - Statistical analysis and reporting
- **Medical Affairs** - Clinical program oversight

## 📁 Data Files

The dashboard uses 5 interconnected clinical datasets:
- `trial_sites.csv` - Research center information
- `trial_patients.csv` - Patient demographics and enrollment
- `trial_efficacy.csv` - Longitudinal efficacy measurements
- `trial_adverse_events.csv` - Safety event tracking
- `trial_laboratory.csv` - Lab values over time

## 💼 Portfolio Highlights

This dashboard demonstrates:
- Advanced statistical analysis capabilities
- Medical/pharmaceutical domain expertise
- Regulatory-compliant visualizations
- Multi-dimensional clinical data modeling
- Interactive filtering and subgroup analysis
- Professional medical aesthetic design
- Understanding of clinical trial methodology

Perfect for roles in:
- Pharmaceutical data analytics
- Clinical data science
- Biostatistics
- Medical research
- Healthcare consulting

## 👤 Contact

**[Alexy Louis]**
- Email: alexy.louis.scholar@gmail.com
- LinkedIn: [Your LinkedIn]
- Portfolio: [Your Website]

## 📜 License

MIT License - Free to use for educational and portfolio purposes

## ⚠️ Disclaimer

This dashboard uses simulated clinical trial data for demonstration purposes only. Not for actual clinical or regulatory use.
```

### **.gitignore**
```
# Python
__pycache__/
*.py[cod]
*$py.class
.Python
env/
venv/

# Streamlit
.streamlit/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db