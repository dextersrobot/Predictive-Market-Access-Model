# Market Access Predictor

A machine learning model that predicts coverage barriers for newly approved drugs based on regulatory, clinical, and economic characteristics.

## 🎯 Project Purpose

This project demonstrates skills relevant to life sciences market access consulting by:
- Collecting and integrating multiple government data sources (FDA, CMS, Medicaid)
- Engineering features that capture payer decision-making factors
- Building predictive models for coverage outcomes
- Creating interactive visualizations for stakeholder communication

## 📊 Data Sources

| Source | Description | Access |
|--------|-------------|--------|
| openFDA Drugs@FDA | Drug approvals, review types, submission data | REST API |
| FDA Orphan Drug Database | Rare disease designations | Manual download |
| CMS Coverage API | National Coverage Determinations (NCDs) | REST API |
| NADAC (Medicaid.gov) | National drug acquisition costs | REST API |

## 🏗️ Project Structure

```
market-access-predictor/
├── data/
│   ├── raw/              # Downloaded source files
│   ├── processed/        # Cleaned, merged datasets  
│   └── features/         # Engineered feature sets
├── src/
│   ├── data_collection/  # API scripts for each source
│   ├── preprocessing/    # Data cleaning pipelines
│   ├── features/         # Feature engineering code
│   ├── models/           # Model training & evaluation
│   └── visualization/    # Streamlit dashboard
├── models/               # Saved model artifacts
├── notebooks/            # Exploratory analysis
├── tests/                # Unit tests
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/market-access-predictor.git
cd market-access-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Collect Data

```bash
# Quick mode (recent data only, ~5 minutes)
python src/data_collection/collect_all_data.py --quick

# Full mode (comprehensive historical data, ~30 minutes)
python src/data_collection/collect_all_data.py
```

### 3. Process & Engineer Features

```bash
# Merge data sources
python src/preprocessing/merge_data.py

# Engineer features
python src/features/engineer_features.py
```

### 4. Train Models

```bash
python src/models/train_model.py
```

### 5. Launch Dashboard

```bash
streamlit run src/visualization/dashboard.py
```

## 📈 Model Features

### Regulatory Features
- **Priority Review**: 6-month vs standard 10-month FDA review
- **Orphan Designation**: Rare disease status with 7-year exclusivity
- **New Molecular Entity**: First-in-class compounds
- **Accelerated Approval**: Based on surrogate endpoints

### Clinical Features
- **Therapeutic Area**: Oncology, immunology, neurology, etc.
- **Dosage Form**: Oral, injectable, complex formulations
- **Biologic Status**: BLA vs NDA submission type

### Economic Features
- **Price Tier**: Based on NADAC percentile distribution
- **Market Size Proxy**: Estimated patient population
- **Launch Price**: Log-transformed acquisition cost

### Temporal Features
- **Approval Year**: Trend effects over time
- **Quarter**: Payer budget cycle effects
- **Election Year**: Policy uncertainty indicator

## 🎯 Model Performance

| Model | Accuracy | Precision | Recall | ROC AUC |
|-------|----------|-----------|--------|---------|
| Logistic Regression | ~0.70 | ~0.68 | ~0.72 | ~0.75 |
| Random Forest | ~0.75 | ~0.73 | ~0.77 | ~0.82 |
| Gradient Boosting | ~0.76 | ~0.74 | ~0.78 | ~0.83 |

*Note: Performance varies based on data availability and target variable definition.*

## 📚 Key Concepts Learned

### Regulatory Pathways
- **Priority Review**: Faster FDA review for significant advances
- **Accelerated Approval**: Based on surrogate endpoints; requires confirmatory trials
- **Breakthrough Therapy**: Intensive FDA guidance for substantial improvements
- **Orphan Designation**: Rare disease incentives (<200K patients)

### Coverage Mechanisms
- **NCD**: National Coverage Determination (CMS policy for all Medicare)
- **LCD**: Local Coverage Determination (regional contractor decisions)
- **Prior Authorization**: Payer approval required before coverage
- **Step Therapy**: Must try cheaper alternatives first

### Pricing Concepts
- **WAC**: Wholesale Acquisition Cost (list price)
- **ASP**: Average Sales Price (Medicare Part B basis)
- **NADAC**: National Average Drug Acquisition Cost
- **ICER Threshold**: $50K-$150K per QALY for cost-effectiveness

## 🔮 Future Enhancements

1. **Real coverage data**: Integrate actual formulary data from commercial payers
2. **NLP for indications**: Extract therapeutic indications from drug labels
3. **ICER integration**: Scrape and incorporate value assessments
4. **Time-to-coverage prediction**: Regression model for coverage timeline
5. **Geographic analysis**: Predict LCD variations by MAC region

## 📝 Portfolio Presentation Tips

When discussing this project:

1. **Lead with the business problem**: "I built a model to predict drug coverage timelines—the exact question pharma companies ask consultants."

2. **Show domain knowledge**: Discuss which features matter most and why (e.g., "Orphan drugs face fewer access barriers because payers have limited alternatives").

3. **Acknowledge limitations**: "Public data can't capture rebates or contract negotiations, which significantly affect real-world access."

4. **Suggest extensions**: "With access to proprietary data, this model could incorporate payer-specific policies and real-time formulary changes."

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

- FDA for openFDA API and public data access
- CMS for Coverage API and NADAC pricing data
- ICER for value assessment methodology documentation
