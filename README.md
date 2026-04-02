# 🎯 Customer Segmentation Analytics

## Project Overview
**Objective:** Categorize customers based on behaviour and demographics using clustering techniques.

**Expected Outcome:** Customer groups for targeted marketing strategies.

---

## 📁 Project Files

| File | Description |
|------|-------------|
| `customer_segmentation.py` | Complete Python analysis script |
| `Customer_Segmentation_Presentation.ipynb` | Interactive Jupyter notebook for presentation |
| `Mall_Customers.csv` | Source dataset (200 customers) |
| `Customer_Segments_Result.csv` | Output: Customers with segment assignments |
| `Cluster_Summary.csv` | Output: Segment statistics summary |

### Visualizations Generated
- `EDA_Visualizations.png` - Data distributions and correlations
- `Cluster_Optimization.png` - Elbow method & silhouette analysis
- `Cluster_Profiles.png` - Segment characteristics comparison
- `PCA_Clusters.png` - 2D PCA visualization of segments
- `Income_vs_Spending_Clusters.png` - Main segmentation plot

---

## 🚀 How to Run

### Option 1: Run Python Script (Quick)
```bash
cd C:\Users\shyle\OneDrive\Desktop\DA-Customer-Segmentation
python customer_segmentation.py
```

### Option 2: Interactive Presentation (Recommended for Demo)
```bash
# Open in Jupyter Notebook
jupyter notebook Customer_Segmentation_Presentation.ipynb

# Or open in VS Code and run cells sequentially (Shift+Enter)
```

### Option 3: View Pre-generated Results
Simply open the CSV files and PNG images directly - no code execution needed!

---

## 📊 Key Findings

### 5 Customer Segments Identified:

| Cluster | Segment | Size | Profile |
|---------|---------|------|---------|
| 0 | Budget-Conscious, Conservative | 10% | Low income, low spending, older |
| 1 | Young Active Spenders | 27% | Mid income, high spending, youngest |
| 2 | High Earners, Active Spenders | 20% | High income, high spending |
| 3 | High Earners, Conservative | 19.5% | High income, low spending |
| 4 | Mature Balanced Spenders | 23.5% | Mid income, moderate spending, oldest |

---

## 🎯 Marketing Recommendations

### Priority Actions:
1. **Cluster 2** (Highest Priority): Target with premium/luxury campaigns
2. **Cluster 1** (High Priority): Retention & loyalty programs for largest segment
3. **Cluster 3** (High Priority): Value proposition campaigns to unlock spending
4. **Clusters 0 & 4** (Medium): Bundle deals and essential products

---

## 📋 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter (for notebook)
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

---

## 👤 Author
Customer Segmentation Analysis Project

---

## 📞 For Questions
Present this notebook to walk through the analysis step-by-step with your coordinator.
