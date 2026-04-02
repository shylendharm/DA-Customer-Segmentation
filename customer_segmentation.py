"""
Customer Segmentation Analytics
================================
This script performs customer segmentation using clustering techniques
to identify distinct customer groups for targeted marketing strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================================

print("=" * 60)
print("CUSTOMER SEGMENTATION ANALYTICS")
print("=" * 60)

# Load the dataset
df = pd.read_csv('Mall_Customers.csv')

print("\n[1] DATA OVERVIEW")
print("-" * 40)
print(f"Dataset Shape: {df.shape[0]} customers × {df.shape[1]} features")
print(f"\nColumn Names: {list(df.columns)}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nBasic Statistics:\n{df.describe()}")

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n[2] EXPLORATORY DATA ANALYSIS")
print("-" * 40)

# Gender distribution
gender_dist = df['Genre'].value_counts()
print(f"\nGender Distribution:\n{gender_dist}")

# Create comprehensive EDA visualizations
fig = plt.figure(figsize=(16, 12))

# Gender distribution
ax1 = plt.subplot(2, 3, 1)
plt.pie(gender_dist.values, labels=gender_dist.index, autopct='%1.1f%%', 
        colors=['#3498db', '#e74c3c'], startangle=90)
plt.title('Gender Distribution', fontsize=12, fontweight='bold')

# Age distribution
ax2 = plt.subplot(2, 3, 2)
plt.hist(df['Age'], bins=20, color='#2ecc71', edgecolor='black', alpha=0.7)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution', fontsize=12, fontweight='bold')

# Annual Income distribution
ax3 = plt.subplot(2, 3, 3)
plt.hist(df['Annual Income (k$)'], bins=20, color='#9b59b6', edgecolor='black', alpha=0.7)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Frequency')
plt.title('Annual Income Distribution', fontsize=12, fontweight='bold')

# Spending Score distribution
ax4 = plt.subplot(2, 3, 4)
plt.hist(df['Spending Score (1-100)'], bins=20, color='#e67e22', edgecolor='black', alpha=0.7)
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Frequency')
plt.title('Spending Score Distribution', fontsize=12, fontweight='bold')

# Box plots for outliers
ax5 = plt.subplot(2, 3, 5)
df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].boxplot(ax=ax5)
ax5.set_title('Outlier Detection (Box Plot)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Values')

# Correlation heatmap
ax6 = plt.subplot(2, 3, 6)
numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', ax=ax6, square=True)
ax6.set_title('Correlation Heatmap', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('EDA_Visualizations.png', dpi=300, bbox_inches='tight')
print("\n✓ EDA visualizations saved to 'EDA_Visualizations.png'")

# ============================================================================
# 3. FEATURE ENGINEERING & PREPROCESSING
# ============================================================================

print("\n[3] DATA PREPROCESSING")
print("-" * 40)

# Select features for clustering
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features].values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Features used for clustering: {features}")
print(f"Data standardized using StandardScaler")
print(f"Scaled data mean: {np.mean(X_scaled, axis=0).round(2)}")
print(f"Scaled data std: {np.std(X_scaled, axis=0).round(2)}")

# ============================================================================
# 4. DETERMINE OPTIMAL NUMBER OF CLUSTERS
# ============================================================================

print("\n[4] DETERMINING OPTIMAL NUMBER OF CLUSTERS")
print("-" * 40)

# Calculate inertia and silhouette scores for different k values
k_range = range(2, 11)
inertias = []
silhouette_scores = []
davies_bouldin_scores_list = []

for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = kmeans_temp.fit_predict(X_scaled)
    inertias.append(kmeans_temp.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels_temp))
    davies_bouldin_scores_list.append(davies_bouldin_score(X_scaled, labels_temp))

# Find optimal k using silhouette score
optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
optimal_k_db = k_range[np.argmin(davies_bouldin_scores_list)]

print(f"\nSilhouette Scores by K: {dict(zip(k_range, [round(s, 3) for s in silhouette_scores]))}")
print(f"Davies-Bouldin Scores by K: {dict(zip(k_range, [round(d, 3) for d in davies_bouldin_scores_list]))}")
print(f"\nOptimal K (Silhouette): {optimal_k_silhouette} (Score: {max(silhouette_scores):.3f})")
print(f"Optimal K (Davies-Bouldin): {optimal_k_db} (Score: {min(davies_bouldin_scores_list):.3f})")

# Create elbow method and silhouette score plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Elbow plot
axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)', fontsize=11)
axes[0].set_ylabel('Inertia', fontsize=11)
axes[0].set_title('Elbow Method', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Silhouette score plot
axes[1].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].axhline(y=max(silhouette_scores), color='r', linestyle='--', 
                label=f'Max: {max(silhouette_scores):.3f}')
axes[1].set_xlabel('Number of Clusters (K)', fontsize=11)
axes[1].set_ylabel('Silhouette Score', fontsize=11)
axes[1].set_title('Silhouette Analysis', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Davies-Bouldin score plot
axes[2].plot(k_range, davies_bouldin_scores_list, 'ro-', linewidth=2, markersize=8)
axes[2].axhline(y=min(davies_bouldin_scores_list), color='g', linestyle='--',
                label=f'Min: {min(davies_bouldin_scores_list):.3f}')
axes[2].set_xlabel('Number of Clusters (K)', fontsize=11)
axes[2].set_ylabel('Davies-Bouldin Score', fontsize=11)
axes[2].set_title('Davies-Bouldin Index', fontsize=12, fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Cluster_Optimization.png', dpi=300, bbox_inches='tight')
print("\n✓ Cluster optimization plots saved to 'Cluster_Optimization.png'")

# Use 5 clusters as it's commonly optimal for customer segmentation
# and provides good balance between granularity and interpretability
n_clusters = 5
print(f"\nSelected K = {n_clusters} for final clustering")

# ============================================================================
# 5. APPLY K-MEANS CLUSTERING
# ============================================================================

print("\n[5] APPLYING K-MEANS CLUSTERING")
print("-" * 40)

# Final K-Means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Add cluster labels based on characteristics
cluster_labels = {
    0: 'High Income, Low Spending',
    1: 'Low Income, Low Spending', 
    2: 'High Income, High Spending',
    3: 'Low Income, High Spending',
    4: 'Medium Income, Medium Spending'
}

print(f"Clustering completed with {n_clusters} clusters")
print(f"\nCluster Distribution:")
print(df['Cluster'].value_counts().sort_index())

# ============================================================================
# 6. CLUSTER PROFILING
# ============================================================================

print("\n[6] CLUSTER PROFILING")
print("-" * 40)

# Calculate cluster statistics
cluster_stats = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 
                                        'Spending Score (1-100)']].agg(['mean', 'std', 'count'])
print("\nCluster Statistics (Mean ± Std):")
print(cluster_stats)

# Gender distribution per cluster
gender_cluster = pd.crosstab(df['Cluster'], df['Genre'], normalize='index') * 100
print(f"\nGender Distribution per Cluster (%):")
print(gender_cluster.round(1))

# Create cluster profile visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Cluster centers comparison
cluster_means = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 
                                        'Spending Score (1-100)']].mean()

x = np.arange(len(cluster_means))
width = 0.25

axes[0, 0].bar(x - width, cluster_means['Age'], width, label='Age', color='#3498db')
axes[0, 0].bar(x, cluster_means['Annual Income (k$)'], width, 
               label='Annual Income (k$)', color='#2ecc71')
axes[0, 0].bar(x + width, cluster_means['Spending Score (1-100)'], width,
               label='Spending Score', color='#e74c3c')
axes[0, 0].set_xlabel('Cluster', fontsize=11)
axes[0, 0].set_ylabel('Mean Values', fontsize=11)
axes[0, 0].set_title('Cluster Characteristics Comparison', fontsize=12, fontweight='bold')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels([f'C{i}' for i in range(n_clusters)])
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Cluster size distribution
cluster_sizes = df['Cluster'].value_counts().sort_index()
colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))
axes[0, 1].bar(cluster_sizes.index.astype(str), cluster_sizes.values, 
               color=colors, edgecolor='black')
axes[0, 1].set_xlabel('Cluster', fontsize=11)
axes[0, 1].set_ylabel('Number of Customers', fontsize=11)
axes[0, 1].set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
for i, v in enumerate(cluster_sizes.values):
    axes[0, 1].text(i, v + 2, str(v), ha='center', fontweight='bold')

# Gender distribution per cluster
gender_cluster_plot = gender_cluster.plot(kind='bar', ax=axes[1, 0], 
                                           color=['#3498db', '#e74c3c'])
axes[1, 0].set_xlabel('Cluster', fontsize=11)
axes[1, 0].set_ylabel('Percentage (%)', fontsize=11)
axes[1, 0].set_title('Gender Distribution per Cluster', fontsize=12, fontweight='bold')
axes[1, 0].legend(title='Gender')
axes[1, 0].set_xticklabels([f'C{i}' for i in range(n_clusters)], rotation=0)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Age distribution per cluster
df.boxplot(column='Age', by='Cluster', ax=axes[1, 1])
axes[1, 1].set_xlabel('Cluster', fontsize=11)
axes[1, 1].set_ylabel('Age', fontsize=11)
axes[1, 1].set_title('Age Distribution per Cluster', fontsize=12, fontweight='bold')
plt.suptitle('')  # Remove automatic title

plt.tight_layout()
plt.savefig('Cluster_Profiles.png', dpi=300, bbox_inches='tight')
print("\n✓ Cluster profile visualizations saved to 'Cluster_Profiles.png'")

# ============================================================================
# 7. 2D VISUALIZATION USING PCA
# ============================================================================

print("\n[7] PCA VISUALIZATION")
print("-" * 40)

# Apply PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA Explained Variance Ratio:")
print(f"  PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"  PC2: {pca.explained_variance_ratio_[1]:.2%}")
print(f"  Total: {sum(pca.explained_variance_ratio_):.2%}")

# Create PCA scatter plot
fig, ax = plt.subplots(figsize=(10, 8))

cluster_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

for cluster in range(n_clusters):
    mask = df['Cluster'] == cluster
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               c=cluster_colors[cluster], label=f'Cluster {cluster}',
               s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

# Plot cluster centers in PCA space
centers_pca = pca.transform(kmeans.cluster_centers_)
ax.scatter(centers_pca[:, 0], centers_pca[:, 1], 
           c='red', s=300, marker='X', 
           label='Cluster Centers', edgecolors='black', linewidth=2)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
ax.set_title('Customer Segments (PCA Visualization)', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('PCA_Clusters.png', dpi=300, bbox_inches='tight')
print("\n✓ PCA visualization saved to 'PCA_Clusters.png'")

# ============================================================================
# 8. 2D SCATTER PLOTS (Income vs Spending)
# ============================================================================

print("\n[8] INCOME vs SPENDING ANALYSIS")
print("-" * 40)

# Create detailed scatter plot with cluster assignments
fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'],
                     c=df['Cluster'].map(dict(enumerate(cluster_colors))),
                     s=100, alpha=0.7, edgecolors='black', linewidth=0.5, cmap='Set2')

# Plot cluster centers
ax.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2],
           c='red', s=400, marker='X', label='Cluster Centers',
           edgecolors='black', linewidth=2)

ax.set_xlabel('Annual Income (k$)', fontsize=12)
ax.set_ylabel('Spending Score (1-100)', fontsize=12)
ax.set_title('Customer Segments: Income vs Spending Behavior', fontsize=14, fontweight='bold')
ax.legend(title='Cluster', loc='best')
ax.grid(True, alpha=0.3)

# Add quadrant lines
ax.axvline(x=df['Annual Income (k$)'].median(), color='gray', 
           linestyle='--', alpha=0.5, label='Median Income')
ax.axhline(y=df['Spending Score (1-100)'].median(), color='gray',
           linestyle='--', alpha=0.5, label='Median Spending')

plt.tight_layout()
plt.savefig('Income_vs_Spending_Clusters.png', dpi=300, bbox_inches='tight')
print("\n✓ Income vs Spending visualization saved to 'Income_vs_Spending_Clusters.png'")

# ============================================================================
# 9. DETAILED CLUSTER ANALYSIS & MARKETING RECOMMENDATIONS
# ============================================================================

print("\n[9] CLUSTER INTERPRETATION & MARKETING STRATEGIES")
print("=" * 60)

# Analyze each cluster in detail
cluster_analysis = df.groupby('Cluster').agg({
    'Age': ['mean', 'min', 'max'],
    'Annual Income (k$)': ['mean', 'min', 'max'],
    'Spending Score (1-100)': ['mean', 'min', 'max'],
    'Genre': 'count',
    'CustomerID': 'count'
}).round(2)

# Rename columns for clarity
cluster_analysis.columns = ['Age_Mean', 'Age_Min', 'Age_Max',
                            'Income_Mean', 'Income_Min', 'Income_Max',
                            'Spending_Mean', 'Spending_Min', 'Spending_Max',
                            'Genre_Count', 'Total_Customers']

print("\nDETAILED CLUSTER ANALYSIS:")
print("-" * 60)

# Define segment names based on characteristics
segment_names = []

for cluster_id in range(n_clusters):
    cluster_data = df[df['Cluster'] == cluster_id]
    
    avg_income = cluster_data['Annual Income (k$)'].mean()
    avg_spending = cluster_data['Spending Score (1-100)'].mean()
    avg_age = cluster_data['Age'].mean()
    
    # Determine segment characteristics
    income_level = "High" if avg_income > 60 else "Medium" if avg_income > 40 else "Low"
    spending_level = "High" if avg_spending > 60 else "Medium" if avg_spending > 40 else "Low"
    
    # Create segment name
    if income_level == "High" and spending_level == "Low":
        segment_name = "High Earners, Conservative Spenders"
        strategy = """
        📊 STRATEGY: Focus on value proposition and long-term investments
        - Emphasize quality and durability
        - Promote savings and investment products
        - Offer premium but practical items
        - Use data-driven, logical marketing appeals"""
    elif income_level == "High" and spending_level == "High":
        segment_name = "High Earners, Active Spenders"
        strategy = """
        📊 STRATEGY: Premium products and exclusive offers
        - Target with luxury and premium products
        - Offer exclusive memberships and VIP treatment
        - Promote latest trends and new arrivals
        - Use aspirational marketing messages"""
    elif income_level == "Low" and spending_level == "High":
        segment_name = "Budget-Conscious, Active Spenders"
        strategy = """
        📊 STRATEGY: Discounts and value-for-money offers
        - Focus on discounts, sales, and promotions
        - Offer installment payment options
        - Highlight affordability and value
        - Use urgency-based marketing (limited time offers)"""
    elif income_level == "Low" and spending_level == "Low":
        segment_name = "Budget-Conscious, Conservative Spenders"
        strategy = """
        📊 STRATEGY: Essential products and maximum savings
        - Promote essential, everyday products
        - Emphasize maximum savings and best deals
        - Offer bundle deals and bulk discounts
        - Use practical, needs-based marketing"""
    else:  # Medium income/spending
        segment_name = "Moderate Earners, Balanced Spenders"
        strategy = """
        📊 STRATEGY: Balanced value and quality offerings
        - Offer mid-range products with good value
        - Promote seasonal sales and rewards programs
        - Focus on quality-to-price ratio
        - Use balanced emotional and rational appeals"""
    
    segment_names.append(segment_name)
    
    print(f"\n{'='*60}")
    print(f"CLUSTER {cluster_id}: {segment_name}")
    print(f"{'='*60}")
    print(f"  📈 Size: {len(cluster_data)} customers ({len(cluster_data)/len(df)*100:.1f}%)")
    print(f"  👤 Avg Age: {avg_age:.1f} years (Range: {cluster_data['Age'].min()}-{cluster_data['Age'].max()})")
    print(f"  💰 Avg Income: ${avg_income:.1f}k (Range: ${cluster_data['Annual Income (k$)'].min()}-${cluster_data['Annual Income (k$)'].max()}k)")
    print(f"  🛒 Avg Spending Score: {avg_spending:.1f} (Range: {cluster_data['Spending Score (1-100)'].min()}-{cluster_data['Spending Score (1-100)'].max()})")
    
    # Gender breakdown
    male_pct = (cluster_data['Genre'] == 'Male').sum() / len(cluster_data) * 100
    female_pct = 100 - male_pct
    print(f"  ⚧ Gender: {female_pct:.1f}% Female, {male_pct:.1f}% Male")
    print(f"\n  🎯 MARKETING STRATEGY:{strategy}")

# ============================================================================
# 10. SAVE RESULTS
# ============================================================================

print("\n\n[10] SAVING RESULTS")
print("-" * 60)

# Save segmented data
df.to_csv('Customer_Segments_Result.csv', index=False)
print("✓ Segmented customer data saved to 'Customer_Segments_Result.csv'")

# Save cluster summary
cluster_summary = df.groupby('Cluster').agg({
    'Age': 'mean',
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean',
    'CustomerID': 'count'
}).round(2)
cluster_summary.columns = ['Avg_Age', 'Avg_Annual_Income_k', 'Avg_Spending_Score', 'Customer_Count']
cluster_summary['Segment_Name'] = segment_names
cluster_summary.to_csv('Cluster_Summary.csv')
print("✓ Cluster summary saved to 'Cluster_Summary.csv'")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)
print(f"""
📊 KEY FINDINGS:
   • Total Customers Analyzed: {len(df)}
   • Optimal Number of Segments: {n_clusters}
   • Clustering Algorithm: K-Means
   • Features Used: Age, Annual Income, Spending Score

📁 OUTPUT FILES GENERATED:
   1. EDA_Visualizations.png - Exploratory data analysis charts
   2. Cluster_Optimization.png - Elbow method and validation metrics
   3. Cluster_Profiles.png - Cluster characteristics comparison
   4. PCA_Clusters.png - 2D PCA visualization of segments
   5. Income_vs_Spending_Clusters.png - Income vs Spending scatter plot
   6. Customer_Segments_Result.csv - Full data with cluster assignments
   7. Cluster_Summary.csv - Summary statistics per segment

💡 RECOMMENDATIONS:
   • Use the 5 identified segments for targeted marketing campaigns
   • Tailor product recommendations based on segment characteristics
   • Develop segment-specific pricing and promotion strategies
   • Monitor segment migration over time for customer lifecycle management
""")

# Display final cluster summary table
print("\n📋 CLUSTER SUMMARY TABLE:")
print("-" * 60)
print(cluster_summary.to_string())
print("\n")
