import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from scipy.fft import fft
from scipy.stats import iqr

# قراءة البيانات من ملف CSV
try:
    df = pd.read_csv('sensor_data.csv', parse_dates=['timestamp'], index_col='timestamp')
except FileNotFoundError:
    print("Error: Please ensure 'sensor_data.csv' exists with columns: timestamp, temperature, vibration, pressure")
    exit()

# ------------------------------
# 1. تحليل البيانات المتقدم (Advanced Data Analysis)
# ------------------------------

# أ. تحليل السلاسل الزمنية لجميع الميزات
features = ['temperature', 'vibration', 'pressure']
for feature in features:
    decomposition = seasonal_decompose(df[feature].dropna(), model='additive', period=60)
    fig = decomposition.plot()
    plt.suptitle(f'Time-Series Decomposition for {feature.capitalize()}', y=1.05)
    plt.savefig(f'time_series_decomposition_{feature}.png')
    plt.close()

# ب. PCA لتقليل الأبعاد
X = df[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
explained_variance = pca.explained_variance_ratio_
components = pca.components_

# تحليل أوزان المكونات
pc1_weights = dict(zip(features, components[0]))
pc2_weights = dict(zip(features, components[1]))
print("PCA Components Weights:\nPC1:", pc1_weights, "\nPC2:", pc2_weights)

# رسم التباين المفسر (تحسين إضافي)
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Principal Components')
plt.savefig('pca_explained_variance.png')
plt.close()

# ج. اكتشاف الشذوذات باستخدام Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df['anomaly_iso'] = iso_forest.fit_predict(X_scaled)
anomalies_iso = df[df['anomaly_iso'] == -1]
print("Detected Anomalies (Isolation Forest):\n", anomalies_iso[features])

# د. اكتشاف الشذوذات باستخدام Z-Score لكل ميزة
for feature in features:
    df[f'{feature}_zscore'] = (df[feature] - df[feature].mean()) / df[feature].std()
    df[f'{feature}_outlier'] = df[f'{feature}_zscore'].abs() > 3

# هـ. اكتشاف الشذوذات باستخدام DBSCAN (تحسين إضافي)
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['anomaly_dbscan'] = dbscan.fit_predict(X_scaled)
anomalies_dbscan = df[df['anomaly_dbscan'] == -1]
print("Detected Anomalies (DBSCAN):\n", anomalies_dbscan[features])

# ------------------------------
# 2. صناعة الميزات (Feature Engineering)
# ------------------------------

# أ. معدل التغير
for feature in features:
    df[f'{feature}_diff'] = df[feature].diff()

# ب. أنماط التدهور
for feature in features:
    df[f'{feature}_ema_short'] = df[feature].ewm(span=10, adjust=False).mean()
    df[f'{feature}_ema_long'] = df[feature].ewm(span=50, adjust=False).mean()
    df[f'{feature}_degradation'] = df[f'{feature}_ema_short'] - df[f'{feature}_ema_long']

# ج. ميزات إحصائية مشتقة
window_size = 20
for feature in features:
    df[f'{feature}_rolling_mean'] = df[feature].rolling(window=window_size, min_periods=1).mean()
    df[f'{feature}_rolling_std'] = df[feature].rolling(window=window_size, min_periods=1).std()
    df[f'{feature}_rolling_min'] = df[feature].rolling(window=window_size, min_periods=1).min()
    df[f'{feature}_rolling_max'] = df[feature].rolling(window=window_size, min_periods=1).max()
    df[f'{feature}_rolling_median'] = df[feature].rolling(window=window_size, min_periods=1).median()
    df[f'{feature}_rolling_iqr'] = df[feature].rolling(window=window_size, min_periods=1).apply(iqr)

# د. ميزات المجال الترددي للاهتزاز
vibration_freq = fft(df['vibration'].dropna().values)
df['vibration_freq_magnitude'] = np.abs(vibration_freq)[:len(df)]

# ------------------------------
# 3. تصور البيانات (Data Visualization)
# ------------------------------

# أ. Heatmap للارتباطات بين الميزات الأساسية
correlation_matrix = df[features].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Sensor Features')
plt.savefig('correlation_heatmap.png')
plt.close()

# ب. Time-Series Plots لجميع الميزات مع EMA
for feature in features:
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df[feature], label=feature.capitalize())
    plt.plot(df.index, df[f'{feature}_ema_short'], label='EMA Short (10)')
    plt.plot(df.index, df[f'{feature}_ema_long'], label='EMA Long (50)')
    plt.xlabel('Time')
    plt.ylabel(feature.capitalize())
    plt.legend()
    plt.savefig(f'{feature}_time_series.png')
    plt.close()

# ج. Box Plots للميزات الأساسية والمشتقة
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[features + [f'{feature}_diff' for feature in features]])
plt.title('Box Plot for Outlier Detection')
plt.xticks(rotation=45)
plt.savefig('box_plot_features.png')
plt.close()

# د. Scatter Plot لنتائج PCA مع تلوين الشذوذات
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['anomaly_iso'], cmap='coolwarm', alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Scatter Plot with Anomalies (Isolation Forest)')
plt.colorbar(label='Anomaly (1: Normal, -1: Anomaly)')
plt.savefig('pca_scatter_anomalies_iso.png')
plt.close()

# هـ. تصور إضافي للميزات المشتقة (تحسين إضافي)
for feature in features:
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df[f'{feature}_rolling_std'], label=f'{feature.capitalize()} Rolling Std')
    plt.xlabel('Time')
    plt.ylabel(f'{feature.capitalize()} Rolling Std')
    plt.legend()
    plt.savefig(f'{feature}_rolling_std.png')
    plt.close()

# ------------------------------
# إنشاء التقارير والملخصات
# ------------------------------

# تقرير التحليل المتقدم
with open('advanced_analysis_report.txt', 'w') as f:
    f.write("Advanced Analysis Report\n")
    f.write("========================\n")
    for feature in features:
        f.write(f"Time-Series Decomposition for {feature.capitalize()}:\n")
        f.write(f" - Trend: General trend over time.\n")
        f.write(f" - Seasonal: Periodic patterns observed.\n")
        f.write(f" - Residual: Anomalies in residuals indicate unexpected behavior.\n\n")
    f.write("PCA Analysis:\n")
    f.write(f" - Explained Variance Ratio: {explained_variance}\n")
    f.write(f" - PC1 Weights: {pc1_weights}\n")
    f.write(f" - PC2 Weights: {pc2_weights}\n")
    f.write(" - Key Components: PC1 likely captures combined effects of features.\n\n")
    f.write("Anomaly Detection:\n")
    f.write(f" - Number of anomalies (Isolation Forest): {len(anomalies_iso)}\n")
    f.write(f" - Number of anomalies (DBSCAN): {len(anomalies_dbscan)}\n")
    for feature in features:
        outliers = df[df[f'{feature}_outlier']]
        f.write(f" - Number of Z-Score outliers in {feature}: {len(outliers)}\n")

# ملخص صناعة الميزات
with open('feature_engineering_summary.txt', 'w') as f:
    f.write("Feature Engineering Summary\n")
    f.write("===========================\n")
    f.write("Engineered Features:\n")
    for feature in features:
        f.write(f" - {feature}_diff: Rate of change to capture sudden shifts.\n")
        f.write(f" - {feature}_ema_short & {feature}_ema_long: Exponential moving averages for trend analysis.\n")
        f.write(f" - {feature}_degradation: Difference between short and long EMA to indicate degradation.\n")
        f.write(f" - {feature}_rolling_mean, _std, _min, _max, _median, _iqr: Statistical features to capture trends and variability.\n")
    f.write(" - vibration_freq_magnitude: FFT magnitude to capture frequency patterns in vibrations.\n")
    f.write("\nExpected Impact:\n")
    f.write(" - These features provide insights into trends, variability, and frequency patterns, enhancing failure prediction accuracy.\n")
    f.write("\nRationale:\n")
    f.write(" - Rate of change and EMA help detect sudden and gradual changes.\n")
    f.write(" - Statistical features provide robust inputs for machine learning models.\n")
    f.write(" - Frequency features capture periodic behavior in vibrations.\n")

# حفظ البيانات المعدلة
df.to_csv('engineered_features.csv')

print("Stage 2 completed: Enhanced Analysis, Feature Engineering, Visualization, and Reports generated.")