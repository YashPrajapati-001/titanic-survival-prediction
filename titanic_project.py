# ============================================================
#   TITANIC DATA SCIENCE PROJECT - Complete Code
#   Steps: Load → Clean → Visualize → Engineer → Model → Evaluate
# ============================================================

# ── 1. IMPORTS ───────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set plot style
sns.set(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)

print("=" * 60)
print("  TITANIC SURVIVAL PREDICTION - DATA SCIENCE PROJECT")
print("=" * 60)


# ── 2. DATA COLLECTION ───────────────────────────────────────
print("\n[STEP 1] Loading Titanic Dataset...")

# Load directly from public URL (no Kaggle account needed)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(f"  ✔  Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())
print("\nBasic Info:")
print(df.info())


# ── 3. DATA CLEANING ─────────────────────────────────────────
print("\n" + "=" * 60)
print("[STEP 2] Data Cleaning & Preprocessing")
print("=" * 60)

# Check missing values
print("\nMissing values BEFORE cleaning:")
print(df.isnull().sum())

# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin (too many missing), Name, Ticket, PassengerId
df.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId'], inplace=True)

print("\nMissing values AFTER cleaning:")
print(df.isnull().sum())
print("\nDataset shape after cleaning:", df.shape)


# ── 4. DATA VISUALIZATION ────────────────────────────────────
print("\n" + "=" * 60)
print("[STEP 3] Data Visualization")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Titanic Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold')

# Plot 1: Survival Count
sns.countplot(x='Survived', data=df, palette=['#e74c3c', '#2ecc71'], ax=axes[0, 0])
axes[0, 0].set_title('Survival Count')
axes[0, 0].set_xticklabels(['Did Not Survive', 'Survived'])

# Plot 2: Survival by Gender
sns.countplot(x='Sex', hue='Survived', data=df, palette=['#e74c3c', '#2ecc71'], ax=axes[0, 1])
axes[0, 1].set_title('Survival by Gender')
axes[0, 1].legend(['Did Not Survive', 'Survived'])

# Plot 3: Survival by Pclass
sns.countplot(x='Pclass', hue='Survived', data=df, palette=['#e74c3c', '#2ecc71'], ax=axes[0, 2])
axes[0, 2].set_title('Survival by Passenger Class')
axes[0, 2].legend(['Did Not Survive', 'Survived'])

# Plot 4: Age Distribution
df['Age'].hist(bins=30, color='steelblue', edgecolor='white', ax=axes[1, 0])
axes[1, 0].set_title('Age Distribution')
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Count')

# Plot 5: Age vs Survival (KDE)
df[df['Survived'] == 0]['Age'].plot(kind='kde', label='Did Not Survive', color='#e74c3c', ax=axes[1, 1])
df[df['Survived'] == 1]['Age'].plot(kind='kde', label='Survived', color='#2ecc71', ax=axes[1, 1])
axes[1, 1].set_title('Age Distribution by Survival')
axes[1, 1].legend()

# Plot 6: Fare Distribution
df['Fare'].hist(bins=40, color='#9b59b6', edgecolor='white', ax=axes[1, 2])
axes[1, 2].set_title('Fare Distribution')
axes[1, 2].set_xlabel('Fare')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✔  EDA plots saved as 'eda_plots.png'")

# Correlation Heatmap
plt.figure(figsize=(10, 6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✔  Heatmap saved as 'heatmap.png'")


# ── 5. FEATURE ENGINEERING ───────────────────────────────────
print("\n" + "=" * 60)
print("[STEP 4] Feature Engineering")
print("=" * 60)

# Create FamilySize feature
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
print("  ✔  Created 'FamilySize' = SibSp + Parch + 1")

# Create IsAlone feature
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
print("  ✔  Created 'IsAlone' feature")

# Encode categorical variables
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])          # male=1, female=0
df['Embarked'] = le.fit_transform(df['Embarked'])  # C=0, Q=1, S=2
print("  ✔  Encoded 'Sex' (female=0, male=1)")
print("  ✔  Encoded 'Embarked' (C=0, Q=1, S=2)")

print("\nFeature preview:")
print(df.head())


# ── 6. MODEL BUILDING ────────────────────────────────────────
print("\n" + "=" * 60)
print("[STEP 5] Building Machine Learning Model")
print("=" * 60)

# Define features and target
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']
X = df[features]
y = df['Survived']

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Training samples : {len(X_train)}")
print(f"  Testing  samples : {len(X_test)}")

# Train Logistic Regression
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)
print("  ✔  Logistic Regression model trained")


# ── 7. EVALUATION ────────────────────────────────────────────
print("\n" + "=" * 60)
print("[STEP 6] Model Evaluation")
print("=" * 60)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n  ✔  Accuracy : {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Did Not Survive', 'Survived']))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Did Not Survive', 'Survived'],
            yticklabels=['Did Not Survive', 'Survived'])
plt.title(f'Confusion Matrix  (Accuracy: {acc*100:.1f}%)', fontsize=14, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✔  Confusion matrix saved as 'confusion_matrix.png'")

# Feature Importance (coefficients)
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', ascending=False)

plt.figure(figsize=(9, 5))
colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in coef_df['Coefficient']]
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Feature Coefficients (Logistic Regression)', fontsize=14, fontweight='bold')
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✔  Feature importance plot saved as 'feature_importance.png'")


# ── 8. SAMPLE PREDICTION ─────────────────────────────────────
print("\n" + "=" * 60)
print("[STEP 7] Sample Prediction Demo")
print("=" * 60)

# Example: 25-year-old female, 1st class, fare=100, alone
sample = pd.DataFrame([{
    'Pclass': 1, 'Sex': 0, 'Age': 25, 'Fare': 100,
    'Embarked': 0, 'FamilySize': 1, 'IsAlone': 1
}])
pred = model.predict(sample)[0]
prob = model.predict_proba(sample)[0][1]

print(f"\n  Sample: 25-yr-old female | 1st class | Fare=100 | Alone")
print(f"  Prediction : {'SURVIVED ✔' if pred == 1 else 'DID NOT SURVIVE ✘'}")
print(f"  Survival Probability : {prob*100:.1f}%")

print("\n" + "=" * 60)
print("  PROJECT COMPLETE! All outputs saved.")
print("=" * 60)
