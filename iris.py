import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dataset
file_path = "IRIS.csv"
iris_df = pd.read_csv(file_path)
iris_df_raw = pd.read_csv(file_path)
iris_df = iris_df_raw.copy()

# Encode target labels
le = LabelEncoder()
iris_df['species'] = le.fit_transform(iris_df['species'])

# Features and target
X = iris_df.drop('species', axis=1)
y = iris_df['species']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Models to train
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', probability=True, random_state=42)
}

# Train & Evaluate
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

# Save the best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
joblib.dump(best_model, "iris_best_model.pkl")

# Prediction function
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    sample_scaled = scaler.transform(sample)
    prediction = best_model.predict(sample_scaled)
    return le.inverse_transform(prediction)[0]

# ---------------------------
# Streamlit Web App
# ---------------------------
st.title("🌸 Iris Flower Classification")
st.write("Predict Iris species based on sepal and petal measurements.")

# User input sliders
sepal_length = st.slider("Sepal Length (cm)", float(X['sepal_length'].min()), float(X['sepal_length'].max()), 5.1)
sepal_width = st.slider("Sepal Width (cm)", float(X['sepal_width'].min()), float(X['sepal_width'].max()), 3.5)
petal_length = st.slider("Petal Length (cm)", float(X['petal_length'].min()), float(X['petal_length'].max()), 1.4)
petal_width = st.slider("Petal Width (cm)", float(X['petal_width'].min()), float(X['petal_width'].max()), 0.2)

if st.button("Predict"):
    species = predict_species(sepal_length, sepal_width, petal_length, petal_width)
    st.success(f"🌼 The predicted species is: **{species}**")

# ---------------------------
# Data Visualization
# ---------------------------
st.subheader("📈 Data Visualization")

# Pie Chart of Species Distribution
st.write("#### Pie Chart of Species Distribution")
fig_pie, ax_pie = plt.subplots()
species_counts = iris_df_raw['species'].value_counts()
ax_pie.pie(species_counts, labels=species_counts.index, autopct='%1.1f%%', startangle=90)
ax_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig_pie)

# Scatter Plot of Features
st.write("#### Scatter Plot of Features")
col1, col2 = st.columns(2)
with col1:
    x_axis = st.selectbox("Select X-axis Feature", X.columns, index=0)
with col2:
    y_axis = st.selectbox("Select Y-axis Feature", X.columns, index=2)

fig_scatter, ax_scatter = plt.subplots()
sns.scatterplot(data=iris_df_raw, x=x_axis, y=y_axis, hue='species', ax=ax_scatter)
st.pyplot(fig_scatter)

# Show model accuracy comparison
st.subheader("📊 Model Performance")
st.bar_chart(pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy']))
