import pandas as pd

# Load the uploaded Excel file to examine its content
file_path = r"C:\Users\hoang\Downloads\diabetes_datasets\Shanghai_T1DM_Summary.xlsx"
xls = pd.ExcelFile(file_path)

# Display sheet names to understand the structure
sheet_names = xls.sheet_names
sheet_names

['T1DM']

import matplotlib.pyplot as plt

# Convert necessary columns to numeric, if not already
df_t1dm['BMI (kg/m2)'] = pd.to_numeric(df_t1dm['BMI (kg/m2)'], errors='coerce')
df_t1dm['Age (years)'] = pd.to_numeric(df_t1dm['Age (years)'], errors='coerce')
df_t1dm['Duration of Diabetes  (years)'] = pd.to_numeric(df_t1dm['Duration of Diabetes  (years)'], errors='coerce')
df_t1dm['Glycated Albumin (%)'] = pd.to_numeric(df_t1dm['Glycated Albumin (%)'], errors='coerce')

# Plot 1: BMI distribution
plt.figure(figsize=(8, 5))
plt.hist(df_t1dm['BMI (kg/m2)'].dropna(), bins=20)
plt.title('Distribution of BMI in T1DM Patients')
plt.xlabel('BMI (kg/m²)')
plt.ylabel('Number of Patients')
plt.grid(True)
plt.show()

# Plot 2: Age vs. Duration of Diabetes
plt.figure(figsize=(8, 5))
plt.scatter(df_t1dm['Age (years)'], df_t1dm['Duration of Diabetes  (years)'], alpha=0.6)
plt.title('Age vs. Duration of Diabetes')
plt.xlabel('Age (years)')
plt.ylabel('Duration of Diabetes (years)')
plt.grid(True)
plt.show()

# Plot 3: Glycated Albumin vs. BMI
plt.figure(figsize=(8, 5))
plt.scatter(df_t1dm['BMI (kg/m2)'], df_t1dm['Glycated Albumin (%)'], alpha=0.6, color='green')
plt.title('BMI vs. Glycated Albumin (%)')
plt.xlabel('BMI (kg/m²)')
plt.ylabel('Glycated Albumin (%)')
plt.grid(True)
plt.show()

# Convert additional columns for plotting
df_t1dm['Gender'] = df_t1dm['Gender (Female=1, Male=2)'].map({1: 'Female', 2: 'Male'})
df_t1dm['Hypoglycemia (yes/no)'] = df_t1dm['Hypoglycemia (yes/no)'].astype(str)

# Plot 4: Glycated Albumin by Gender
plt.figure(figsize=(8, 5))
df_t1dm.boxplot(column='Glycated Albumin (%)', by='Gender')
plt.title('Glycated Albumin (%) by Gender')
plt.suptitle('')
plt.xlabel('Gender')
plt.ylabel('Glycated Albumin (%)')
plt.grid(True)
plt.show()

# Plot 5: Age distribution by Hypoglycemia occurrence
plt.figure(figsize=(8, 5))
df_t1dm[df_t1dm['Hypoglycemia (yes/no)'].isin(['yes', 'no'])].boxplot(column='Age (years)', by='Hypoglycemia (yes/no)')
plt.title('Age Distribution by Hypoglycemia Occurrence')
plt.suptitle('')
plt.xlabel('Hypoglycemia')
plt.ylabel('Age (years)')
plt.grid(True)
plt.show()

# Plot 6: Total Cholesterol vs. Triglyceride colored by Hypoglycemia
df_t1dm['Total Cholesterol (mmol/L)'] = pd.to_numeric(df_t1dm['Total Cholesterol (mmol/L)'], errors='coerce')
df_t1dm['Triglyceride (mmol/L)'] = pd.to_numeric(df_t1dm['Triglyceride (mmol/L)'], errors='coerce')

plt.figure(figsize=(8, 5))
for label in ['yes', 'no']:
    subset = df_t1dm[df_t1dm['Hypoglycemia (yes/no)'] == label]
    plt.scatter(subset['Total Cholesterol (mmol/L)'], subset['Triglyceride (mmol/L)'], label=label, alpha=0.6)

plt.title('Cholesterol vs. Triglyceride by Hypoglycemia Status')
plt.xlabel('Total Cholesterol (mmol/L)')
plt.ylabel('Triglyceride (mmol/L)')
plt.legend(title='Hypoglycemia')
plt.grid(True)
plt.show()