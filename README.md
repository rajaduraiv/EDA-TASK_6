# EDA-TASK_6

# Exploratory Data Analysis (EDA) – Healthcare Dataset

## 1. Project Overview

This project performs Exploratory Data Analysis (EDA) on a healthcare dataset using Python. It explores patient demographics, medical conditions, admission types, admission and discharge dates, length of stay, and billing amounts. The notebook includes data inspection, data cleaning, summary statistics, group-based analysis, and visualizations.

## 2. Objectives

- Inspect the dataset structure, columns, data types, and size.
- Identify and handle missing values.
- Standardize admission-type labels and convert date columns to datetime.
- Calculate patient length of stay.
- Summarize billing amounts and average age by medical condition.
- Compare medical conditions by gender and admission type.
- Visualize admission counts, billing distributions, monthly admissions, and correlations.

## 3. Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook / Google Colab

## 4. Dataset

The notebook reads a CSV file named `healthcare_data_for_task.csv`. The dataset has **1,000 rows and 10 original columns**. The columns are `Patient_ID`, `Age`, `Gender`, `Blood_Type`, `Medical_Condition`, `Medical_Code`, `Date_of_Admission`, `Discharge_Date`, `Admission_Type`, and `Billing_Amount`. After creating `Stay_Date`, the DataFrame contains 11 columns.

**Before running:** place `healthcare_data_for_task.csv` in the same folder as the notebook/README, or update the CSV path in the loading cell. The original notebook uses `/content/healthcare_data_for_task.csv`, which is a Google Colab path.

## 5. Analysis Workflow

1. Import libraries and load the CSV file.
2. Preview the data and inspect shape, columns, and data types.
3. Check missing values and fill missing `Medical_Code` values with `Unknown`.
4. Standardize `Admission_Type` text.
5. Convert admission/discharge dates to datetime and calculate stay duration in days.
6. Explore billing statistics, average age by condition, and gender counts by condition.
7. Plot admission-type counts, total billing by condition/admission type, billing distributions, monthly admissions, and a correlation heatmap.

## 6. Key Results

- Dataset size: **1,000 rows × 10 columns** before feature engineering.
- Missing values: **30** in `Medical_Code`; the other listed columns have no missing values in the notebook output. These missing codes are replaced with `Unknown`.
- Admission types after standardization: `Emergency`, `Elective`, and `Urgent`.
- Admission-type counts: Emergency **381**, Elective **368**, Urgent **251**.
- Billing amount: mean **77,117.52**, median **77,293.90**, minimum **5,568.16**, maximum **149,921.80**.
- Mean age by medical condition: Arthritis **52.67**, Asthma **49.92**, Diabetes **53.13**, Heart Disease **52.18**, Hypertension **51.40**.
- `Stay_Date` is calculated as discharge date minus admission date, in days.

## 7. Code and Outputs

The following sections reproduce the notebook code cells and the saved textual outputs. Plot cells are included; chart images themselves remain visible in the executed notebook rather than being embedded in this README.

### Cell 1

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

**Output**

_No text output saved for this cell. Any plot is generated when the cell is run in Jupyter/Colab._

### Cell 2

```python
df=pd.read_csv("/content/healthcare_data_for_task.csv")
```

**Output**

_No text output saved for this cell. Any plot is generated when the cell is run in Jupyter/Colab._

### Cell 3

```python
df.head()
```

**Output**

```text
  Patient_ID  Age  Gender Blood_Type Medical_Condition Medical_Code  \
0      P0001   47    Male        AB+     Heart Disease       HEA004   
1      P0002   62  Female        AB-          Diabetes       DIA001   
2      P0003   36    Male        AB-     Heart Disease       HEA004   
3      P0004   51    Male         A+     Heart Disease       HEA004   
4      P0005   30    Male         A+          Diabetes       DIA001   

  Date_of_Admission Discharge_Date Admission_Type  Billing_Amount  
0        2025-05-10     2025-05-16      Emergency        23218.31  
1        2025-12-31     2026-01-02      Emergency        47010.56  
2        2025-11-10     2025-11-27      Emergency        39619.01  
3        2023-05-02     2023-05-07       Elective        35963.71  
4        2024-04-11     2024-04-17         Urgent        64615.93
```

### Cell 4

```python
df.shape
```

**Output**

```text
(1000, 10)
```

### Cell 5

```python
df.columns
```

**Output**

```text
Index(['Patient_ID', 'Age', 'Gender', 'Blood_Type', 'Medical_Condition',
       'Medical_Code', 'Date_of_Admission', 'Discharge_Date', 'Admission_Type',
       'Billing_Amount'],
      dtype='object')
```

### Cell 6

```python
df.info()
```

**Output**

```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 10 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   Patient_ID         1000 non-null   object 
 1   Age                1000 non-null   int64  
 2   Gender             1000 non-null   object 
 3   Blood_Type         1000 non-null   object 
 4   Medical_Condition  1000 non-null   object 
 5   Medical_Code       970 non-null    object 
 6   Date_of_Admission  1000 non-null   object 
 7   Discharge_Date     1000 non-null   object 
 8   Admission_Type     1000 non-null   object 
 9   Billing_Amount     1000 non-null   float64
dtypes: float64(1), int64(1), object(8)
memory usage: 78.3+ KB
```

### Cell 7

```python
df['Age']
```

**Output**

```text
0      47
1      62
2      36
3      51
4      30
       ..
995    24
996    63
997    63
998    33
999    41
Name: Age, Length: 1000, dtype: int64
```

### Cell 8

```python
df['Medical_Code']
```

**Output**

```text
0      HEA004
1      DIA001
2      HEA004
3      HEA004
4      DIA001
        ...  
995    DIA001
996    DIA001
997    DIA001
998    DIA001
999    AST003
Name: Medical_Code, Length: 1000, dtype: object
```

### Cell 9

```python
df.isnull().sum()
```

**Output**

```text
Patient_ID            0
Age                   0
Gender                0
Blood_Type            0
Medical_Condition     0
Medical_Code         30
Date_of_Admission     0
Discharge_Date        0
Admission_Type        0
Billing_Amount        0
dtype: int64
```

### Cell 10

```python
df['Medical_Code'] = df['Medical_Code'].fillna('Unknown')
```

**Output**

_No text output saved for this cell. Any plot is generated when the cell is run in Jupyter/Colab._

### Cell 11

```python
df['Admission_Type'] = (
    df['Admission_Type']
    .str.strip()
    .str.title()
)
```

**Output**

_No text output saved for this cell. Any plot is generated when the cell is run in Jupyter/Colab._

### Cell 12

```python
df['Admission_Type'].unique()
```

**Output**

```text
array(['Emergency', 'Elective', 'Urgent'], dtype=object)
```

### Cell 13

```python
df['Admission_Type'].value_counts()
```

**Output**

```text
Admission_Type
Emergency    381
Elective     368
Urgent       251
Name: count, dtype: int64
```

### Cell 14

```python
df['Date_of_Admission'] = pd.to_datetime(df['Date_of_Admission'])
```

**Output**

_No text output saved for this cell. Any plot is generated when the cell is run in Jupyter/Colab._

### Cell 15

```python
df['Discharge_Date'] = pd.to_datetime(df['Discharge_Date'])
```

**Output**

_No text output saved for this cell. Any plot is generated when the cell is run in Jupyter/Colab._

### Cell 16

```python
df['Date_of_Admission'].dt.year
```

**Output**

```text
0      2025
1      2025
2      2025
3      2023
4      2024
       ... 
995    2023
996    2025
997    2023
998    2023
999    2023
Name: Date_of_Admission, Length: 1000, dtype: int32
```

### Cell 17

```python
df['Date_of_Admission'].dt.month
```

**Output**

```text
0       5
1      12
2      11
3       5
4       4
       ..
995     3
996     4
997     5
998     2
999     2
Name: Date_of_Admission, Length: 1000, dtype: int32
```

### Cell 18

```python
df['Date_of_Admission'].dt.day
```

**Output**

```text
0      10
1      31
2      10
3       2
4      11
       ..
995    16
996    17
997     8
998    11
999    10
Name: Date_of_Admission, Length: 1000, dtype: int32
```

### Cell 19

```python
df['Stay_Date'] = (
    df['Discharge_Date'] - df['Date_of_Admission']
).dt.days
```

**Output**

_No text output saved for this cell. Any plot is generated when the cell is run in Jupyter/Colab._

### Cell 20

```python
df.info()
```

**Output**

```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 11 columns):
 #   Column             Non-Null Count  Dtype         
---  ------             --------------  -----         
 0   Patient_ID         1000 non-null   object        
 1   Age                1000 non-null   int64         
 2   Gender             1000 non-null   object        
 3   Blood_Type         1000 non-null   object        
 4   Medical_Condition  1000 non-null   object        
 5   Medical_Code       1000 non-null   object        
 6   Date_of_Admission  1000 non-null   datetime64[ns]
 7   Discharge_Date     1000 non-null   datetime64[ns]
 8   Admission_Type     1000 non-null   object        
 9   Billing_Amount     1000 non-null   float64       
 10  Stay_Date          1000 non-null   int64         
dtypes: datetime64[ns](2), float64(1), int64(2), object(6)
memory usage: 86.1+ KB
```

### Cell 21

```python
df['Billing_Amount'].describe()
```

**Output**

```text
count      1000.000000
mean      77117.515720
std       41955.242269
min        5568.160000
25%       40318.785000
50%       77293.900000
75%      113089.267500
max      149921.800000
Name: Billing_Amount, dtype: float64
```

### Cell 22

```python
df.groupby('Medical_Condition')['Age'].mean()
```

**Output**

```text
Medical_Condition
Arthritis        52.674641
Asthma           49.919598
Diabetes         53.130653
Heart Disease    52.180995
Hypertension     51.395349
Name: Age, dtype: float64
```

### Cell 23

```python
pd.crosstab(
    df['Medical_Condition'],
    df['Gender']
)
```

**Output**

```text
Gender             Female  Male
Medical_Condition              
Arthritis              95   114
Asthma                101    98
Diabetes              110    89
Heart Disease         104   117
Hypertension           80    92
```

### Cell 24

```python
df['Admission_Type'].value_counts().plot(kind='bar')
```

**Output**

<img width="552" height="502" alt="image" src="https://github.com/user-attachments/assets/d8350fdd-fd54-4e87-8a76-11314304d6e6" />


```text
<Axes: xlabel='Admission_Type'>

<Figure size 640x480 with 1 Axes>

[Plot output: the chart is displayed in the executed Jupyter Notebook. Re-run the cell to regenerate it.]
```

### Cell 25

```python
import pandas as pd
import matplotlib.pyplot as plt

billing_summary = (
    df.groupby(['Medical_Condition','Admission_Type'])['Billing_Amount']
    .sum()
    .unstack()
)

billing_summary.plot(
    kind='bar',
    stacked=True,
    figsize=(12,6),
    edgecolor='black'
)

plt.title('Total Billing Amount by Medical Condition and Admission Type',fontsize=16)
plt.xlabel('Medical Condition',fontsize=12)
plt.ylabel('Total Billing Amount($)',fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Admission Type')
plt.show()
```

**Output**

<img width="1014" height="618" alt="image" src="https://github.com/user-attachments/assets/cb8b014c-c687-42f0-9a60-ecc6c0aa8b9f" />


```text
<Figure size 1200x600 with 1 Axes>

[Plot output: the chart is displayed in the executed Jupyter Notebook. Re-run the cell to regenerate it.]
```

### Cell 26

```python
plt.figure(figsize=(12, 5))

sns.violinplot(
    data=df,
    x='Medical_Condition',
    y='Billing_Amount'
)

plt.title('billing amount distribution by medical condition', fontsize=16)

plt.xlabel('Medical_condition')
plt.ylabel('billing_amount ($)')

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

**Output**

<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/5615ce06-b64e-4455-8d13-13982799c4ba" />


```text
<Figure size 1200x500 with 1 Axes>

[Plot output: the chart is displayed in the executed Jupyter Notebook. Re-run the cell to regenerate it.]
```

### Cell 27

```python
df['Date_of_Admission']=pd.to_datetime(
    df['Date_of_Admission']
)
```

**Output**

_No text output saved for this cell. Any plot is generated when the cell is run in Jupyter/Colab._

### Cell 28

```python
monthly_admissions=(
    df.set_index('Date_of_Admission')
    .resample('ME')
    .size()
)
print(monthly_admissions)
```

**Output**

```text
Date_of_Admission
2023-01-31    35
2023-02-28    29
2023-03-31    26
2023-04-30    33
2023-05-31    31
2023-06-30    25
2023-07-31    23
2023-08-31    27
2023-09-30    23
2023-10-31    19
2023-11-30    19
2023-12-31    23
2024-01-31    23
2024-02-29    24
2024-03-31    24
2024-04-30    25
2024-05-31    31
2024-06-30    23
2024-07-31    29
2024-08-31    19
2024-09-30    23
2024-10-31    33
2024-11-30    32
2024-12-31    30
2025-01-31    33
2025-02-28    28
2025-03-31    39
2025-04-30    34
2025-05-31    34
2025-06-30    26
2025-07-31    25
2025-08-31    33
2025-09-30    24
2025-10-31    36
2025-11-30    28
2025-12-31    31
Freq: ME, dtype: int64
```

### Cell 29

```python
plt.figure(figsize=(14,6))
plt.plot(
    monthly_admissions.index,
    monthly_admissions.values,
    marker='o',
)
plt.title('Monthly Paitient Admissions',fontsize=16)
plt.xlabel('Month',fontsize=12)
plt.ylabel('Number of Admissions',fontsize=12)
plt.grid(True,alpha=0.3)
plt.tight_layout()
plt.show()
```

**Output**

<img width="1390" height="590" alt="image" src="https://github.com/user-attachments/assets/b731b6e4-24e8-4344-a098-fbbe2f5bcbf1" />


```text
<Figure size 1400x600 with 1 Axes>

[Plot output: the chart is displayed in the executed Jupyter Notebook. Re-run the cell to regenerate it.]
```

### Cell 30

```python
columns_to_check = ['Age', 'Stay_Date', 'Billing_Amount']
correlation_subset = df[columns_to_check].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(correlation_subset, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation: Age, Stay Days & Billing Amount', fontsize=20)
plt.show()
```

**Output**

<img width="645" height="382" alt="image" src="https://github.com/user-attachments/assets/cca91514-a84b-4a80-965c-529d5937c9cd" />


```text
<Figure size 600x400 with 2 Axes>

[Plot output: the chart is displayed in the executed Jupyter Notebook. Re-run the cell to regenerate it.]
```

## 8. Visualizations Included

1. **Admission Type Counts** – bar chart comparing Emergency, Elective, and Urgent admissions.
2. **Total Billing Amount by Medical Condition and Admission Type** – stacked bar chart of summed billing amounts.
3. **Billing Amount Distribution by Medical Condition** – violin plot comparing billing distributions across conditions.
4. **Monthly Patient Admissions** – line chart of admissions resampled by month.
5. **Correlation Heatmap** – correlation between `Age`, `Stay_Date`, and `Billing_Amount`.

## 9. Conclusion

This EDA provides an overview of the healthcare records and highlights admission patterns, billing variation, age differences across medical conditions, and the relationship between selected numeric variables. The analysis also performs basic data preparation by filling missing medical codes, standardizing admission-type labels, converting date fields, and calculating length of stay. Further analysis could investigate outliers, validate date consistency, and explore factors associated with higher billing amounts.

## 10. How to Run

1. Download or place `healthcare_data_for_task.csv` in the working directory.
2. Open `EDA_TASK_5 (1).ipynb` in Jupyter Notebook or Google Colab.
3. Update the CSV path if necessary.
4. Run all cells from top to bottom.
5. Commit this `README.md`, the notebook, and (if permitted) the dataset to your project repository.

---

**Note:** The results above reflect outputs saved in the supplied notebook. If the CSV file or code changes, rerun the notebook and update this README so the reported results stay consistent.
