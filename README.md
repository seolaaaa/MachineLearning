#  Students Academic Performance Analysis (Unsupervised ML)

This project analyzes the **Students Academic Performance Dataset** from Kaggle using **unsupervised machine learning**.  
Instead of predicting labels like “Pass/Fail,” the program discovers natural groupings of students based on academic scores and demographic features.

---

##  Dataset

The dataset is available on Kaggle: [Students Performance in Exams](https://www.kaggle.com/datasets/sadiajavedd/students-academic-performance-dataset).  

Columns include:

- Gender  
- Race/Ethnicity  
- Parental Level of Education  
- Lunch  
- Test Preparation Course  
- Math Score  
- Reading Score  
- Writing Score  

---

##  Features of the Program

- **Unsupervised Learning:** Uses K-Means clustering to automatically group students.  
- **Preprocessing:**  
  - Label encoding for categorical features (gender, parental education, etc.)  
  - Standardization for numeric features (scores)  
- **Visualization:** Scatter plot using Matplotlib with color-coded clusters and a legend.  

---

##  How the Program Works

1. **Load Dataset**: Read CSV into a Pandas DataFrame.  
2. **Encode Categorical Features**: Convert text features to numeric using `LabelEncoder`.  
3. **Select Features**: Use all input columns (demographics + scores) for clustering.  
4. **Scale Features**: Standardize values so all features contribute equally.  
5. **Apply K-Means Clustering**:  
   - Choose `n_clusters=3` (or any number)  
   - Assign each student to a cluster automatically  
6. **Visualize Clusters**:  
   - Plot Math Score vs Reading Score  
   - Color points by cluster  
   - Add a legend for clarity  

---

##  Example Visualization

The scatter plot shows clusters:

- Cluster 0  
- Cluster 1  
- Cluster 2  

Each cluster represents students with similar academic and demographic profiles.

---

##  Requirements

- Python 3.7+  
- Pandas  
- Scikit-learn  
- Matplotlib  

Install dependencies:

```bash
pip install pandas scikit-learn matplotlib
```
## Usage
1. Clone the repository:
```
git clone https://github.com/YourUsername/Students-Academic-ML.git
```
2. Navigate to the project folder:
```
cd Students-Academic-ML
```
3. Run the program:
```
python student_unsupervised.py
```

## Notes
- This is **unsupervised learning**: there is no target label like "Pass/Fail."
- Cluster interpretation is based on visualization and feature analysis.
- Preprocessing ensures categorical and numeric data are properly scaled for clustering.
  
