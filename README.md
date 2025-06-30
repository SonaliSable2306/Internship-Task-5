# Decision Tree & Random Forest Classifier - Heart Disease Dataset

## Objective
Build and evaluate machine learning models using **Decision Tree** and **Random Forest** classifiers to detect the presence of heart disease.

---

##  Steps Performed

- Installed and imported required libraries.
- Uploaded and loaded the dataset (`heart.csv`) using `Google Colab` file upload.
- Performed initial data inspection using `df.head()` and `df.info()`.
- Checked for missing values and understood feature distributions.
- Split data into **train** and **test** sets.
- Standardized numerical features using `StandardScaler`.
- Trained a **Decision Tree Classifier**.
  - Visualized the tree using `plot_tree` and `graphviz`.
  - Evaluated the model using Accuracy, Confusion Matrix, and Classification Report.
- Trained a **Random Forest Classifier**.
  - Plotted **feature importances** using bar graphs.
  - Compared performance with the Decision Tree.
- Tuned hyperparameters like `max_depth` and `n_estimators` to control overfitting.
- Applied **Cross-Validation** to assess model robustness.

---

##  Tools Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Graphviz
- Google Colab

---

##  Evaluation Metrics

- **Accuracy Score**
- **Confusion Matrix**
- **Precision, Recall, F1-score**
- **Feature Importance Plot**
- **Cross-Validation Score**

---

##  Key Takeaways

- **Decision Trees** are interpretable and easy to visualize but prone to overfitting.
- **Random Forests**, being an ensemble of trees, reduce overfitting and improve performance.
- **Feature importance plots** help identify which features contribute most to model predictions.
- **Cross-validation** ensures that model performance is not dependent on a single train-test split.
- **Hyperparameter tuning** (e.g., `max_depth`, `n_estimators`) is essential to balance bias-variance.

---

##  Dataset
The dataset used is `heart.csv`, containing patient attributes such as:
- Age, Sex, Cholesterol, Blood Pressure, Chest Pain Type, etc.
---

##  Conclusion

Tree-based models like Decision Trees and Random Forests are powerful for structured classification tasks. This notebook demonstrates both training and evaluation with a focus on interpretability and performance.
