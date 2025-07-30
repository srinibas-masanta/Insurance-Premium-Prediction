## 🧾 Insurance Premium Prediction

This project aims to predict insurance premium charges using machine learning models based on an individual’s demographic and lifestyle attributes. The dataset includes information such as age, BMI, number of children, region, sex, and smoking status. The goal is to build interpretable models that help identify key drivers of insurance costs and make accurate predictions.


### 📁 Dataset

* The dataset used is from Kaggle's [US Health Insurance Dataset](https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset/data).
* It contains 1,300+ records with the following features:

  * `age`: Age of the primary beneficiary
  * `sex`: Gender
  * `bmi`: Body mass index
  * `children`: Number of dependents
  * `smoker`: Whether the person smokes
  * `region`: Residential region in the US
  * `charges`: Medical insurance charges (target variable)


### 🔄 Data Preprocessing & Feature Engineering

- Removed outliers using the IQR method to retain meaningful data.
- Created an `age_group` feature by binning ages into categories (Young, Adult, Middle-aged, Senior).
- Encoded categorical variables:
  - One-hot encoding for `region`
  - Label encoding for `smoker` and `sex`
- Selected relevant numeric features based on correlation analysis.


### 📊 Exploratory Data Analysis (EDA)

Conducted univariate and bivariate analysis to understand feature distributions and relationships:
- **Univariate Plots**: Histograms of `age` and `bmi`, pie chart of smoker distribution, count plot of gender.
- **Bivariate Plots**: 
  - Scatter plots of `age` vs. `charges` and `bmi` vs. `charges` (colored by smoking status)
  - Violin plot for `charges` by number of children
  - Box plot comparing charges for smokers vs non-smokers
  - Bar plots showing total charges and smoker counts by region


### 📈 Correlation Analysis

* Pearson correlation was used to analyze relationships between numerical features.
* Only `age` and `smoker_encoded` showed a strong correlation with `charges`.


### 🤖 Modeling and Evaluation

Three regression models were trained using the selected features (`age` and `smoker_encoded`) to predict insurance charges:

| Model             | Mean Squared Error (MSE) | Mean Absolute Error (MAE) |
| ----------------- | ------------------------ | ------------------------- |
| Decision Tree     | 19,994,952.54            | 2,757.66                  |
| Random Forest     | 19,946,316.36            | 2,749.04                  |
| Gradient Boosting | 19,405,842.52            | 2,694.16                  |

📌 **Gradient Boosting** achieved the lowest error values among the three, indicating better performance on the test data.

Each model’s predictions were also visualized using **Actual vs Predicted** scatter plots to assess how well the models captured the underlying pattern in the data.


### 🔍 Feature Importance

Feature importance was extracted from all three models.
**Observation**: Smoking status consistently had a higher importance than age in predicting insurance charges.


### 🔮 Inference on New Data

The trained models were used to predict insurance charges for new input data.
Example:

```python
new_data = pd.DataFrame({'age': [31], 'smoker_encoded': [1]})
```

**Predicted Charges for a 31-year-old smoker:**

* Decision Tree: ₹19,275.16
* Random Forest: ₹19,190.37
* Gradient Boosting: ₹19,103.24


### 📦 Libraries Used

* `pandas`, `numpy`
* `matplotlib`, `seaborn`
* `sklearn`: preprocessing, model\_selection, tree, ensemble, metrics


### 📝 License

This project is licensed under the MIT License.
