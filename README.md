# Titanic Categorical Data Pipeline & Probabilistic Classification

An end-to-end data engineering and statistical classification pipeline. This project focuses on the precise data transformation, feature discretization (binning) methodologies, and out-of-sample inferencing workflows required to optimize inputs explicitly for a **Categorical Naive Bayes** probabilistic model.

## 🚀 Business Application & Value
In industry, raw transactional data often presents a challenging mix of high-variance numerical fields and sparse categories. This pipeline models real-world data preparation workflows by:
1. **Resolving Nullity:** Imputing missing values using central tendency markers without skewing data distributions.
2. **Feature Discretization:** Segmenting continuous features into deterministic population bins.
3. **Out-of-Sample Evaluation:** Running clean inference partitions across unlabelled target splits to mirror production deployment behaviors.
4. **Optimized Model Alignment:** Structuring input architecture to match the mathematical assumptions of a specific distribution engine (`CategoricalNB`).

---

## 📐 Pipeline Architecture

### 1. Dimensionality Selection & Pruning
Irrelevant or high-cardinality identity trackers (`PassengerId`, `Name`, `Ticket`, `Fare`, `Cabin`) were dropped to prevent overfitting, leaving a clean core matrix: `['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']`.

### 2. Statistical Imputation Matrix
* **Continuous Features (`Age`):** Filled missing entries by computing the rolling **median**, protecting the data from outlier biases.
* **Categorical Features (`Embarked`):** Repaired missing records using the dominant localized **mode** statistic.

### 3. Quantitative Discretization (Feature Binning)
Continuous age fields were mapped into 5 logical lifecycle cohorts using fixed numerical edge criteria via `pd.cut()`:
* **Child:** 0 to 12 years old
* **Teen:** 12 to 18 years old
* **YoungAdult:** 18 to 35 years old
* **Adult:** 35 to 50 years old
* **Senior:** 50 to 80 years old

### 4. Categorical Integer Encoding & Testing Alignment
An automated programmatic loop passed every categorical column through a `LabelEncoder` instance. To guarantee complete consistency during out-of-sample data modeling, tracking classes are saved and strictly mapped to prevent data leakage between the isolated training splits and the test dataset partition.

---

## 📊 Feature Matrix Transformation Profile

### Data Layout Pre-Encoding:



| Survived | Pclass | Sex | Age | SibSp | Parch | Embarked |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 3 | male | YoungAdult | 1 | 0 | S |
| 1 | 1 | female | Adult | 1 | 0 | C |

### Data Layout Post-Encoding (Fed into Machine Learning):



| Pclass | Sex | Age | SibSp | Parch | Embarked |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 2 | 1 | 4 | 1 | 0 | 2 |
| 0 | 0 | 0 | 1 | 0 | 0 |

---

## 📈 Model Performance & Evaluation

The pipeline processes validation steps across internal splits and successfully generates predictions on external data:

* **Validation Split Testing:** The trained `CategoricalNB` engine is cross-examined using standard indicators (Accuracy, Precision, Recall, and F1-Score) to ensure training stability.
* **Titanic Test Dataset Evaluation:** The complete, unified engineering pipeline has been rigorously verified and tested against `titanic_test.csv` to confirm stability on out-of-sample records without throwing data leakage or class shape syntax errors.

---

## ⚡ Setup & Local Execution

1. **Clone this pipeline repository:**
   ```bash
   git clone https://github.com/usmanali9999/Titanic-Categorical-Data-Pipeline.git
   cd Titanic-Categorical-Data-Pipeline
   ```

2. **Install environment dependencies instantly using the requirements file:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the pipeline:**
   Ensure both **`titanic_train.csv`** and **`titanic_test.csv`** are present in the root project directory before executing your Jupyter environment. This ensures that the model script can fit training weights on historical markers and smoothly evaluate unknown rows inside the test partition.
