# SmartMail — ML-Based Email Spam Detection & Automated Triage

SmartMail is a Python-based machine learning system that detects spam messages and automatically routes incoming emails into **Inbox, Review, or Spam** categories based on their predicted spam probability.

The project combines **TF-IDF text representation, multiple machine learning classifiers, probability calibration, model evaluation, and Python-based automated mailbox triage** into a single workflow.

---

## 🚀 Key Features

* Cleans and preprocesses labeled message data
* Converts text into numerical features using **TF-IDF**
* Compares three machine learning models:

  * Multinomial Naive Bayes
  * Logistic Regression
  * Linear SVM
* Evaluates models using:

  * Accuracy
  * Precision
  * Recall
  * F1 Score
  * Confusion Matrix
* Selects the best model based on **F1 Score**
* Uses calibrated Linear SVM probabilities for risk-based triage
* Automatically classifies incoming `.txt` email messages
* Routes messages into:

  * `Inbox`
  * `Review`
  * `Spam`
* Maintains a CSV log of automated triage decisions
* Saves the trained model and TF-IDF vectorizer for reuse

---

## 🧠 System Architecture

```text
                 Raw SMS Dataset
                       │
                       ▼
               Data Cleaning
                       │
                       ▼
                 Train / Test Split
                       │
                       ▼
                    TF-IDF
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    Naive Bayes   Logistic Reg.   Linear SVM
          │            │            │
          └────────────┼────────────┘
                       ▼
                Model Evaluation
                       │
                       ▼
               Best Model Selection
                       │
                       ▼
              Calibrated SVM Model
                       │
                       ▼
              Spam Probability
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       INBOX         REVIEW         SPAM
        <40%        40–69%         ≥70%
                       │
                       ▼
                 Triage Log
```

---

## 📊 Dataset

The project uses the **UCI SMS Spam Collection**, containing labeled SMS messages classified as either `ham` or `spam`.

The dataset can be downloaded from the official **UCI Machine Learning Repository**:

**Dataset:** [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection?utm_source=chatgpt.com)

After downloading, place the `SMSSpamCollection` file inside:

```text
data/
└── SMSSpamCollection
```

After cleaning:

* **5,169 messages** retained

* **4,135 training samples**

* **1,034 testing samples**

* **7,680 TF-IDF features**

The dataset was cleaned by removing missing values, duplicate messages, invalid labels, and resetting the resulting index.

**Dataset License:** Creative Commons Attribution 4.0 International (CC BY 4.0).

---

## 🔤 TF-IDF Feature Extraction

Text messages are converted into numerical feature vectors using **Term Frequency–Inverse Document Frequency (TF-IDF)**.

The vectorizer is fitted only on the training data and then applied to the test data to prevent information leakage.

```python
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

The resulting feature space contains **7,680 TF-IDF features**.

---

## 🤖 Model Comparison

Three classification algorithms were trained and evaluated on the same train/test split.

| Model               |   Accuracy |  Precision |     Recall |   F1 Score |
| ------------------- | ---------: | ---------: | ---------: | ---------: |
| Naive Bayes         |     95.26% |    100.00% |     62.60% |     77.00% |
| Logistic Regression |     96.42% |     96.08% |     74.81% |     84.12% |
| **Linear SVM**      | **98.26%** | **95.93%** | **90.08%** | **92.91%** |

The **Linear SVM achieved the highest F1 score in this experiment**, so it was selected as the deployed classifier.

---

## 📈 Confusion Matrix

The selected model produced the following results on the 1,034-message test set:

```text
                 Predicted
              Ham       Spam
Actual Ham    898        5
Actual Spam    13      118
```

This corresponds to:

* **898** ham messages correctly classified
* **118** spam messages correctly classified
* **5** ham messages classified as spam
* **13** spam messages classified as ham

The confusion matrix is saved at:

```text
plots/confusion_matrix.png
```

---

## ⚙️ Automated Email Triage

SmartMail includes a Python-based simulated mailbox system that automatically classifies incoming email messages and routes them into **Inbox, Review, or Spam** folders.

### 📂 Mailbox Structure

Create the following folder structure inside the project root:

```text
mailbox/
│
├── incoming/
├── inbox/
├── review/
└── spam/
```

The folders have the following purpose:

| Folder      | Purpose                                  |
| ----------- | ---------------------------------------- |
| `incoming/` | New emails waiting to be analyzed        |
| `inbox/`    | Messages classified as low-risk          |
| `review/`   | Messages with uncertain spam probability |
| `spam/`     | Messages classified as high-risk         |

The `automate.py` script automatically creates these folders if they do not already exist.

---

### 📊 Triage Rules

SmartMail uses the calibrated model's spam probability to determine the appropriate destination.

| Spam Probability | Decision | Destination       |
| ---------------: | -------- | ----------------- |
|          `< 40%` | INBOX    | `mailbox/inbox/`  |
|     `40% – <70%` | REVIEW   | `mailbox/review/` |
|          `≥ 70%` | SPAM     | `mailbox/spam/`   |

---

### 🧪 Testing the Mailbox Automation

#### Step 1 — Create the mailbox folders

From the project root, create:

```text
mailbox/
├── incoming/
├── inbox/
├── review/
└── spam/
```

On Windows, the folders can also be created using:

```bash
mkdir mailbox
mkdir mailbox\incoming
mkdir mailbox\inbox
mkdir mailbox\review
mkdir mailbox\spam
```

---

#### Step 2 — Add test emails

Create `.txt` files inside:

```text
mailbox/incoming/
```

For example:

```text
mailbox/
├── incoming/
│   ├── email_001.txt
│   ├── email_002.txt
│   └── email_003.txt
│
├── inbox/
├── review/
└── spam/
```

Example `email_001.txt`:

```text
Hey, are we still meeting at 6 pm today?
Let me know if the timing changes.
```

Example `email_002.txt`:

```text
CONGRATULATIONS!

You have been selected to receive a FREE cash reward.
Click now to claim your prize.
Limited time offer!
```

Example `email_003.txt`:

```text
URGENT: Your account has been selected for a special reward.

You have won a cash bonus.
Reply WIN immediately to claim your reward.
```

---

#### Step 3 — Run the automation

From the project root:

```bash
python src/automate.py
```

SmartMail will:

1. Read every `.txt` file from `mailbox/incoming/`
2. Load the saved TF-IDF vectorizer
3. Convert the message into TF-IDF features
4. Load the trained ML model
5. Predict whether the message is `ham` or `spam`
6. Calculate the calibrated spam probability
7. Assign a triage decision
8. Move the email to the corresponding mailbox folder
9. Record the decision in `triage_log.csv`

---

### 📬 Example Result

Before running the automation:

```text
mailbox/
├── incoming/
│   ├── email_001.txt
│   ├── email_002.txt
│   └── email_003.txt
│
├── inbox/
├── review/
└── spam/
```

After running the automation, the messages are automatically distributed according to their predicted risk:

```text
mailbox/
├── incoming/
│
├── inbox/
│   └── email_001.txt
│
├── review/
│
└── spam/
    ├── email_002.txt
    └── email_003.txt
```

The exact destination depends on the spam probability generated by the trained model.

---

### 📝 Triage Logging

Every processed message is recorded in:

```text
triage_log.csv
```

The log contains:

```text
Timestamp
Filename
ML Prediction
Spam Probability
Triage Decision
```

This provides a record of how the automated triage system handled each incoming message.

---

### 🔄 Testing Again

After the first test, place new `.txt` files into:

```text
mailbox/incoming/
```

and run:

```bash
python src/automate.py
```

The system will process only the messages currently present in the `incoming` folder and move them to their corresponding destinations.

---

## 📁 Project Structure

```text
SmartMail/
│
├── data/
│   ├── SMSSpamCollection
│   └── cleaned_SMSSpamCollection.csv
│
├── models/
│   ├── smartmail_model.joblib
│   └── tfidf_vectorizer.joblib
│
├── plots/
│   ├── class_distribution.png
│   ├── model_comparison.png
│   └── confusion_matrix.png
│
├── mailbox/
│   ├── incoming/          # New messages for classification
│   ├── inbox/             # Low-risk messages
│   ├── review/            # Uncertain messages
│   └── spam/              # High-risk messages
│
├── src/
│   ├── inspect_data.py
│   ├── clean_data.py
│   ├── train_models.py
│   ├── predict.py
│   └── automate.py
│
├── triage_log.csv         # Automated triage history
├── README.md
├── .gitignore
└── .venv/
```

### Mailbox Processing Flow

```text
New Email
    │
    ▼
mailbox/incoming/
    │
    ▼
automate.py
    │
    ▼
TF-IDF Vectorization
    │
    ▼
Trained Linear SVM
    │
    ▼
Spam Probability
    │
    ├───────────────┬────────────────┐
    ▼               ▼                ▼
 < 40%           40–69%            ≥ 70%
    │               │                │
    ▼               ▼                ▼
 mailbox/         mailbox/         mailbox/
 inbox/           review/          spam/
    │               │                │
    └───────────────┴────────────────┘
                    │
                    ▼
             triage_log.csv
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** — data loading and preprocessing
* **NumPy** — numerical operations
* **Scikit-learn** — machine learning and evaluation
* **Matplotlib** — visualization
* **Joblib** — model persistence

### Machine Learning

* TF-IDF
* Multinomial Naive Bayes
* Logistic Regression
* Linear SVM
* Probability Calibration
* Train/Test Split
* Confusion Matrix
* Precision, Recall and F1 Evaluation

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Shouvanik21/SmartMail__ML-Based-Email-Spam-Detection-and-Automated-Triage.git
cd SmartMail
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

### 5. Prepare the dataset

```bash
python src/clean_data.py
```

### 6. Train and evaluate the models

```bash
python src/train_models.py
```

This trains the three classifiers, compares their performance, generates the model comparison and confusion matrix plots, and saves the selected model.

### 7. Test individual messages

```bash
python src/predict.py
```

### 8. Run automated email triage

Place `.txt` messages inside:

```text
mailbox/incoming/
```

Then run:

```bash
python src/automate.py
```

---

## 📌 Results

On the current cleaned dataset and fixed 80/20 stratified train/test split:

**Linear SVM**

* Accuracy: **98.26%**
* Precision: **95.93%**
* Spam Recall: **90.08%**
* F1 Score: **92.91%**
* TF-IDF Features: **7,680**
* Training Samples: **4,135**
* Testing Samples: **1,034**

These metrics describe the performance obtained in this specific experiment and may vary with different preprocessing, splits, or datasets.

---

## 👨‍💻 Project Goal

SmartMail was developed to demonstrate how traditional machine learning can be combined with automated decision workflows to build a practical **email risk detection and triage system**.

The project focuses on the complete ML lifecycle:

**Data → Preprocessing → Feature Engineering → Model Comparison → Evaluation → Model Persistence → Prediction → Automated Triage**
