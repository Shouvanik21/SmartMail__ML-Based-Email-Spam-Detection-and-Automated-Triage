import pandas as pd
import os
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# --------------------------------------------------
# 1. FIND PROJECT DIRECTORY
# --------------------------------------------------

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------
# 2. LOAD CLEANED DATASET
# --------------------------------------------------

cleaned_data_path = os.path.join(project_dir, "data", "cleaned_SMSSpamCollection.csv")

df = pd.read_csv(cleaned_data_path)

print("Dataset loaded:", df.shape)


# --------------------------------------------------
# 3. SEPARATE INPUT AND TARGET
# --------------------------------------------------

X = df["message"]
y = df["label"]


# --------------------------------------------------
# 4. TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))


# --------------------------------------------------
# 5. TF-IDF
# --------------------------------------------------

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF features:", X_train_tfidf.shape[1])


# --------------------------------------------------
# 6. CREATE THREE MODELS
# --------------------------------------------------

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": CalibratedClassifierCV(LinearSVC(), cv=5),
}


# --------------------------------------------------
# 7. TRAIN AND EVALUATE ALL MODELS
# --------------------------------------------------

results = []

trained_models = {}


for model_name, model in models.items():

    print("\nTraining:", model_name)

    # Train
    model.fit(X_train_tfidf, y_train)

    # Predict
    y_pred = model.predict(X_test_tfidf)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred, pos_label="spam")

    recall = recall_score(y_test, y_pred, pos_label="spam")

    f1 = f1_score(y_test, y_pred, pos_label="spam")

    # Store results
    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
        }
    )

    # Store trained model
    trained_models[model_name] = model


# --------------------------------------------------
# 8. CREATE COMPARISON TABLE
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n")
print("=" * 15)
print("MODEL COMPARISON")
print("=" * 15)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format,
        },
    )
)


# --------------------------------------------------
# 9. FIND MODEL WITH HIGHEST F1 SCORE
# --------------------------------------------------

best_model_name = results_df.loc[results_df["F1 Score"].idxmax(), "Model"]

print("\nBest model based on F1 Score:")
print(best_model_name)

# --------------------------------------------------
# 10. SAVE BEST MODEL
# --------------------------------------------------

best_model = trained_models[best_model_name]

model_path = os.path.join(project_dir, "models", "smartmail_model.joblib")

vectorizer_path = os.path.join(project_dir, "models", "tfidf_vectorizer.joblib")

joblib.dump(best_model, model_path)

joblib.dump(vectorizer, vectorizer_path)

print("\nBest model saved to:")
print(model_path)

print("\nTF-IDF vectorizer saved to:")
print(vectorizer_path)

# --------------------------------------------------
# 11. CREATE COMPARISON GRAPH
# --------------------------------------------------

results_plot = results_df.set_index("Model")

results_plot[["Accuracy", "Precision", "Recall", "F1 Score"]].plot(
    kind="bar", figsize=(10, 6)
)

plt.title("ML Model Performance Comparison")

plt.xlabel("Model")

plt.ylabel("Score")

plt.ylim(0, 1)

plt.xticks(rotation=0)

plt.legend()

plt.tight_layout()

# --------------------------------------------------
# CONFUSION MATRIX FOR BEST MODEL
# --------------------------------------------------

best_model = trained_models[best_model_name]

best_predictions = best_model.predict(X_test_tfidf)

cm = confusion_matrix(y_test, best_predictions, labels=["ham", "spam"])

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print("\n                 Predicted")
print("              Ham       Spam")

print(f"Actual Ham    {cm[0][0]:<8} {cm[0][1]}")

print(f"Actual Spam   {cm[1][0]:<8} {cm[1][1]}")


# --------------------------------------------------
# CONFUSION MATRIX GRAPH
# --------------------------------------------------

plt.figure(figsize=(7, 5))

plt.imshow(cm)

plt.title(f"Confusion Matrix - {best_model_name}")

plt.xlabel("Predicted Label")

plt.ylabel("Actual Label")

plt.xticks([0, 1], ["Ham", "Spam"])

plt.yticks([0, 1], ["Ham", "Spam"])


# Add numbers to cells
for i in range(2):

    for j in range(2):

        plt.text(j, i, cm[i, j], ha="center", va="center")


plt.tight_layout()


cm_plot_path = os.path.join(project_dir, "plots", "confusion_matrix.png")

plt.savefig(cm_plot_path)

plt.show()

print("\nConfusion matrix saved to:")
print(cm_plot_path)

# --------------------------------------------------
# 12. SAVE GRAPH
# --------------------------------------------------

plot_path = os.path.join(project_dir, "plots", "model_comparison.png")

plt.savefig(plot_path)
print("\nComparison graph saved to:", plot_path)

plt.show()
