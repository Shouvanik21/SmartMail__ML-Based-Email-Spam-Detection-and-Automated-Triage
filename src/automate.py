import os
import shutil
import joblib
import pandas as pd
from datetime import datetime

# ==================================================
# 1. PROJECT PATHS
# ==================================================

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ==================================================
# 2. MODEL PATHS
# ==================================================

model_path = os.path.join(project_dir, "models", "smartmail_model.joblib")

vectorizer_path = os.path.join(project_dir, "models", "tfidf_vectorizer.joblib")


# ==================================================
# 3. MAILBOX PATHS
# ==================================================

mailbox_dir = os.path.join(project_dir, "mailbox")

incoming_dir = os.path.join(mailbox_dir, "incoming")

inbox_dir = os.path.join(mailbox_dir, "inbox")

review_dir = os.path.join(mailbox_dir, "review")

spam_dir = os.path.join(mailbox_dir, "spam")


# ==================================================
# 4. LOG PATH
# ==================================================

log_path = os.path.join(project_dir, "triage_log.csv")


# ==================================================
# 5. CREATE MAILBOX DIRECTORIES
# ==================================================

os.makedirs(incoming_dir, exist_ok=True)

os.makedirs(inbox_dir, exist_ok=True)

os.makedirs(review_dir, exist_ok=True)

os.makedirs(spam_dir, exist_ok=True)


# ==================================================
# 6. LOAD TRAINED MODEL
# ==================================================

model = joblib.load(model_path)

vectorizer = joblib.load(vectorizer_path)


print("\n==========================================")
print("SMARTMAIL AUTOMATED TRIAGE")
print("==========================================")

print("\nSaved ML model loaded successfully.")


# ==================================================
# 7. CLASSIFICATION FUNCTION
# ==================================================


def classify_email(message):

    # Convert email to TF-IDF
    message_tfidf = vectorizer.transform([message])

    # Predict
    prediction = model.predict(message_tfidf)[0]

    # Probability
    probabilities = model.predict_proba(message_tfidf)[0]

    spam_index = list(model.classes_).index("spam")

    spam_probability = probabilities[spam_index] * 100

    # Determine destination
    if spam_probability >= 70:

        decision = "SPAM"
        destination = spam_dir

    elif spam_probability >= 40:

        decision = "REVIEW"
        destination = review_dir

    else:

        decision = "INBOX"
        destination = inbox_dir

    return (prediction, spam_probability, decision, destination)


# ==================================================
# 8. PROCESS INCOMING EMAILS
# ==================================================

log_entries = []


for filename in os.listdir(incoming_dir):

    file_path = os.path.join(incoming_dir, filename)

    # Only process TXT files
    if not filename.endswith(".txt"):
        continue

    # Read email
    with open(file_path, "r", encoding="utf-8") as file:

        message = file.read()

    # Classify
    prediction, spam_probability, decision, destination = classify_email(message)

    # Move email
    destination_path = os.path.join(destination, filename)

    shutil.move(file_path, destination_path)

    # Display result
    print("\nEmail:", filename)

    print(f"Prediction: {prediction}")

    print(f"Spam Probability: {spam_probability:.2f}%")

    print(f"Decision: {decision}")

    print(f"Moved to: {os.path.basename(destination)}")

    # Log
    log_entries.append(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": filename,
            "prediction": prediction,
            "spam_probability": round(spam_probability, 2),
            "decision": decision,
        }
    )


# ==================================================
# 9. SAVE TRIAGE LOG
# ==================================================

if log_entries:

    log_df = pd.DataFrame(log_entries)

    if os.path.exists(log_path):

        log_df.to_csv(log_path, mode="a", header=False, index=False)

    else:

        log_df.to_csv(log_path, index=False)


print("\n==========================================")
print("TRIAGE COMPLETE")
print("==========================================")

print("\nTriage log:")
print(log_path)
