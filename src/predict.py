import os
import joblib

# ==================================================
# 1. FIND PROJECT DIRECTORY
# ==================================================

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ==================================================
# 2. MODEL PATHS
# ==================================================

model_path = os.path.join(project_dir, "models", "smartmail_model.joblib")

vectorizer_path = os.path.join(project_dir, "models", "tfidf_vectorizer.joblib")


# ==================================================
# 3. LOAD TRAINED MODEL
# ==================================================

model = joblib.load(model_path)

vectorizer = joblib.load(vectorizer_path)


print("\n==========================================")
print("SMARTMAIL AI PREDICTION ENGINE")
print("==========================================")

print("\nSaved model loaded successfully.")


# ==================================================
# 4. PREDICTION FUNCTION
# ==================================================


def predict_message(message):

    # Convert message into TF-IDF
    message_tfidf = vectorizer.transform([message])

    # Predict class
    prediction = model.predict(message_tfidf)[0]

    # Get probabilities
    probabilities = model.predict_proba(message_tfidf)[0]

    # Find spam probability
    spam_index = list(model.classes_).index("spam")

    spam_probability = probabilities[spam_index] * 100

    return (prediction, spam_probability)


# ==================================================
# 5. TRIAGE FUNCTION
# ==================================================


def get_triage_decision(spam_probability):

    if spam_probability >= 70:

        return "SPAM"

    elif spam_probability >= 40:

        return "REVIEW"

    else:

        return "INBOX"


# ==================================================
# 6. TEST MESSAGES
# ==================================================

messages = [
    "Congratulations! You have won a free prize. Call now to claim.",
    "Hey, are we still meeting at 6 pm today?",
    "URGENT! You have been selected for a cash reward. Reply WIN now!",
    "Can you send me the notes from today's class?",
    "Your account has been selected for a special cash reward. Click now!",
]


# ==================================================
# 7. RUN PREDICTIONS
# ==================================================

for message in messages:

    prediction, spam_probability = predict_message(message)

    decision = get_triage_decision(spam_probability)

    print("\nMessage:")
    print(message)

    print(f"\nML Prediction: {prediction}")

    print(f"Spam Probability: {spam_probability:.2f}%")

    print(f"Triage Decision: {decision}")

    print("-" * 60)
