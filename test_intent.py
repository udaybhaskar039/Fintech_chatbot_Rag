
from src.intent_classifier import classify_intent

questions = [
    "Explain KYC policy",
    "What is a failed payment?",
    "Hello Finexa",
    "What is Python?"
]

for question in questions:
    intent = classify_intent(question)

    print("Question:", question)
    print("Detected Intent:", intent)
    print("-------------------------")
