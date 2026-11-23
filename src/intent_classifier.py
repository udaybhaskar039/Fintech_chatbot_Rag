
import re


def classify_intent(question):

    question = question.lower().strip()
    question = re.sub(r"\s+", " ", question)

    # 1. Greeting keywords
    greeting_keywords = [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening"
    ]

    # 2. Financial policy keywords
    policy_keywords = [
        "policy",
        "pol",
        "kyc",
        "chargeback",
        "dispute",
        "duplicate transaction",
        "duplicate debit",
        "pos",
        "pos transaction",
        "failed payment",
        "auto reversal",
        "auto-reversal",
        "auto refund",
        "auto-refund",
        "refund",
        "transaction limit",
        "credit limit",
        "blocked card limit",
        "beneficiary",
        "compliance",
        "suspicious transaction",
        "high value transaction",
        "account closure",
        "loan emi",
        "emi failure",
        "emi auto debit",
        "settlement cycle",
        "reversal",
        "overdue kyc"
    ]

    # 3. Financial operations keywords
    operations_keywords = [
        "account",
        "payment",
        "transaction",
        "failed",
        "pending",
        "settlement",
        "bank",
        "debit",
        "credit",
        "transfer",
        "balance",
        "withdrawal",
        "deposit",
        "customer",
        "money"
    ]

    # Helper function for whole-word / phrase matching
    def contains_keyword(keywords):
        return any(
            re.search(
                r"(?<!\w)" + re.escape(word) + r"(?!\w)",
                question
            )
            for word in keywords
        )

    # 4. Greeting detection
    # Only classify short greeting messages as greetings.
    if contains_keyword(greeting_keywords):
        if question in greeting_keywords or question in [
            "hello finexa",
            "hi finexa",
            "hey finexa"
        ]:
            return "greeting"

    # 5. Policy detection
    # Policy intent gets priority over general financial operations.
    if contains_keyword(policy_keywords):
        return "policy"

    # 6. Financial operations detection
    if contains_keyword(operations_keywords):
        return "financial_operations"

    # 7. Unsupported questions
    return "out_of_domain"

