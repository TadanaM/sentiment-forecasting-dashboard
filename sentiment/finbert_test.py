# =============================================================================
# finbert_test.py
# PURPOSE: Validate that FinBERT correctly classifies financial sentiment.
#          This file is NOT part of the forecasting pipeline.
#          It exists to demonstrate that FinBERT works correctly before
#          we use it to generate the sentiment index.
# =============================================================================

from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch

def load_finbert():
    """Load FinBERT model and tokenizer from HuggingFace."""
    print("Loading FinBERT model...")
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
    nlp = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1  # use GPU if available
    )
    print("FinBERT loaded successfully.\n")
    return nlp


def run_validation(nlp):
    """
    Run FinBERT on a set of clearly positive, negative, and neutral
    financial sentences to confirm correct behaviour.
    """

    test_sentences = [
        # --- Positive ---
        ("Apple reported record quarterly earnings, beating analyst expectations.",   "POSITIVE"),
        ("The company announced a significant increase in revenue and profit margins.", "POSITIVE"),
        ("Investors responded positively after the dividend increase was confirmed.",  "POSITIVE"),

        # --- Negative ---
        ("The firm reported a major loss and is facing potential bankruptcy.",          "NEGATIVE"),
        ("Apple's stock fell sharply after weaker-than-expected iPhone sales.",        "NEGATIVE"),
        ("Rising inflation and interest rates weighed heavily on the tech sector.",    "NEGATIVE"),

        # --- Neutral ---
        ("The company will hold its annual general meeting next Tuesday.",             "NEUTRAL"),
        ("Apple Inc. is scheduled to release its quarterly earnings report on Friday.","NEUTRAL"),
        ("The board has acknowledged receipt of the shareholder proposal.",            "NEUTRAL"),
    ]

    print("=" * 65)
    print(f"{'Sentence':<50} {'Expected':<10} {'Predicted':<10} {'Score':<6}")
    print("=" * 65)

    correct = 0
    for sentence, expected_label in test_sentences:
        result = nlp(sentence, truncation=True, max_length=512)[0]
        predicted = result["label"].upper()
        score     = result["score"]
        match     = "✓" if predicted == expected_label else "✗"
        correct  += (predicted == expected_label)

        # Truncate sentence for display
        display = sentence[:47] + "..." if len(sentence) > 50 else sentence
        print(f"{display:<50} {expected_label:<10} {predicted:<10} {score:.3f}  {match}")

    print("=" * 65)
    print(f"\nValidation complete: {correct}/{len(test_sentences)} correct.\n")

    if correct >= 7:
        print("✅ FinBERT is working correctly. Ready to generate sentiment index.")
    else:
        print("⚠️  Some classifications unexpected. Check model loading.")


if __name__ == "__main__":
    nlp = load_finbert()
    run_validation(nlp)