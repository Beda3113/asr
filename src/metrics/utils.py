import editdistance


def calc_cer(target_text: str, predicted_text: str) -> float:
    """
    Calculate Character Error Rate (CER).

    Args:
        target_text: ground truth text
        predicted_text: recognized text from model

    Returns:
        CER value between 0.0 and 1.0
    """
    # Handle empty strings
    if target_text is None or predicted_text is None:
        return 1.0

    if len(target_text) == 0:
        return 0.0 if len(predicted_text) == 0 else 1.0

    # Calculate Levenshtein distance at character level
    distance = editdistance.eval(target_text, predicted_text)
    return distance / len(target_text)


def calc_wer(target_text: str, predicted_text: str) -> float:
    """
    Calculate Word Error Rate (WER).

    Args:
        target_text: ground truth text
        predicted_text: recognized text from model

    Returns:
        WER value between 0.0 and 1.0
    """
    # Handle empty strings
    if target_text is None or predicted_text is None:
        return 1.0

    # Split into words
    target_words = target_text.split()
    pred_words = predicted_text.split()

    if len(target_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0

    # Calculate Levenshtein distance at word level
    distance = editdistance.eval(target_words, pred_words)
    return distance / len(target_words)
