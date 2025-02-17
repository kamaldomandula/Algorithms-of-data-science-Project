def calculate_score(content_relevance: int, bias_score: int) -> int:
    """
    Calculates a simple final score based on content relevance and bias score.

    Input args:
      content_relevance: int - Score indicating how relevant the content is to the query.
      bias_score: int - Score indicating the neutrality or bias of the content.

    Return:
      final_score: int - A weighted average of the two scores.
    """
    return int((0.5 * content_relevance) + (0.5 * bias_score))

if __name__ == "__main__":
    relevance = int(input("Enter a content relevance score (0-100): "))
    bias = int(input("Enter a bias score (0-100): "))

    final_score = calculate_score(relevance, bias)
    print(f"Computed final validity score: {final_score}")
