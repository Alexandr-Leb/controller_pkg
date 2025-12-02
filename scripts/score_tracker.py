class ScoreClient:
    """
    Small helper for formatting and publishing messages to /score_tracker.

    Responsibilities:
    - Encodes team name, password, clue id, and clue text into the exact
      string format required by the competition.
    - Provides helpers for sending start/stop timer messages and clue
      predictions.
    - Centralizes scoring logic so the rest of the code never hard-codes
      score strings.
    """
