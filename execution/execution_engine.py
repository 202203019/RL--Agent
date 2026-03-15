from typing import Dict

class ExecutionEngine:
    def __init__(self, threshold: float = 0.40):
        self.threshold = threshold

    def evaluate_candidate(self, probs: Dict[str, float]) -> bool:
        """Step 3: Check Candidate Threshold"""
        return max(probs["P_DOWN"], probs["P_UP"]) >= self.threshold

    def get_signal_direction(self, probs: Dict[str, float]) -> str:
        """Step 3: BUY if P_UP >= P_DOWN, else SELL"""
        return "BUY" if probs["P_UP"] >= probs["P_DOWN"] else "SELL"
