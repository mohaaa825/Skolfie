"""
ml_model.py
------------
The core AIDetectorModel class: an ensemble that combines

    1. Burrows' Delta         (stylometric distance to human/AI profiles)
    2. Traditional ML         (Random Forest + Logistic Regression over
                                stylometric + punctuation + POS + char
                                n-gram features)
    3. A small neural network (MLPClassifier -- the "basic neural features"
                                component) folded into the same soft-voting
                                ensemble

Final probability = weighted blend of the Burrows' Delta signal and the
ML ensemble's soft-voted probability. This mirrors how a human stylometry
expert would work: use robust distributional statistics (Delta) as a
sanity check alongside a more flexible learned model.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from feature_extraction import FeatureExtractor, tokenize_words, FUNCTION_WORDS, features_to_vector
from burrows_delta import BurrowsDeltaModel

# Weight given to the Burrows' Delta signal vs. the ML ensemble signal when
# blending the final probability. The ML ensemble generally becomes more
# accurate as more training data accumulates, while Burrows' Delta is more
# robust with very little data -- hence a meaningful but minority weight.
BURROWS_WEIGHT = 0.35
ML_WEIGHT = 1.0 - BURROWS_WEIGHT

# Friendly display names for the more cryptic internal feature keys.
_FRIENDLY_NAMES = {
    "avg_sentence_length": "Average sentence length",
    "sentence_length_std": "Sentence length variability",
    "sentence_length_burstiness": "Sentence length burstiness",
    "type_token_ratio": "Vocabulary diversity (type-token ratio)",
    "hapax_legomena_ratio": "Rare-word usage (hapax legomena ratio)",
    "yules_k": "Vocabulary richness (Yule's K)",
    "punct_density": "Overall punctuation density",
    "avg_word_length": "Average word length",
    "avg_syllables_per_word": "Average syllables per word",
    "flesch_reading_ease": "Flesch reading-ease score",
    "burrows_ai_probability": "Burrows' Delta stylistic distance",
    "burrows_delta_human": "Distance to human stylistic profile",
    "burrows_delta_ai": "Distance to AI stylistic profile",
    "noun_ratio": "Noun usage ratio",
    "verb_ratio": "Verb usage ratio",
    "adj_ratio": "Adjective usage ratio",
    "adv_ratio": "Adverb usage ratio",
    "pronoun_ratio": "Pronoun usage ratio",
    "repeated_bigram_ratio": "Repeated phrase (bigram) ratio",
    "stopword_ratio": "Stopword ratio",
    "comma_ratio": "Comma usage rate",
    "period_ratio": "Period usage rate",
    "exclaim_ratio": "Exclamation mark usage rate",
    "question_ratio": "Question mark usage rate",
    "semicolon_ratio": "Semicolon usage rate",
}


class AIDetectorModel:
    """Ensemble AI-vs-Human text detector (Burrows' Delta + ML ensemble)."""

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.burrows_model = BurrowsDeltaModel()
        self.tfidf = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), max_features=150, sublinear_tf=True
        )
        self.scaler = StandardScaler()
        self.ensemble: VotingClassifier = None
        self.feature_names: List[str] = []
        self.is_trained = False

    # -----------------------------------------------------------------
    # Feature matrix construction
    # -----------------------------------------------------------------
    def _build_feature_matrix(self, texts: List[str], fit: bool = False) -> np.ndarray:
        # 1. Stylometric feature dicts -> fixed-order numeric matrix
        stylo_dicts = [self.feature_extractor.extract_features(t) for t in texts]
        if fit or not self.feature_names:
            self.feature_names = sorted(stylo_dicts[0].keys())
        stylo_matrix = np.vstack(
            [features_to_vector(d, self.feature_names) for d in stylo_dicts]
        )

        # 2. Burrows' Delta signal as three extra numeric features
        burrows_rows = []
        for t in texts:
            tokens = tokenize_words(t)
            if self.burrows_model.vocabulary:
                s = self.burrows_model.score(tokens)
                burrows_rows.append([s["delta_human"], s["delta_ai"], s["ai_probability"]])
            else:
                burrows_rows.append([0.0, 0.0, 0.5])
        burrows_matrix = np.array(burrows_rows)

        # 3. Character n-gram TF-IDF (captures lower-level stylistic/typo signal)
        if fit:
            tfidf_matrix = self.tfidf.fit_transform(texts).toarray()
        else:
            tfidf_matrix = self.tfidf.transform(texts).toarray()

        return np.hstack([stylo_matrix, burrows_matrix, tfidf_matrix])

    def _all_feature_names(self) -> List[str]:
        burrows_names = ["burrows_delta_human", "burrows_delta_ai", "burrows_ai_probability"]
        tfidf_names = [f"char-pattern '{t}'" for t in self.tfidf.get_feature_names_out()]
        return list(self.feature_names) + burrows_names + tfidf_names

    @staticmethod
    def _friendly_name(name: str) -> str:
        if name.startswith("fw_"):
            return f"frequency of the word '{name[3:]}'"
        if name in _FRIENDLY_NAMES:
            return _FRIENDLY_NAMES[name]
        if name.startswith("char-pattern"):
            return name
        return name.replace("_", " ").capitalize()

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------
    def train(self, texts: List[str], labels: List[str],
              n_estimators: int = 200, random_state: int = 42) -> Dict:
        """
        Train the full ensemble.

        texts  : list of raw text strings
        labels : matching list of "human" / "ai" string labels
        """
        tokens_list = [tokenize_words(t) for t in texts]
        self.burrows_model.fit(
            tokens_list, labels, n_words=60, candidate_vocab=FUNCTION_WORDS
        )

        combined = self._build_feature_matrix(texts, fit=True)
        scaled = self.scaler.fit_transform(combined)

        rf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=12,
            random_state=random_state, class_weight="balanced",
        )
        lr = LogisticRegression(max_iter=2000, class_weight="balanced")
        mlp = MLPClassifier(  # the "basic neural features" ensemble member
            hidden_layer_sizes=(64, 16), max_iter=1000,
            random_state=random_state, early_stopping=True,
        )

        self.ensemble = VotingClassifier(
            estimators=[("rf", rf), ("lr", lr), ("mlp", mlp)], voting="soft"
        )
        self.ensemble.fit(scaled, labels)
        self.is_trained = True

        # Quick self-eval on the training data. This is NOT a substitute for
        # a held-out test set -- it's just a sanity-check number shown in
        # the UI after retraining.
        preds = self.ensemble.predict(scaled)
        acc = float(np.mean(preds == np.array(labels)))
        return {"train_accuracy": acc, "n_samples": len(texts)}

    # -----------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------
    def predict(self, text: str) -> Dict:
        if not self.is_trained:
            raise RuntimeError(
                "Model has not been trained yet. Call train() or load a saved model."
            )

        tokens = tokenize_words(text)
        burrows_result = self.burrows_model.score(tokens)

        combined = self._build_feature_matrix([text], fit=False)
        scaled = self.scaler.transform(combined)

        ml_proba = self.ensemble.predict_proba(scaled)[0]
        classes = list(self.ensemble.classes_)
        ai_idx = classes.index("ai")
        ml_ai_probability = float(ml_proba[ai_idx])

        combined_ai_probability = (
            BURROWS_WEIGHT * burrows_result["ai_probability"] + ML_WEIGHT * ml_ai_probability
        )
        combined_ai_probability = float(np.clip(combined_ai_probability, 0.0, 1.0))

        label = "AI-Generated" if combined_ai_probability >= 0.5 else "Human-Written"
        confidence = self._confidence_level(combined_ai_probability)

        top_features = self._top_contributing_features(scaled[0])

        per_word = burrows_result["per_word_diff"]
        top_words = sorted(per_word.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]

        return {
            "label": label,
            "ai_probability": combined_ai_probability,
            "human_probability": 1.0 - combined_ai_probability,
            "confidence": confidence,
            "ml_ai_probability": ml_ai_probability,
            "burrows_ai_probability": burrows_result["ai_probability"],
            "burrows_delta_human": burrows_result["delta_human"],
            "burrows_delta_ai": burrows_result["delta_ai"],
            "burrows_top_words": top_words,
            "top_features": top_features,
            "raw_features": self.feature_extractor.extract_features(text),
        }

    # -----------------------------------------------------------------
    def _top_contributing_features(self, scaled_row: np.ndarray, top_k: int = 8) -> List[Dict]:
        """
        Approximate per-prediction feature contributions using the fitted
        Random Forest's global feature importances, weighted by how far
        this particular text's (scaled) feature value sits from the mean.
        This is a lightweight explainability heuristic, not a formal
        method like SHAP -- but it's fast and dependency-free, and good
        enough to point users toward the features that mattered most.
        """
        try:
            rf = self.ensemble.named_estimators_["rf"]
            importances = rf.feature_importances_
        except Exception:
            return []

        contributions = importances * np.abs(scaled_row)
        names = self._all_feature_names()
        idx_sorted = np.argsort(contributions)[::-1][:top_k]

        results = []
        for idx in idx_sorted:
            if idx >= len(names):
                continue
            name = names[idx]
            direction = "above average" if scaled_row[idx] > 0 else "below average"
            results.append({
                "feature": self._friendly_name(name),
                "importance": float(contributions[idx]),
                "direction": direction,
            })
        return results

    @staticmethod
    def _confidence_level(ai_probability: float) -> str:
        distance = abs(ai_probability - 0.5)
        if distance >= 0.35:
            return "High"
        elif distance >= 0.15:
            return "Medium"
        return "Low"

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------
    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "AIDetectorModel":
        return joblib.load(path)
