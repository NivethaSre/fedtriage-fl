import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

NUM_CLASSES = 3

TRIAGE_COST_MATRIX = np.array([
    [0.0, 0.5, 2.0],
    [1.5, 0.0, 0.5],
    [5.0, 2.0, 0.0],
], dtype=np.float32)


class ClinicTrainer:
    def __init__(self, clinic_id, learning_rate=0.1):
        self.clinic_id = clinic_id
        self.model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=learning_rate,
            max_depth=4, random_state=42
        )
        self.is_trained = False

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=None, batch_size=None, verbose=1):
        sample_weights = self._compute_sample_weights(y_train)
        self.model.fit(X_train, y_train, sample_weight=sample_weights)
        self.is_trained = True
        if verbose and X_val is not None:
            report, _, _ = self.evaluate(X_val, y_val)
            print(f"    {self.clinic_id} val accuracy: {report['accuracy']:.4f}")
        return self

    def _compute_sample_weights(self, y):
        weights = np.ones(len(y), dtype=np.float32)
        weights[y == 2] = 3.0
        weights[y == 1] = 1.5
        weights[y == 0] = 1.0
        return weights

    def get_weights(self):
        return pickle.dumps(self.model)

    def set_weights(self, serialized):
        self.model = pickle.loads(serialized)
        self.is_trained = True

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        labels = ["Green", "Yellow", "Red"]
        report = classification_report(
            y_test, y_pred, target_names=labels,
            output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred)
        return report, cm, y_pred

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path=None):
        if path is None:
            path = f"models/saved/{self.clinic_id}_model.pkl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"✓ {self.clinic_id} model saved → {path}")

    def load(self, path=None):
        if path is None:
            path = f"models/saved/{self.clinic_id}_model.pkl"
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True


class TriageRLAgent:
    def __init__(self, n_classes=3):
        self.thresholds = np.array([0.4, 0.35, 0.25])
        self.rewards_history = []

    def act(self, probs):
        if probs[2] >= self.thresholds[2]: return 2
        elif probs[1] >= self.thresholds[1]: return 1
        else: return 0

    def compute_reward(self, predicted, true_label):
        return -TRIAGE_COST_MATRIX[true_label][predicted]

    def update_thresholds(self, probs_batch, true_labels, learning_rate=0.01):
        total_reward = 0
        for probs, true in zip(probs_batch, true_labels):
            pred = self.act(probs)
            reward = self.compute_reward(pred, int(true))
            total_reward += reward
            if true == 2 and pred < 2:
                self.thresholds[2] = max(0.1, self.thresholds[2] - learning_rate)
            elif true == 0 and pred == 2:
                self.thresholds[2] = min(0.6, self.thresholds[2] + learning_rate * 0.5)
        avg_reward = total_reward / max(len(probs_batch), 1)
        self.rewards_history.append(avg_reward)
        return avg_reward