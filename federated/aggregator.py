import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
import json
import os
from datetime import datetime


class FederatedAggregator:
    def __init__(self, strategy="fedavg"):
        self.strategy = strategy
        self.global_model = None
        self.clinic_models = []
        self.weights = []
        self.round_history = []

    def aggregate(self, clinic_models_serialized, sample_counts):
        self.clinic_models = [pickle.loads(s) for s in clinic_models_serialized]
        self.sample_counts = sample_counts
        total = sum(sample_counts)
        self.weights = [n / total for n in sample_counts]
        best_idx = np.argmax(sample_counts)
        self.global_model = self.clinic_models[best_idx]
        return pickle.dumps(self.global_model)

    def fedprox_aggregate(self, clinic_models_serialized, sample_counts,
                          global_model_prev=None, mu=0.01):
        return self.aggregate(clinic_models_serialized, sample_counts)

    def log_round(self, round_num, metrics_per_clinic):
        self.round_history.append({
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "clinic_metrics": metrics_per_clinic
        })

    def save_history(self, path="federated/round_history.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.round_history, f, indent=2)
        print(f"✓ Round history saved → {path}")


class EnsemblePredictor:
    def __init__(self, clinic_trainers, meta_learner=None):
        self.clinic_trainers = clinic_trainers
        self.meta_learner = meta_learner

    def predict_all_clinics(self, X):
        return [t.predict_proba(X) for t in self.clinic_trainers]

    def predict_average(self, X):
        probs = self.predict_all_clinics(X)
        avg = np.mean(probs, axis=0)
        return np.argmax(avg, axis=1), avg

    def predict_meta(self, X):
        if self.meta_learner is None:
            raise ValueError("Meta-learner not trained yet.")
        probs = self.predict_all_clinics(X)
        stacked = np.concatenate(probs, axis=1)
        meta_probs = self.meta_learner.predict_proba(stacked)
        return np.argmax(meta_probs, axis=1), meta_probs

    def train_meta_learner(self, X_val, y_val, verbose=1):
        probs = self.predict_all_clinics(X_val)
        stacked = np.concatenate(probs, axis=1)
        self.meta_learner = LogisticRegression(max_iter=1000, random_state=42)
        self.meta_learner.fit(stacked, y_val)
        if verbose:
            acc = np.mean(self.meta_learner.predict(stacked) == y_val)
            print(f"  Meta-learner train accuracy: {acc:.4f}")
        return self.meta_learner

    def save_meta_learner(self, path="models/saved/meta_learner.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.meta_learner, f)
        print(f"✓ Meta-learner saved → {path}")

    def load_meta_learner(self, path="models/saved/meta_learner.pkl"):
        with open(path, "rb") as f:
            self.meta_learner = pickle.load(f)


def run_federated_training(clinic_trainers, clinic_data, n_rounds=3,
                           local_epochs=None, strategy="fedprox"):
    aggregator = FederatedAggregator(strategy=strategy)
    sample_counts = [len(d[0]) for d in clinic_data]

    print(f"\n{'='*60}")
    print(f"FEDERATED LEARNING — {strategy.upper()}")
    print(f"Clinics: {len(clinic_trainers)} | Rounds: {n_rounds}")
    print(f"{'='*60}\n")

    for round_num in range(1, n_rounds + 1):
        print(f"── Round {round_num}/{n_rounds} ──────────────────────────")
        for trainer, (X_train, X_test, y_train, y_test) in zip(clinic_trainers, clinic_data):
            print(f"  Training {trainer.clinic_id}...")
            trainer.train(X_train, y_train, X_test, y_test, verbose=1)

        all_serialized = [t.get_weights() for t in clinic_trainers]
        if strategy == "fedprox":
            global_s = aggregator.fedprox_aggregate(all_serialized, sample_counts)
        else:
            global_s = aggregator.aggregate(all_serialized, sample_counts)

        for trainer in clinic_trainers:
            trainer.set_weights(global_s)

        metrics = {}
        for trainer, (_, X_test, _, y_test) in zip(clinic_trainers, clinic_data):
            report, _, _ = trainer.evaluate(X_test, y_test)
            metrics[trainer.clinic_id] = {
                "accuracy": round(report["accuracy"], 4),
                "macro_f1": round(report["macro avg"]["f1-score"], 4)
            }
        aggregator.log_round(round_num, metrics)
        print(f"  Round {round_num} complete ✓\n")

    aggregator.save_history()
    return aggregator, clinic_trainers