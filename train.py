import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import json
import pickle

from utils.preprocessing import load_clinic_data, save_scaler
from models.clinic_model import ClinicTrainer, TriageRLAgent
from federated.aggregator import run_federated_training, EnsemblePredictor

CLINIC_IDS = ["clinic_1", "clinic_2", "clinic_3"]
N_FEDERATED_ROUNDS = 3
FEDERATED_STRATEGY = "fedprox"


def main():
    os.makedirs("models/saved", exist_ok=True)
    os.makedirs("federated", exist_ok=True)

    print("STEP 1: Loading clinic data...")
    clinic_data = []
    for clinic_id in CLINIC_IDS:
        X_train, X_test, y_train, y_test, scaler = load_clinic_data(clinic_id)
        clinic_data.append((X_train, X_test, y_train, y_test))
        save_scaler(scaler, f"models/saved/{clinic_id}_scaler.pkl")
        print(f"  {clinic_id}: {X_train.shape[0]} train, {X_test.shape[0]} test")

    print("\nSTEP 2: Baseline training...")
    clinic_trainers = [ClinicTrainer(cid) for cid in CLINIC_IDS]
    baseline_results = {}
    for trainer, (X_train, X_test, y_train, y_test) in zip(clinic_trainers, clinic_data):
        trainer.train(X_train, y_train, X_test, y_test, verbose=0)
        report, _, _ = trainer.evaluate(X_test, y_test)
        baseline_results[trainer.clinic_id] = {
            "accuracy": round(report["accuracy"], 4),
            "macro_f1": round(report["macro avg"]["f1-score"], 4)
        }
        print(f"  {trainer.clinic_id} baseline: {report['accuracy']:.4f}")

    print(f"\nSTEP 3: Federated Learning ({FEDERATED_STRATEGY})...")
    aggregator, clinic_trainers = run_federated_training(
        clinic_trainers, clinic_data,
        n_rounds=N_FEDERATED_ROUNDS,
        strategy=FEDERATED_STRATEGY
    )

    print("\nSTEP 4: Post-federated evaluation...")
    federated_results = {}
    for trainer, (_, X_test, _, y_test) in zip(clinic_trainers, clinic_data):
        report, _, _ = trainer.evaluate(X_test, y_test)
        federated_results[trainer.clinic_id] = {
            "accuracy": round(report["accuracy"], 4),
            "macro_f1": round(report["macro avg"]["f1-score"], 4)
        }
        print(f"  {trainer.clinic_id} federated: {report['accuracy']:.4f}")
        trainer.save()

    print("\nSTEP 5: Training RL agents...")
    rl_agents = {}
    for trainer, (_, X_test, _, y_test) in zip(clinic_trainers, clinic_data):
        agent = TriageRLAgent()
        probs = trainer.predict_proba(X_test)
        for _ in range(10):
            avg_reward = agent.update_thresholds(probs, y_test)
        rl_agents[trainer.clinic_id] = agent
        print(f"  {trainer.clinic_id} RL reward: {avg_reward:.4f}")
    with open("models/saved/rl_agents.pkl", "wb") as f:
        pickle.dump(rl_agents, f)

    print("\nSTEP 6: Training meta-learner...")
    X_val = np.concatenate([d[1] for d in clinic_data])
    y_val = np.concatenate([d[3] for d in clinic_data])
    ensemble = EnsemblePredictor(clinic_trainers)
    ensemble.train_meta_learner(X_val, y_val, verbose=1)
    ensemble.save_meta_learner()

    y_pred_avg, _ = ensemble.predict_average(X_val)
    y_pred_meta, _ = ensemble.predict_meta(X_val)
    avg_acc = float(np.mean(y_pred_avg == y_val))
    meta_acc = float(np.mean(y_pred_meta == y_val))
    print(f"  Ensemble average : {avg_acc:.4f}")
    print(f"  Meta-learner     : {meta_acc:.4f}")

    summary = {
        "baseline": baseline_results,
        "federated": federated_results,
        "ensemble": {
            "average_accuracy": round(avg_acc, 4),
            "meta_learner_accuracy": round(meta_acc, 4)
        },
        "improvement": {
            cid: round(federated_results[cid]["accuracy"] - baseline_results[cid]["accuracy"], 4)
            for cid in CLINIC_IDS
        }
    }
    with open("federated/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(json.dumps(summary, indent=2))
    print("\nNext: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()