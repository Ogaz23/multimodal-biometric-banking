import csv
import numpy as np
import matplotlib.pyplot as plt

def load_scores(log_file, score_col="score", lower_is_better=False):
    """Read a verification log CSV and separate genuine vs impostor scores."""
    genuine_scores = []
    impostor_scores = []

    with open(log_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = float(row[score_col])
            if row["attempt_type"] == "genuine":
                genuine_scores.append(score)
            elif row["attempt_type"] == "impostor":
                impostor_scores.append(score)

    return np.array(genuine_scores), np.array(impostor_scores)

def compute_far_frr(genuine_scores, impostor_scores, thresholds, lower_is_better=False):
    """For each threshold, compute FAR and FRR."""
    far_list = []
    frr_list = []

    for t in thresholds:
        if lower_is_better:
            # Accept if score <= threshold
            false_accepts = np.sum(impostor_scores <= t)
            false_rejects = np.sum(genuine_scores > t)
        else:
            # Accept if score >= threshold
            false_accepts = np.sum(impostor_scores >= t)
            false_rejects = np.sum(genuine_scores < t)

        far = false_accepts / len(impostor_scores) if len(impostor_scores) > 0 else 0
        frr = false_rejects / len(genuine_scores) if len(genuine_scores) > 0 else 0

        far_list.append(far)
        frr_list.append(frr)

    return np.array(far_list), np.array(frr_list)

def find_eer(thresholds, far_list, frr_list):
    """Find the threshold where FAR and FRR are closest (EER point)."""
    diffs = np.abs(far_list - frr_list)
    idx = np.argmin(diffs)
    eer = (far_list[idx] + frr_list[idx]) / 2
    return thresholds[idx], eer

def evaluate_modality(name, log_file, score_col="score", lower_is_better=False, num_thresholds=200):
    print(f"\n{'='*55}")
    print(f"  {name.upper()} MODALITY EVALUATION")
    print(f"{'='*55}")

    genuine, impostor = load_scores(log_file, score_col, lower_is_better)

    print(f"Genuine attempts: {len(genuine)}  |  Impostor attempts: {len(impostor)}")
    print(f"Genuine scores  -> mean: {genuine.mean():.4f}, std: {genuine.std():.4f}, range: [{genuine.min():.4f}, {genuine.max():.4f}]")
    print(f"Impostor scores -> mean: {impostor.mean():.4f}, std: {impostor.std():.4f}, range: [{impostor.min():.4f}, {impostor.max():.4f}]")

    all_scores = np.concatenate([genuine, impostor])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), num_thresholds)

    far_list, frr_list = compute_far_frr(genuine, impostor, thresholds, lower_is_better)
    eer_threshold, eer = find_eer(thresholds, far_list, frr_list)

    print(f"\n📊 EER (Equal Error Rate): {eer*100:.2f}%")
    print(f"📊 Optimal threshold (at EER): {eer_threshold:.4f}")

    return {
        "name": name,
        "genuine": genuine,
        "impostor": impostor,
        "thresholds": thresholds,
        "far": far_list,
        "frr": frr_list,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "lower_is_better": lower_is_better
    }

def plot_results(results_list):
    fig, axes = plt.subplots(2, len(results_list), figsize=(6*len(results_list), 10))

    if len(results_list) == 1:
        axes = axes.reshape(2, 1)

    for i, r in enumerate(results_list):
        # Top row: score distributions
        ax1 = axes[0, i]
        ax1.hist(r["genuine"], bins=15, alpha=0.6, label="Genuine", color="green")
        ax1.hist(r["impostor"], bins=15, alpha=0.6, label="Impostor", color="red")
        ax1.set_title(f"{r['name']} - Score Distribution")
        ax1.set_xlabel("Score" if not r["lower_is_better"] else "Hamming Distance")
        ax1.set_ylabel("Frequency")
        ax1.legend()

        # Bottom row: FAR/FRR curves
        ax2 = axes[1, i]
        ax2.plot(r["thresholds"], r["far"]*100, label="FAR", color="red")
        ax2.plot(r["thresholds"], r["frr"]*100, label="FRR", color="blue")
        ax2.axvline(r["eer_threshold"], color="gray", linestyle="--", label=f"EER threshold")
        ax2.set_title(f"{r['name']} - FAR/FRR (EER={r['eer']*100:.1f}%)")
        ax2.set_xlabel("Threshold")
        ax2.set_ylabel("Error Rate (%)")
        ax2.legend()

    plt.tight_layout()
    plt.savefig("evaluation_results.png", dpi=150)
    print("\n📈 Plots saved to evaluation_results.png")
    plt.show()

if __name__ == "__main__":
    results = []

    try:
        results.append(evaluate_modality("Face", "face_verification_log.csv",
                                          score_col="score", lower_is_better=False))
    except FileNotFoundError:
        print("⚠️ face_verification_log.csv not found, skipping.")

    try:
        results.append(evaluate_modality("Voice", "voice_verification_log.csv",
                                          score_col="score", lower_is_better=False))
    except FileNotFoundError:
        print("⚠️ voice_verification_log.csv not found, skipping.")

    try:
        results.append(evaluate_modality("Iris", "iris_verification_log.csv",
                                          score_col="hamming_distance", lower_is_better=True))
    except FileNotFoundError:
        print("⚠️ iris_verification_log.csv not found, skipping.")

    if results:
        print(f"\n{'='*55}")
        print("  SUMMARY")
        print(f"{'='*55}")
        for r in results:
            print(f"  {r['name']}: EER = {r['eer']*100:.2f}%")

        plot_results(results)