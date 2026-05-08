# %%
# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

SEED      = 42
N_BINS    = 10          # bins for reliability diagram / ECE
FIGURES   = "figures"   
os.makedirs(FIGURES, exist_ok=True)

np.random.seed(SEED)

# %%
# =============================================================================
# SECTION 1 ─ DATA LOADING
# =============================================================================

def load_breast_cancer(path: str):
    """
    WDBC format: id, label (M/B), feature_1 … feature_30
    Label encoding: M (malignant) → 1,  B (benign) → 0
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 12:
                continue
            label = 1 if parts[1].strip() == "M" else 0
            feats = [float(v) for v in parts[2:]]
            rows.append([label] + feats)
    arr = np.array(rows, dtype=float)
    return arr[:, 1:], arr[:, 0].astype(int)   


def load_heart_disease(path: str):
    """
    Cleveland format: 13 features + target (0 = no disease, 1-4 = disease)
    Rows with '?' are dropped.  Target is binarised: disease = (target > 0).
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or "?" in line:
                continue
            vals = [float(v) for v in line.split(",")]
            rows.append(vals)
    arr = np.array(rows, dtype=float)
    y = (arr[:, -1] > 0).astype(int)
    return arr[:, :-1], y                       


# %%
# =============================================================================
# SECTION 2 ─ PRE-PROCESSING UTILITIES
# =============================================================================

def train_test_split(X, y, test_size=0.25, seed=42):
    """Stratified-ish split via random permutation."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = int(len(y) * test_size)
    return (X[idx[n_test:]], X[idx[:n_test]],
            y[idx[n_test:]], y[idx[:n_test]])


def standardise(X_tr, X_te):
    """Z-score normalisation fitted on train only (no leakage)."""
    mu  = X_tr.mean(0)
    std = X_tr.std(0) + 1e-8
    return (X_tr - mu) / std, (X_te - mu) / std

# %%
# =============================================================================
# SECTION 3 ─ METRICS (all from scratch)
# =============================================================================

def accuracy(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    return float(np.mean((p_hat >= 0.5).astype(int) == y_true))


def log_loss(y_true: np.ndarray, p_hat: np.ndarray, eps=1e-15) -> float:
    p = np.clip(p_hat, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def brier_score(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    return float(np.mean((y_true - p_hat) ** 2))


def ece(y_true: np.ndarray, p_hat: np.ndarray, n_bins=N_BINS) -> float:
    """
    Expected Calibration Error.
    ECE = Σ_m (|B_m| / n) * |acc(B_m) − conf(B_m)|
    """
    n = len(y_true)
    total = 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p_hat >= lo) & (p_hat < hi)
        if hi == 1.0:
            mask = (p_hat >= lo) & (p_hat <= hi)
        if mask.sum() == 0:
            continue
        conf = p_hat[mask].mean()
        acc  = y_true[mask].mean()
        total += (mask.sum() / n) * abs(acc - conf)
    return total


def all_metrics(y_true: np.ndarray, p_hat: np.ndarray) -> dict:
    return {
        "accuracy":  accuracy(y_true, p_hat),
        "log_loss":  log_loss(y_true, p_hat),
        "brier":     brier_score(y_true, p_hat),
        "ece":       ece(y_true, p_hat),
    }

# %%

# =============================================================================
# SECTION 4 ─ LOGISTIC REGRESSION 
# =============================================================================

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegression:
    """
    Binary logistic regression trained with mini-batch stochastic gradient descent.

    Loss:  L(w,b) = -mean[ y log σ(w·x+b) + (1−y) log(1−σ(w·x+b)) ] + λ||w||²/2
    Update rule:
        g_w = (1/|B|) X_B^T (p_B − y_B)  +  λ w
        g_b = (1/|B|) sum(p_B − y_B)
        w ← w − η g_w
        b ← b − η g_b
    """

    def __init__(self, lr=0.05, n_epochs=600, batch_size=32,
                 l2=1e-3, seed=42):
        self.lr         = lr
        self.n_epochs   = n_epochs
        self.batch_size = batch_size
        self.l2         = l2
        self.seed       = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0

        for _ in range(self.n_epochs):
            perm = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx   = perm[start : start + self.batch_size]
                Xb    = X[idx];  yb = y[idx]
                p     = _sigmoid(Xb @ self.w + self.b)
                err   = p - yb
                gw    = Xb.T @ err / len(yb) + self.l2 * self.w
                gb    = err.mean()
                self.w -= self.lr * gw
                self.b -= self.lr * gb
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(X @ self.w + self.b)



# %%
# =============================================================================
# SECTION 5 ─ RANDOM FOREST (from scratch — bootstrap + feature subsampling)
# =============================================================================

class RandomForest:
    """
    Ensemble of decision trees trained on bootstrap samples with random feature
    subsets.  Individual trees use sklearn's DecisionTreeClassifier as the
    splitting primitive; all ensemble logic is implemented here.

    Probability estimate: average of per-tree class-1 probabilities.
    """

    def __init__(self, n_trees=150, max_depth=8,
                 max_features="sqrt", seed=42):
        self.n_trees     = n_trees
        self.max_depth   = max_depth
        self.max_features = max_features
        self.seed        = seed

    def _n_features(self, d):
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(d)))
        if self.max_features == "log2":
            return max(1, int(np.log2(d)))
        return d

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
        rng    = np.random.default_rng(self.seed)
        n, d   = X.shape
        k      = self._n_features(d)
        self.estimators_  = []   

        for i in range(self.n_trees):
            # Bootstrap sample
            boot_idx  = rng.integers(0, n, size=n)
            feat_idx  = rng.choice(d, size=k, replace=False)

            tree = DecisionTreeClassifier(
                max_depth   = self.max_depth,
                random_state= int(rng.integers(0, 99999)),
            )
            tree.fit(X[boot_idx][:, feat_idx], y[boot_idx])
            self.estimators_.append((tree, feat_idx))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns class-1 probability (average over all trees)."""
        agg = np.zeros(len(X))
        for tree, feat_idx in self.estimators_:
            p = tree.predict_proba(X[:, feat_idx])
            if p.shape[1] == 2:
                agg += p[:, 1]
            else:
                agg += float(tree.classes_[0]) * np.ones(len(X))
        return agg / self.n_trees


# %%

# =============================================================================
# SECTION 6 ─ PLATT SCALING (from scratch — gradient descent on sigmoid)
# =============================================================================

class PlattScaling:
    """
    Post-hoc calibration via a learned sigmoid:
        p_cal(f) = σ(A·f + B)
    Parameters A, B are fitted on a held-out calibration set by minimising
    log-loss using gradient descent.

    Reference: Platt (1999), "Probabilistic Outputs for SVMs and Comparisons
               to Regularized Likelihood Methods."
    """

    def __init__(self, lr=0.005, n_iter=3000):
        self.lr     = lr
        self.n_iter = n_iter
        self.A      = 1.0    
        self.B      = 0.0

    def fit(self, f: np.ndarray, y: np.ndarray) -> "PlattScaling":
        """
        f : raw scores (logits or uncalibrated probabilities)
        y : true binary labels
        """
        f = np.asarray(f, float)
        y = np.asarray(y, float)
        for _ in range(self.n_iter):
            p   = _sigmoid(self.A * f + self.B)
            err = p - y
            dA  = np.dot(err, f) / len(y)
            dB  = err.mean()
            self.A -= self.lr * dA
            self.B -= self.lr * dB
        return self

    def predict_proba(self, f: np.ndarray) -> np.ndarray:
        return _sigmoid(self.A * np.asarray(f, float) + self.B)



# %%
# =============================================================================
# SECTION 7 ─ ISOTONIC REGRESSION (from scratch — PAV algorithm)
# =============================================================================

class IsotonicRegression:
    """
    Non-parametric monotone calibration via the Pool-Adjacent-Violators (PAV)
    algorithm.  Fits the best (in MSE sense) non-decreasing step function
    g: ℝ → [0, 1] on the calibration set.

    New predictions use piecewise-constant interpolation (nearest-neighbour).
    """

    def fit(self, f: np.ndarray, y: np.ndarray) -> "IsotonicRegression":
        f = np.asarray(f, float)
        y = np.asarray(y, float)
        order   = np.argsort(f)
        f_sorted = f[order]
        y_sorted = y[order]

        # Each block: [pooled_mean, count]
        blocks = [[float(y_sorted[0]), 1]]
        for val in y_sorted[1:]:
            blocks.append([float(val), 1])
            # Merge adjacent violating blocks
            while len(blocks) >= 2 and blocks[-2][0] >= blocks[-1][0]:
                m1, c1 = blocks[-2]
                m2, c2 = blocks[-1]
                merged = (m1 * c1 + m2 * c2) / (c1 + c2)
                blocks[-2:] = [[merged, c1 + c2]]

        # Expand blocks back into a full fitted array
        fitted = np.concatenate([[b[0]] * b[1] for b in blocks])

        # Store (sorted scores, fitted values) for interpolation
        self._f_sorted  = f_sorted
        self._fitted    = np.clip(fitted, 0.0, 1.0)
        return self

    def predict_proba(self, f: np.ndarray) -> np.ndarray:
        """Piecewise-constant (step-function) interpolation."""
        return np.clip(
            np.interp(np.asarray(f, float),
                      self._f_sorted, self._fitted,
                      left  = self._fitted[0],
                      right = self._fitted[-1]),
            0.0, 1.0
        )


# %%

# =============================================================================
# SECTION 8 ─ RELIABILITY DIAGRAM HELPER
# =============================================================================

def reliability_diagram_data(y_true: np.ndarray, p_hat: np.ndarray,
                              n_bins=N_BINS):
    """
    Returns:
      mean_pred  : mean predicted probability per bin  (nan if empty)
      frac_pos   : fraction of positives per bin       (nan if empty)
      counts     : number of samples per bin
    """
    edges      = np.linspace(0, 1, n_bins + 1)
    mean_pred  = np.full(n_bins, np.nan)
    frac_pos   = np.full(n_bins, np.nan)
    counts     = np.zeros(n_bins, dtype=int)

    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (p_hat >= lo) & (p_hat < hi)
        if hi == 1.0:
            mask = (p_hat >= lo) & (p_hat <= hi)
        if mask.sum() > 0:
            mean_pred[i] = p_hat[mask].mean()
            frac_pos[i]  = y_true[mask].mean()
            counts[i]    = mask.sum()
    return mean_pred, frac_pos, counts



# %%

# =============================================================================
# SECTION 9 ─ FULL EXPERIMENT PIPELINE
# =============================================================================

def run_experiment(X_tr, X_te, y_tr, y_te,
                   dataset_name: str, verbose=True):
    """
    Trains LR and RF, applies Platt + Isotonic on a calibration split,
    evaluates all combinations, and returns a nested results dict.
    """
    rng = np.random.default_rng(SEED)

    # ── train both base models ────────────────────────────────────────────────
    print(f"\n  [{dataset_name}] Training Logistic Regression …")
    lr = LogisticRegression(lr=0.05, n_epochs=700, batch_size=32, l2=1e-3)
    lr.fit(X_tr, y_tr)

    print(f"  [{dataset_name}] Training Random Forest (150 trees) …")
    rf = RandomForest(n_trees=150, max_depth=8, seed=SEED)
    rf.fit(X_tr, y_tr)

    # ── split test → calibration (50 %) + evaluation (50 %) ─────────────────
    n_te    = len(y_te)
    cal_idx = rng.choice(n_te, size=n_te // 2, replace=False)
    ev_idx  = np.setdiff1d(np.arange(n_te), cal_idx)

    y_cal,  y_ev  = y_te[cal_idx],  y_te[ev_idx]

    results = {}
    for model_name, model in [("Logistic Regression", lr),
                               ("Random Forest",       rf)]:
        p_te  = model.predict_proba(X_te)
        p_cal = p_te[cal_idx]
        p_ev  = p_te[ev_idx]

        # ── Platt scaling ────────────────────────────────────────────────────
        platt = PlattScaling(lr=0.005, n_iter=3000)
        platt.fit(p_cal, y_cal)
        p_ev_platt = platt.predict_proba(p_ev)

        # ── Isotonic regression ──────────────────────────────────────────────
        iso = IsotonicRegression()
        iso.fit(p_cal, y_cal)
        p_ev_iso = iso.predict_proba(p_ev)

        results[model_name] = {
            "Uncalibrated": {"probs": p_ev,       "metrics": all_metrics(y_ev, p_ev)},
            "Platt":        {"probs": p_ev_platt,  "metrics": all_metrics(y_ev, p_ev_platt)},
            "Isotonic":     {"probs": p_ev_iso,    "metrics": all_metrics(y_ev, p_ev_iso)},
            "y_eval": y_ev,
        }

        if verbose:
            print(f"\n  ── {model_name} on {dataset_name} ──────────────────────────")
            header = f"  {'Variant':<18} {'Accuracy':>10} {'Log-loss':>10} {'Brier':>10} {'ECE':>10}"
            print(header)
            print("  " + "─" * 56)
            for v in ["Uncalibrated", "Platt", "Isotonic"]:
                m = results[model_name][v]["metrics"]
                print(f"  {v:<18} {m['accuracy']:>10.4f} {m['log_loss']:>10.4f}"
                      f" {m['brier']:>10.4f} {m['ece']:>10.4f}")

    return results



# %%
# =============================================================================
# SECTION 10 ─ PLOTTING
# =============================================================================

VARIANT_COLORS = {
    "Uncalibrated": "#e74c3c",
    "Platt":        "#2980b9",
    "Isotonic":     "#27ae60",
}


def _draw_reliability_panel(ax, y_true, p_hat, variant, metrics_dict, n_bins=N_BINS):
    """Draw one reliability diagram panel (bars + histogram + diagonal)."""
    mp, fp, cnt = reliability_diagram_data(y_true, p_hat, n_bins)
    valid = ~np.isnan(mp)

    # Perfect-calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.55, label="Perfect")

    # Reliability bars
    bar_w = 0.9 / n_bins
    ax.bar(mp[valid], fp[valid], width=bar_w,
           color=VARIANT_COLORS[variant], alpha=0.65,
           edgecolor="white", label="Empirical")

    # Confidence histogram (right axis, greyed out)
    ax2 = ax.twinx()
    ax2.hist(p_hat, bins=n_bins, range=(0, 1), alpha=0.18,
             color="gray", density=True)
    ax2.set_yticks([])
    ax2.set_ylabel("density", fontsize=7, color="grey")
    ax2.set_ylim(0, ax2.get_ylim()[1] * 4)   

    m = metrics_dict
    ax.set_title(
        f"{variant}\n"
        f"ECE={m['ece']:.3f}   Brier={m['brier']:.3f}\n"
        f"Log-loss={m['log_loss']:.3f}   Acc={m['accuracy']:.3f}",
        fontsize=8.5
    )
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted prob.", fontsize=8)
    ax.set_ylabel("Fraction positives", fontsize=8)
    ax.legend(fontsize=7, loc="upper left")


def plot_reliability_diagrams(results: dict, dataset_name: str):
    """
    Grid of reliability diagrams:
        rows = models (LR, RF)
        cols = variants (Uncalibrated, Platt, Isotonic)
    """
    models   = list(results.keys())          # ["Logistic Regression", "Random Forest"]
    variants = ["Uncalibrated", "Platt", "Isotonic"]
    n_rows, n_cols = len(models), len(variants)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 9))
    fig.suptitle(f"Reliability Diagrams — {dataset_name}",
                 fontsize=14, fontweight="bold", y=1.01)

    for r, model_name in enumerate(models):
        res   = results[model_name]
        y_ev  = res["y_eval"]
        for c, variant in enumerate(variants):
            ax = axes[r, c]
            p  = res[variant]["probs"]
            m  = res[variant]["metrics"]
            _draw_reliability_panel(ax, y_ev, p, variant, m)
            if c == 0:
                ax.set_ylabel(f"{model_name}\nFraction positives", fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGURES, f"reliability_{dataset_name.replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {path}")


def plot_calibration_overlay(results: dict, dataset_name: str):
    """
    Overlay of calibration curves for all variants on two subplots (one per model).
    Easier to see improvement at a glance.
    """
    models   = list(results.keys())
    variants = ["Uncalibrated", "Platt", "Isotonic"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Calibration Curve Overlay — {dataset_name}",
                 fontsize=13, fontweight="bold")

    for ax, model_name in zip(axes, models):
        res  = results[model_name]
        y_ev = res["y_eval"]
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Perfect")
        for variant in variants:
            p   = res[variant]["probs"]
            m   = res[variant]["metrics"]
            mp, fp, _ = reliability_diagram_data(y_ev, p)
            valid = ~np.isnan(mp)
            ax.plot(mp[valid], fp[valid], "o-",
                    color=VARIANT_COLORS[variant], lw=2, ms=6,
                    label=f"{variant}  ECE={m['ece']:.3f}")
        ax.set_title(model_name, fontsize=11)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES, f"overlay_{dataset_name.replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {path}")


def plot_metrics_bar(results: dict, dataset_name: str):
    """
    Side-by-side bar charts for all four metrics across models × variants.
    """
    models   = list(results.keys())
    variants = ["Uncalibrated", "Platt", "Isotonic"]
    metric_keys    = ["accuracy", "log_loss", "brier", "ece"]
    metric_labels  = ["Accuracy ↑", "Log-loss ↓", "Brier Score ↓", "ECE ↓"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(f"Metric Comparison — {dataset_name}",
                 fontsize=13, fontweight="bold")

    x      = np.arange(len(variants))
    width  = 0.35
    colors = ["#c0392b", "#1a5276"]

    for ax, mkey, mlabel in zip(axes, metric_keys, metric_labels):
        for i, model_name in enumerate(models):
            vals = [results[model_name][v]["metrics"][mkey] for v in variants]
            bars = ax.bar(x + (i - 0.5) * width, vals, width,
                          label=model_name, color=colors[i], alpha=0.78)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=20, ha="right", fontsize=9)
        ax.set_title(mlabel, fontsize=11)
        ax.legend(fontsize=8)
        ax.set_ylim(0, min(ax.get_ylim()[1] * 1.18, 1.1))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(FIGURES, f"metrics_{dataset_name.replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {path}")


def plot_probability_histograms(results: dict, dataset_name: str):
    """
    Show how the probability distribution shifts after calibration.
    Rows = models, Cols = variants; histograms coloured by true class.
    """
    models   = list(results.keys())
    variants = ["Uncalibrated", "Platt", "Isotonic"]

    fig, axes = plt.subplots(len(models), len(variants), figsize=(15, 8))
    fig.suptitle(f"Predicted Probability Distributions — {dataset_name}",
                 fontsize=13, fontweight="bold", y=1.01)

    for r, model_name in enumerate(models):
        res  = results[model_name]
        y_ev = res["y_eval"]
        for c, variant in enumerate(variants):
            ax = axes[r, c]
            p  = res[variant]["probs"]
            ax.hist(p[y_ev == 0], bins=20, range=(0, 1), alpha=0.6,
                    color="#2980b9", label="Negative (y=0)", density=True)
            ax.hist(p[y_ev == 1], bins=20, range=(0, 1), alpha=0.6,
                    color="#e74c3c", label="Positive (y=1)", density=True)
            m = res[variant]["metrics"]
            ax.set_title(f"{variant}\nECE={m['ece']:.3f}", fontsize=9)
            ax.set_xlabel("Predicted probability", fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{model_name}\nDensity", fontsize=8)
            ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(FIGURES, f"histograms_{dataset_name.replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {path}")

# %%

# =============================================================================
# SECTION 11 ─ MAIN
# =============================================================================

def main():
    BC_PATH = "wdbc.data"
    HD_PATH = "processed_cleveland.data"

    print("=" * 65)
    print("  Assignment 2 — Calibration of Probabilities")
    print("=" * 65)

    # ── Load datasets ─────────────────────────────────────────────────────────
    print("\nLoading datasets …")
    X_bc, y_bc = load_breast_cancer('./Datasets/breast_cancer/wdbc.data')
    X_hd, y_hd = load_heart_disease('./Datasets/heart_disease/processed.cleveland.data')
    print(f"  Breast Cancer  : {X_bc.shape[0]} samples, {X_bc.shape[1]} features,"
          f"  positive rate = {y_bc.mean():.2f}")
    print(f"  Heart Disease  : {X_hd.shape[0]} samples, {X_hd.shape[1]} features,"
          f"  positive rate = {y_hd.mean():.2f}")

    # ── Train/test split + normalisation ─────────────────────────────────────
    Xtr_bc, Xte_bc, ytr_bc, yte_bc = train_test_split(X_bc, y_bc, test_size=0.25)
    Xtr_hd, Xte_hd, ytr_hd, yte_hd = train_test_split(X_hd, y_hd, test_size=0.25)
    Xtr_bc, Xte_bc = standardise(Xtr_bc, Xte_bc)
    Xtr_hd, Xte_hd = standardise(Xtr_hd, Xte_hd)

    # ── Run experiments ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    res_bc = run_experiment(Xtr_bc, Xte_bc, ytr_bc, yte_bc,
                            dataset_name="Breast Cancer")
    print("\n" + "=" * 65)
    res_hd = run_experiment(Xtr_hd, Xte_hd, ytr_hd, yte_hd,
                            dataset_name="Heart Disease")

    # ── Generate all plots ────────────────────────────────────────────────────
    print("\n\nGenerating figures …")
    for res, dname in [(res_bc, "Breast Cancer"), (res_hd, "Heart Disease")]:
        plot_reliability_diagrams(res, dname)
        plot_calibration_overlay(res, dname)
        plot_metrics_bar(res, dname)
        plot_probability_histograms(res, dname)

    print(f"\nAll figures saved to: ./{FIGURES}/")
    print("\nDone.")


if __name__ == "__main__":
    main()


