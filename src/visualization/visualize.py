import pandas as pd
import numpy as np

def plot_drift(reference: pd.DataFrame, current: pd.DataFrame, feature: str, out_path: str):
    """Histogram overlay of reference vs. current for a single feature."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ref = pd.to_numeric(reference[feature], errors="coerce").dropna()
    cur = pd.to_numeric(current[feature], errors="coerce").dropna()
    bins = np.linspace(min(ref.min(), cur.min()), max(ref.max(), cur.max()), 30)
    ax.hist(ref, bins=bins, alpha=0.5, label="reference (train)", density=True)
    ax.hist(cur, bins=bins, alpha=0.5, label="current (prod)", density=True)
    ax.set_title(f"Drift: {feature}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
