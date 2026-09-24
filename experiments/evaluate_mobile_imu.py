"""
Safe Road AI - Mobile Sensor IMU Dataset (8,000 Records) Evaluator
Evaluates machine learning classifiers on the newly uploaded 8,000-sample smartphone IMU dataset.
Generates metrics JSON, feature importance charts, and crash transition visualizations.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.config import RESULTS_DIR

def run_mobile_imu_evaluation():
    csv_path = PROJECT_ROOT / "sensordata" / "road_accident_imu_dataset_8000.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    print("Loading 8,000-record Mobile IMU Dataset...")
    df = pd.read_csv(csv_path)

    features = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Speed_kmh', 'Motion_Intensity']
    X = df[features]
    y = df['Crash_Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    results = {
        "dataset_name": "Mobile Sensor IMU Dataset (8,000 Smartphone Records)",
        "total_samples": len(df),
        "features": features,
        "class_distribution": {
            "normal_samples": int((y == 0).sum()),
            "crash_samples": int((y == 1).sum()),
            "normal_pct": round(float((y == 0).mean()) * 100, 1),
            "crash_pct": round(float((y == 1).mean()) * 100, 1)
        },
        "models": {}
    }

    print("\n--- Training and Evaluating Sensor Models ---")
    fitted_rf = None
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        p = precision_score(y_test, preds)
        r = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        cm = confusion_matrix(y_test, preds).tolist()

        if name == 'Random Forest':
            fitted_rf = clf

        results["models"][name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm
        }
        print(f"[{name}] Acc: {acc*100:.2f}% | F1: {f1:.4f} | Prec: {p*100:.2f}% | Rec: {r*100:.2f}%")

    # Feature Importance
    if fitted_rf:
        fi = pd.Series(fitted_rf.feature_importances_, index=features).sort_values(ascending=True)
        results["feature_importances"] = {k: round(float(v), 4) for k, v in fi.items()}

        # Plot 1: Feature Importance Chart
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
        fig.patch.set_facecolor('#0F172A')
        ax.set_facecolor('#1E293B')

        colors = ['#38BDF8' if 'Acc' in feat else '#F59E0B' if 'Speed' in feat else '#10B981' if 'Motion' in feat else '#94A3B8' for feat in fi.index]
        bars = ax.barh(fi.index, fi.values * 100, color=colors, edgecolor='#334155', height=0.6)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.8, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', ha='left', color='#F8FAFC', fontsize=9, fontweight='bold')

        ax.set_xlim(0, max(fi.values * 100) + 7)
        ax.set_xlabel("Relative Importance (%)", color='#F8FAFC', fontweight='bold', fontsize=10)
        ax.set_title("Random Forest Feature Importance on Mobile Sensor Data", color='#F8FAFC', fontweight='bold', fontsize=11, pad=10)
        ax.tick_params(colors='#F1F5F9', labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('#334155')
        ax.grid(True, linestyle="--", alpha=0.3, color='#475569', axis='x')
        plt.tight_layout()
        chart_p1 = RESULTS_DIR / "mobile_imu_feature_importance.png"
        fig.savefig(chart_p1, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {chart_p1}")

    # Plot 2: Crash Transition Telemetry Plot (Window around index 6970 to 7030)
    crash_idx = df[df['Crash_Label'] == 1].index.min()
    window_df = df.iloc[max(0, crash_idx - 30): min(len(df), crash_idx + 30)].copy()
    window_df['Relative_Time_Sec'] = np.arange(-30, len(window_df) - 30)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True, dpi=180)
    fig.patch.set_facecolor('#0F172A')
    for ax in (ax1, ax2):
        ax.set_facecolor('#1E293B')
        ax.tick_params(colors='#F1F5F9', labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('#334155')
        ax.grid(True, linestyle="--", alpha=0.3, color='#475569')

    # Top: Speed Drop
    ax1.plot(window_df['Relative_Time_Sec'], window_df['Speed_kmh'], color="#F59E0B", lw=2.2, label="GPS Vehicle Speed (km/h)")
    ax1.axvline(0, color="#EF4444", linestyle="--", lw=2.0, label="Crash Impact Onset (t=0)")
    ax1.set_ylabel("Speed (km/h)", color='#F8FAFC', fontweight='bold')
    ax1.legend(loc="upper right", fontsize=8, facecolor='#0F172A', edgecolor='#334155', labelcolor='#F8FAFC')
    ax1.set_title("Crash Dynamics: Speed Drop & Acceleration Shock Profile", color='#F8FAFC', fontweight='bold', pad=8)

    # Bottom: G-force / Accelerations
    ax2.plot(window_df['Relative_Time_Sec'], window_df['Motion_Intensity'], color="#38BDF8", lw=2.2, label="Total G-Force / Motion Intensity (m/s²)")
    ax2.plot(window_df['Relative_Time_Sec'], window_df['Acc_Z'], color="#A855F7", lw=1.5, linestyle=":", label="Vertical Acc Z (m/s²)")
    ax2.axvline(0, color="#EF4444", linestyle="--", lw=2.0)
    ax2.axhline(9.81, color="#94A3B8", linestyle="--", alpha=0.6, label="Normal Gravity Baseline (9.81 m/s²)")
    ax2.set_xlabel("Time Relative to Crash Onset (seconds)", color='#F8FAFC', fontweight='bold')
    ax2.set_ylabel("Acceleration (m/s²)", color='#F8FAFC', fontweight='bold')
    ax2.legend(loc="upper left", fontsize=8, facecolor='#0F172A', edgecolor='#334155', labelcolor='#F8FAFC')

    plt.tight_layout()
    chart_p2 = RESULTS_DIR / "mobile_imu_crash_transition.png"
    fig.savefig(chart_p2, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {chart_p2}")

    # Plot 3: Confusion Matrix
    cm_arr = np.array(results["models"]["Random Forest"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4), dpi=180)
    fig.patch.set_facecolor('#0F172A')
    ax.set_facecolor('#1E293B')
    im = ax.imshow(cm_arr, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ['Normal (0)', 'Crash (1)']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, color='#F8FAFC', fontweight='bold')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, color='#F8FAFC', fontweight='bold')
    ax.set_ylabel('True Label', color='#F8FAFC', fontweight='bold')
    ax.set_xlabel('Predicted Label', color='#F8FAFC', fontweight='bold')
    ax.set_title('Random Forest Confusion Matrix (8,000 Records)', color='#F8FAFC', fontweight='bold', pad=10)

    thresh = cm_arr.max() / 2.
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, format(cm_arr[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_arr[i, j] > thresh else "#F8FAFC",
                    fontweight='bold', fontsize=12)
    plt.tight_layout()
    chart_p3 = RESULTS_DIR / "mobile_imu_confusion_matrix.png"
    fig.savefig(chart_p3, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {chart_p3}")

    # Save summary JSON
    json_path = RESULTS_DIR / "metrics_summary_mobile_imu.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results summary: {json_path}")

    return results

if __name__ == "__main__":
    run_mobile_imu_evaluation()
