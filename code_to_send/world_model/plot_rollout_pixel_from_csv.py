import csv
import matplotlib.pyplot as plt

# -------------------------
# file paths
# -------------------------
original_csv = "world_model_out/April6th/original_results.csv"
ds4_csv = "world_model_out/April6th/all_eval_results.csv"          
ds5_csv = "world_model_out/April6th/all_eval_results2.csv"             

TARGET_DATASETS = ["dataset1", "dataset2", "dataset3"]
TARGET_K = 32

def load_rollout_pixel(csv_path, label):
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row["Dataset"].strip()
            k = int(row["K"])
            if dataset in TARGET_DATASETS and k == TARGET_K:
                rows.append({
                    "Dataset": dataset,
                    "Rollout Pixel": float(row["Rollout Pixel"])
                })

    order = {name: i for i, name in enumerate(TARGET_DATASETS)}
    rows.sort(key=lambda r: order[r["Dataset"]])

    return {
        "label": label,
        "x": [r["Dataset"] for r in rows],
        "y": [r["Rollout Pixel"] for r in rows],
    }


series1 = load_rollout_pixel(original_csv, "Best model after ds3")
series2 = load_rollout_pixel(ds4_csv, "After adding ds4")
series3 = load_rollout_pixel(ds5_csv, "After adding ds5")

plt.figure(figsize=(8, 5))
plt.plot(series1["x"], series1["y"], marker="o", label=series1["label"])
plt.plot(series2["x"], series2["y"], marker="o", label=series2["label"])
plt.plot(series3["x"], series3["y"], marker="o", label=series3["label"])

plt.xlabel("Dataset")
plt.ylabel("Rollout Pixel MSE (K=32)")
plt.title("Effect of Adding More Datasets on Previous Environments")
plt.legend()
plt.tight_layout()

out_path = "world_model_out/rollout_pixel_comparison_from_csv.png"
plt.savefig(out_path, dpi=200)
plt.show()

print("Saved:", out_path)