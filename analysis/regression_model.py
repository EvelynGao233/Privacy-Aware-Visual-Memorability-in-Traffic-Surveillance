import pandas as pd
import statsmodels.formula.api as smf
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "../datasets/all_videos_full_object_stats.csv")
model_dir = os.path.join(current_dir, "../models")
os.makedirs(model_dir, exist_ok=True)

df = pd.read_csv(csv_path)
df["color"] = df["color"].str.capitalize()
df["video"] = df["video"].astype(str)

top_objects = df.loc[df.groupby(["video", "object_id"])["memo_score"].idxmax()]
area_low = top_objects["area"].quantile(0.01)
area_high = top_objects["area"].quantile(0.99)
filtered_df = top_objects[
    (top_objects["area"] >= area_low) &
    (top_objects["area"] <= area_high) &
    (top_objects["memo_score"] > 0)
]

models = {
    "Area only": "memo_score ~ area",
    "Color only": "memo_score ~ C(color)",
    "Position only": "memo_score ~ bbox_x + bbox_y",
    "Area + Color": "memo_score ~ area + C(color)",
    "Area + Position": "memo_score ~ area + bbox_x + bbox_y",
    "Color + Position": "memo_score ~ C(color) + bbox_x + bbox_y",
    "Area + Color + Position": "memo_score ~ area + C(color) + bbox_x + bbox_y",
    "Area * Color": "memo_score ~ area * C(color)",
    "Area * Position": "memo_score ~ area * (bbox_x + bbox_y)",
    "Color * Position": "memo_score ~ C(color) * (bbox_x + bbox_y)",
    "Area * Color + Position": "memo_score ~ area * C(color) + bbox_x + bbox_y",
    "Area * Position + Color": "memo_score ~ area * (bbox_x + bbox_y) + C(color)",
    "Color * Position + Area": "memo_score ~ C(color) * (bbox_x + bbox_y) + area",
    "Area * bbox_y + Color": "memo_score ~ area * bbox_y + C(color)",
    "Area * Color * Position": "memo_score ~ area * C(color) * bbox_x * bbox_y",
}

model_results = {}
for name, formula in models.items():
    model = smf.ols(formula, data=filtered_df).fit()
    model_results[name] = model

    filename = name.lower().replace(" ", "_").replace("*", "interaction").replace("+", "plus").replace("(", "").replace(")", "").replace(":", "X")
    with open(os.path.join(model_dir, f"regression_model_{filename}.txt"), "w") as f:
        f.write(model.summary().as_text())

def extract_model_metrics(model, name):
    return {
        "Model": name,
        "R_squared": round(model.rsquared, 3),
        "Adj_R_squared": round(model.rsquared_adj, 3),
        "AIC": round(model.aic, 1),
        "BIC": round(model.bic, 1),
        "F_statistic": round(model.fvalue, 2),
        "F_pvalue": model.f_pvalue
    }

metrics = [extract_model_metrics(m, name) for name, m in model_results.items()]
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(os.path.join(model_dir, "regression_model_comparison_full.csv"), index=False)
print(metrics_df)