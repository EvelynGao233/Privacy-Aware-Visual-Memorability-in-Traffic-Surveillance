
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import os
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "../datasets/all_videos_full_object_stats.csv")
visuals_dir = os.path.join(current_dir, "../visuals")
model_dir = os.path.join(current_dir, "../models")
os.makedirs(visuals_dir, exist_ok=True)
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

df_unique = df.sort_values("memo_score", ascending=False).drop_duplicates(subset=["video", "object_id"])
color_stats = df_unique.groupby("color").agg(
    count=("object_id", "count"),
    mean_score=("memo_score", "mean")
).reset_index()
color_stats["label"] = color_stats["color"] + " (n=" + color_stats["count"].astype(str) + ")"
color_stats = color_stats.sort_values("mean_score", ascending=False)


plt.figure(figsize=(10, 6))
sns.barplot(data=color_stats, y="label", x="mean_score", palette="viridis", ci=None)
plt.title("Average Memorability by Color (Unique Vehicles, Non-filtered)")
plt.xlabel("Average memo_score")
plt.ylabel("Color (with unique vehicle count)")
plt.tight_layout()
plt.savefig(os.path.join(visuals_dir, "color_vehicle_level_barplot.png"))
plt.close()

sns.lmplot(data=df_unique, x="area", y="memo_score", hue="color",
           scatter_kws={"alpha": 0.3}, height=6, aspect=1.3)
plt.title("Linear Regression: Area vs Score by Color (Non-filtered)")
plt.tight_layout()
plt.savefig(os.path.join(visuals_dir, "lmplot_area_vs_score_by_color_nonfiltered.png"))
plt.close()

model_interaction = smf.ols("memo_score ~ area * C(color)", data=filtered_df).fit()
coefs = model_interaction.params.filter(like="C(color)")
conf_int = model_interaction.conf_int().loc[coefs.index]

plt.figure(figsize=(10, 6))
sns.barplot(x=coefs.values, y=coefs.index, orient="h", palette="coolwarm")
plt.errorbar(x=coefs.values, y=range(len(coefs)),
             xerr=[coefs.values - conf_int[0], conf_int[1] - coefs.values],
             fmt='none', ecolor='black', capsize=4)
plt.axvline(0, color='gray', linestyle='--')
plt.title("Color Coefficients (area * color, Filtered)")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig(os.path.join(visuals_dir, "color_interaction_coefficients_filtered.png"))
plt.close()

model_simple = smf.ols("memo_score ~ area + C(color)", data=filtered_df).fit()
with open(os.path.join(model_dir, "regression_model_area_color.txt"), "w") as f:
    f.write(model_simple.summary().as_text())

with open(os.path.join(model_dir, "regression_model_interaction.txt"), "w") as f:
    f.write(model_interaction.summary().as_text())

model_area_only = smf.ols("memo_score ~ area", data=filtered_df).fit()
with open(os.path.join(model_dir, "regression_model_area_only.txt"), "w") as f:
    f.write(model_area_only.summary().as_text())

model_color_only = smf.ols("memo_score ~ C(color)", data=filtered_df).fit()
with open(os.path.join(model_dir, "regression_model_color_only.txt"), "w") as f:
    f.write(model_color_only.summary().as_text())

def normalize_and_corr(data, cols, method="spearman"):
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(data[cols]), columns=cols)
    return scaled.corr(method=method)

compare_features = ["bbox_x", "bbox_y", "bbox_w", "bbox_h", "area"]
video_corr_matrix = []

for vid in sorted(df["video"].unique()):
    sub_df = df[df["video"] == vid]
    cols = ["memo_score"] + compare_features
    corr = normalize_and_corr(sub_df, cols)
    if "memo_score" in corr.columns:
        row = corr["memo_score"].drop("memo_score")
        row.name = vid
        video_corr_matrix.append(row)

corr_all = normalize_and_corr(df, ["memo_score"] + compare_features)
overall_row = corr_all["memo_score"].drop("memo_score")
overall_row.name = "all"
video_corr_matrix.append(overall_row)

heatmap_df = pd.DataFrame(video_corr_matrix)


SMALL_SIZE = 23
MEDIUM_SIZE = 24
BIGGER_SIZE = 24
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_df, annot=True, cmap="RdBu_r", center=0, fmt=".2f", annot_kws={"size": 22})
#plt.title("Spearman Correlation with Score Across Videos (Non-filtered)")
plt.ylabel("Video")
plt.xlabel("Feature")
plt.tight_layout()
plt.savefig(os.path.join(visuals_dir, "video_feature_score_correlation_matrix.pdf"))
plt.close()

color_stats_full = df.groupby("color").agg(
    count=("object_id", "count"),
    mean_score=("memo_score", "mean")
).reset_index()

plt.figure(figsize=(10, 6))

SMALL_SIZE = 30
MEDIUM_SIZE = 30
BIGGER_SIZE = 30
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.xlim([0, 95000])
sns.scatterplot(data=color_stats_full, x="count", y="mean_score")
for i in range(len(color_stats_full)):
    row = color_stats_full.iloc[i]
    plt.text(row["count"] + 1000, row["mean_score"] - 1, row["color"], fontsize=22)


plt.xlabel("Unique Object Count (per Color)")
plt.ylabel("Avg Memorability Score")
#plt.title("Color Frequency vs. Memorability (All Frames)")
plt.tight_layout()
plt.savefig(os.path.join(visuals_dir, "color_frequency_vs_memorability_scatter_nonfiltered.pdf"))
plt.close()