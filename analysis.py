import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("IRIS.csv")

# distritbution plots
sns.set(style="whitegrid")

plt.figure(figsize=(12, 8))
for i, feature in enumerate(df.columns[:-1], 1):  # assuming last column is the class
    plt.subplot(2, 2, i)
    sns.histplot(data=df, x=feature, hue=df.columns[-1], kde=True, palette="Set2")
    plt.title(f"Distribution of {feature}")

plt.tight_layout()
plt.savefig("distribution_plot.png", dpi=300, bbox_inches='tight')

plt.show()

# pairplot
sns.pairplot(df, hue=df.columns[-1], palette="Set2", diag_kind="kde")
plt.suptitle("Feature Pair Matrix (Scatter Plots)", y=1.02)
plt.show()
plt.savefig("pairplot_matrix.png", dpi=300, bbox_inches='tight')
