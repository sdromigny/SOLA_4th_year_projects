import pandas as pd
import os

input_G = "/workspaces/SOLA_4th_year_projects/sola-base/sola_inputs_cornwall/G"
output_G = "/workspaces/SOLA_4th_year_projects/sola-base/sola_inputs_cornwall/G"

# read header
with open(input_G, "r") as f:
    nnz = int(f.readline().strip())
    N, M = map(int, f.readline().split())

# read ijv
Gdf = pd.read_table(
    input_G,
    sep=r"\s+",
    skiprows=2,
    header=None,
    names=["i", "j", "val"],
)

# convert 1-based -> 0-based
Gdf["i"] -= 1
Gdf["j"] -= 1

# sanity check
if Gdf["i"].min() < 0 or Gdf["j"].min() < 0:
    raise ValueError("Negative indices after conversion — input G was not 1-based?")
if Gdf["i"].max() >= N or Gdf["j"].max() >= M:
    raise ValueError("Index out of bounds after conversion")

# write new G
with open(output_G, "w") as f:
    f.write(f"{nnz}\n")
    f.write(f"{N} {M}\n")
    for i, j, v in zip(Gdf["i"], Gdf["j"], Gdf["val"]):
        f.write(f"{i} {j} {v:.6e}\n")

print("Wrote 0-based G to:", output_G)
