import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import display

CSV_PATH = "KAG_conversion_data.csv"
df = pd.read_csv(CSV_PATH)

print("Dataset shape:", df.shape)
display(df.head())
print("\\nMissing values:")
display(df.isnull().sum())

data = df[["ad_id", "Spent", "Approved_Conversion"]].copy()
data = data.rename(columns={
    "Spent": "cost",
    "Approved_Conversion": "value"
})

data = data[(data["cost"] >= 0) & (data["value"] >= 0)].copy()
data["cost_units"] = np.rint(data["cost"] * 10).astype(int)

print("Usable records:", len(data))
display(data.head())

N_ITEMS = 10
RANDOM_STATE = 42

experiment_data = data.sample(n=N_ITEMS, random_state=RANDOM_STATE).reset_index(drop=True)
total_sample_cost = int(experiment_data["cost_units"].sum())
budget = int(total_sample_cost * 0.30)

print("Selected advertisements:")
display(experiment_data[["ad_id", "cost", "value", "cost_units"]])
print(f"Total sample cost (scaled units): {total_sample_cost}")
print(f"Budget (30%): {budget}")

costs = experiment_data["cost_units"].astype(int).tolist()
values = experiment_data["value"].astype(int).tolist()
ad_ids = experiment_data["ad_id"].tolist()

def divide_conquer(costs, values, capacity, i=0):
    # Base case: no items left or no remaining budget
    if i == len(costs) or capacity <= 0:
        return 0, ()

    # Option 1: do not select item i
    exclude_value, exclude_items = divide_conquer(
        costs, values, capacity, i + 1
    )

    # Option 2: select item i, only if it fits
    if costs[i] <= capacity:
        remaining_value, remaining_items = divide_conquer(
            costs, values, capacity - costs[i], i + 1
        )
        include_value = values[i] + remaining_value
        include_items = (i,) + remaining_items
    else:
        include_value, include_items = -1, ()

    # Maximize value; if tied, prefer the lower total cost
    if include_value > exclude_value:
        return include_value, include_items
    if include_value < exclude_value:
        return exclude_value, exclude_items

    include_cost = sum(costs[j] for j in include_items)
    exclude_cost = sum(costs[j] for j in exclude_items)
    if include_cost < exclude_cost:
        return include_value, include_items
    return exclude_value, exclude_items

start = time.perf_counter()
dc_value, dc_items = divide_conquer(costs, values, budget)
dc_time = time.perf_counter() - start
dc_cost = sum(costs[i] for i in dc_items)

print("Divide and Conquer")
print("Selected row positions:", dc_items)
print("Selected ad IDs:", [ad_ids[i] for i in dc_items])
print("Total cost:", dc_cost)
print("Total approved conversions:", dc_value)
print("Execution time (seconds):", dc_time)

greedy_table = experiment_data[["ad_id", "cost", "value", "cost_units"]].copy()
greedy_table["value_per_cost"] = np.where(
    greedy_table["cost_units"] > 0,
    greedy_table["value"] / greedy_table["cost_units"],
    np.where(greedy_table["value"] > 0, np.inf, 0.0)
)
greedy_table = greedy_table.sort_values(
    "value_per_cost", ascending=False, kind="stable"
)
display(greedy_table)

def greedy_knapsack(costs, values, capacity):
    ratios = []
    for i, (cost, value) in enumerate(zip(costs, values)):
        if cost == 0:
            ratio = float("inf") if value > 0 else 0.0
        else:
            ratio = value / cost
        ratios.append((ratio, i))

    ratios.sort(key=lambda item: item[0], reverse=True)

    selected = []
    total_cost = 0
    total_value = 0

    for ratio, i in ratios:
        if values[i] <= 0:
            continue
        if total_cost + costs[i] <= capacity:
            selected.append(i)
            total_cost += costs[i]
            total_value += values[i]

    return total_value, tuple(selected), total_cost

start = time.perf_counter()
greedy_value, greedy_items, greedy_cost = greedy_knapsack(
    costs, values, budget
)
greedy_time = time.perf_counter() - start

print("Greedy")
print("Selected row positions:", greedy_items)
print("Selected ad IDs:", [ad_ids[i] for i in greedy_items])
print("Total cost:", greedy_cost)
print("Total approved conversions:", greedy_value)
print("Execution time (seconds):", greedy_time)

def dynamic_programming(costs, values, capacity):
    n = len(costs)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        item_cost = costs[i - 1]
        item_value = values[i - 1]

        for w in range(capacity + 1):
            # Exclude current item
            dp[i][w] = dp[i - 1][w]

            # Include current item if it fits
            if item_cost <= w:
                dp[i][w] = max(
                    dp[i][w],
                    item_value + dp[i - 1][w - item_cost]
                )

    # Trace backwards to recover one optimal selection
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(i - 1)
            w -= costs[i - 1]

    selected.reverse()
    total_cost = sum(costs[i] for i in selected)
    return dp[n][capacity], tuple(selected), total_cost, dp

start = time.perf_counter()
dp_value, dp_items, dp_cost, dp_table = dynamic_programming(
    costs, values, budget
)
dp_time = time.perf_counter() - start

print("Dynamic Programming")
print("Selected row positions:", dp_items)
print("Selected ad IDs:", [ad_ids[i] for i in dp_items])
print("Total cost:", dp_cost)
print("Total approved conversions:", dp_value)
print("Execution time (seconds):", dp_time)

results = pd.DataFrame({
    "Algorithm": ["Divide and Conquer", "Greedy", "Dynamic Programming"],
    "Selected ad IDs": [
        [ad_ids[i] for i in dc_items],
        [ad_ids[i] for i in greedy_items],
        [ad_ids[i] for i in dp_items]
    ],
    "Total Cost (scaled)": [dc_cost, greedy_cost, dp_cost],
    "Approved Conversions": [dc_value, greedy_value, dp_value],
    "Execution Time (seconds)": [dc_time, greedy_time, dp_time]
})

display(results)
print("Budget:", budget)
print("All solutions within budget:",
      all(c <= budget for c in [dc_cost, greedy_cost, dp_cost]))
print("Exact algorithms agree:",
      dc_value == dp_value)

plt.figure(figsize=(9, 5))
plt.bar(results["Algorithm"], results["Approved Conversions"])
plt.title("Approved Conversions by Algorithm")
plt.ylabel("Approved conversions")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
plt.bar(results["Algorithm"], results["Execution Time (seconds)"])
plt.title("Execution Time by Algorithm")
plt.ylabel("Time (seconds)")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.show()