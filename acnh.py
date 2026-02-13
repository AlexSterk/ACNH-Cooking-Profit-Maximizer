import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

products = pd.read_csv("recipes.csv")
storage = pd.read_csv("storage.csv")
ingredients = pd.read_csv("ingredients.csv")

price = dict(zip(products["item"], products["price per item"]))
output_qty = dict(zip(products["item"], products["quantity crafted"]))
inventory = dict(zip(storage["item"], storage["storage"]))

recipe_names = products["item"].tolist()
n = len(recipe_names)

c = np.array([-price[r] for r in recipe_names])

all_items = set()

all_items.update(inventory.keys())
all_items.update(recipe_names)
all_items.update(ingredients["ingredient"].unique())

A = []
b = []

for item in all_items:
    row = np.zeros(n)

    for i, recipe in enumerate(recipe_names):

        # consumption
        used = ingredients[
            (ingredients["recipe"] == recipe) &
            (ingredients["ingredient"] == item)
            ]

        if not used.empty:
            row[i] += used["quantity"].values[0]

        # production
        if recipe == item:
            row[i] -= output_qty.get(recipe, 1)

    A.append(row)
    b.append(inventory.get(item, 0))

bounds = Bounds(lb=np.zeros(n), ub=np.full(n, np.inf))
integrality = np.ones(n)
constraints = LinearConstraint(A, -np.inf, b)

res = milp(
    c=c,
    constraints=constraints,
    integrality=integrality,
    bounds=bounds
)

print("Max profit:", -res.fun)

for name, val in zip(recipe_names, res.x):
    if val > 0.5:
        print(name, int(round(val)))


solution = {
    name: int(round(val))
    for name, val in zip(recipe_names, res.x)
}

from collections import defaultdict

consumed = defaultdict(int)
produced = defaultdict(int)

for recipe, times in solution.items():
    if times == 0:
        continue

    # production
    produced[recipe] += times * output_qty.get(recipe, 1)

    # consumption
    recipe_ingredients = ingredients[ingredients["recipe"] == recipe]

    for _, row in recipe_ingredients.iterrows():
        item = row["ingredient"]
        qty = row["quantity"]
        consumed[item] += times * qty

for item in ["flour", "sugar", "whole-wheat flour", "brown sugar", "tomato puree"]:
    print(item)
    print("  produced:", produced[item])
    print("  consumed:", consumed[item])
    print("  net:", produced[item] - consumed[item])

print("\n=== Item Flow Summary ===")

all_items = set(consumed.keys()) | set(produced.keys())

for item in sorted(all_items):
    prod = produced[item]
    cons = consumed[item]
    net = prod - cons

    if prod > 0 or cons > 0:
        print(f"{item:25} produced={prod:4} consumed={cons:4} net={net:4}")
