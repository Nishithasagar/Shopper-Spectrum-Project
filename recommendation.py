import pandas as pd

df = pd.read_csv(
    "cleaned_online_retail.csv"
)

print(df.shape)

pivot = df.pivot_table(
    index="CustomerID",
    columns="Description",
    values="Quantity",
    fill_value=0
)

print(pivot.shape)

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(
    pivot.T
)

print(similarity.shape)

similarity_df = pd.DataFrame(
    similarity,
    index=pivot.columns,
    columns=pivot.columns
)

print(similarity_df.head())

def recommend(product):

    # exact match first
    exact = [
        col for col in similarity_df.columns
        if col.lower() == product.lower()
    ]

    if exact:
        selected = exact[0]

    else:
        # partial match
        matches = [
            col for col in similarity_df.columns
            if product.lower() in col.lower()
        ]

        if len(matches) == 0:
            return ["Product not found"]

        print("\nMatching Products:")

        for i, p in enumerate(matches[:10]):
            print(f"{i+1}. {p}")

        choice = int(
            input(
                "\nChoose number: "
            )
        )

        selected = matches[
            choice - 1
        ]

    rec = (
        similarity_df[selected]
        .sort_values(
            ascending=False
        )[1:6]
        .index
        .tolist()
    )

    return selected, rec


product = input(
    "Enter Product Name: "
)

selected, rec = recommend(product)

print("\nSelected Product:")
print(selected)

print("\nRecommended Products:")

for i, p in enumerate(rec, 1):
    print(f"{i}. {p}")

# Save similarity matrix
similarity_df.to_csv(
    "product_similarity.csv"
)

print(
    "\nproduct_similarity.csv created successfully"
)

