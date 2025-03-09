import numpy as np

def find_top_k_similarity(embeddings, k):
    embeddings = embeddings / np.sqrt((embeddings**2).sum(1, keepdims=True)) # L2 normalize the rows, as is common

    query = np.random.randn(1536) # the query vector
    query = query / np.sqrt((query**2).sum())   
    similarities = embeddings.dot(query)
    sorted_ix = np.argsort(-similarities)
    print("top 10 results:")
    for k in sorted_ix[:10]:
        print(f"row {k}, similarity {similarities[k]}")

if __name__ == "__main__":
    np.random.seed(42)
    embeddings = np.random.randn(1000, 1536) # 1000 documents, 1536-dimensional embeddings
    res = find_top_k_similarity(embeddings, 10)
    print(res)