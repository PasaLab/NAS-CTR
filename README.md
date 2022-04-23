# NAS-CTR

The codes for *NAS-CTR: Efficient Neural Architecture Search  for Click-Through Rate Prediction*

# Dataset

- Use `python codes/datasets/avazu.py dataset_path` to preprocess [Avazu](https://www.kaggle.com/c/avazu-ctr-prediction/data).
- Use `python codes/datasets/criteo.py dataset_path` to preprocess [Criteo](https://www.kaggle.com/c/criteo-display-ad-challenge).

# Search & Evaluate

Use `sh search_evaluate.sh  ${gpu_idx}` to search and evaluate on chosen dataset.

# License
The codes and models in this repo are released under the GNU GPLv3 license.