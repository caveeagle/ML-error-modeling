# Error Modeling

Correct and safe pipeline for error modeling and ensemble routing
=================================================================

0\. First split: create a clean meta-test set
---------------------------------------------

Start by splitting the original dataset into two parts:

*   **70% — meta\_train** (used for training all models and for computing error values)
*   **30% — meta\_test** (a completely untouched evaluation set; none of the models should see it during training)

```python
X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

All further steps use **meta\_train only** until the final evaluation.

* * *

1\. Build a loss column using OOF predictions on meta\_train
------------------------------------------------------------

Instead of computing loss only on 30% of the data, compute a **clean error estimate for every row** in meta\_train using out-of-fold predictions:

1.  Apply `KFold(n_splits=5, shuffle=True, random_state=42)` to `meta_train`.
2.  For each fold:
    *   train `best_model` on 4 folds,
    *   predict on the remaining fold.
3.  Collect OOF predictions into a single vector `y_pred_oof`.
4.  Compute per-row error:
    ```python
    loss = np.abs(y_meta_train - y_pred_oof)
    ```
5.  Construct `loss_dataset`:
    *   features: `X_meta_train`
    *   optional extra feature: `y_pred_oof`
    *   target: `loss`

This gives you **around 14k rows with clean, leakage-free error labels**, instead of only 30% of the data.

* * *

2\. Train the error\_predict\_model
-----------------------------------

Now train `error_predict_model` on `loss_dataset`:

*   Input features: the same 50 features (plus optionally `y_pred_oof`)
*   Target: `loss` (regression), or a binary flag of “high error” (classification)

This model is trained **only on meta\_train**.

There is no leakage because every loss value was computed using a model that did **not** see the corresponding row during training (OOF).

* * *

3\. Train the alternative\_model
--------------------------------

Two possible approaches:

### Simple version (recommended first):

Train `alternative_model` on **all meta\_train**, just like you trained best\_model.

### More advanced:

Train `alternative_model` specifically on the high-loss rows (based on real OOF loss).  
But this is optional and can be done in a second iteration; the simple version is good enough for the initial proof-of-concept.

* * *

4\. Evaluate the ensemble on meta\_test (the clean 30%)
-------------------------------------------------------

Now use the untouched 30% meta\_test to check whether the idea actually improves accuracy.

For evaluation:

1.  Retrain `best_model` on **all meta\_train**.
2.  Retrain `error_predict_model` on the full `loss_dataset`.
3.  Retrain `alternative_model` on all meta\_train.

Then on `meta_test`:

*   `y_pred_best = best_model.predict(X_meta_test)`
*   `y_pred_alt = alternative_model.predict(X_meta_test)`
*   `predicted_loss = error_predict_model.predict(X_meta_test)`

Select a threshold (e.g., top 30% by predicted\_loss):

*   For rows with predicted\_loss above the threshold → use `y_pred_alt`
*   For the rest → use `y_pred_best`

Finally:

*   compute the global metric (R², MAE, MAPE, etc.)
*   compare it to the plain `best_model` baseline on the same meta\_test

If the ensemble gives better accuracy — the idea works.

* * *


