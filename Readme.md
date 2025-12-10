# Project Overview

## Concept

In this project, the following approach was implemented:

1. **Baseline Error Estimation**  
   For the primary model that delivered the best performance, its per-row prediction error on the test dataset was computed.

2. **Error Prediction Model**  
   A second model was trained to predict this error. Based on its output, the dataset was divided into two groups:  
   - well-predictable observations  
   - poorly predictable observations

3. **Re-prediction for Hard Cases**  
   For the poorly predictable group, a third model from a different model family was used to re-predict the target.  
   This strategy aims to improve the overall performance across the entire dataset.

---

## Implementation

All stages were **completed**:

- **Stage 1:** Gradient Boosting  
- **Stage 2:** Gradient Boosting + XGBoost, including feature-importance analysis  
- **Stage 3:** Simple neural networks (FNN models)

A final ensemble consisting of all three models was constructed and evaluated on the original dataset.

---

## Results

Although the ensemble achieved accuracy metrics comparable to the baseline, **<u>the overall prediction error was higher</u>** than expected.  

Further optimisation and validation would be required to improve performance, but this was not feasible within the available time frame.

---

## Conclusions

The concept is viable, but it requires substantial time to fine-tune all models in the ensemble to achieve a meaningful improvement in predictive performance.
