# Evaluation and Optimization


Underfitting

We need more data or train more.

Overfitting

Introduce more data

Early stopping

Sprinkle in more noise

Regularization

Ensembles (combine different models together)

Ditch some features

| Training Error | Testing Error | |
|----------------|---------------|-|
| Low | Low | Its good |
| Low | High | Overfitting |
| High | High | Reject |
| High | Low | Rarely happens, Reject |

Regression accuracy is measured by RMSE.

Lower RMSE (Root mean square Error) is better.

| | True | False |
|-|------|-------|
| True | Predicted correct | False positive </br> Type I Error |
| False | False Negative </br> Type II Error | Predicted Correctly |

ML model performance metric for binary classification model

AUC (Area under the Curve)

value will be between 0 to 1, values closer to 1 means model is more accurate.

Recall = We guessed right / We guessed right + We should have flagged these too

eg: spam gets through

Precision = We guessed right / We guessed right + We flagged these but we were wrong

eg: Legitimate email gets blocked.

F1 score = 2 * (Precision * Recall) / (Precision + Recall)

The hight the F1 score the better.