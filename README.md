# Decision Tree vs Neural Network: Which Predicts Titanic Survival Better?

A side-by-side comparison of a traditional ML model (Decision Tree) and a feedforward neural network on the same binary classification problem, following the full ML life cycle: problem definition, data prep, training and tuning both models, then head-to-head comparison.

This project builds directly on my earlier [Titanic EDA](https://github.com/WayneS0628/titanic-eda), which explored the dataset and found that class and sex were the dominant predictors of survival, that finding drove the feature selection here.

Everything lives in one notebook: [`Titanic_Survival_Prediction-DT_vs_NN.ipynb`](./Titanic_Survival_Prediction-DT_vs_NN.ipynb)

## The Question

Is a decision tree or a neural network better at predicting Titanic survival, and is the neural network's added complexity actually worth it?

## Short Answer

The neural network won on both metrics tested, and by a real margin, not a rounding error.

| Metric | Decision Tree | Neural Network |
|---|---|---|
| Accuracy | 0.811 | 0.867 |
| F1 Score | 0.712 | 0.818 |

The F1 gap (0.712 vs 0.818) matters more than the accuracy gap. F1 punishes a model that leans on class imbalance to look good, so the neural network isn't just "more often right," it's genuinely better at correctly identifying survivors, the harder and more meaningful part of this problem.

## Dataset

Kaggle's [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/overview) competition training set: 891 passengers, mixed numeric and categorical features. Pulled programmatically via `kagglehub`.

## Data Prep

- **Age** (~20% missing): filled with the median *within each Pclass/Sex group*, rather than a single global median. This preserves the real age skew across class and sex instead of flattening everyone toward the center, an improvement over the naive global-median approach used in the original EDA project.
- **Embarked** (2 rows missing): filled with the mode, then one-hot encoded (`drop_first=True`) to avoid the dummy variable trap.
- **Cabin** (~77% missing): filled with `'Unknown'` as its own category rather than imputed, imputing that much missing data would be fabricating information.
- **Family_Size**: engineered from `SibSp + Parch + 1`, replacing both original columns since they both measure the same underlying thing (family aboard).
- **Sex**: mapped to binary (male=0, female=1) for use in both models and the correlation matrix.

### Chosen Features

Based on the resulting correlation matrix against `Survived`:

```
Sex            0.543
Fare           0.257
Family_Size    0.017
Age           -0.060
Pclass        -0.338
```

Final feature set: **Sex, Fare, Family_Size, Age, Pclass**

## Decision Tree

- Baseline (untuned) 5-fold CV accuracy: 0.775
- `GridSearchCV` over `max_depth` and `min_samples_leaf` → best params: `max_depth=4`, `min_samples_leaf=50` → CV accuracy: 0.804
- Final test set: **0.811 accuracy, 0.712 F1**
- Top features by importance: Sex, Pclass, Fare, Age, Family_Size, matching the original EDA's findings almost exactly

A decision tree fits this problem well because the strongest signal is an interaction (class and sex together, not either alone) and trees naturally capture that kind of threshold-based logic. The shallow depth GridSearchCV landed on (4) suggests the untuned tree was overfitting a small, 891-row dataset, and constraining it to simpler splits actually helped rather than hurt.

**A fairness note:** the tree is explicitly learning that sex and class predict survival, which is historically accurate for this specific event, but would be a real problem if that pattern were ever generalized to a different context (lending, hiring, healthcare, etc.).

## Neural Network

Feedforward network built in Keras:

- Architecture: `Dense(64, relu) → Dense(32, relu) → Dense(16, relu) → Dense(1, sigmoid)`
- Optimizer: SGD, learning rate 0.1
- Loss: Binary Crossentropy
- 200 epochs, trained in ~7.7 seconds
- Inputs scaled with `StandardScaler` (fit on train, applied to test, never re-fit)

At epoch 200: train accuracy 0.848 vs val accuracy 0.820, train loss 0.346 vs val loss 0.417, a real but modest gap indicating mild overfitting, not anything severe. Final test set: **0.867 accuracy, 0.818 F1**, both slightly better than the validation curves alone predicted.

## Comparative Analysis

**Was the added complexity worth it?** Yes. Training took under 8 seconds, so the usual "NNs are expensive" argument doesn't really apply at this data scale, and a 10-point F1 improvement on a life-or-death classification problem is not a rounding error.

**Which would I deploy?** The neural network, on pure performance. The one place I'd reconsider is interpretability: the decision tree's feature importances give a clean, explainable story a non-technical stakeholder can follow at a glance ("sex and class mattered most"), while the NN is a black box. In a regulated or high-stakes explainability context, I'd lean toward the tree despite the performance gap. For a pure prediction task like this one, the NN wins.

**What I'd try next:** benchmark against a random forest before trusting a single decision tree as the "traditional ML" baseline, handle the ~62/38 class imbalance more deliberately (class weights or a tuned decision threshold instead of a flat 0.5 cutoff), and try Adam plus dropout on the NN side, along with engineered features like passenger title extracted from name (Mr/Mrs/Master), which the original EDA flagged as promising but never tested.

## Kaggle Leaderboard Submission

Predictions from the neural network were submitted to Kaggle's official leaderboard for scoring against their true (unseen) holdout set. Preprocessing and submission logic for that lives in a separate notebook, kept isolated from this one since it exists purely for leaderboard scoring, not analysis: see `Titanic Kaggle Submission.ipynb`.

**Public leaderboard score: 0.7608**

Worth noting: the top of Kaggle's Titanic leaderboard is well known to be inflated by leaderboard hacking (submissions built from the real historical passenger manifest rather than actual model predictions), so a legitimate score generally falls in the 0.75-0.85 range. 0.7608 is a solid, honest result for a single 5-feature neural network with no ensembling.

It's also noticeably lower than the 0.867 accuracy this same model hit on its local held-out test set, a real and informative gap. With only 90 rows in that local test split, the 0.867 number carries a wide margin of uncertainty and likely reflects an easier-than-average split, on top of the mild overfitting already visible in the training curves. It's a good reminder that a single small local test set can overstate how well a model generalizes.

## Try It Live

*(Coming soon: an interactive Streamlit app where you can enter passenger details and see both models predict survival in real time.)*

## Tools

Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, Kaggle API, seaborn, matplotlib, uv

## Running It Yourself

Requires [uv](https://docs.astral.sh/uv/) and a [Kaggle API token](https://www.kaggle.com/docs/api):

```bash
git clone https://github.com/WayneS0628/Titanic-DTvsNN.git
cd Titanic-DTvsNN
uv sync
uv run jupyter notebook Titanic_Survival_Prediction-DT_vs_NN.ipynb
```

The dataset downloads automatically via `kagglehub` on first run, no manual download needed (requires accepting the competition rules on Kaggle's site first).

---

Part of my data science portfolio at [waynesimmonsjr.com](https://waynesimmonsjr.com)
