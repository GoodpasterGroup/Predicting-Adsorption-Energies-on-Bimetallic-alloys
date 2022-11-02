import catboost
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error
import pandas as pd
from sklearn.model_selection import train_test_split
import gc

def objective(trial):
    X = pd.read_pickle('./x_train.pkl')
    y = pd.read_pickle('./y_train.pkl')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)

    param = {
        "objective": trial.suggest_categorical("objective", ["MAE", "RMSE"]),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "depth" : trial.suggest_int('depth', 6, 12),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
    }
    

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_uniform("subsample", 0.1, 1)

    gbm = catboost.CatBoostRegressor(**param, iterations = 1000)

    gbm.fit(X_train, y_train, eval_set = [(X_val, y_val)], verbose = 0, early_stopping_rounds = 100)

    preds = gbm.predict(X_val)
    #pred_labels = np.rint(preds)
    mse = max_error(y_val, preds)
    
    return mse

study = optuna.create_study(direction = "minimize")
study.optimize(objective, n_trials = 5, show_progress_bar = True, gc_after_trial=True)
