import optuna


def f(x, y):
    return 2 * (x - 10) ** 2 + 3 * (y - 5) ** 2


# Define the objective function
def objective(trial):
    # Example of hyperparameters to optimize
    param1 = trial.suggest_float("param1", -100.0, 100.0)
    param2 = trial.suggest_float("param2", -100.0, 100.0)

    # Your function f(...) that you want to optimize using the parameters
    result = f(param1, param2)

    # The objective function should return a value to minimize or maximize
    return result


storage_path = "sqlite:///optuna_study.db"

# Create an Optuna study
study = optuna.create_study(
    direction="minimize",
    storage=storage_path,
    study_name="regressor_optimization",
    load_if_exists=True,
)  # Use 'maximize' if you want to maximize the result

# Optimize the objective function
study.optimize(objective, n_trials=10)  # Number of trials you want to run

# Print the best parameters found
print(f"Best parameters: {study.best_params}")
print(f"Best value: {study.best_value}")


import optuna.visualization as vis

# Visualize optimization history
fig = vis.plot_optimization_history(study)
fig.show()

# Visualize parameter importance
fig2 = vis.plot_param_importances(study)
fig2.show()
