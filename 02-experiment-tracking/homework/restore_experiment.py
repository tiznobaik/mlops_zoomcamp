import mlflow
from mlflow.exceptions import MlflowException

client = mlflow.tracking.MlflowClient()
experiment_name = "random-forest-hyperopt"

# Search for all experiments including the deleted ones
try:
    experiments = client.search_experiments(view_type=mlflow.entities.ViewType.ALL)
except MlflowException as e:
    print(f"Error retrieving experiments: {e}")
    experiments = []

# Find the deleted experiment by name
deleted_experiment = None
for exp in experiments:
    if exp.name == experiment_name and exp.lifecycle_stage == 'deleted':
        deleted_experiment = exp
        break

if deleted_experiment is not None:
    # Restore the experiment
    client.restore_experiment(deleted_experiment.experiment_id)
    print(f"Experiment '{experiment_name}' has been restored.")
else:
    print(f"Experiment '{experiment_name}' not found among deleted experiments.")