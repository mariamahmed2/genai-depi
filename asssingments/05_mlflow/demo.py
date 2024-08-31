import mlflow
import argparse


"""

try:
    exp_id = mlflow.create_experiment('test')
except:
    pass

exp = mlflow.set_experiment('test') # better
with mlflow.start_run(run_name= 'test smth') as run:
    mlflow.log_param('param_1', 10)
    mlflow.log_param('param_2', 20)

    mlflow.log_params({'param_1': 10, 'param_2': 20 })

    mlflow.log_metric('acc', 0.8)
"""

def eval(param_1: int, param_2: int):
    return

