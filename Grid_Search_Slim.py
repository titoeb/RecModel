import numpy as np
import scipy.sparse
import yaml
import mlflow
import time
import utils.config_helpers
import os
import RecModel
import datetime
import hydra
import logging


log = logging.getLogger(__name__)

# Work around to get the working directory (after release use hydra.utils.get_original_cwd())
from hydra.plugins.common.utils import HydraConfig
def get_original_cwd():
    return HydraConfig().hydra.runtime.cwd

@hydra.main(config_path='configs/config.yaml')
def my_app(cfg):
    # Main 

    # Load mat.
    # Be aware that hydra changes the working directory
    train_mat = scipy.sparse.load_npz(os.path.join(get_original_cwd(), cfg.model.train_mat_path))
    n_users, n_items = train_mat.shape
    eval_mat = scipy.sparse.load_npz(os.path.join(get_original_cwd(), cfg.model.eval_mat_path))

    train_mat = train_mat.astype(np.float64)
    eval_mat = eval_mat.astype(np.float64)
    eval_mat.sort_indices()
    train_mat.sort_indices()
    
    # Set up MLFlow experiment
    experiment_name = f"Grid_Search_Slim_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M').replace(' ', '_')}"
    experiment = mlflow.create_experiment(experiment_name)
    
    log.info("Starting Grid_Search")
    np.random.seed(seed=cfg.model.seed)
    for alpha in  [float(i) for i in cfg.gridsearch.alpha]:
        for l1_ratio in [float(i) for i in cfg.gridsearch.l1_ratio]:
            with mlflow.start_run(experiment_id=experiment):
                log.info(f"\nCurrently fitting model with  alpha: {alpha} and l1_ratio: {l1_ratio}")

                # Log all params in cfg.model
                utils.config_helpers.log_config(dict(cfg.model))
    
                mlflow.log_param("alpha", alpha)
                mlflow.log_param("l1_ratio", l1_ratio)
                
                start = time.time()
                
                slim = RecModel.Slim(num_items=n_items, num_users=n_users)
                slim.train(train_mat.copy(), alpha, l1_ratio, int(cfg.model.max_iter), float(cfg.model.tol), int(cfg.model.cores), int(cfg.model.verbose))
                
                mlflow.log_metric("Runtime", int(round(time.time() - start, 0)))
                # Evaluate top-n metric:
                perf_all = slim.eval_topn(eval_mat.copy(), int(cfg.model.rand_sampled),  np.array(cfg.model.top_n_performances, dtype=np.int32), int(cfg.model.seed), int(cfg.model.cores))

                # Log the computed top recalls at different ns.
                for pos in range(len(cfg.model.top_n_performances)):
                    mlflow.log_metric(f"recallAT{cfg.model.top_n_performances[pos]}_of_{cfg.model.rand_sampled}", perf_all[f"Recall@{cfg.model.top_n_performances[pos]}"])
                mlflow.log_metric('MAE_train', slim.eval_prec(train_mat.copy(), 'MSE', int(cfg.model.cores)))
                mlflow.log_metric('MAE_eval', slim.eval_prec(eval_mat.copy(), 'MSE', int(cfg.model.cores)))

                # Remove object again so that next time a truly new object is created.
                del slim
    log.info("Grid_Search finished\n")
    # Shutdown VM when grid-search is finished
    os.system("shutdown now -h")
    
    
if __name__ == "__main__":
    my_app()
