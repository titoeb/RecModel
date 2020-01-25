
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
import hyperopt as hp
from functools import partial
import pickle

log = logging.getLogger(__name__)

# Helper functions
def eval_Slim(params, cfg, train_mat, eval_mat, experiment):
    # This function is what Hyperopt is going to optimize (minimize 'loss' value)
    print(experiment)
    with mlflow.start_run(experiment_id=experiment):

        # Log the config
        utils.config_helpers.log_config(dict(cfg.model))        

        n_users, n_items = train_mat.shape
        np.random.seed(seed=cfg.model.seed)

        # Log relevant parameters for this run.
        mlflow.log_param("alpha", params['alpha'])
        mlflow.log_param("l1_ratio", params['l1_ratio'])
        mlflow.log_param("max_iter", params['max_iter'])
        mlflow.log_param("tol", params['tol'])

        # Log this run
        log.info(f"Testing  alpha: {params['alpha']},  l1_ratio: {params['l1_ratio']}, max_iter: {params['max_iter']} and tol: {params['tol']}")
        
        start = time.time()       
        # Create model
        slim = RecModel.Slim(num_items=n_items, num_users=n_users)

        # Train Model
        slim.train(X=train_mat.copy(), alpha=params['alpha'], l1_ratio=params['l1_ratio'], max_iter=params['max_iter'], tolerance=params['tol'], cores = 1, verbose=int(cfg.model.verbose))
        
        # Log run-time
        mlflow.log_metric("Runtime", int(round(time.time() - start, 0)))

        # Evaluate model
        perf_all = slim.eval_topn(eval_mat.copy(), rand_sampled=int(cfg.model.rand_sampled),  topn=np.array(cfg.model.top_n_performances, dtype=np.int32), random_state=int(cfg.model.seed),
         cores=int(cfg.model.cores))

        # Log the performance of the model
        for pos in range(len(cfg.model.top_n_performances)):
            mlflow.log_metric(f"recallAT{cfg.model.top_n_performances[pos]}_of_{cfg.model.rand_sampled}", perf_all[f"Recall@{cfg.model.top_n_performances[pos]}"])
        mlflow.log_metric('MAE_train', slim.eval_prec(train_mat.copy()))
        mlflow.log_metric('MAE_eval', slim.eval_prec(eval_mat.copy()))

        #We will always choose the first topn performance. Hopefully, that is also the smallest is most relevant for us.g
        rel_topn_perf = perf_all[f"Recall@{cfg.model.top_n_performances[0]}"]
        
        log.info(f"Current recallAT{cfg.model.top_n_performances[0]}_of_{cfg.model.rand_sampled} performance was {rel_topn_perf}")
        loss = -rel_topn_perf
        return {'loss': loss, 'status': hp.STATUS_OK, 'eval_time': time.time()}

def hyper_opt_fmin(space, fun, additional_evals, verbose = 0, trials_path='../trials.p', **kwargs):
    # This is a wrapper around the training process that enables warm starts from file.

    objective = partial(fun, **kwargs)
    
    # Try to recover trials object, else create new one!
    try:
        trials = pickle.load(open(trials_path, "rb"))
        if verbose > 0:
            print(f"Loaded trails from {trials_path}")
    except FileNotFoundError:
        trials = hp.Trials()
        
    # Compute the effect number of new trials that have to be run.
    past_evals = len(trials.losses())
    new_evals = past_evals + additional_evals

    best = hp.fmin(fn = objective, space=space, algo=hp.tpe.suggest,  max_evals = new_evals, trials=trials)
    if verbose > 0:
        print(f"HyperOpt got best loss {trials.best_trial['result']['loss']} with the following hyper paramters: \n{trials.best_trial['misc']['vals']}")
        
    # Store the trials object
    pickle.dump(trials, open(trials_path, "wb"))
    
    return best, trials

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

    # Setup HyperOpt
    space = space = {'alpha': hp.hp.lognormal('alpha',0, 1),
        'l1_ratio' : hp.hp.lognormal('l1_ratio', 0, 1),
         'max_iter': hp.hp.choice('max_iter', np.arange(1, 40)),
         'tol': hp.hp.lognormal('tol', -3, 1.0)}

    # Set up MLFlow experiment
    experiment_name = f"HyperOpt_Slim_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M').replace(' ', '_')}"
    experiment = mlflow.create_experiment(experiment_name)

    # Log the config
    log.info("Starting Optimization")

    hyper_opt_fmin(space, eval_Slim, cfg.gridsearch.num_evals, verbose = 0, cfg=cfg, train_mat=train_mat, eval_mat=eval_mat, experiment=experiment)
         

    log.info("Optimization finished\n")
    # Shutdown VM when grid-search is finished
    if cfg.model.shutdown == 1:
        os.system("shutdown now -h")
    
    
if __name__ == "__main__":
    my_app()
