import os
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
def eval_Ease(params, cfg, train_mat_bin, train_mat_count, eval_mat, experiment):
    # This function is what Hyperopt is going to optimize (minimize 'loss' value)
    print(experiment)
    with mlflow.start_run(experiment_id=experiment):

        # Log the config
        utils.config_helpers.log_config(dict(cfg.model))        

        n_users, n_items = train_mat_bin.shape
        np.random.seed(seed=cfg.model.seed)

        # Log relevant parameters for this run.
        mlflow.log_param("alpha", params['alpha'])
        mlflow.log_param("mat", params['mat'])

        # Log this run
        log.info(f"Testing  alpha: {params['alpha']}, and mat: {params['mat']}")
        
        start = time.time()       

        # Create model
        ease = RecModel.Ease(num_items=n_items, num_users=n_users)
        print(f"start training!, number of cores {int(cfg.model.cores)}")
        
        if params['mat'] == 'count':
            ease.train(train_mat_count.copy(), alpha=params['alpha'], verbose=int(cfg.model.verbose), cores=int(cfg.model.cores))
        elif params['mat'] == 'bin':
            ease.train(train_mat_bin.copy(), alpha=params['alpha'], verbose=int(cfg.model.verbose), cores=int(cfg.model.cores))
        else:
            raise ValueError(f"mat can only take values 'count' or 'bin' and not {params['mat']}")
        
        print('trained model')
        # Log run-time
        mlflow.log_metric("Runtime", int(round(time.time() - start, 0)))

        # Evaluate model
        perf_all = ease.eval_topn(test_mat=eval_mat.copy(), topn=np.array(cfg.model.top_n_performances,
         dtype=np.int32), rand_sampled=int(cfg.model.rand_sampled), cores=int(cfg.model.cores), random_state= int(cfg.model.seed))
        print('estimated performance')

        # Log the performance of the model
        for pos in range(len(cfg.model.top_n_performances)):
            mlflow.log_metric(f"recallAT{cfg.model.top_n_performances[pos]}_of_{cfg.model.rand_sampled}", perf_all[f"Recall@{cfg.model.top_n_performances[pos]}"])

        print('logged recall')
        if params['mat'] == 'count':
            mlflow.log_metric('MSE_train', ease.eval_prec(utility_mat = train_mat_count.copy()))
            
        elif params['mat'] == 'bin':
            mlflow.log_metric('MSE_train', ease.eval_prec(utility_mat = train_mat_bin.copy()))
        else:
            raise ValueError(f"mat can only take values 'count' or 'bin' and not {params['mat']}")

        print('estimated mse')

        #We will always choose the first topn performance. Hopefully, that is also the smallest is most relevant for us.
        rel_topn_perf = perf_all[f"Recall@{cfg.model.top_n_performances[0]}"]
        print('extracted performance')        
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
    train_mat_bin = scipy.sparse.load_npz(os.path.join(get_original_cwd(), cfg.model.train_mat_bin_path))
    train_mat_count = scipy.sparse.load_npz(os.path.join(get_original_cwd(), cfg.model.train_mat_count_path))
    n_users, n_items = train_mat_bin.shape
    eval_mat = scipy.sparse.load_npz(os.path.join(get_original_cwd(), cfg.model.eval_mat_path))

    train_mat_bin = train_mat_bin.astype(np.float64)
    train_mat_count = train_mat_count.astype(np.float64)
    eval_mat = eval_mat.astype(np.float64)
    eval_mat.sort_indices()
    train_mat_bin.sort_indices()
    train_mat_count.sort_indices()

    # Setup HyperOpt
    space = space = {'alpha': hp.hp.lognormal('alpha',0, 1), 'mat': hp.hp.choice('mat', ['count', 'bin'])}

    # Set up MLFlow experiment
    experiment_name = f"HyperOpt_Ease_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M').replace(' ', '_')}"
    experiment = mlflow.create_experiment(experiment_name)

    # Log the config
    log.info("Starting Optimization")
    hyper_opt_fmin(space, eval_Ease, cfg.gridsearch.num_evals, verbose = 0, cfg=cfg, train_mat_count=train_mat_count, train_mat_bin=train_mat_bin, eval_mat=eval_mat, experiment=experiment)
         

    log.info("Optimization finished\n")
    # Shutdown VM when grid-search is finished
    if cfg.model.shutdown == 1:
        os.system("shutdown now -h")
    
    
if __name__ == "__main__":
    my_app()
