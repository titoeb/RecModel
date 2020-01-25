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
from RecModel import unfold_config

log = logging.getLogger(__name__)

# Helper functions
def eval_MF(params, cfg, train_mat_bin, train_mat_count, eval_mat, experiment):
    # This function is what Hyperopt is going to optimize (minimize 'loss' value)
    print(experiment)
    with mlflow.start_run(experiment_id=experiment):

        # flatten the config.
        params = unfold_config(params)

        # Log the config in hydra
        utils.config_helpers.log_config(dict(cfg.model))   

        # Log the params in mlflow
        utils.config_helpers.log_config(params) 
        
        n_users, n_items = train_mat_bin.shape
        np.random.seed(seed=cfg.model.seed)

        # Create model and train and evaluate it.
        if params['weighted'] == 'weighted':
            if params['bias'] == 1:
                MF = RecModel.WMF(num_items=n_items, num_users=n_users, dim=params['dim'], gamma=params['gamma'], weighted=True, bias=True, seed=int(cfg.model.seed))
            elif params['bias'] == 0:
                MF = RecModel.WMF(num_items=n_items, num_users=n_users, dim=params['dim'], gamma=params['gamma'], weighted=True, bias=False, seed=int(cfg.model.seed))
        elif params['weighted'] == 'non_weighted':
            MF = RecModel.WMF(num_items=n_items, num_users=n_users, dim=params['dim'], gamma=params['gamma'], weighted=False, bias=False, seed=int(cfg.model.seed))

        start = time.time()
        if params['weighted'] == 'non_weighted':
            if params['mat'] == 'count':
                MF.train(utility_mat=train_mat_count.copy(), iterations=int(cfg.model.iterations), verbose=int(cfg.model.verbose), eval_mat=eval_mat.copy(), cores=int(cfg.model.cores), 
                        alpha=params['alpha'], stopping_rounds=int(cfg.model.stopping_rounds), dtype='float32', min_improvement=float(cfg.model.min_improvement),
                        pre_process_count=params['pre_process'], beta=params['beta'], preprocess_mat = params['pre_process'] != "None")

            elif params['mat'] == 'bin':
                MF.train(utility_mat=train_mat_count.copy(), iterations=int(cfg.model.iterations), verbose=int(cfg.model.verbose), eval_mat=eval_mat.copy(), cores=int(cfg.model.cores), 
                    alpha=params['alpha'], stopping_rounds=int(cfg.model.stopping_rounds), dtype='float32', min_improvement=float(cfg.model.min_improvement),
                    pre_process_count=params['pre_process'], beta=params['beta'], preprocess_mat = False)
            else:
                raise ValueError(f"mat can only be one of ['count', 'binary'] not {params['mat']}")

        elif params['weighted'] == 'weighted':
            MF.train(utility_mat=train_mat_bin.copy(), count_mat=train_mat_count.copy(), iterations=int(cfg.model.iterations), verbose=int(cfg.model.verbose), eval_mat=eval_mat.copy(),
                cores=int(cfg.model.cores), alpha=params['alpha'], stopping_rounds=int(cfg.model.stopping_rounds), dtype='float32',
                min_improvement=float(cfg.model.min_improvement), pre_process_count=params['pre_process'], beta=params['beta'], preprocess_mat = False)
        
        else:
            raise ValueError(f"weighted can only be one of ['weighted', 'non_weighted'] not {params['weighted']}")
        
        # Log the training time
        mlflow.log_metric("training_time", int(round(time.time() - start, 0)))

        start = time.time()
        perf_all = MF.eval_topn(test_mat=eval_mat.copy(), topn=np.array(cfg.model.top_n_performances,
         dtype=np.int32), rand_sampled=int(cfg.model.rand_sampled), cores=int(cfg.model.cores), random_state= int(cfg.model.seed))
        mlflow.log_metric("Topn_evaluation_time", int(round(time.time() - start, 0)))

        mse_train = MF.eval_prec(utility_mat = train_mat_count.copy())
        mse_test = MF.eval_prec(utility_mat = eval_mat.copy())

        # Log the topn performance of the model
        for pos in range(len(cfg.model.top_n_performances)):
            mlflow.log_metric(f"recallAT{cfg.model.top_n_performances[pos]}_of_{cfg.model.rand_sampled}", perf_all[f"Recall@{cfg.model.top_n_performances[pos]}"])

        # Log the accuracy
        mlflow.log_metric("mse_train", mse_train)
        mlflow.log_metric("mse_test", mse_test)

        # We will always choose the first topn performance. Hopefully, that is also the smallest is most relevant for us.
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
    train_mat_bin = scipy.sparse.load_npz(os.path.join(get_original_cwd(), cfg.model.train_mat_bin_path))
    train_mat_count = scipy.sparse.load_npz(os.path.join(get_original_cwd(), cfg.model.train_mat_count_path))
    n_users, n_items = train_mat_bin.shape
    eval_mat = scipy.sparse.load_npz(os.path.join(get_original_cwd(), cfg.model.eval_mat_path))

    # The algorithms rely on the sparse matrices being sorted by idx.
    train_mat_bin = train_mat_bin.astype(np.float32)
    train_mat_count = train_mat_count.astype(np.float32)
    eval_mat = eval_mat.astype(np.float32)
    eval_mat.sort_indices()
    train_mat_bin.sort_indices()
    train_mat_count.sort_indices()

    # Setup HyperOpt
    # Space with alpha and beta
    space = {'weighted' : hp.hp.choice('solver', [
                {'type': 'weighted',
                'dim' : hp.hp.choice('dim_weighted', np.arange(10, 200)),
                'gamma': hp.hp.uniform('gamma_weighted', 0, 1000),
                'bias': hp.hp.choice('bias', [0, 1]),
                'pre_process': hp.hp.choice('pre_process_weighted', ['log', 'linear']),
                'alpha': hp.hp.lognormal('alpha_weighted', 0, 1),
                'beta': hp.hp.lognormal('beta', 0, 1)},

                {'type': 'non_weighted',
                'dim' : hp.hp.choice('dim_non_weighted', np.arange(10, 200)),
                'gamma': hp.hp.uniform('gamma_non_weighted', 0, 1000),
                'pre_process': hp.hp.choice('pre_process_non_weighted', ['log', 'linear', 'None']),
                'mat' : hp.hp.choice('mat', ['count', 'bin']),
                'alpha': hp.hp.lognormal('alpha_non_weighted', 0, 1),
                'beta': hp.hp.lognormal('beta_non_weighted', 0, 1)}
            ])}
    # Space without alpha and beta:
    """space = {'weighted' : hp.hp.choice('solver', [
                {'type': 'weighted',
                'dim' : hp.hp.choice('dim_weighted', np.arange(10, 200)),
                'gamma': hp.hp.lognormal('gamma_weighted', 0, 1),
                'bias': hp.hp.choice('bias', [0, 1]),
                'pre_process': hp.hp.choice('pre_process_weighted', ['log', 'linear'])},

                {'type': 'non_weighted',
                'dim' : hp.hp.choice('dim_non_weighted', np.arange(10, 200)),
                'gamma': hp.hp.lognormal('gamma_non_weighted', 0, 1),
                'pre_process': hp.hp.choice('pre_process_non_weighted', ['log', 'linear', 'None']),
                'mat' : hp.hp.choice('mat', ['count', 'bin'])}
            ])}"""

    # Set up MLFlow experiment
    experiment_name = f"HyperOpt_MF_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M').replace(' ', '_')}"
    experiment = mlflow.create_experiment(experiment_name)

    # Log the config
    log.info("Starting Optimization")
    hyper_opt_fmin(space, eval_MF, cfg.gridsearch.num_evals, verbose = 0, cfg=cfg, train_mat_count=train_mat_count, train_mat_bin=train_mat_bin, eval_mat=eval_mat, experiment=experiment)  

    log.info("Optimization finished\n")
    # Shutdown VM when grid-search is finished
    if cfg.model.shutdown == 1:
        os.system("shutdown now -h")
    
    
if __name__ == "__main__":
    my_app()
