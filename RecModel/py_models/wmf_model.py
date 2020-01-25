import numpy as np
import scipy.sparse
import time
from multiprocessing import Pool
from functools import partial
from RecModel.py_models.base_model import RecModel, MKLThreads

class WMF(RecModel):

    def __init__(self, num_items, num_users, dim, gamma, weighted=None, bias=False, seed=1993, dtype='float32'):
        """
        :param num_items:
        :param num_users:
        :param dim:
        :param gamma:
        :param weighted:
        :param bias:
        :param seed:
        """
        np.random.seed(seed)
        self.bias = bias
        self.gamma = gamma
        if self.bias is False:
            self.items = np.random.random((num_items, dim)).astype(dtype=dtype)
        elif self.bias is True:
            self.items = np.random.random((num_items, (dim + 1))).astype(dtype=dtype)
        self.users = None
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.weighted = weighted
        self.dtype = dtype

    def rank(self, items, users, topn=None):
        if topn is None:
            topn = len(items)

        if isinstance(users, list):
            if not type(items) == np.ndarray:
                users = np.array(users)
            return[self.rank(items, user, topn) for user in users]
        else:
            if not type(items) == np.ndarray:
                items = np.array(items)
            predictions = self.predict(users=users, items=items)

            # simple heuristic: If the number of topn is higher than half the number of prediction it may useful to
            # sort the whole vector.
            if len(predictions) * 0.5 > topn:
                # Ranking optimized for sorting only the first few entries. This is useful if topn << |item| and sorting
                # all ranked items would take significantly longer.
                return items[np.argpartition(predictions, list(range(-topn, 0, 1)))[-topn:]][::-1]

            else:
                # Ranking sorting all item, useful if topn near |items|.
                return items[np.argsort(predictions)[-topn:]][::-1]

    def train(self, utility_mat, iterations, verbose=0, eval_mat=None, count_mat=None, alpha=10,
              cores=4, stopping_rounds=3, dtype='float64', min_improvement=0.0001,
              pre_process_count='log', beta=1, preprocess_mat=False):

        # Make hard copies of input matrices.
        utility_mat = utility_mat.copy()
        if count_mat is not None:
            count_mat = count_mat.copy()

        if self.bias is True and self.weighted is False:
            print(f"Bias computation is only implemented for weighted matrix factorization.")

        if eval_mat is None and verbose > 1:
            print(f"Since no explicit evaluation was provided the train matrix is used for evaluation.")
            eval_mat = utility_mat

        if preprocess_mat==True:
            if pre_process_count == 'log':
                utility_mat.data = alpha * np.log(1 + beta * utility_mat.data)

            elif pre_process_count == 'linear':
                utility_mat.data = alpha * utility_mat.data

        past_mse = []
        if self.weighted is not True:

            # Non-Weighted MF
            # Initiallize data structure for Early stopping.
            last_mse = - np.inf
            count_improvement = 0

            for iter in range(iterations):
                if verbose > 0:
                    print(f"Starting fitting iteration {iter}")

                # Recomputing user vectors assuming item vector to be fixed
                self.users = scipy.sparse.csr_matrix.dot(np.dot(np.linalg.inv(np.dot(self.items.T, self.items) + self.gamma * np.eye(self.dim, dtype=self.dtype)), self.items.T), utility_mat.T).T.copy()

                # Recomputing item vectors assuming user vector to be fixed.
                self.items = scipy.sparse.csr_matrix.dot(np.dot(np.linalg.inv(np.dot(self.users.T, self.users) + self.gamma * np.eye(self.dim, dtype=self.dtype)), self.users.T), utility_mat).T.copy()

                mse_eval = self.eval_prec(eval_mat)
                past_mse.append(mse_eval)
                if verbose > 0:
                    print(f"Current eval mse is {mse_eval}")

                if verbose > 1:
                    print(f"\tMSE Eval: {mse_eval}")
                    print(f"\tMSE Train: {self.eval_prec(utility_mat)}")

                mse_eval = self.eval_prec(eval_mat)
                if mse_eval * (1 + min_improvement) > last_mse:
                    count_improvement += 1
                else:
                    count_improvement = 0
                last_mse = mse_eval

                # Early stopping: If the mean mse of the past 2 rounds were below stopping_percentage lower abort training.
                if count_improvement >= stopping_rounds:
                    break
                    print(f"Training was aborted as the precision did not improve for {count_improvement} rounds")

            if verbose > 0:
                print("Training was completed.")
            if verbose > 1:
                print(f"MSE Eval at iteration {iter}: {self.eval_prec(eval_mat)}")
                print(f"MSE Train at iteration {iter}: {self.eval_prec(utility_mat)}")
        else:
            # Weighted MF
            # Pre-Process Count Matrix:
            if pre_process_count == 'log':
                count_mat.data = alpha * np.log(1 + beta * count_mat.data)

            elif pre_process_count == 'linear':
                count_mat.data = alpha * count_mat.data

            else:
                raise ValueError(f"Pre_process_count {pre_process_count} is not implement please use log or linear.")

            count_mat_T = count_mat.T.tocsr()

            # Initiallize data structure for Early stopping.
            last_mse = - np.inf
            count_improvement = 0

            for iter in range(iterations):

                if verbose > 0:
                    print(f"Starting fitting iteration {iter}")

                # Recomputing item vectors assuming user vector to be fixed.
                if cores >= 1:
                    if self.bias is False:
                        if cores == 1:
                            self.users = self.recompute_factors(self.items, count_mat, self.gamma)
                            self.items = self.recompute_factors(self.users, count_mat_T, self.gamma)
                        else:
                            self.users = self.recompute_factors_par(self.items, count_mat, self.gamma, cores=cores)
                            self.items = self.recompute_factors_par(self.users, count_mat_T, self.gamma,cores=cores)
                    elif self.bias is True:
                        if cores == 1:
                            start = time.time()
                            self.users = self.recompute_factors_bias(self.items.copy(), count_mat.copy(), self.gamma, cores=cores)
                            self.items = self.recompute_factors_bias(self.users.copy(), count_mat_T.copy(), self.gamma, cores=cores)
                            print(f"Iteration {iter} took {round(time.time() - start, 4)} seconds.")
                        else:
                            self.users = self.recompute_factors_bias_par(self.items.copy(), count_mat.copy(), self.gamma,cores=cores)
                            self.items = self.recompute_factors_bias_par(self.users.copy(), count_mat_T.copy(), self.gamma, cores=cores)
                    else:
                        raise ValueError(f"self.bias = {self.bias} is unknown. Only True / False are allowed.")

                else:
                    raise ValueError(f"Values of cores has to be positive not {cores}")

                mse_eval = self.eval_prec(eval_mat)
                if mse_eval * (1 + min_improvement) > last_mse:
                    count_improvement += 1
                else:
                    count_improvement = 0
                last_mse = mse_eval


                if verbose > 0:
                    print(f"Current eval mse is {mse_eval}")

                if verbose > 1:
                    print(f"\tMSE Eval: {mse_eval}")
                    print(f"\tMSE Train: {self.eval_prec(utility_mat)}")

                # Early stopping: If the mean mse of the past 2 rounds were below stopping_percentage lower abort training.
                if count_improvement >= stopping_rounds:
                    break
                    print(f"Training was aborted as the precision did not improve for {count_improvement} rounds")

            if verbose > 0:
                print("Training was completed.")

            if verbose > 1:
                print(f"MSE Eval at iteration {iter}: {self.eval_prec(eval_mat)}")
                print(f"MSE Train at iteration {iter}: {self.eval_prec(utility_mat)}")
        return iter

    def predict(self, users, items):
        """
        Gives the predicted scores for the user and items. User and Item need to have the same shape except if only one
        user or item was provided. Given input ([user1, user2], [item1, item2]) was provided the prediction
        (f(user1, item1), f(user2, item2)) are given. In the case of ([user1], [item1, ..., itemn]) or
        ([user1, ... , usern], (item1)) the predictions f([user1, item1], ... [user1, itemn]) or
        f([user1, item1], [usern, item1]) are given respectively.
        """
        # Input checks
        if (type(users) == list or type(users) == np.ndarray) and (type(items) == list or type(items) == np.ndarray):
            if len(users) != len(items):
                if not (len(users) == 1 or len(items) == 0):
                    raise ValueError("users and items need to have the same length or only one user / item needs to be provided.")

        if self.bias is False:
            return (self.users[users, :] * self.items[items, :]).sum(axis=1)
        else:
            # Incorporate the biases!
            user_bias = self.users[:, 0][users]
            item_bias = self.items[:, 0][items]
            return (self.users[:, 1:][users, :] * self.items[:, 1:][items, :]).sum(axis=1) + user_bias + item_bias

    def recompute_factors(self, Y, C, lambda_reg):

        YTY_I = np.dot(Y.T, Y) + lambda_reg * np.eye(Y.shape[1], dtype=self.dtype)

        X_new = np.empty((C.shape[0], Y.shape[1]), dtype=self.dtype)

        # For each row in Count matrix C, compute the corresponding output vector.
        for row in range(len(C.indptr) - 1):
            start_elem = C.indptr[row]
            end_elem = C.indptr[row + 1]
            if end_elem - start_elem == 0:
                # There are not entries in that row in C.
                X_new[row, :] = 0

            else:
                # There are some entries in C.

                # Select the indices and data for the relevant column.
                idx = C.indices[start_elem:end_elem]
                data = C.data[start_elem:end_elem]
                Y_rel = Y[idx, :]

                # This is Y'C_uY. But we use the fact that C_u is diagonal only if it is in idx
                # does have positive value.
                YT_C_Y = np.dot(Y_rel.T, Y_rel * data[:, np.newaxis])

                X_new[row, :] = np.linalg.solve(YT_C_Y + YTY_I, np.dot(data + 1, Y_rel))
        return X_new

    def recompute_factors_par(self, Y, C, lambda_reg, cores=4):

        YTY_I = np.dot(Y.T, Y) + lambda_reg * np.eye(Y.shape[1], dtype=self.dtype)
        pool = Pool(cores)
        X_new = np.stack(pool.map(self.recompute_factors_intern,
                                  ((YTY_I, row, Y, C) for row in range(len(C.indptr) - 1))))
        pool.close()
        pool.join()
        return X_new

    def recompute_factors_bias_par(self, Y, C, lambda_reg,  cores=3):
        # Extract item / user bias
        bias = Y[:, 0].copy()

        # Set old bias to one (needed to compute the correct bias implictly later on.)
        Y[:, 0] = 1
        YTY_I = np.dot(Y.T, Y) + lambda_reg * np.eye(Y.shape[1], dtype=self.dtype)

        pool = Pool(cores)
        X_new = np.stack(pool.map(self.recompute_factors_bias_intern,
                                  ((YTY_I, row, Y, C, bias) for row in range(len(C.indptr) - 1))))
        pool.close()
        pool.join()
        return X_new

    def recompute_factors_bias_intern(self, args):
        with MKLThreads(1):
            YTY_I, row, Y, C, bias = args
            num_fac = Y.shape[1]
            start_elem = C.indptr[row]
            end_elem = C.indptr[row + 1]

            if end_elem - start_elem == 0:
                # There are not entries in that row in C.
                return np.zeros(num_fac, dtype=self.dtype)

            else:
                # Select the indicees and data for the relevant column.
                idx = C.indices[start_elem:end_elem]
                data = C.data[start_elem:end_elem] - bias[idx]
                Y_rel = Y[idx, :]
                # This is Y'C_uY. But we use the fact that C_u is diagonal only if it is in idx
                # does have positive value.
                YT_C_Y = np.dot(Y_rel.T, Y_rel * data[:, np.newaxis])

                return np.linalg.solve(YT_C_Y + YTY_I, np.dot(data + 1, Y_rel))

    def recompute_factors_intern(self, args):
        with MKLThreads(1):
            YTY_I, row, Y, C = args
            num_fac = Y.shape[1]
            start_elem = C.indptr[row]
            end_elem = C.indptr[row + 1]

            if end_elem - start_elem == 0:
                # There are not entries in that row in C.
                return np.zeros(num_fac,dtype=self.dtype)

            else:
                # Select the indicees and data for the relevant column.
                idx = C.indices[start_elem:end_elem]
                data = C.data[start_elem:end_elem]
                Y_rel = Y[idx, :]
                # This is Y'C_uY. But we use the fact that C_u is diagonal only if it is in idx
                # does have positive value.
                YT_C_Y = np.dot(Y_rel.T, Y_rel * data[:, np.newaxis])

                return np.linalg.solve(YT_C_Y + YTY_I, np.dot(data + 1, Y_rel))

    def recompute_factors_bias(self, Y, C, lambda_reg, cores=1):
        """
               CODE Insprired by recompute_factors but used the formulas from:
               http://activisiongamescience.github.io/2016/01/11/Implicit-Recommender-Systems-Biased-Matrix-Factorization/

               under the header
               ALS with Biases for Implicit Feedback

               recompute matrix X from Y.
               X = recompute_factors(Y, S, lambda_reg)
               This can also be used for the reverse operation as follows:
               Y = recompute_factors(X, ST, lambda_reg)

               The comments are in terms of X being the users and Y being the items.
               """
        with MKLThreads(cores):
            # Extract item / user bias
            bias = Y[:, 0].copy()

            # Set old bias to one (needed to compute the correct bias implictly later on.)
            Y[:, 0] = 1
            YTY_I = np.dot(Y.T, Y) + lambda_reg * np.eye(Y.shape[1], dtype=self.dtype)

            X_new = np.empty((C.shape[0], Y.shape[1]), dtype=self.dtype)

            # For each row in Count matrix C, compute the corresponding output vector.
            for row in range(len(C.indptr) - 1):
                start_elem = C.indptr[row]
                end_elem = C.indptr[row + 1]

                # Select the indicees and data for the relevant column.
                idx = C.indices[start_elem:end_elem]
                data = C.data[start_elem:end_elem] - bias[idx]
                Y_rel = Y[idx, :]

                # This is Y'C_uY. But we use the fact that C_u is diagonal only if it is in idx
                # does have positive value.
                YT_C_Y = np.dot(Y_rel.T, Y_rel * data[:, np.newaxis])

                X_new[row, :] = np.linalg.solve(YT_C_Y + YTY_I, np.dot(data + 1, Y_rel))
        return X_new