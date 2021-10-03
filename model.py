import pickle
import pandas as pd
from collections import Counter
from scipy.optimize import newton
import numpy as np
from scipy.stats import norm, multivariate_normal
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler

def getScaledSum(similarity_features):
    feature_sums = np.sum(similarity_features, axis=1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(feature_sums.reshape(-1,1))
    return scaled


def get_y_init_given_threshold(similarity_features_df, threshold=0.8):
    x = similarity_features_df.values
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    scaled_sum = getScaledSum(x_scaled)
    training_labels_ = scaled_sum > threshold
    y_init = [int(val) for val in training_labels_]
    return y_init


DEL = 1e-300

def _get_results(true_labels, predicted_labels):
    p = precision_score(true_labels, predicted_labels)
    r = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    return p, r, f1

def bay_coeff(a,b,u):
    return np.exp(-(np.log(a/(b+DEL)+b/(a+DEL)+2)+u/(a+b+DEL)))


class ConvergenceMeter:
    def __init__(self, num_converged, rate_threshold,
                 diff_fn=lambda a, b: abs(a - b)):
        self._num_converged = num_converged
        self._rate_threshold = rate_threshold
        self._diff_fn = diff_fn
        self._diff_history = list()
        self._last_val = None

    def offer(self, val):
        if self._last_val is not None:
            self._diff_history.append(
                self._diff_fn(val, self._last_val))

        self._last_val = val

    @property
    def is_converged(self):
        if len(self._diff_history) < self._num_converged:
            return False

        return np.mean(
            self._diff_history[-self._num_converged:]) \
               <= self._rate_threshold



class ZeroerModel:
    class Gaussian:
        def __init__(self, mu, std):
            self.mu = mu
            self.std = (std + DEL)

        def plot(self, axis):
            x = np.linspace(0, 1, 1000)
            pdf = norm.pdf(x, self.mu, self.std)
            axis.plot(x, pdf, linewidth=4)

        def pdf(self, s):
            return norm.pdf(s, loc=self.mu, scale=self.std)

        def logpdf(self, s):
            return norm.logpdf(s, loc=self.mu, scale=self.std)


    def __init__(self, similarity_matrix, feature_names, y,id_df, c_bay,pi_M=None, hard=False):
        self.c_bay = c_bay
        self.y = get_y_init_given_threshold(pd.DataFrame(similarity_matrix))
        self.X = np.array(similarity_matrix)
        self.id_tuple_to_index = {}
        if id_df is not None:
            self.ids = id_df.values
            for i in range(self.ids.shape[0]):
                self.id_tuple_to_index[(self.ids[i,0],self.ids[i,1])] = i
                self.id_tuple_to_index[(self.ids[i,1], self.ids[i,0])] = i

        Mu_all = np.mean(self.X,axis=0)
        self.Cov_all = np.dot(np.transpose(self.X - Mu_all),(self.X - Mu_all))/self.X.shape[0]
        self.corr = pd.DataFrame(similarity_matrix).corr().values
        self.sigma = np.zeros_like(self.corr)
        for i in range(self.corr.shape[0]):
            self.sigma[i,i] = np.std(self.X[:,i])
        self.P_M = np.zeros(self.X.shape[0])  # M is class 1
        self.Q_avg = 0
        self.feature_names = feature_names

        self.col_index_2_group_name = []
        self.group_name_2_col_indices = defaultdict(list)
        for i_col,name in enumerate(feature_names):
            self.col_index_2_group_name.append(name.split("_")[0])
            self.group_name_2_col_indices[self.col_index_2_group_name[-1]].append(i_col)
        self.group_names = list(set(self.col_index_2_group_name))

        if pi_M is None:
            pi_M = Counter(list(y))[1] / float(len(y))

        self._hard = hard
        self._num_rows = self.X.shape[0]
        self._num_cols = self.X.shape[1]
        self._labels = list(sorted(np.unique(y)))
        self.y_step = y

        self.pi_M = pi_M
        self.pi_M_l = pi_M
        self.pi_M_r = pi_M
        self.params = []
        self.Mu_M = np.zeros((self._num_cols,))
        self.Mu_U = np.zeros((self._num_cols,))
        self.Cov_M = np.zeros((self._num_cols,self._num_cols))
        self.Cov_U = np.zeros((self._num_cols,self._num_cols))
        for i in range(self._num_cols):
            self.params.append(self.fit_conditional_parameters(i))
            self.Mu_U[i] = self.params[-1][0].mu
            self.Mu_M[i] = self.params[-1][1].mu
            self.Cov_U[i,i] = self.params[-1][0].std**2
            self.Cov_M[i,i] = self.params[-1][1].std**2
        self.P_M_2_dimen = None
        self.log_P_M_2_dimen = None
        self.log_P_U_2_dimen = None


    def get_class_wise_scores(self, i_cols):
        class_wise_scores = dict()
        for label in self._labels:
            class_wise_scores[label] = \
                self.X[np.where(self.y == label), i_cols]

        return class_wise_scores


    def fit_conditional_parameters(self, i):
        class_wise_scores = self.get_class_wise_scores(i)

        class_wise_parameters = dict()
        for label in self._labels:
            gmm = GaussianMixture(n_components=1)
            gmm.fit(class_wise_scores[label].reshape(-1, 1))

            class_wise_parameters[label] = \
                self.Gaussian(mu=gmm.means_.flatten()[0],
                              std=np.sqrt(gmm.covariances_.flatten()[0]))

        return class_wise_parameters


    def e_step(self, model_l = None,model_r = None):
        self.model_l = model_l
        self.model_r = model_r
        N = self._num_rows
        M = self._num_cols

        reg_cov = 1e-8 * np.identity(len(self.X[0]))
        self.Cov_M += reg_cov
        self.Cov_U += reg_cov

        min_eig = np.min(np.real(np.linalg.eigvals(self.Cov_M)))
        if min_eig < 0:
            self.Cov_M -= 10 * min_eig * np.eye(*self.Cov_M.shape)
            #self.Cov_M += reg_cov
        min_eig = np.min(np.real(np.linalg.eigvals(self.Cov_U)))
        if min_eig < 0:
            self.Cov_U -= 10 * min_eig * np.eye(*self.Cov_U.shape)
            #self.Cov_U += reg_cov
        log_prods_dup = multivariate_normal.logpdf(self.X, mean=self.Mu_M, cov=self.Cov_M,allow_singular=True)
        log_prods_non_dup = multivariate_normal.logpdf(self.X, mean=self.Mu_U, cov=self.Cov_U,allow_singular=True)

        pi_M = self.pi_M
        pi_U = 1 - pi_M

        prob_non_dup_over_dup = np.exp(np.clip(log_prods_non_dup - log_prods_dup, -500, 500))

        self.Q_M = log_prods_dup
        self.Q_U = log_prods_non_dup


        self.P_M = pi_M/ (pi_M + pi_U * prob_non_dup_over_dup)
        self.P_U = 1-self.P_M
        if self._hard:
            self.P_M = np.round(np.clip(self.P_M, 0., 1.))

    def free_energy(self):
        return self.P_M*(np.log(self.pi_M+DEL)-np.log(self.P_M+DEL)+self.Q_M)+self.P_U*(np.log(1-self.pi_M+DEL)-np.log(self.P_U+DEL)+self.Q_U)

    def predict_PM(self,X_test):
        reg_cov = 1e-8 * np.identity(len(self.X[0]))
        self.Cov_M += reg_cov
        self.Cov_U += reg_cov
        min_eig = np.min(np.real(np.linalg.eigvals(self.Cov_M)))
        if min_eig < 0:
            self.Cov_M -= 10 * min_eig * np.eye(*self.Cov_M.shape)
        min_eig = np.min(np.real(np.linalg.eigvals(self.Cov_U)))
        if min_eig < 0:
            self.Cov_U -= 10 * min_eig * np.eye(*self.Cov_U.shape)
        log_prods_dup = multivariate_normal.logpdf(X_test, mean=self.Mu_M, cov=self.Cov_M)
        log_prods_non_dup = multivariate_normal.logpdf(X_test, mean=self.Mu_U, cov=self.Cov_U)

        pi_M = self.pi_M
        pi_U = 1 - pi_M

        prob_non_dup_over_dup = np.exp(np.clip(log_prods_non_dup - log_prods_dup, -500, 500))


        P_M_test = pi_M / (pi_M + pi_U * prob_non_dup_over_dup)
        P_M_test = np.round(np.clip(P_M_test, 0., 1.))
        return P_M_test

    def enforce_transitivity(self, P_M, ids, id_tuple_to_index, model_l, model_r,LR_dup_free=False,LR_identical=False):
        model_l_P_M=None
        model_r_P_M=None
        if model_l is not None:
            model_l_P_M = model_l.P_M
            model_r_P_M = model_r.P_M
            id_tuple_to_index_l = model_l.id_tuple_to_index
            id_tuple_to_index_r = model_r.id_tuple_to_index
        P_M = P_M.copy()
        pred_tuples = []

        for i in range(P_M.shape[0]):
            if P_M[i]>0.5:
                pred_tuples.append((ids[i,0],ids[i,1]))
        pred_tuples = sorted(pred_tuples)

        for i in range(len(pred_tuples)):
            for j in range(i+1, len(pred_tuples)):
                if pred_tuples[j][0] == pred_tuples[i][0]:
                    p1 = P_M[id_tuple_to_index[pred_tuples[i]]]
                    p2 = P_M[id_tuple_to_index[pred_tuples[j]]]
                    p_r = 0
                    id1 = id_tuple_to_index[pred_tuples[i]]
                    id2 = id_tuple_to_index[pred_tuples[j]]
                    if LR_dup_free:
                        p_r = 0
                        idr = -1
                    elif LR_identical:
                        if (pred_tuples[i][1], pred_tuples[j][1]) not in id_tuple_to_index:
                            p_r = 0
                            idr = -1
                        else:
                            p_r = P_M[id_tuple_to_index[(pred_tuples[i][1],pred_tuples[j][1])]]
                            idr = id_tuple_to_index[(pred_tuples[i][1],pred_tuples[j][1])]
                    elif model_r_P_M is not None:
                            if (pred_tuples[i][1], pred_tuples[j][1]) not in id_tuple_to_index_r:
                                p_r = 0
                                idr = -1
                            else:
                                p_r = model_r_P_M[id_tuple_to_index_r[(pred_tuples[i][1],pred_tuples[j][1])]]
                                idr = id_tuple_to_index_r[(pred_tuples[i][1],pred_tuples[j][1])]

                    if p1*p2 > p_r:
                        delta_ls = [self.delta_L(p_r/p2,id1),self.delta_L(p_r/p1,id2)]
                        if idr != -1:
                            if LR_identical:
                                delta_ls.append(self.delta_L(p1 * p2, idr))
                            else:
                                delta_ls.append(model_r.delta_L(p1 * p2, idr))
                        i_max = np.argmax(delta_ls)
                        if delta_ls[i_max]>-1e100:
                            if i_max == 0:
                                P_M[id1] = p_r / p2
                            elif i_max == 1:
                                P_M[id2] = p_r / p1
                            elif i_max == 2:
                                if LR_identical:
                                    P_M[idr] = p1 * p2
                                else:
                                    model_r_P_M[idr] = p1*p2
                else:
                    break

        pred_tuples = sorted(pred_tuples,key=lambda x:(x[1],x[0]))
        for i in range(len(pred_tuples)):
            for j in range(i+1, len(pred_tuples)):
                if pred_tuples[j][1] == pred_tuples[i][1]:
                    p1 = P_M[id_tuple_to_index[pred_tuples[i]]]
                    p2 = P_M[id_tuple_to_index[pred_tuples[j]]]
                    p_l=0
                    id1 = id_tuple_to_index[pred_tuples[i]]
                    id2 = id_tuple_to_index[pred_tuples[j]]
                    if LR_dup_free:
                        p_l = 0
                        idl = -1
                    elif LR_identical:
                        if (pred_tuples[i][0], pred_tuples[j][0]) not in id_tuple_to_index:
                            p_l = 0
                            idl = -1
                        else:
                            p_l = P_M[id_tuple_to_index[(pred_tuples[i][0],pred_tuples[j][0])]]
                            idl = id_tuple_to_index[(pred_tuples[i][0],pred_tuples[j][0])]
                    elif model_l_P_M is not None:
                            if (pred_tuples[i][0], pred_tuples[j][0]) not in id_tuple_to_index_l:
                                p_l = 0
                                idl = -1
                            else:
                                p_l = model_l_P_M[id_tuple_to_index_l[(pred_tuples[i][0],pred_tuples[j][0])]]
                                idl = id_tuple_to_index_l[(pred_tuples[i][0],pred_tuples[j][0])]
                                #p_l = 0
                                #idl = -1
                    if p1*p2 > p_l:
                        delta_ls = [self.delta_L(p_l / p2, id1), self.delta_L(p_l / p1, id2)]
                        if idl != -1:
                            if LR_identical:
                                delta_ls.append(self.delta_L(p1 * p2, idl))
                            else:
                                delta_ls.append(model_l.delta_L(p1 * p2, idl))
                        i_max = np.argmax(delta_ls)
                        if delta_ls[i_max]>-1e100:
                            if i_max == 0:
                                P_M[id1] = p_l / p2
                            elif i_max == 1:
                                P_M[id2] = p_l / p1
                            elif i_max == 2:
                                if LR_identical:
                                    P_M[idl] = p1*p2
                                else:
                                    model_l_P_M[idl] = p1 * p2
                else:
                    break
        if model_r_P_M is not None:
            model_l.P_M = model_l_P_M
            model_r.P_M = model_r_P_M
        return P_M

    def m_step(self):
        N = self._num_rows
        M = self._num_cols

        X = self.X
        P_M = self.P_M
        P_U = 1. - P_M

        if self._hard:
            P_M = P_M.astype(int)
            P_U = P_U.astype(int)

        N_M = np.sum(P_M, axis=0)
        N_U = N - N_M

        self.pi_M = N_M / N


        P_M = P_M.reshape(N, 1)
        P_U = P_U.reshape(N, 1)

        self.Mu_M = np.sum(P_M * X, axis=0) / (N_M + DEL)
        self.Mu_U = np.sum(P_U * X, axis=0) / (N_U + DEL)

        smooth_factor = abs((self.Mu_M - self.Mu_U))**2

        std_M = (np.sqrt(np.sum(
            P_M * ((X - np.tile(self.Mu_M, (N, 1))) ** 2), axis=0) / (N_M + DEL))) + 1e-100
        std_U = (np.sqrt(np.sum(
            P_U * ((X - np.tile(self.Mu_U, (N, 1))) ** 2), axis=0) / (N_U + DEL))) + 1e-100

        Cov_M = np.dot(np.transpose(self.X - self.Mu_M),P_M*(self.X - self.Mu_M))/(N_M + DEL)
        Cov_U = np.dot(np.transpose(self.X - self.Mu_U),P_U*(self.X - self.Mu_U))/(N_U + DEL)

        a = np.diag(Cov_M)
        b = np.diag(Cov_U)
        u = (self.Mu_M - self.Mu_U)**2
        c=0.15

        c_bay = self.c_bay
        bay_ori =  bay_coeff(a,b,u)
        target_bay =bay_ori + c_bay
        target_bay[target_bay>=1] = bay_ori[target_bay>=1]/2+0.5
        def bay_coeff_equ(x):
            return bay_coeff(a + x, b + x, u) - target_bay
        x0=c*smooth_factor
        x1 = np.zeros_like(x0)
        kappas = newton(bay_coeff_equ,x0=x0,x1=x1,maxiter=5,tol=1)
        kappas[kappas<0] = 0
        kappas[kappas>1] = 1
        kappas = np.nan_to_num(kappas,posinf=0,neginf=0)
        self.Cov_M = np.zeros_like(Cov_M)
        self.Cov_U = np.zeros_like(Cov_U)

        for g_name in self.group_names:
            i_cols = self.group_name_2_col_indices[g_name]

            for col_1 in i_cols:
                for col_2 in i_cols:
                    if col_2 == col_1:
                        self.Cov_M[col_1, col_2] = Cov_M[col_1, col_2]+kappas[col_1]
                        self.Cov_U[col_1, col_2] = Cov_U[col_1, col_2]+kappas[col_1]
                    else:
                        self.Cov_M[col_1, col_2] = self.corr[col_1,col_2]*std_M[col_1]*std_M[col_2]
                        self.Cov_U[col_1, col_2] = self.corr[col_1,col_2]*std_U[col_1]*std_U[col_2]
    def L(self,q,i):
        return q*(np.log(self.pi_M+DEL) + self.Q_M[i] - np.log(q+DEL)) +(1-q)*(np.log(1-self.pi_M+DEL)+self.Q_U[i]-np.log(1-q+DEL))

    def delta_L(self,q,i):
        delta = self.L(q,i) - self.L(self.P_M[i],i)
        if delta > 0.00001:
            return -1e200
        return delta

    def save_model(self, filepath):
        pickle.dump(self, open(filepath, 'wb'))

    @staticmethod
    def load_model(filepath):
        return pickle.load(open(filepath, 'rb'))

    @classmethod
    def run_em(cls, similarity_matrixs, feature_names, y_inits,id_dfs,LR_dup_free,LR_identical,run_trans,
               c_bay=0.015,
               y_true=None,
               pi_M=None,
               hard=False,
               max_iter=40):
        sims, sims_l, sims_r = similarity_matrixs
        y_init,y_init_l,y_init_r = y_inits
        id_df, id_df_l, id_df_r = id_dfs
        model = cls(sims, feature_names,y_init,id_df,pi_M=pi_M, hard=hard,c_bay=c_bay)
        if run_trans and LR_dup_free==False and LR_identical==False:
            model_l = cls(sims_l, feature_names,y_init_l,id_df_l,c_bay=c_bay)
            model_r = cls(sims_r, feature_names,y_init_r,id_df_r,c_bay=c_bay)

        convergence = ConvergenceMeter(10, 0.01, diff_fn=lambda a, b: np.linalg.norm(a - b))

        with tqdm(range(max_iter)) as pbar:
            for i in pbar:
                model.e_step()
                if run_trans:
                    if LR_dup_free==False and LR_identical==False:
                        model_r.e_step()
                        model_l.e_step()
                    for i in range(4):
                        if LR_dup_free == False and LR_identical==False:
                            model_l.P_M = model_l.enforce_transitivity(model_l.P_M, model_l.ids, model_l.id_tuple_to_index, model_l, model_l)
                            model_r.P_M = model_r.enforce_transitivity(model_r.P_M, model_r.ids, model_r.id_tuple_to_index, model_r, model_r)
                            model.P_M = model.enforce_transitivity(model.P_M, model.ids, model.id_tuple_to_index, model_l, model_r)
                        else:
                            model.P_M = model.enforce_transitivity(model.P_M, model.ids, model.id_tuple_to_index, None, None,LR_dup_free,LR_identical)
                model.m_step()
                if run_trans and LR_dup_free == False and LR_identical==False:
                    model_r.m_step()
                    model_l.m_step()

                convergence.offer(model.free_energy())
                if convergence.is_converged:
                    break
                if y_true is not None:
                    y_pred = np.round(np.clip(model.P_M + DEL, 0., 1.)).astype(int) \
                            if not hard else model.P_M.astype(int)
                    p, r, f1 = _get_results(y_true, y_pred)
                    result_str = (
                        "norm: {:0.2f}, "
                        "F1: {:0.2f}, "
                        "Precision: {:0.2f}, "
                        "Recall: {:0.2f}".format(
                            np.linalg.norm(model.P_M),
                            f1, p, r))
                    pbar.set_description_str(result_str)

        return model, model.P_M


if __name__ == '__main__':
    pass




