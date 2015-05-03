"""
Created on Sun Apr 12 20:20:48 2015

@author: Ziang
"""
import time
import numpy as np
from math import exp, sqrt, pi
from numpy.random import normal, choice
from scipy import linalg
from matplotlib import pyplot as plt
from GPSVI.util.inverse import cho_inverse
from GPSVI.util.kernel import compute_kernel
from GPSVI.util.quadrature import GaussHermiteQuadrature

class GPClassifier:

    def __init__(self, xTr, yTr, xTe=None, yTe=None, **kwargs):
        self.xTr = xTr
        self.yTr = yTr
        self.xTe = xTe
        self.yTe = yTe
        if 'num_inducing_points' in kwargs.keys():
            self.num_inducing_points = min(xTr.shape[0], \
                                           kwargs['num_inducing_points'])
        else:
            self.num_inducing_points = min(xTr.shape[0], 10)
        if 'quad_deg' in kwargs.keys():
            self.quad = GaussHermiteQuadrature(kwargs['quad_deg'])
        else:
            self.quad = GaussHermiteQuadrature(30)
        if 'kernel_type' in kwargs.keys():
            self.kernel_type = kwargs['kernel_type']
        else:
            self.kernel_type = 'rbf'
        if 'kernel_args' in kwargs.keys():
            self.kernel_args = kwargs['kernel_args']
        else:
            self.kernel_args = {'gamma':None}
        if 'learning_rate' in kwargs.keys():
            self.learning_rate = kwargs['learning_rate']
        else:
            self.learning_rate = 0.01
        self.r0 = self.learning_rate
        if 'alpha' in kwargs.keys():
            self.alpha = kwargs['alpha']
        else:
            self.alpha = 0.2
        if 'verbose' in kwargs.keys():
            self.verbose = kwargs['verbose']
        else:
            self.verbose = 0
        if 'max_iter' in kwargs.keys():
            self.max_iter = kwargs['max_iter']
        else:
            self.max_iter = 10000
        if 'tolerance' in kwargs.keys():
            self.tolerance = kwargs['tolerance']
        else:
            self.tolerance = 1.0
        if xTr is None or yTr is None:
            raise Exception('None training data error')
        else:
            M = self.num_inducing_points
            self.labels = np.unique(yTr)
            self.num_classes = len(self.labels)
            C = self.num_classes
            self.labels_dist = []
            for i in range(C):
                indices, = np.where(self.yTr == self.labels[i])
                self.labels_dist.append(indices)
            self.inducing_points = xTr[self.sample_x(M)]
            if self.verbose > 0:
                print('computing kernel matrices...')
            self.Kmm = self.kernel(self.inducing_points, self.inducing_points)
            self.Kmm_inv = cho_inverse(self.Kmm)
            self.Knn = self.kernel(xTr, xTr)
            self.Knm = self.kernel(xTr, self.inducing_points)
            self.A = self.Knm.dot(self.Kmm_inv)
            self.mask = np.tril(np.ones((M, M))).ravel()
            if self.verbose > 0:
                print('finished.')
            self.parameters = np.zeros(M*C+M*M*C)
            self.parameters_best = None
            for j in range(C):
                self.parameters[M*j:M*(j+1)] = np.ones(M)
                self.parameters[M*C+M*M*j:M*C+M*M*(j+1)] = np.eye(M).ravel()

    def fit(self):
        self.optimize()

    def score(self, xTe, yTe):
        pd = self.predict(xTe)
        return 1 - len(np.where(pd != yTe)[0]) / float(xTe.shape[0])

    def objective(self, z, indices):
        val = 0
        target = self.yTr[indices]
        N = len(indices)
        M = self.num_inducing_points
        C = self.num_classes
        S = self.quad.degree
        Knn = self.Knn[indices, indices]
        Kmm = self.Kmm
        A = self.A[indices, :]
#new
        samples = self.quad.get_samples().reshape((S, 1))
        weights = self.quad.get_weights().reshape((S, 1))
        f = []
        for c in range(C):
            m = self.parameters[M*c:M*(c+1)].reshape(M, 1)
            L = self.parameters[M*C+M*M*c:M*C+M*M*(c+1)].reshape(M, M)
            mu = A.dot(m)
            sigma = np.abs((Knn + A.dot(L.dot(L.T) - Kmm).dot(A.T)).diagonal()).reshape((1,N))
            f.append(np.sum(np.exp(samples.dot(sigma) + mu.T) * weights, axis=0)/sqrt(pi))
        sumf = np.sum(f, axis=0)
        for i in range(N):
            y = target[i]
            proba = f[y][i]/sumf[y]
            val += np.log(proba)
        
        
##old
#        mu = []
#        sigma = []
#        for j in range(C):
#            m = self.parameters[M*j:M*(j+1)].reshape(M, 1)
#            L = self.parameters[M*C+M*M*j:M*C+M*M*(j+1)].reshape(M, M)
#            mu.append(A.dot(m).ravel())
#            sigma.append(np.abs((Knn + A.dot(L.dot(L.T) - Kmm).dot(A.T)).diagonal()))
#        func = lambda c: exp(samples[k]*sigma[c][i] + mu[c][i])
#        samples = self.quad.get_samples()
#        weights = self.quad.get_weights()
#        for i in range(N):
#            y = target[i]
#            sample_sum = 0
#            for k in range(S):
##                expf_map = map(lambda c:exp(samples[k]*abs(sigma[c][i]) + mu[c][i]), range(C))
#                expf_map = map(func, range(C))
#                exp_sum = 0
#                for expf, num in zip(expf_map, range(C)):
#                    exp_sum += expf
#                    if num == y:
#                        expf_y = expf
##                sample_sum += weights[k]**C*expf_y/exp_sum
#                sample_sum += weights[k] * expf_y / exp_sum
#            sample_sum /= sqrt(pi)
#            val += np.log(sample_sum)
        return val/N

    def gradient(self, z, indices):
        target = self.yTr[indices]
        N = len(indices)
        M = self.num_inducing_points
        C = self.num_classes
        S = self.quad.degree
        grad = np.zeros((M*C+M*M*C))
        u = [np.array(self.parameters[M*C+M*M*j:M*C+M*M*(j+1)])\
                          .reshape(M, M).dot(z) \
           + np.array(self.parameters[M*j:M*(j+1)])\
                          .reshape(M, 1) for j in range(C)] # global variables
        Knn = self.Knn[indices, indices]
        Knm = self.Knm[indices, :]
        Kmm_inv = self.Kmm_inv
        A = self.A[indices]
        delta = [np.diag(1.0/self.parameters[M*C+M*M*c:M*C+M*M*(c+1)]\
                                 .reshape(M, M).diagonal()) for c in range(C)]
# new
        sigma = np.abs(np.diag(Knn - A.dot(Knm.T))).reshape((1, N))
        samples = self.quad.get_samples().reshape((S, 1))
        weights = self.quad.get_weights().reshape((S, 1))
        f = []
        for c in range(C):
            mu = A.dot(u[c])
            f.append(np.sum(np.exp(samples.dot(sigma) + mu.T) * weights, axis=0)/sqrt(pi))
        sumf = np.sum(f, axis=0)
        for i in range(N):
            y = target[i]
            approx = f[y][i]/sumf[y]
            grad[M*y:M*(y+1)] += (1-approx)*A[i, :]
            grad[M*C+M*M*y:M*C+M*M*(y+1)] += \
                          (1-approx)*A[i, :].reshape(M, 1).dot(z.T).ravel()
## old
#        sigma = np.abs(np.diag(Knn - A.dot(Knm.T)))
#        mu = [A.dot(u[j]).ravel() for j in range(C)]
#        func = lambda c: exp(samples[k]*sigma[i] + mu[c][i])
#        samples = self.quad.get_samples()
#        weights = self.quad.get_weights()
#        for i in range(N):
#            y = target[i]
#            sample_sum = 0
#            for k in range(S):
#                expf_map = map(func, range(C))
#                exp_sum = 0
#                for expf, num in zip(expf_map, range(C)):
#                    exp_sum += expf
#                    if num == y:
#                        expf_y = expf
##                sample_sum += weights[k]**C * expf_y/exp_sum
#                sample_sum += weights[k] * expf_y / exp_sum
#            sample_sum /= sqrt(pi)
#            grad[M*y:M*(y+1)] += (1-sample_sum)*A[i, :]
#            grad[M*C+M*M*y:M*C+M*M*(y+1)] += \
#                (1-sample_sum)*A[i, :].reshape(M, 1).dot(z.T).ravel()

        for j in range(C):
            grad[M*j:M*(j+1)] -= Kmm_inv.dot(u[j]).ravel()
            grad[M*C+M*M*j:M*C+M*M*(j+1)] -= Kmm_inv.dot(u[j]).dot(z.T).ravel()
            grad[M*C+M*M*j:M*C+M*M*(j+1)] += delta[j].ravel()
        return grad

    def sample_x(self, N):
        total_size = self.xTr.shape[0]
        if N > total_size:
            raise Exception('Exceed size of data')
        C = self.num_classes
        num_samples = []
        for i in range(C):
            label_size = len(self.labels_dist[i])
            num = min(int(N*(label_size)/total_size*1.0), label_size)
            num_samples.append(num)
        while sum(num_samples) < N:
            i = (i + 1) % C
            label_size = len(self.labels_dist[i])
            if num_samples[i] < label_size:
                num_samples[i] += 1
        indices = []
        for i in range(C):
            label_size = len(self.labels_dist[i])
            indices += self.labels_dist[i][choice(range(label_size), \
                                                  size=num_samples[i], \
                                                  replace=False)].tolist()
        return indices

    def sample_z(self):
        return normal(0, 1, (self.num_inducing_points, 1))

    def update(self, z, indices, value, gradient, t):
        M = self.num_inducing_points
        C = self.num_classes
        rho1 = self.learning_rate
        rho2 = self.learning_rate*0.1
#        rho2 = self.learning_rate*(1.0/self.xTr.shape[1])
        self.parameters[0:M*C] += rho1 * gradient[0:M*C]
        self.parameters[M*C: ] += rho2 * gradient[M*C: ]
        for c in range(C):
            self.parameters[M*C+M*M*c:M*C+M*M*(c+1)] *= self.mask
        self.learning_rate = self.r0*(1+self.r0*0.05*t)**(-3/4)
        h = self.inducing_points
        self.inducing_points = self.xTr[self.sample_x(M)]
        if value <= self.objective(z, indices):
            self.inducing_points = h

    def predict_proba(self, xTe):
        N, _ = xTe.shape
        M = self.num_inducing_points
        C = self.num_classes
        S = self.quad.degree
        h = self.inducing_points
        probs = np.zeros((N, C))
        Knn = self.kernel(xTe, xTe)
        Knm = self.kernel(xTe, h)
        Kmm = self.kernel(h, h)
        Kmm_inv = cho_inverse(Kmm)
        A = Knm.dot(Kmm_inv)
        
# new
        samples = self.quad.get_samples().reshape((S, 1))
        weights = self.quad.get_weights().reshape((S, 1))
        f = []
        for c in range(C):
            m = self.parameters[M*c:M*(c+1)].reshape(M, 1)
            L = self.parameters[M*C+M*M*c:M*C+M*M*(c+1)].reshape(M, M)
            mu = A.dot(m)
            sigma = np.abs((Knn + A.dot(L.dot(L.T) - Kmm).dot(A.T)).diagonal()).reshape((1,N))
            f.append(np.sum(np.exp(samples.dot(sigma) + mu.T) * weights, axis=0)/sqrt(pi))
        sumf = np.sum(f, axis=0)
        for i in range(N):
            probs[i, :] = [f[c][i]/sumf[c] for c in range(C)]
## old
#        mu = []
#        sigma = []
#        for j in range(C):
#            m = self.parameters[M*j:M*(j+1)].reshape(M, 1)
#            L = self.parameters[M*C+M*M*j:M*C+M*M*(j+1)].reshape(M, M)
#            mu.append(A.dot(m).ravel())
#            sigma.append(np.abs((Knn + A.dot(L.dot(L.T) - Kmm).dot(A.T)).diagonal()))
#        func = lambda c: exp(samples[k]*sigma[c][i] + mu[c][i])
#        samples = self.quad.get_samples()
#        weights = self.quad.get_weights()
#        for i in range(N):
#            sample_sum = 0
#            for j in range(C):
#                for k in range(S):
#                    expf_map = map(func, range(C))
#                    exp_sum = 0
#                    for expf, num in zip(expf_map, range(C)):
#                        exp_sum += expf
#                        if num == j:
#                            expf_y = expf
##                    sample_sum += weights[k]**C * expf_y/exp_sum
#                    sample_sum += weights[k] * expf_y/exp_sum
#                sample_sum /= sqrt(pi)
#                probs[i, j] = sample_sum
        return probs

    def predict(self, xTe):
        N, _ = xTe.shape
        probs = self.predict_proba(xTe)
        return np.array([self.labels[np.argmax(probs[i, :])] for i in range(N)])

    def kernel(self, A, B):
        return compute_kernel(A, B, ktype=self.kernel_type, **self.kernel_args)

    def optimize(self):
        if self.verbose > 0:
            print('start optimizing...')
        # initialize parameters
        t = 0
        z = self.sample_z()
        indices = self.sample_x(int(self.alpha*self.xTr.shape[0]))
        grad = self.gradient(z, indices)
        val = self.objective(z, indices)
        # for debugging
        is_optimizing_on_te = False if self.xTe is None \
                                   and self.yTe is None else True
        if self.verbose > 0:
            self.parameters_best = self.parameters
            if self.verbose > 1:
                plt.figure('Stochastic Variational Inference')
                plt.rc('text', usetex=True)
                xdata = [t] # steps
                vals = [val]
                error_tr = np.sum(len(np.where(self.predict(self.xTr) \
                           != self.yTr)[0])) / float(self.xTr.shape[0])
                ydata1 = [error_tr]
                error_tr_min = error_tr
                if is_optimizing_on_te:
                    error_te = np.sum(len(np.where(self.predict(self.xTe) \
                               != self.yTe)[0])) / float(self.xTe.shape[0])
                    ydata2 = [error_te]
                    error_te_min = error_te
                    plt.subplot(211)
                    plt.title(r'Training performance')
                    line_tr, = plt.plot(xdata, ydata1, 'g', \
                                        label=r'Training Error')
                    line_te, = plt.plot(xdata, ydata2, 'b', \
                                        label=r'Testing Error')
                else:
                    plt.subplot(211)
                    plt.title(r'Training performance')
                    line_tr, = plt.plot(xdata, ydata1, 'g', \
                                        label=r'Training Error')
                plt.legend(loc=1)
                plt.ylabel(r'Error')
                plt.ylim(0, 1)
                plt.grid(True)
                plt.subplot(212)
                plt.xlabel(r'Number of iterations')
                plt.ylabel(r'log($\textbf{y}$)')
                line_val, = plt.plot(xdata, vals, 'r', \
                                     label=r'Log marginal likelihood')
                plt.legend(loc=4)
                plt.ylim(np.min(vals), np.max(vals))
                plt.grid(True)

        # perform stochastic gradient descent (max(func))
        t0 = time.time()
        while t < self.max_iter and linalg.norm(grad) > self.tolerance:
            t = t + 1
            if self.verbose > 0:
                print('{} iterations:'.format(t))
            self.update(z, indices, val, grad, t)
            # sample z
            z = self.sample_z()
            # sample X
            indices = self.sample_x(int(self.alpha*self.xTr.shape[0]))
            grad = self.gradient(z, indices)
            val = self.objective(z, indices)
            # for debugging
            if self.verbose > 0:
                if self.verbose > 1:
                    interval = 50
                    if self.verbose > 2:
                        interval = 1
                    if t % interval == 0 or t == self.max_iter:
                        xdata.append(t)
                        error_tr = np.sum(len(np.where(self.predict(self.xTr) \
                                   != self.yTr)[0])) / float(self.xTr.shape[0])
                        ydata1.append(error_tr)
                        if error_tr < error_tr_min:
                            error_tr_min = error_tr
                        line_tr.set_data(xdata, ydata1)
                        if is_optimizing_on_te:
                            error_te = np.sum(len(np.where(self.predict(self.xTe) \
                                       != self.yTe)[0])) / float(self.xTe.shape[0])
                            ydata2.append(error_te)
                            if error_te < error_te_min:
                                error_te_min = error_te
                                self.parameters_best = self.parameters
                            line_te.set_data(xdata, ydata2)
                        plt.subplot(211)
                        plt.xlim(xmax=np.max(xdata))
                        plt.draw()
                        vals.append(val)
                        line_val.set_data(xdata, vals)
                        plt.subplot(212)
                        plt.xlim(xmax=np.max(xdata))
                        plt.ylim(np.min(vals), np.max(vals))
                        plt.draw()
                if self.verbose > 1:
                    if is_optimizing_on_te:
                        print('tr_error = {} \nte_error = {}'\
                              .format(ydata1[-1], ydata2[-1]))
                    else:
                        print('tr_error = {}'.format(ydata1[-1]))
            if self.verbose > 0:
                print('val={:.16f}, time = {:.5f}s'.format(val, time.time()-t0))
        if is_optimizing_on_te and self.verbose > 2:
            self.parameters = self.parameters_best
            if self.verbose > 2:
                print('Best Training Error = {} \nBest Testing  Error = {}'\
                      .format(error_tr_min, error_te_min))
        