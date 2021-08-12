# -*- coding: utf-8 -*-
import time
import numpy as np


class CheckinModule(object):
    def __init__(self, num_factors, reg_lambda=0.6, gamma_lmf=1.0, iters=30):
        self.num_factors = num_factors
        self.iters = iters
        self.reg_lambda = reg_lambda
        self.gamma_lmf = gamma_lmf

    def save_result(self, path):
        ctime = time.time()
        print("Saving CheckinModule result...")
        np.save(path + "user_vectors", self.user_vectors)
        np.save(path + "user_biases", self.user_biases)
        np.save(path + "poi_vectors", self.poi_vectors)
        np.save(path + "poi_biases", self.poi_biases)

        print("Done. Elapsed time:", time.time() - ctime, "s")

    def load_result(self, path):
        ctime = time.time()
        print("Loading CheckinModule result...")
        self.user_vectors = np.load(path + "user_vectors.npy")
        self.user_biases = np.load(path + "user_biases.npy")
        self.poi_vectors = np.load(path + "poi_vectors.npy")
        self.poi_biases = np.load(path + "poi_biases.npy")
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def train(self, check_in_matrix):
        self.check_in_matrix = check_in_matrix
        self.num_users, self.num_pois = self.check_in_matrix.shape[0], self.check_in_matrix.shape[1]
        self.ones = np.ones((self.num_users, self.num_pois))
        self.user_vectors = np.random.normal(size=(self.num_users, self.num_factors))
        self.poi_vectors = np.random.normal(size=(self.num_pois, self.num_factors))
        self.user_biases = np.random.normal(size=(self.num_users, 1))
        self.poi_biases = np.random.normal(size=(self.num_pois, 1))

        user_vec_deriv_sum = np.zeros((self.num_users, self.num_factors))
        poi_vec_deriv_sum = np.zeros((self.num_pois, self.num_factors))
        user_bias_deriv_sum = np.zeros((self.num_users, 1))
        poi_bias_deriv_sum = np.zeros((self.num_pois, 1))

        for i in range(self.iters):
            ctime = time.time()
            user_vec_deriv, user_bias_deriv = self.deriv(True)
            user_vec_deriv_sum += np.square(user_vec_deriv)
            user_bias_deriv_sum += np.square(user_bias_deriv)
            vec_step_size = self.gamma_lmf / np.sqrt(user_vec_deriv_sum)
            bias_step_size = self.gamma_lmf / np.sqrt(user_bias_deriv_sum)
            self.user_vectors += vec_step_size * user_vec_deriv
            self.user_biases += bias_step_size * user_bias_deriv

            poi_vec_deriv, poi_bias_deriv = self.deriv(False)
            poi_vec_deriv_sum += np.square(poi_vec_deriv)
            poi_bias_deriv_sum += np.square(poi_bias_deriv)
            vec_step_size = self.gamma_lmf / np.sqrt(poi_vec_deriv_sum)
            bias_step_size = self.gamma_lmf / np.sqrt(poi_bias_deriv_sum)
            self.poi_vectors += vec_step_size * poi_vec_deriv
            self.poi_biases += bias_step_size * poi_bias_deriv
            print('iteration %i finished in %f seconds' % (i + 1, time.time() - ctime))

    def deriv(self, user):
        if user:
            vec_deriv = np.dot(self.check_in_matrix, self.poi_vectors)
            bias_deriv = np.expand_dims(np.sum(self.check_in_matrix, axis=1), 1)
        else:
            vec_deriv = np.dot(self.check_in_matrix.T, self.user_vectors)
            bias_deriv = np.expand_dims(np.sum(self.check_in_matrix, axis=0), 1)

        A = np.dot(self.user_vectors, self.poi_vectors.T)
        A += self.user_biases
        A += self.poi_biases.T
        A = np.exp(A)
        A /= (A + self.ones)
        A = (self.check_in_matrix + self.ones) * A

        if user:
            vec_deriv -= np.dot(A, self.poi_vectors)
            bias_deriv -= np.expand_dims(np.sum(A, axis=1), 1)
            vec_deriv -= self.reg_lambda * self.user_vectors
        else:
            vec_deriv -= np.dot(A.T, self.user_vectors)
            bias_deriv -= np.expand_dims(np.sum(A, axis=0), 1)
            vec_deriv -= self.reg_lambda * self.poi_vectors
        return (vec_deriv, bias_deriv)

    def predict(self, uid, pid):
        A = (self.user_vectors[uid].dot(self.poi_vectors[pid])) + self.user_biases[uid] + self.poi_biases[pid]
        A = np.exp(A)
        pro = A / (1 + A)
        return pro
