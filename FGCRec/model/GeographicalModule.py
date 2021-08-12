# -*- coding: utf-8 -*-
import time
import numpy as np
from utils.utils import euclidean_dist
from utils.utils import gaussian_fun


class GeographicalModule(object):
    def __init__(self, alpha = 0.3):
        self.upsilon = None
        self.user_normal_dist = None
        self.phi = None
        self.poi_normal_dist = None

        self.alpha = alpha
        self.check_in_matrix = None
        self.poi_coos = None
        
    def save_result(self, path):
        ctime = time.time()   
        print("Saving GeographicalModule result...")
        print("1.Saving UserModel result...")
        np.save(path + "upsilon", self.upsilon)
        np.save(path + "user_normal_dist", self.user_normal_dist)

        print("2.Saving POIModel result...")
        np.save(path + "phi", self.phi)
        np.save(path + "poi_normal_dist", self.poi_normal_dist)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def load_result(self, path):
        ctime = time.time()
        print("Loading GeographicalModule result...")
        print("1.Loading UserModel result...")
        self.upsilon = np.load(path + "upsilon.npy")
        self.user_normal_dist = np.load(path + "user_normal_dist.npy")

        print("2.Loading POIModel result...")
        self.phi = np.load(path + "phi.npy")
        self.poi_normal_dist = np.load(path + "poi_normal_dist.npy")
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def UserModel(self, check_in_matrix, poi_coos, user_centers):
        ctime = time.time()
        print("Training UserModel...")
        normal_dist = []
        upsilon = []
        for uid in range(check_in_matrix.shape[0]):
            uid_pids = check_in_matrix[uid, :].nonzero()[0]
            tohome_dist = []
            pid_normal_dist = []
            for i in uid_pids:
                h_dist = np.mean([euclidean_dist(poi_coos[i], poi_coos[cid]) for cid in user_centers[uid]])
                tohome_dist.append(h_dist)
            uid_activity_dist = np.max(tohome_dist)

            for i in uid_pids:
                for cid in user_centers[uid]:
                    h_dist = uid_activity_dist * euclidean_dist(poi_coos[i], poi_coos[cid])
                    pid_normal_dist.append(h_dist)
            
            upsilon.append(uid_activity_dist)
            normal_dist.append(gaussian_fun(pid_normal_dist))
        self.upsilon = np.array(upsilon)
        self.user_normal_dist = np.array(normal_dist)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def POIModel(self, check_in_matrix, poi_coos):
        self.poi_coos = poi_coos
        self.check_in_matrix = check_in_matrix
        ctime = time.time()
        print("Training POIModel...", )
        normal_dist = []
        phi = []
        for uid in range(check_in_matrix.shape[0]):
            uid_pids = check_in_matrix[uid, :].nonzero()[0]
            visited_dist = []
            lid_normal_dist = []
            for i in range(len(uid_pids)):
                for j in range(i + 1, len(uid_pids)):
                    v_dist = euclidean_dist(poi_coos[uid_pids[i]], poi_coos[uid_pids[j]])
                    visited_dist.append(v_dist)
            uid_mean_visited_dist = np.mean(visited_dist)

            for i in uid_pids:
                for k in range(check_in_matrix.shape[1]):
                    if i != k:
                        v_dist = uid_mean_visited_dist * euclidean_dist(poi_coos[i], poi_coos[k])
                        lid_normal_dist.append(v_dist)

            phi.append(uid_mean_visited_dist)
            normal_dist.append(gaussian_fun(lid_normal_dist))
        self.phi = np.array(phi)
        self.poi_normal_dist = np.array(normal_dist)
        print("Done. Elapsed time:", time.time() - ctime, "s")

    def predict(self, uid, pid, check_in_matrix, poi_coos, user_centers):
        uid_pids = check_in_matrix[uid, :].nonzero()[0]
        user_centers_dist = []
        user_visited_dist = []
        for cid in user_centers[uid]:
            f_dist = self.upsilon[uid] * euclidean_dist(poi_coos[cid], poi_coos[pid])
            user_centers_dist.append(f_dist)
        pro_user = gaussian_fun(user_centers_dist) / self.user_normal_dist[uid]

        for j in uid_pids:
            f_dist = self.phi[uid] * euclidean_dist(poi_coos[j], poi_coos[pid])
            user_visited_dist.append(f_dist)
        pro_poi = gaussian_fun(user_visited_dist) / self.poi_normal_dist[uid]

        pro = self.alpha * pro_user + (1-self.alpha) * pro_poi
        return pro
