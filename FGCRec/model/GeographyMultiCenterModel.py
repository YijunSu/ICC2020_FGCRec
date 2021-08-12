# -*- coding: utf-8 -*-
from collections import defaultdict
from utils.utils import euclidean_dist


class POI(object):
    def __init__(self, pid, lat, lng, freq, center=-1):
        self.pid = pid
        self.lat = lat
        self.lng = lng
        self.freq = freq
        self.center = center


class Center(object):
    def __init__(self):
        self.pois = []
        self.total_freq = 0
        self.lat = None
        self.lng = None

    def add(self, poi):
        self.pois.append(poi)
        self.total_freq += poi.freq


class GeographyMultiCenterModel(object):
    def __init__(self, alpha=0.2, theta=0.02, dmax=50):
        self.alpha = alpha
        self.theta = theta
        self.dmax = dmax
        self.poi_coos = None
        self.center_list = None
        self.hist = None

    def build_user_check_in_profile(self, sparse_check_in_matrix):
        Hist = defaultdict(list)
        for (uid, pid), freq in sparse_check_in_matrix.items():
            lat, lng = self.poi_coos[pid]
            Hist[uid].append(POI(pid, lat, lng, freq))
        return Hist

    def discover_user_centers(self, hist_u):
        center_min_freq = max(sum([poi.freq for poi in hist_u]) * self.theta, 2)
        hist_u.sort(key=lambda k: k.freq, reverse=True)
        center_list = []
        center_num = 0
        for i in range(len(hist_u)):
            if hist_u[i].center == -1:
                center_num += 1
                center = Center()
                center.add(hist_u[i])
                hist_u[i].center = center_num
                for j in range(i + 1, len(hist_u)):
                    if hist_u[j].center == -1 and euclidean_dist(hist_u[i], hist_u[j]) <= self.dmax:
                        hist_u[j].center = center_num
                        center.add(hist_u[j])
                if center.total_freq >= center_min_freq:
                    center_list.append(center)
        return center_list

    def multi_center_discover(self, sparse_check_in_matrix, poi_coos):
        self.poi_coos = poi_coos
        self.hist = self.build_user_check_in_profile(sparse_check_in_matrix)
        center_list = {}
        u_centers = {}
        for uid in range(len(self.hist)):
            center_pids = []
            center_list[uid] = self.discover_user_centers(self.hist[uid])
            for it in center_list[uid]:
                center_pids.append(it.pois[0].pid)
            u_centers[uid] = center_pids
        return u_centers
