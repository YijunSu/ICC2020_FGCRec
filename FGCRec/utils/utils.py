# -*- coding: utf-8 -*-
import math
import numpy as np


def euclidean_dist(poi1, poi2):
    if isinstance(poi1, tuple) and isinstance(poi2, tuple):
        lat1, lng1 = poi1[0], poi1[1]
        lat2, lng2 = poi2[0], poi2[1]
    else:
        lat1, lng1 = poi1.lat, poi1.lng
        lat2, lng2 = poi2.lat, poi2.lng

    if abs(lat1 - lat2) < 1e-6 and abs(lng1 - lng2) < 1e-6:
        return 0.0
    return (lat1 - lat2)**2 + (lng1 - lng2)**2

def gaussian_fun(uid_distance):
    distance = np.sum(math.exp(-1 / 2 * d) for d in uid_distance)
    return distance


def haversine_dist(poi1, poi2):
    if isinstance(poi1, tuple) and isinstance(poi2, tuple):
        lat1, lng1 = poi1[0], poi1[1]
        lat2, lng2 = poi2[0], poi2[1]
    else:
        lat1, lng1 = poi1.lat, poi1.lng
        lat2, lng2 = poi2.lat, poi2.lng

    if abs(lat1 - lat2) < 1e-6 and abs(lng1 - lng2) < 1e-6:
        return 0.0
    degrees_to_radians = math.pi / 180.0
    phi1 = (90.0 - lat1) * degrees_to_radians
    phi2 = (90.0 - lat2) * degrees_to_radians
    theta1 = lng1 * degrees_to_radians
    theta2 = lng2 * degrees_to_radians

    cos = (math.sin(phi1) * math.sin(phi2) * math.cos(theta1 - theta2) +
           math.cos(phi1) * math.cos(phi2))
    arc = math.acos(cos)
    earth_radius = 6371
    return arc * earth_radius