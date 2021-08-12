# -*- coding: utf-8 -*-
import scipy.sparse as sparse
import numpy as np
import time
from collections import defaultdict
from model.GeographyMultiCenterModel import GeographyMultiCenterModel
from model.GeographicalModule import GeographicalModule
from model.CheckinModule import CheckinModule
from metric.metrics import precisionk, recallk


def read_training_data():
    train_data = open(train_file, 'r').readlines()
    sparse_training_matrix = sparse.dok_matrix((num_users, num_pois))
    user_poi_matrix = np.zeros((num_users, num_pois))
    user_poi_weighted_matrix = np.zeros((num_users, num_pois))
    training_tuples = set()
    for eachline in train_data:
        uid, pid, freq = eachline.strip().split()
        uid, pid, freq = int(uid), int(pid), int(freq)
        sparse_training_matrix[uid, pid] = freq
        user_poi_matrix[uid, pid] = freq
        user_poi_weighted_matrix[uid, pid] = 1 + gamma * np.log(1 + freq)
        training_tuples.add((uid, pid))
    return sparse_training_matrix, training_tuples, user_poi_matrix, user_poi_weighted_matrix


def read_ground_truth():
    ground_truth = defaultdict(set)
    truth_data = open(test_file, 'r').readlines()
    for eachline in truth_data:
        uid, pid, _ = eachline.strip().split()
        uid, pid = int(uid), int(pid)
        ground_truth[uid].add(pid)
    return ground_truth


def read_poi_coos():
    poi_coos = {}
    poi_data = open(poi_file, 'r').readlines()
    for eachline in poi_data:
        pid, lat, lng = eachline.strip().split()
        pid, lat, lng = int(pid), float(lat), float(lng)
        poi_coos[pid] = (lat, lng)
    return poi_coos


def main():
    sparse_training_matrix, training_tuples, user_poi_matrix, user_poi_weighted_matrix = read_training_data()
    ground_truth = read_ground_truth()
    poi_coos = read_poi_coos()

    print("Start Training FGCRec....")
    start_time = time.time()
    user_centers = GMM.multi_center_discover(sparse_training_matrix, poi_coos)
    GM.UserModel(user_poi_matrix, poi_coos, user_centers)
    GM.POIModel(user_poi_matrix, poi_coos)
    GM.save_result("./tmp/")
    GM.load_result("./tmp/")
    CM.train(user_poi_weighted_matrix)
    CM.save_result("./tmp/")
    CM.load_result("./tmp/")
    elapsed_time = time.time() - start_time
    print("Training Done. Elapsed time:", elapsed_time, "s")

    result_10 = open("./result/result_top_" + str(10) + ".txt", 'w')
    result_20 = open("./result/result_top_" + str(20) + ".txt", 'w')
    all_uids = list(range(num_users))
    all_pids = list(range(num_pois))
    np.random.shuffle(all_uids)

    precision_10, recall_10, nDCG_10, MAP_10 = [], [], [], []
    precision_20, recall_20, nDCG_20, MAP_20 = [], [], [], []
    print("Start Predicting...")
    for cnt, uid in enumerate(all_uids):
        if uid in ground_truth:
            GM_scores = [GM.predict(uid, pid, user_poi_matrix, poi_coos, user_centers) * CM.predict(uid, pid)
                                   if (uid, pid) not in training_tuples else -1
                                   for pid in all_pids]

            overall_scores = np.array(GM_scores)
            predicted = list(reversed(overall_scores.argsort()))[:top_n]
            actual = ground_truth[uid]

            precision_10.append(precisionk(actual, predicted[:10]))
            recall_10.append(recallk(actual, predicted[:10]))
            precision_20.append(precisionk(actual, predicted[:20]))
            recall_20.append(recallk(actual, predicted[:20]))

            result_10.write('\t'.join([str(cnt), str(uid), str(np.mean(precision_10)), str(np.mean(recall_10))]) + '\n')
            result_20.write('\t'.join([str(cnt), str(uid), str(np.mean(precision_20)), str(np.mean(recall_20))]) + '\n')

    print("Task Finished!")


if __name__ == '__main__':
    data_dir = "../datasets/Foursquare/"
    datsize_file = data_dir + "Foursquare_data_size.txt"
    check_in_file = data_dir + "Foursquare_checkins.txt"
    train_file = data_dir + "Foursquare_train.txt"
    tune_file = data_dir + "Foursquare_tune.txt"
    test_file = data_dir + "Foursquare_test.txt"
    poi_file = data_dir + "Foursquare_poi_coos.txt"

    num_users, num_pois = open(datsize_file, 'r').readlines()[0].strip('\n').split()
    num_users, num_pois = int(num_users), int(num_pois)

    top_n = 50
    gamma = 50.0

    GMM = GeographyMultiCenterModel()
    GM = GeographicalModule(alpha=0.9)
    CM = CheckinModule(num_factors=10, reg_lambda=0.6, gamma_lmf=1.0, iters=30)

    main()
