# from collections import defaultdict
# from itertools import combinations
# import os, pickle
# import numpy as np

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import sortedcontainers
# import pandas as pd
# from datetime import datetime

# def get_user_item_relationship(config):
#     input_folder = config.input_folder
#     with open(os.path.join(input_folder, '5foldsfeatures.pickle'), 'rb') as file:
#         folds_dict = pickle.load(file)
#     fold_id = config.fold_id
#     temp_dict = folds_dict[fold_id]
#     bf, uf, _ = temp_dict['train_bf'], temp_dict['train_uf'], temp_dict['train_pd']

#     num_reviews = len(bf)
#     user2Business = {}
#     business2User = {}
#     for i in range(num_reviews):
#         user_id, business_id = uf[i][0], bf[i][0]
#         if user_id not in user2Business:
#             user2Business[user_id] = []
#         user2Business[user_id].append(business_id)

#     for i in range(num_reviews):
#         user_id, business_id = uf[i][0], bf[i][0]
#         if business_id not in business2User:
#             business2User[business_id] = []
#         business2User[business_id].append(user_id)    
        
        
#     return user2Business, business2User

# # user_item_dict with key being user id and value being list of business ids visited by the user
# # item_user_dict with key being business id and value being the list of users who have visited the business being considered.
# # alpha (temp) is a tuning params

# # calculate swing similarity between business.
# def calculate_swing_items(user_item_dict, item_user_dict, alpha=1):
#     similarity_matrix = defaultdict(lambda: defaultdict(float))

#     for item_a, users_a in item_user_dict.items():
#         for item_b, users_b in item_user_dict.items():
#             if item_a != item_b:
#                 common_users = set(users_a).intersection(set(users_b))

#                 similarity = 0
#                 for u, v in combinations(common_users, 2):
#                     overlap = len(set(user_item_dict[u]).intersection(set(user_item_dict[v])))
#                     similarity += 1 / (alpha + overlap)

#                 similarity_matrix[item_a][item_b] = similarity
#                 similarity_matrix[item_b][item_a] = similarity

#     for item in similarity_matrix:
#       similarity_matrix[item] = sortedcontainers.SortedDict(similarity_matrix[item])


#     return similarity_matrix


# def calculate_swing_users(user_item_dict, item_user_dict, alpha=1):
#   similarity_matrix = defaultdict(lambda: defaultdict(float))

#   for ui, item_i in user_item_dict.items():
#     for uj, item_j in user_item_dict.items():
#       if ui != uj:
#         common_items = set(item_i).intersection(set(item_j))

#         similarity = 0
#         for i, j in combinations(common_items, 2):
#           overlap = len(set(item_user_dict[i]).intersection(item_user_dict[j]))
#           similarity += 1 / (alpha + overlap)

#         similarity_matrix[ui][uj] = similarity
#         similarity_matrix[uj][ui] = similarity

#   for user in similarity_matrix:
#     similarity_matrix[user] = sortedcontainers.SortedDict(similarity_matrix[user])

#   return similarity_matrix

# ## example usage 
# # item_sim = calculate_swing_similarity(user2Business, business2User, 1)
