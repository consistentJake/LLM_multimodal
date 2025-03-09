import random


class SwingSimilarityData():
    def __init__(self, train_b_idxs, train_u_idxs, val_b_idxes, val_u_idxs, 
                 top5_similar_items,bottom5_similar_items, top5_similar_users, bottom5_similar_users,
                 b_idx2b_emb, u_idx2u_emb):
        ## given the data pre-processing logic, here the business_ids and users_ids are int based indexes
        self.train_business_ids = train_b_idxs
        self.train_user_ids = train_u_idxs
        self.val_business_ids = val_b_idxes
        self.val_user_ids= val_u_idxs
        self.top5_similar_items = top5_similar_items
        self.bottom5_similar_items = bottom5_similar_items
        self.top5_similar_users = top5_similar_users
        self.bottom5_similar_users = bottom5_similar_users
        self.b_idx2b_emb = b_idx2b_emb
        self.u_idx2u_emb = u_idx2u_emb

class SwingSimilarityData2():
    def __init__(self, train_b_idxs, train_u_idxs, val_b_idxes, val_u_idxs, 
                 sorted_similar_items_map, sorted_similar_users_map, num_items, num_users,
                 b_idx2b_emb, u_idx2u_emb):
        ## given the data pre-processing logic, here the business_ids and users_ids are int based indexes
        self.train_business_ids = train_b_idxs
        self.train_user_ids = train_u_idxs
        self.val_business_ids = val_b_idxes
        self.val_user_ids= val_u_idxs
        self.b_idx2b_emb = b_idx2b_emb
        self.u_idx2u_emb = u_idx2u_emb
        ## if there is at least one item in the sorted similar map for a key, meaning there is at least one similar item for the key
        ## therefore we take the first one of it as top similar
        ## for the bottom similar, we take random one from the rest of the list

        def find_idx_out_of_range(input_set, total_num):
            while True:
                idx = random.randint(0, total_num-1)
                if idx not in input_set:
                    return idx

        def find_top_bottom_similar(input_list, total_num):
            num_input = len(input_list)
            if num_input > 0:
                # if num_input > 20:
                #     print(num_input)
                return (input_list[0], find_idx_out_of_range(set(input_list), total_num))
            else:
                top_element = find_idx_out_of_range(set(), total_num)
                bottom_element = find_idx_out_of_range(set([top_element]), total_num)
                return (top_element, bottom_element)
            
        print("start to find top bottom similar items and users, num_items is {}, num_users is {}".format(num_items, num_users))
        self.top_bottom_similar_businesses = {}
        ## similar list of each business is a list of (business_id, similarity_score) pairs, same for users
        for b, similar_list in sorted_similar_items_map.items():
            self.top_bottom_similar_businesses[b] = find_top_bottom_similar([pair[0] for pair in similar_list], num_items)
        self.top_bottom_similar_users = {}
        for u, similar_list in sorted_similar_users_map.items():
            self.top_bottom_similar_users[u] = find_top_bottom_similar([pair[0] for pair in similar_list], num_users)
        print("finish finding top bottom similar items and users")