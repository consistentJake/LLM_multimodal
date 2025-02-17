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
