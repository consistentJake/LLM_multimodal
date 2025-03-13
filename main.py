import torch
from config import Config
from models import DLRM
from dataset import ContrastiveDataset
from train import train_contrastive_model
from torch.utils.data import DataLoader
import pickle
import numpy as np
import os
import itertools
from swing_similarity_data import SwingSimilarityData, SwingSimilarityData2
import utils
# from swing import get_user_item_relationship, calculate_swing_items, calculate_swing_users

def load_data(config):
    # Load features and labels
    with open(f"{config.input_folder}/features.pickle", 'rb') as file:
        features_data = pickle.load(file)
    
    with open(f"{config.input_folder}/5foldsfeatures.pickle", 'rb') as file:
        folds_dict = pickle.load(file)
    
    # Get fold data
    temp_dict = folds_dict[config.fold_id]
    picked_train2originidx = temp_dict['train2originidx']
    picked_val2originidx = temp_dict['val2originidx']
    
    # Load embeddings

    with open(f"{config.input_folder}/user_vectors.npy", 'rb') as file:
        user_data = np.load(file)
    with open(f"{config.input_folder}/bus_vectors.npy", 'rb') as file:
        business_data = np.load(file)

    with open(f"{config.input_folder}/imgtxtemb_avg_pooling_vectors_train_160719.pickle", 'rb') as f:
        image_embs_dict = pickle.load(f)
    imgtxtemb_avg_pooling_vectors_all = image_embs_dict['imgtxtemb_avg_pooling_vectors']

    if config.is_contrastive_loss:
        # eventually we get a list of n element, each is (3, 384) tensor, n is the number of samples, user_emb, business_emb, image_emb are all 384 dimension
        origin_emb_features_all = []
        for i in range(len(business_data)):
            one_line = [user_data[i], business_data[i], imgtxtemb_avg_pooling_vectors_all[i]]
            origin_emb_features_all.append(torch.stack([torch.tensor(e) for e in one_line]))

        # if we use following stack, the list will become a tensor of shape (n, 3, 384)    
        # origin_emb_features_all = torch.stack(origin_emb_features_all)
        
    else:
        # TODO: fix here, we did not include image features in the pickle file
        origin_emb_features_all = torch.tensor(np.load(f"{config.input_folder}/llm_reduced_vec.npy"))

    # print("type of user_data is ", type(user_data[0]))
    print(f"origin_emb_features_all shape is {origin_emb_features_all[0].shape}")

    # map from user id to user embedding
    user2emb = {}
    for i, uf_list in enumerate(features_data['uf']):
        user_id = uf_list[0]
        user2emb[user_id] = torch.tensor(user_data[i])
    # map from business id to business embedding
    business2emb = {}
    for i, bf_list in enumerate(features_data['bf']):
        business_id = bf_list[0]
        business2emb[business_id] = torch.tensor(business_data[i])


    labels_all = torch.tensor([1 if x >= 4.0 else 0 for x in features_data['pd'][:len(features_data['bf'])]], dtype=torch.float)
    business_ids_all = [b[0] for b in features_data['bf']]
    user_id_all = [u[0] for u in features_data['uf']]

    # # load similarity data
    # with open(f"{config.input_folder}/similarity_data.pickle", "rb") as f:
    #     similarity_data = pickle.load(f)
    # # b_idx_to_id = similarity_data['b_idx_to_id']
    # id_to_b_idx = similarity_data['id_to_b_idx']
    # # u_idx_to_id = similarity_data['u_idx_to_id']
    # id_to_u_idx = similarity_data['id_to_u_idx']
    # top5_similar_items = similarity_data['top5_similar_items']
    # bottom5_similar_items = similarity_data['bottom5_similar_items']
    # top5_similar_users = similarity_data['top5_similar_users']
    # bottom5_similar_users = similarity_data['bottom5_similar_users']

    # load similarity data simple 
    with open(f"{config.input_folder}/similarity_data_simple.pickle", "rb") as f:
        similarity_data = pickle.load(f)
    b_idx_to_id = similarity_data['b_idx_to_id']
    id_to_b_idx = similarity_data['id_to_b_idx']
    u_idx_to_id = similarity_data['u_idx_to_id']
    id_to_u_idx = similarity_data['id_to_u_idx']
    sorted_similar_items_map = similarity_data['sorted_similar_items_map']
    sorted_similar_users_map = similarity_data['sorted_similar_users_map']
    total_num_distinct_business = len(b_idx_to_id)
    total_num_distinct_users = len(u_idx_to_id)
    # get the mappings: int based index -> user/business  embedding 
    b_idx2b_emb = {}
    for i, b_id in enumerate(business_ids_all):
        b_idx = id_to_b_idx[b_id]
        b_emb = business_data[i]
        b_idx2b_emb[b_idx] = torch.tensor(b_emb)
    
    u_idx2u_emb = {}
    for i, u_id in enumerate(user_id_all):
        u_idx = id_to_u_idx[u_id]
        u_emb = user_data[i]
        u_idx2u_emb[u_idx] = torch.tensor(u_emb)

    # Prepare datasets
    train_labels = torch.stack([labels_all[i] for i in picked_train2originidx])
    train_emb_features = torch.stack([origin_emb_features_all[i] for i in picked_train2originidx])
    print(f"train_emb_features shape is {train_emb_features.shape}")
    train_b_ids = [business_ids_all[i] for i in picked_train2originidx]
    train_u_ids = [user_id_all[i] for i in picked_train2originidx]
    
    val_labels = torch.stack([labels_all[i] for i in picked_val2originidx])
    val_emb_features = torch.stack([origin_emb_features_all[i] for i in picked_val2originidx])
    val_b_ids = [business_ids_all[i] for i in picked_val2originidx]
    val_u_ids = [user_id_all[i] for i in picked_val2originidx]


    # convert to int based index instead of using the original string id
    train_b_idxs = [id_to_b_idx[b] for b in train_b_ids]
    train_u_idxs = [id_to_u_idx[u] for u in train_u_ids]
    val_b_idxs = [id_to_b_idx[b] for b in val_b_ids]
    val_u_idxs = [id_to_u_idx[u] for u in val_u_ids]

    

    
    # Load sparse and dense features
    sparse_features_all = torch.tensor(np.load(f"{config.input_folder}/model_sparse.npy")).long()
    dense_features_all = torch.tensor(np.load(f"{config.input_folder}/model_dense.npy"))
    
    train_sparse = torch.stack([sparse_features_all[i] for i in picked_train2originidx])
    train_dense = torch.stack([dense_features_all[i] for i in picked_train2originidx])
    
    val_sparse = torch.stack([sparse_features_all[i] for i in picked_val2originidx])
    val_dense = torch.stack([dense_features_all[i] for i in picked_val2originidx])

    # # load similarity matrix
    # user2Business, business2User = get_user_item_relationship(config)
    # user_similarity_matrix = calculate_swing_users(user2Business, business2User)
    # business_similarity_matrix = calculate_swing_items(user2Business, business2User)

    # change this into model
    # swing_similarity_data = SwingSimilarityData(train_b_idxs, train_u_idxs, val_b_idxs, val_u_idxs,
    #     top5_similar_items, bottom5_similar_items, top5_similar_users, bottom5_similar_users, b_idx2b_emb, u_idx2u_emb)
    
    swing_similarity_data = SwingSimilarityData2(train_b_idxs, train_u_idxs, val_b_idxs, val_u_idxs,
        sorted_similar_items_map, sorted_similar_users_map, total_num_distinct_business, total_num_distinct_users,
        b_idx2b_emb, u_idx2u_emb)

    print("finished loading data")

    return (train_sparse, train_dense, train_emb_features, train_labels,
            val_sparse, val_dense, val_emb_features, val_labels, 
            swing_similarity_data)



def main():
    # Initialize overall configuration
    config = Config()

    # customerize the config if needed
    # config.epochs = 1000

    start_time = utils.get_formatted_time()
    print(f"Start time: {start_time}")
    
    # Load data
    (train_sparse, train_dense, train_emb, train_labels,
     val_sparse, val_dense, val_emb, val_labels, swing_similarity_data) = load_data(config)
    
    # Create datasets
    train_dataset = ContrastiveDataset(train_sparse, train_dense, train_emb, train_labels, swing_similarity_data.train_business_ids, swing_similarity_data.train_user_ids)
    val_dataset = ContrastiveDataset(val_sparse, val_dense, val_emb, val_labels, swing_similarity_data.val_business_ids, swing_similarity_data.val_user_ids)
    
    print(f"train dataset length is {train_dataset.number_samples}, val dataset length is {val_dataset.number_samples}")

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False
    )
    
    # Get model parameters
    model_params = config.get_model_params(train_dense.shape[1], train_emb.shape[1])
    
    # Initialize model
    model = DLRM(**model_params).to(config.device)
    
    # Get training parameters
    train_params = config.get_training_params()
    
    # Train model
    trained_model, train_history = train_contrastive_model(model, train_labels, train_dataloader,
        val_dataloader,  train_params, swing_similarity_data)
    
    end_time = utils.get_formatted_time()

    save_label = f"{config.fold_id}_{start_time}_{end_time}"

    # Save final model
    torch.save(trained_model.state_dict(), os.path.join(config.model_saved_path,save_label + "_final_model.pth"))
    with open(f"{config.log_folder}/{save_label}_train_history.txt", "w") as f:
        f.write("\n".join(train_history))

if __name__ == "__main__":
    main()


# checkbox:
# review the model variables 
# review if train and validation steps are correct
# better seperating the config for model creation and training
# if the train embedding features are build up correctly
# add a try catch is failed 
