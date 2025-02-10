import torch
from config import Config
from models import DLRM
from dataset import ContrastiveDataset
from train import train_contrastive_model
from torch.utils.data import DataLoader
import pickle
import numpy as np

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
    origin_emb_features_all = torch.tensor(np.load(f"{config.input_folder}/llm_reduced_vec.npy"))
    labels_all = torch.tensor([1 if x >= 4.0 else 0 for x in features_data['pd'][:len(features_data['bf'])]], dtype=torch.float)
    
    # Prepare datasets
    train_labels = torch.stack([labels_all[i] for i in picked_train2originidx])
    train_emb_features = torch.stack([origin_emb_features_all[i] for i in picked_train2originidx])
    
    val_labels = torch.stack([labels_all[i] for i in picked_val2originidx])
    val_emb_features = torch.stack([origin_emb_features_all[i] for i in picked_val2originidx])
    
    # Load sparse and dense features
    sparse_features_all = torch.tensor(np.load(f"{config.input_folder}/model_sparse.npy")).long()
    dense_features_all = torch.tensor(np.load(f"{config.input_folder}/model_dense.npy"))
    
    train_sparse = torch.stack([sparse_features_all[i] for i in picked_train2originidx])
    train_dense = torch.stack([dense_features_all[i] for i in picked_train2originidx])
    
    val_sparse = torch.stack([sparse_features_all[i] for i in picked_val2originidx])
    val_dense = torch.stack([dense_features_all[i] for i in picked_val2originidx])
    
    return (train_sparse, train_dense, train_emb_features, train_labels,
            val_sparse, val_dense, val_emb_features, val_labels)

def main():
    # Initialize overall configuration
    config = Config()

    # customerize the config if needed
    # config.epochs = 1000
    
    # Load data
    (train_sparse, train_dense, train_emb, train_labels,
     val_sparse, val_dense, val_emb, val_labels) = load_data(config)
    
    # Create datasets
    train_dataset = ContrastiveDataset(train_sparse, train_dense, train_emb, train_labels)
    val_dataset = ContrastiveDataset(val_sparse, val_dense, val_emb, val_labels)
    
    print(f"train dataset length is {train_sparse.shape[1]}, val dataset length is {val_sparse.shape[1]}")}")

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
    trained_model = train_contrastive_model(model, train_labels, train_dataloader,
        val_dataloader,  train_params
    )
    
    # Save final model
    torch.save(trained_model.state_dict(), os.path.join(config.model_saved_path,"final_model.pth"))

if __name__ == "__main__":
    main()


# checkbox:
# review the model variables 
# review if train and validation steps are correct
# better seperating the config for model creation and training
# if the train embedding features are build up correctly
# add a try catch is failed 
