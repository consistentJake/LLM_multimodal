import torch

class Config:
    def __init__(self):
        # Data paths
        self.input_folder = "/Users/zhenkai/Documents/personal/2024/LLM_RecSys/LLM_meetup/models/inputs"
        self.model_saved_path = "/Users/zhenkai/Documents/personal/Projects/Paper/LLM_multimodal/saved_models"
        # Model parameters
        self.projection_dim = 128
        self.sparse_dim_before_embedding = 180
        self.sparse_dim_after_embedding = 96
        self.bottom_mlp_dims = (64, 16)
        self.top_mlp_dims = (256, 16)
        self.dropout_prob = 0.5
        
        # Training parameters
        self.epochs = 1000
        self.batch_size = 1024
        self.learning_rate = 3e-4
        self.temperature = 0.1
        self.val_interval = 3
        self.early_stopping_patience = 100
        
        # Cross-validation
        self.fold_id = 2  # 0-4
        
        # Weight calculation
        self.is_sqrt_weight = True
        
        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Logging
        self.log_file = "training_log.txt"
        
    def get_model_params(self):
        return {
            "sparse_feature_num": 1,
            "dense_feature_number": None,  # Update based on actual data
            "emb_feature_dim": 64,
            "sparse_dim_before_embedding": self.sparse_dim_before_embedding,
            "sparse_dim_after_embedding": self.sparse_dim_after_embedding,
            "bottom_mlp_dims": self.bottom_mlp_dims,
            "top_mlp_dims": self.top_mlp_dims,
            "emb_feature_dim_before_project": None,  # Update based on actual data
            "dropout_prob": self.dropout_prob
        }
    
    def get_model_params(self, dense_feature_number, emb_feature_dim_before_project):
        return {
            "sparse_feature_num": 1,
            "dense_feature_number": dense_feature_number,  # Update based on actual data
            "emb_feature_dim": 64,
            "sparse_dim_before_embedding": self.sparse_dim_before_embedding,
            "sparse_dim_after_embedding": self.sparse_dim_after_embedding,
            "bottom_mlp_dims": self.bottom_mlp_dims,
            "top_mlp_dims": self.top_mlp_dims,
            "emb_feature_dim_before_project": emb_feature_dim_before_project,  # Update based on actual data
            "dropout_prob": self.dropout_prob
        }
        
    def get_training_params(self):
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "temperature": self.temperature,
            "val_interval": self.val_interval,
            "early_stopping_patience": self.early_stopping_patience,
            "fold_id": self.fold_id,
            "IS_SQRT_WEIGHT": self.is_sqrt_weight,
            "device": self.device
        }
