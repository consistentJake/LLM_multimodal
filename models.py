import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveNet(nn.Module):
    def __init__(self, input_dim, projection_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim),
            nn.BatchNorm1d(projection_dim)  # Added BatchNorm for better stability
        )

    def forward(self, x):
        # print(f"Input to projection: {x.shape}")
        return self.projection(x)

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        # Initialize learnable weights as a single tensor
        self.weights = nn.Parameter(torch.ones(3))  # One weight for each loss component
        
        # Ensure weights are positive using softplus
        self.softplus = nn.Softplus()

    def forward(self, bce_loss, contrastive_loss1, contrastive_loss2):
        # Apply softplus to ensure weights are positive
        w_bce, w_contrastive1, w_contrastive2 = self.softplus(self.weights)
        
        # Compute the weighted sum of the losses
        total_loss = w_bce * bce_loss + w_contrastive1 * contrastive_loss1 + w_contrastive2 * contrastive_loss2
        
        return total_loss


def compute_contrastive_loss1(user_ids_in_batch, user_relationship_map, model):

    return 0


def compute_contrastive_loss2(business_ids_in_batch, business_relationship_map, model):

    return 0



class InfoNCELoss(nn.Module):
    def __init__(self, temperature, weight=None):
        super().__init__()
        self.temperature = temperature
        self.weight = weight

    def forward(self, anchors, positives, labels):
        """
        anchors: tensor of shape [batch_size, feature_dim]
        positives: tensor of shape [batch_size, feature_dim]
        labels: tensor of shape [batch_size]
        """
        # Normalize embeddings
        anchors = nn.functional.normalize(anchors, dim=1)
        positives = nn.functional.normalize(positives, dim=1)

        # Create the complete set of embeddings [2*batch_size, feature_dim]
        all_embeddings = torch.cat([anchors, positives], dim=0)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(anchors, all_embeddings.T) / self.temperature

        # Create labels matrix for positive pairs
        batch_size = anchors.shape[0]
        # print("labels ", labels.shape)
        labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)

        # Extend labels matrix to account for positives in second half of batch
        labels_matrix = torch.cat([
            labels_matrix,
            labels_matrix
        ], dim=1)

        # Remove self-similarities from positive pairs
        labels_matrix = labels_matrix.fill_diagonal_(False)

        # InfoNCE loss computation
        exp_similarities = torch.exp(similarity_matrix)

        # # Mask out self-similarities
        # mask = torch.eye(batch_size, dtype=torch.bool, device=anchors.device)
        # exp_similarities = exp_similarities.masked_fill(mask, 0)

        # Mask out self-similarities for anchors
        batch_size = anchors.shape[0]
        mask = torch.eye(batch_size, dtype=torch.bool, device=anchors.device)
        full_mask = torch.cat([mask, mask], dim=1)  # Extend mask for positives
        exp_similarities = exp_similarities.masked_fill(full_mask, 0)

        # Compute positive similarities
        positive_similarities = torch.sum(
            exp_similarities * labels_matrix.float(), dim=1
        )

        # Compute denominator (sum of all exp similarities)
        denominator = torch.sum(exp_similarities, dim=1)

        # print(f"Positive similarities: {positive_similarities}")
        # print(f"Denominator: {denominator}")
        # if these two variables contain 0 or nan, can cause losses become nan

        # Compute final loss
        # If positive_similarities or denominator contains small or zero values, add a small epsilon to prevent division by zero or log(0):
        epsilon = 1e-8
        losses = -torch.log((positive_similarities + epsilon) / (denominator + epsilon))

        return losses.mean()

class FeatureInteraction(nn.Module):
    def __init__(self):
        super().__init__()
        #self.self_interaction = self_interaction

    def forward(self, inputs):
        feature_dim = inputs.shape[1]

        concat_features = inputs.view(-1, feature_dim, 1) # B , (dense_dim + emb_dim),1
        #print(f"before matmul is {concat_features.shape}")
        dot_products = torch.matmul(concat_features, concat_features.transpose(1, 2)) # B, (dense_dim + emb_dim), ((dense_dim + emb_dim)
        ones = torch.ones_like(dot_products)

        mask = torch.triu(ones)
        out_dim = feature_dim * (feature_dim + 1) // 2 # dim calc

        flat_result = dot_products[mask.bool()] # (B * output_dim) flatten result for the entire batch
        reshape_result = flat_result.view(-1, out_dim) # (B, output_dim) return result

        return reshape_result

class DLRM(torch.nn.Module):

    def __init__(self,sparse_feature_num, dense_feature_number, emb_feature_dim, sparse_dim_before_embedding, sparse_dim_after_embedding, bottom_mlp_dims, top_mlp_dims, emb_feature_dim_before_project, dropout_prob):
        super(DLRM, self).__init__()
        if sparse_dim_before_embedding > 0:
            self.embedding = torch.nn.Embedding(sparse_dim_before_embedding, sparse_dim_after_embedding, padding_idx=179)
        self.layer_feature_interaction = FeatureInteraction()
        self.projection_model = ContrastiveNet(emb_feature_dim_before_project, emb_feature_dim)  # ContrastiveNet instance

        self.bottom_mlp = torch.nn.Sequential(
            torch.nn.Linear(dense_feature_number, bottom_mlp_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(bottom_mlp_dims[0], bottom_mlp_dims[1]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_prob)
        )
        self.all_dim = sparse_feature_num * sparse_dim_after_embedding + bottom_mlp_dims[1] + emb_feature_dim
        self.output_dim = ((self.all_dim * (self.all_dim + 1)) // 2) + bottom_mlp_dims[1]
        self.top_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.output_dim, top_mlp_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(top_mlp_dims[0], top_mlp_dims[1]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(top_mlp_dims[1], 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x_sparse, x_dense, x_embed_before_projection):
        # is_sparse_features_included = not IS_BASELINE and x_sparse.shape[0] > 0
        # if is_sparse_features_included:
        embed_x = self.embedding(x_sparse)
        embed_x = embed_x.sum(dim = 1) # (B, N, D) -> (B, D) N is the length after padding 11 in our case. 
        #embed_x = embed_x.view(x_sparse.shape[0], -1) # concat all embedding of sparse features together

        x_embed = self.projection_model(x_embed_before_projection)
        bottom_mlp_output = self.bottom_mlp(x_dense)
        
        # TODO change name of variables
        concat_first = torch.concat([bottom_mlp_output, x_embed, embed_x], dim = -1)   
        
        #print(f"bottom_mlp_output is shape {bottom_mlp_output.shape}, input to interaction arch is shape {concat_first.shape}")

        interaction = self.layer_feature_interaction(concat_first) #interaction all features

        concat_second = torch.concat([interaction, bottom_mlp_output], dim=-1) # concat with the output of bottom mlp

        output = self.top_mlp(concat_second)

        output = output.squeeze(-1)  # Ensure the output shape is [batch_size]

        output = output.squeeze().unsqueeze(1)

        return output, x_embed