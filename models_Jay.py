import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb 

epsilon = 1e-10  # Small value to avoid division by zero and log of zero


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


# Define a simple MLP for dimensionality reduction
# TODO finetune the MLP model for image vectors embedding dimension reduction 
class MLPDimReduction(nn.Module):
    def __init__(self, input_dim, hidden_dim = 64, output_dim = 32):
        super(MLPDimReduction, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # Output is the reduced-dimension representation
        return x

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

# users contraistive loss
def compute_contrastive_loss_user(user_ids_in_batch, swing_similarity_data, model):
    x_tensor = []
    x_pos_tensor = []
    x_neg_list_tensor = []
    for user_id in user_ids_in_batch:
        user_id = user_id.item()  # Convert to Python int
        user_emb = swing_similarity_data.u_idx2u_emb[user_id]
        # print(f"shape of swing_similarity_data.top5_similar_users[user_id] is {swing_similarity_data.top5_similar_users[user_id]}")
        # shape of swing_similarity_data.top5_similar_users[user_id] is [(1801, 0), (3907, 0), (14802, 0), (8975, 0), (14174, 0)]
        top_pos_user_id = swing_similarity_data.top5_similar_users[user_id][0][0] # 
        top_pos_user_emb = swing_similarity_data.u_idx2u_emb[top_pos_user_id]
        bottom5_pos_user_id = [pair[0] for pair in swing_similarity_data.bottom5_similar_users[user_id][:5]]
        bottom5_pos_user_emb = [swing_similarity_data.u_idx2u_emb[x] for x in bottom5_pos_user_id]

        # append to tensor list 
        # print("user_emb shape is ", user_emb.shape)
        x_tensor.append(model(user_emb.unsqueeze(0)))
        x_pos_tensor.append(model(top_pos_user_emb.unsqueeze(0)))
        x_neg_list_tensor.append([model(x.unsqueeze(0)) for x in bottom5_pos_user_emb])
    # convert to tensor
    x_tensor = torch.stack(x_tensor)
    x_pos_tensor = torch.stack(x_pos_tensor)
    x_neg_list_tensor = torch.stack([torch.stack(x_neg) for x_neg in x_neg_list_tensor])

    ## TODO, how to make it as a tensor and model eval in batch

    return contrastive_loss_helper(x_tensor, x_pos_tensor, x_neg_list_tensor)

# business contrastive loss
def compute_contrastive_loss_business(business_ids_in_batch, swing_similarity_data, model):

    x_tensor = []
    x_pos_tensor = []
    x_neg_list_tensor = []
    for business_id in business_ids_in_batch:
        business_id = business_id.item()  # Convert to Python int
        business_emb = swing_similarity_data.b_idx2b_emb[business_id]
        top_pos_business_id = swing_similarity_data.top5_similar_items[business_id][0][0]
        top_pos_business_emb = swing_similarity_data.b_idx2b_emb[top_pos_business_id]
        bottom5_pos_business_id = [pair[0] for pair in swing_similarity_data.bottom5_similar_items[business_id][:5]]
        bottom5_pos_business_emb = [swing_similarity_data.b_idx2b_emb[x] for x in bottom5_pos_business_id]

        # append to tensor list 
        x_tensor.append(model(business_emb.unsqueeze(0)))
        x_pos_tensor.append(model(top_pos_business_emb.unsqueeze(0)))
        x_neg_list_tensor.append([model(x.unsqueeze(0)) for x in bottom5_pos_business_emb])
    # convert to tensor
    x_tensor = torch.stack(x_tensor)
    x_pos_tensor = torch.stack(x_pos_tensor)
    x_neg_list_tensor = torch.stack([torch.stack(x_neg) for x_neg in x_neg_list_tensor])

    ## TODO, how to make it as a tensor and model eval in batch

    return contrastive_loss_helper(x_tensor, x_pos_tensor, x_neg_list_tensor)


def contrastive_loss_helper(f_x, f_x_pos, f_x_neg_list, tau=0.1):
    """
    Computes the InfoNCE loss without feature normalization.

    Args:
        f_x (torch.Tensor): The feature representation of the anchor sample (batch_size, dim).
        f_x_pos (torch.Tensor): The feature representation of the positive sample (batch_size, dim).
        f_x_neg_list (list of torch.Tensor): A list of negative samples (each of shape (batch_size, num_neg, dim)).
        tau (float): Temperature parameter.

    Returns:
        torch.Tensor: The computed InfoNCE loss.
    """

    # Compute positive similarity: exp(f_x^T f_x^+ / tau)
    #pos_sim = torch.exp(torch.sum(f_x * f_x_pos, dim=-1) / tau)  # (batch_size)
    
   

    # Compute negative similarities for each negative group and sum them
    
    #neg_sim = sum(torch.exp(torch.sum(f_x.unsqueeze(1) * f_x_neg, dim=-1) / tau) for f_x_neg in f_x_neg_list)  # (batch_size, num_neg) summed
    
    
    # Compute the denominator
    #denominator = pos_sim + torch.sum(neg_sim, dim=-1)  # (batch_size)

    pos_sim = torch.exp(torch.bmm(f_x, f_x_pos.transpose(2, 1)).squeeze(2) / tau) # (B, 1)
    neg_sim = torch.exp(torch.sum(torch.bmm(f_x, f_x_neg_list.squeeze(2).transpose(2, 1)), dim = 2) / tau) # (B, 1)
    denominator = pos_sim + neg_sim 
    

    # Compute the ratio
    ratio = torch.nan_to_num(pos_sim / denominator)
    # print(f"pos_sim: {pos_sim}, neg_sim: {neg_sim}, denominator: {denominator}, ratio: {ratio}")
    # Ensure the ratio is positive
    if torch.isnan(ratio).any():
        pdb.set_trace()
        print("NaN detected in ratio. Applying clamping.")
        print(f"The numerator is {pos_sim} and the denominator is {denominator}, neg_sim is {neg_sim}")
    if torch.isinf(ratio).any():
        print("Inf detected in ratio. Applying clamping.")
    
    # Replace NaN and Inf values in ratio with epsilon
    #ratio = torch.where(torch.isnan(ratio) | torch.isinf(ratio), torch.tensor(epsilon, device=ratio.device), ratio)
    # ratio = torch.clamp(ratio, min=epsilon)


    # Compute loss 
    ## TODO, he .mean() at the end is used to compute the average loss across the batch. make sure we need to do the mean
    loss = -torch.log(ratio).mean()  # Scalar loss

    return loss if not torch.isinf(loss) else 0


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

    def __init__(self,sparse_feature_num, dense_feature_number, emb_feature_dim, sparse_dim_before_embedding, sparse_dim_after_embedding, 
                 bottom_mlp_dims, top_mlp_dims, emb_feature_dim_before_project, dropout_prob, model_config):
        super(DLRM, self).__init__()
        if sparse_dim_before_embedding > 0:
            self.embedding = torch.nn.Embedding(sparse_dim_before_embedding, sparse_dim_after_embedding, padding_idx=179)
        self.layer_feature_interaction = FeatureInteraction()
        self.model_config = model_config
        self.is_contrastive_loss = model_config["is_contrastive_loss"]

        if model_config["is_contrastive_loss"]:
            self.business_projection_model = ContrastiveNet(model_config["business_features_input_dim"], model_config["projection_dim"])
            self.user_projection_model = ContrastiveNet(model_config["user_features_input_dim"], model_config["projection_dim"])
            self.image_projection_model = MLPDimReduction(model_config["image_features_input_dim"], model_config["projection_dim"]) # 
            self.real_emb_features_dim = 3 * model_config["projection_dim"] # we concat 3 projection results
        else:
            self.projection_model = ContrastiveNet(emb_feature_dim_before_project, emb_feature_dim)  # ContrastiveNet instance
            self.real_emb_features_dim = emb_feature_dim

        self.bottom_mlp = torch.nn.Sequential(
            torch.nn.Linear(dense_feature_number, bottom_mlp_dims[0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(bottom_mlp_dims[0], bottom_mlp_dims[1]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_prob)
        )

        self.all_dim = sparse_feature_num * sparse_dim_after_embedding + bottom_mlp_dims[1] + self.real_emb_features_dim
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

    def forward(self, x_sparse, x_dense, x_embed_before_projection, postive_input=None):
        # is_sparse_features_included = not IS_BASELINE and x_sparse.shape[0] > 0
        # if is_sparse_features_included:
        embed_x = self.embedding(x_sparse)
        embed_x = embed_x.sum(dim = 1) # (B, N, D) -> (B, D) N is the length after padding 11 in our case. 
        #embed_x = embed_x.view(x_sparse.shape[0], -1) # concat all embedding of sparse features together

        if self.is_contrastive_loss:
            # projection for business and user features
            # here we expect each x_embed_before_projection contain 3 tensors, each of them is 384 dimension, 
            # first one is business, second one is user, third one is image embedding
            # print("x_embed_before_projection's shape ", x_embed_before_projection.shape)
            # Extract each component separately
            x_embed_business = x_embed_before_projection[:, 0, :]  # Shape (n, 384)
            x_embed_user = x_embed_before_projection[:, 1, :]      # Shape (n, 384)
            x_embed_image = x_embed_before_projection[:, 2, :]     # Shape (n, 384)

            # Process each component separately
            x_embed_business = self.business_projection_model(x_embed_business)
            x_embed_user = self.user_projection_model(x_embed_user)
            x_embed_image = self.image_projection_model(x_embed_image)
            x_embed = torch.concat([x_embed_business, x_embed_user, x_embed_image], dim=-1)
        else:
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

        postive_embed = self.projection_model(postive_input) if postive_input is not None else None
        return output, x_embed, postive_embed