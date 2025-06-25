import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from torchmetrics.classification import ConfusionMatrix
from models import InfoNCELoss, CustomLoss, CustomLoss2, compute_contrastive_loss_user, compute_contrastive_loss_business
from utils import calculate_weights, get_fp_rate, get_accuracy, get_formatted_time

def validate_one_epoch(model, val_loader, device, weight_dict, print_method, epoch, swing_similarity_data, custom_loss_fn, para_dict, temperature):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    info_nce_loss_fn = InfoNCELoss(temperature).to(device)
    bce_loss_fn = nn.BCELoss()

    # with torch.no_grad():
    #     for sparse_f, dense_f, emb_f, positives_f, target_f, in val_loader:
    #         dense_f = dense_f.to(device)
    #         sparse_f = sparse_f.to(device)
    #         emb_f = emb_f.to(device)
    #         target_f = target_f.to(device)

    #         output, x_embed, positive_embed = model(sparse_f, dense_f, emb_f)
    #         cur_weights = torch.tensor([weight_dict[x.item()] for x in target_f], device=device)
    #         criterion = nn.BCELoss(weight=cur_weights)

    #         info_nce_loss = info_nce_loss_fn(x_embed, positive_embed, target_f)
    #         bce_loss = bce_loss_fn(output, target_f.unsqueeze(1).float())
    #         loss = info_nce_loss + bce_loss
    #         val_loss += loss.item() * dense_f.size(0)

    #         predicted = (output > 0.5).float().squeeze(1)
    #         total += target_f.size(0)
    #         correct += (predicted == target_f).sum().item()

    #         all_preds.extend(predicted.cpu().numpy())
    #         all_targets.extend(target_f.cpu().numpy())
    with torch.no_grad():
        for sparse, dense, original_emb, positives, labels, business_ids, user_ids in val_loader:
            # Move data to device
            dense = dense.to(device)
            sparse = sparse.to(device)
            original_emb = original_emb.to(device)
            labels = labels.to(device)

            

            # Forward pass
            if model.is_contrastive_loss:
                # Use the contrastive loss
                output, x_embed, _ = model(sparse, dense, original_emb, None)
            else:
                output, x_embed, positive_embed = model(sparse, dense, original_emb, positives)


            # # Calculate loss
            # info_nce_loss_fn = InfoNCELoss(temperature).to(device)
            # info_nce_loss = info_nce_loss_fn(x_embed, positive_embed, labels)

            # BCE loss with weights
            cur_weights = torch.tensor([weight_dict[x.item()] for x in labels], device=device)
            bce_loss_fn = nn.BCELoss(weight=cur_weights.unsqueeze(1))            
            bce_loss = bce_loss_fn(output, labels.unsqueeze(1).float())

            # print("calculating contrastive loss")
            # switch to model into evaluation mode as we need to compute the 
            # print("in validation user_ids shape", user_ids.shape, "business_ids shape", business_ids.shape)
            if para_dict['is_use_user_contrastive_loss'] and para_dict['is_use_business_contrastive_loss']:
                contrastive_loss_user = compute_contrastive_loss_user(user_ids, swing_similarity_data, model.user_projection_model)
                contrastive_loss_business = compute_contrastive_loss_business(business_ids, swing_similarity_data, model.business_projection_model)
                # combining two losses
                loss = custom_loss_fn(bce_loss, contrastive_loss_user, contrastive_loss_business)
            elif para_dict['is_use_user_contrastive_loss']:
                contrastive_loss_user = compute_contrastive_loss_user(user_ids, swing_similarity_data, model.user_projection_model)
                loss = custom_loss_fn(bce_loss, contrastive_loss_user)
            elif para_dict['is_use_business_contrastive_loss']:
                contrastive_loss_business = compute_contrastive_loss_business(business_ids, swing_similarity_data, model.business_projection_model)
                loss = custom_loss_fn(bce_loss, contrastive_loss_business)
            else:
                loss = bce_loss

            val_loss += loss.item() * output.size(0)

            predicted = (output > 0.5).float().squeeze(1)
            total += output.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    accuracy = correct / total

    all_preds = torch.tensor(all_preds) > 0.5
    all_targets = torch.tensor(all_targets) > 0.5
    # total_pos_num = (all_targets == True).sum().item()
    # total_neg_num = (all_targets == False).sum().item()

    confmat = ConfusionMatrix(task="binary").to(device)
    print("shape of all_preds is {}, all_targets is {}".format(all_preds.shape, all_targets.shape))
    conf_matrix = confmat(all_preds, all_targets)

    ## check out true of false 
    # Count the number of True values
    num_true = all_targets.sum().item()

    # Count the number of False values
    num_false = all_targets.numel() - num_true

    print("all targets num_true", num_true, "num_false", num_false)

        ## check out true of false 
    # Count the number of True values
    num_true = all_preds.sum().item()

    # Count the number of False values
    num_false = all_preds.numel() - num_true

    # print("all preds num_true", num_true, "num_false", num_false)

    # print("all_preds", all_preds)
    # print("all_targets", all_targets)
    print("confusion matrix", conf_matrix)

    fp_rate = get_fp_rate(conf_matrix)
    accuracy = get_accuracy(conf_matrix)
    print_method(f"Validation at epoch {epoch}: Loss={val_loss:.4f}, Accuracy={accuracy:.4f}, FP Rate={fp_rate:.4f}")
    return val_loss, accuracy, conf_matrix

def train_contrastive_model(model, train_labels, train_dataloader, val_dataloader, para_dict, swing_similarity_data):
    start_time = get_formatted_time()
    # batch_size = para_dict['batch_size']
    epochs = para_dict['epochs']
    temperature = para_dict['temperature']
    # dropout_prob = para_dict['dropout_prob']
    val_interval = para_dict['val_interval']

    ## TODO remove 
    # dataset = ContrastiveDataset(sparse_features, dense_features, origin_emb_features, labels)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    train_history = []
    early_stopping_counter = 0

    def add_train_log(log):
        train_history.append(log)
        print(log)

    print_method = add_train_log
    _, _, _, weight_dict = calculate_weights(train_labels, is_use_sqrt=para_dict['IS_SQRT_WEIGHT'])

    # print para_dict
    print_method(str(para_dict))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## TODO remove, we should create the model in main 
    # model = DLRM(
    #     sparse_feature_num=1,
    #     dense_feature_number=dense_features.shape[1],
    #     emb_feature_dim=64,
    #     sparse_dim_before_embedding=180,
    #     sparse_dim_after_embedding=96,
    #     bottom_mlp_dims=(64, 16),
    #     top_mlp_dims=(256, 16),
    #     emb_feature_dim_before_project=origin_emb_features.shape[1],
    #     dropout_prob=dropout_prob
    # ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_model_for_val = None
    best_val_fp_rate = float('inf')

    if para_dict['is_use_user_contrastive_loss'] and para_dict['is_use_business_contrastive_loss']:
        custom_loss_fn = CustomLoss()
    else:
        custom_loss_fn = CustomLoss2()

    print_method(f"Training started at {start_time}")
    for epoch in range(epochs):
        model.train()
        model.user_projection_model.train()
        model.business_projection_model.train()
        counter = 0
        total_loss = 0
        num_train = 0
        correct = 0
        all_preds = []
        all_targets = []
        ## TODO remove passing in positives from dataloader
        for sparse, dense, original_emb, positives, labels, business_ids, user_ids in train_dataloader:
            # Move data to device
            dense = dense.to(device)
            sparse = sparse.to(device)
            original_emb = original_emb.to(device)
            labels = labels.to(device)

            

            # Forward pass
            optimizer.zero_grad()
            if model.is_contrastive_loss:
                # Use the contrastive loss
                output, x_embed, _ = model(sparse, dense, original_emb, None)
            else:
                output, x_embed, positive_embed = model(sparse, dense, original_emb, positives)


            # # Calculate loss
            # info_nce_loss_fn = InfoNCELoss(temperature).to(device)
            # info_nce_loss = info_nce_loss_fn(x_embed, positive_embed, labels)

            # BCE loss with weights
            cur_weights = torch.tensor([weight_dict[x.item()] for x in labels], device=device)
            bce_loss_fn = nn.BCELoss(weight=cur_weights.unsqueeze(1))    
            # print("labels", labels)        
            # print("output", output)
            bce_loss = bce_loss_fn(output, labels.unsqueeze(1).float())

            # print("calculating contrastive loss")
            # switch to model into evaluation mode as we need to compute the 
            model.eval()
            model.user_projection_model.eval()
            model.business_projection_model.eval()
            # print("user_ids shape", user_ids.shape, "business_ids shape", business_ids.shape)

            counter += 1
            if para_dict['is_use_user_contrastive_loss'] and para_dict['is_use_business_contrastive_loss']:
                contrastive_loss_user = compute_contrastive_loss_user(user_ids, swing_similarity_data, model.user_projection_model)
                contrastive_loss_business = compute_contrastive_loss_business(business_ids, swing_similarity_data, model.business_projection_model)
                # combining two losses
                loss = custom_loss_fn(bce_loss, contrastive_loss_user, contrastive_loss_business)
                if counter % 40 == 0:
                    print(f"contrastive_loss_user:{contrastive_loss_user.item()}\t, contrastive_loss_business:{contrastive_loss_business.item()}\t, bce_loss:{bce_loss.item()}\t, loss:{loss.item()}")

            elif para_dict['is_use_user_contrastive_loss']:
                contrastive_loss_user = compute_contrastive_loss_user(user_ids, swing_similarity_data, model.user_projection_model)
                loss = custom_loss_fn(bce_loss, contrastive_loss_user)
                if counter % 40 == 0:
                    print(f"contrastive_loss_user:{contrastive_loss_user.item()}\t, bce_loss:{bce_loss.item()}\t, loss:{loss.item()}")
            elif para_dict['is_use_business_contrastive_loss']:
                contrastive_loss_business = compute_contrastive_loss_business(business_ids, swing_similarity_data, model.business_projection_model)
                loss = custom_loss_fn(bce_loss, contrastive_loss_business)
                if counter % 40 == 0:
                    print(f"contrastive_loss_business:{contrastive_loss_business.item()}\t, bce_loss:{bce_loss.item()}\t, loss:{loss.item()}")
            else:
                loss = bce_loss
                if counter % 40 == 0:
                    print(f"bce_loss:{bce_loss.item()}\t, loss:{loss.item()}")

            # re-enable model train mode
            model.train()
            model.user_projection_model.train()
            model.business_projection_model.train()


            #
            
            


            # Backward pass
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.user_projection_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.business_projection_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # update model parameters
            optimizer.step()

            # Update metrics
            total_loss += loss.item() * dense.size(0)
            predicted = (output > 0.5).float().squeeze(1)
            num_train += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())


        # Step the scheduler
        scheduler.step()

        # Validation
        if epoch % val_interval == 0:
            val_loss, val_accuracy, val_conf_matrix = validate_one_epoch(
                model, val_dataloader, device, weight_dict, print_method, epoch, swing_similarity_data, custom_loss_fn, para_dict, temperature
            )
            
            # Early stopping check
            val_fp_rate = get_fp_rate(val_conf_matrix)
            if val_fp_rate < best_val_fp_rate:
                best_val_fp_rate = val_fp_rate
                best_model_for_val = model
                early_stopping_counter = 0
            else:
                early_stopping_counter += val_interval
                if early_stopping_counter >= para_dict["early_stopping_patience"]:
                    print_method(f"Early stopping triggered at epoch {epoch}")
                    break
            print_method(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Print training stats
        train_loss = total_loss / num_train
        # train_accuracy = correct / num_train
            # accuracy = correct / total

        all_preds = torch.tensor(all_preds) > 0.5
        all_targets = torch.tensor(all_targets) > 0.5
        # total_pos_num = (all_targets == True).sum().item()
        # total_neg_num = (all_targets == False).sum().item()

        confmat = ConfusionMatrix(task="binary").to(device)
        # print("Training: shape of all_preds is {}, all_targets is {}".format(all_preds.shape, all_targets.shape))
        conf_matrix = confmat(all_preds, all_targets)

        ## check out true of false 
        # Count the number of True values
        num_true = all_targets.sum().item()

        # Count the number of False values
        num_false = all_targets.numel() - num_true

        print("Training all targets num_true", num_true, "num_false", num_false)

            ## check out true of false 
        # Count the number of True values
        num_true = all_preds.sum().item()

        # Count the number of False values
        num_false = all_preds.numel() - num_true

        # print("all preds num_true", num_true, "num_false", num_false)

        # print("all_preds", all_preds)
        # print("all_targets", all_targets)
        print_method("Training confusion matrix" + str(conf_matrix))

        fp_rate = get_fp_rate(conf_matrix)
        accuracy = get_accuracy(conf_matrix)
        print_method(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {accuracy:.4f}, fp_rate: {fp_rate:.4f}")

    end_time = get_formatted_time()
    print_method(f"Training finished at {end_time}")
    return best_model_for_val, train_history
