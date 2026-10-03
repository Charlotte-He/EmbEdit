# Original MSE function excerpt; see reference/README.md.
# SPDX-License-Identifier: MIT
import torch
import torch.nn as nn
device = torch.device("cpu")  # Set to the encoder device if using this reference.

def change_emd(text_encoder, tokenizer, row, learning_rate=0.05, num_iterations=50):
    """
    Optimize the embedding weights for a specific word in the given text encoder.

    Args:
        text_encoder: The text encoder module from the pipeline.
        tokenizer: The tokenizer associated with the text encoder.
        row (dict): A dictionary containing 'old' and 'new' words.
        learning_rate (float): Learning rate for the optimizer.
        num_iterations (int): Maximum number of optimization iterations.
        fid_threshold_factor (float): Factor to determine the FID threshold.

    Returns:
        str: The old word that was optimized.
    """
    # print("### Optimizing embeddings for:", row['old'], "->", row['new'])
    # old_word = "CEO" #ceo
    # new_word = "female CEO" #female ceo
    old_word = row['old']
    new_word = row['new']

    if new_word.startswith('female '):
        bias_word = 'male ' + old_word
    elif new_word.startswith('male '):
        bias_word = 'female ' + old_word
    else:
        bias_word = None
    if bias_word is None:
        print("cannot know bias word")

    print(f"bias_word:{bias_word}, old word:{old_word},new word:{new_word}")


    # 定义固定的最大序列长度
    max_length = 77

    # Encode old word to get token IDs, excluding special tokens
    old_token_ids = tokenizer.encode(old_word, add_special_tokens=True)[1:-1]
    print(old_token_ids)
    embedding_weight = text_encoder.text_model.embeddings.token_embedding.weight
    embedding_weight.requires_grad_(True)

    # Encode the words "female" and "male" to get their token IDs
    female_token_id = tokenizer.encode("female")[1:-1]
    male_token_id = tokenizer.encode("male")[1:-1]

    print(f"Token ID for 'female': {female_token_id}")
    print(f"Token ID for 'male': {male_token_id}")

    # Get the WTE for "female" and "male"
    female_wte = embedding_weight[female_token_id]
    male_wte = embedding_weight[male_token_id]
    old_token_wte = embedding_weight[old_token_ids]
    # print(f"WTE for 'female': {female_wte}")
    # print(f"WTE for 'male': {male_wte}")
    print(f"WTE for 'profession': {old_token_wte}")

    old_token_wte = (female_wte + male_wte + old_token_wte)/3
    print(f"Edit WTE for 'profession': {old_token_wte}")

    # Create the optimizer for the embedding weights
    optimizer = torch.optim.Adam([embedding_weight], lr=learning_rate)
    mse_loss = torch.nn.MSELoss()
    old_texts = [old_word]
    new_texts = [new_word]

    for old_text, new_text in zip(old_texts, new_texts):
        # Get conditioning embeddings
        encoded_old = tokenizer(
            [old_word],
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_length
        ).input_ids.to(device)
        oc = text_encoder(encoded_old)[-1].detach()

        with torch.no_grad():
            encoded_new = tokenizer(
                [new_text],
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=max_length
            ).input_ids.to(device)
            d = text_encoder(encoded_new)[-1]

            # Calculate initial FID to set the threshold
            encoded_old_initial = tokenizer(
                [bias_word],
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=max_length
            ).input_ids.to(device)
            e = text_encoder(encoded_old_initial)[-1]  # shape [batch_size, seq_length, hidden_dim]
        
        loss_ocd = mse_loss(oc, d)
        loss_oce = mse_loss(oc, e)
        print("Loss_ocd:",loss_ocd)
        print("Loss_oce:",loss_oce)

        stopping_con = abs(2*(loss_ocd-loss_oce)/(loss_ocd+loss_oce))
        difference = abs(2*(loss_ocd-loss_oce)/(loss_ocd+loss_oce))
        normalized_difference = difference / 1
        normalized_difference = min(normalized_difference, 1.0)  # 防止超出范围

        print(difference)

        for i in range(num_iterations):
            # Get conditioning embeddings
            encoded_old = tokenizer(
                [old_text],
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=max_length
            ).input_ids.to(device)
            c = text_encoder(encoded_old)[-1]  # shape [batch_size, seq_length, hidden_dim]

            with torch.no_grad():
                encoded_new = tokenizer(
                    [new_text],
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                    max_length=max_length
                ).input_ids.to(device)
                d = text_encoder(encoded_new)[-1]

                # Calculate initial FID to set the threshold
                encoded_old_initial = tokenizer(
                    [bias_word],
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                    max_length=max_length
                ).input_ids.to(device)
                e = text_encoder(encoded_old_initial)[-1]  # shape [batch_size, seq_length, hidden_dim]

           
            # Calculate loss and compute gradients
            optimizer.zero_grad()  # Clear previous gradients
            # print(f"Shape of c: {c.shape}")
            # print(f"Shape of d: {d.shape}")
            # 计算 c 和 d 的余弦距离
            # cos_sim_cd = F.cosine_similarity(c, d, dim=1)
            # cos_dist_cd = (1 - cos_sim_cd).mean()  # 余弦距离


            # 计算 c 和 e 的余弦距离
            loss_cd = mse_loss(c, d)
            loss_ce = mse_loss(c, e)
            # cos_sim_ce = F.cosine_similarity(c, e, dim=1)
            # cos_dist_ce = (1 - cos_sim_ce).mean()  # 余弦距离
            min_weight = 2
            
            adjust = max(min_weight, 10 * normalized_difference)
            print("adjust:",adjust)
            total_loss =  adjust**2*loss_cd + (1/adjust)**2*loss_ce
            # 输出余弦距离
            print(f" distance between c and d : {loss_cd.item()}")
            print(f" distance between c and e : {loss_ce.item()}")

            total_loss.backward()  # Compute gradients

            # Zero out gradients for all tokens except the target
            tmp = embedding_weight.grad[old_token_ids]
            embedding_weight.grad.zero_()
            embedding_weight.grad[old_token_ids] = tmp  # Retain only target gradient
            optimizer.step()  # Update weights
            
            bias_con =abs(2 * (loss_cd - loss_ce) / (loss_cd + loss_ce))
            print({bias_con.item()})
            print("stopping:",stopping_con)
            # if i > 1:
            #     if cos_dist_cd <= stopping_con and cos_dist_cd <= cos_dist_ce:
            #         print(f"Stopping early for '{cos_dist_cd}'")
            #         break

            if i > 1:
                if  bias_con >= 0.*stopping_con:
                    print(f"Stopping early for '{loss_cd}'")
                    break
