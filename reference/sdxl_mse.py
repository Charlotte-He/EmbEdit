# Original MSE function excerpt; see reference/README.md.
# SPDX-License-Identifier: MIT
import torch
import torch.nn as nn
device = torch.device("cpu")  # Set to the encoder device if using this reference.

def change_emd(text_encoder, tokenizer, row, learning_rate=0.001, num_iterations=100, fid_threshold_factor=0.2):
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
    print("### Optimizing embeddings for:", row['old'], "->", row['new'])
    old_word = row['old']
    new_word = row['new']

    # 定义固定的最大序列长度
    max_length = 77

    # Encode old word to get token IDs, excluding special tokens
    old_token_ids = tokenizer.encode(old_word, add_special_tokens=True)[1:-1]
    print("Optimizing embedding for token IDs: {old_token_ids}")

    embedding_weight = text_encoder.text_model.embeddings.token_embedding.weight
    embedding_weight.requires_grad_(True)

    # Create the optimizer for the embedding weights
    optimizer = torch.optim.Adam([embedding_weight], lr=learning_rate)
    mse_loss = torch.nn.MSELoss()
    old_texts = [old_word]
    new_texts = [new_word]

    for old_text, new_text in zip(old_texts, new_texts):
        # Calculate initial FID to set the threshold
        encoded_old_initial = tokenizer(
            [old_word],
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_length
        ).input_ids.to(device)
        c = text_encoder(encoded_old_initial)[0]  # shape [batch_size, seq_length, hidden_dim]

        with torch.no_grad():
            encoded_new_initial = tokenizer(
                [new_word],
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=max_length
            ).input_ids.to(device)
            d = text_encoder(encoded_new_initial)[0]

        #fid_threshold = calculate_fid(c, d) * fid_threshold_factor
        #print(f"FID threshold set to: {fid_threshold}")

        mse_threshold1 = (mse_loss(c,d).item()) * 0.3

        for i in range(num_iterations):
            # Get conditioning embeddings
            encoded_old = tokenizer(
                [old_word],
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=max_length
            ).input_ids.to(device)
            c = text_encoder(encoded_old)[0]  # shape [batch_size, seq_length, hidden_dim]

            with torch.no_grad():
                encoded_new = tokenizer(
                    [new_word],
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                    max_length=max_length
                ).input_ids.to(device)
                d = text_encoder(encoded_new)[0]

            # Calculate loss and compute gradients
            optimizer.zero_grad()  # Clear previous gradients
            # print(f"Shape of c: {c.shape}")
            # print(f"Shape of d: {d.shape}")
            loss = mse_loss(c, d)
            loss.backward()  # Compute gradients

            # Zero out gradients for all tokens except the target
            tmp = embedding_weight.grad[old_token_ids]
            embedding_weight.grad.zero_()
            embedding_weight.grad[old_token_ids] = tmp  # Retain only target gradient

            optimizer.step()  # Update weights
            '''
            # Calculate current FID score
            fid_score = calculate_fid(c, d)
            print(f"Iteration {i + 1}, FID Score: {fid_score}")

            # Stop if FID score is below threshold
            if fid_score <= fid_threshold:
                print(f"Stopping early for '{old_text}' and '{new_text}' at iteration {i + 1}, FID Score: {fid_score}")
                break
            '''
            if i > 0:
                print(f"####i:{i}#####")
                #fid_score = calculate_fid(c, d)
                mse_score = loss.item()
                print(f"Iteration {i}, mse Score: {mse_score}")
                    # Stop if FID score is below threshold
                if mse_threshold1 is not None and mse_score <= mse_threshold1 :
                    print(f"Stopping early for '{old_text}' and '{new_text}' at iteration {i}, mse Score: {mse_score}")
                    break
        print("Embedding update complete for '{old_text}' and '{new_text}'.")
