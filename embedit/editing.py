"""Consolidated token-row optimization; source variants are in reference/."""

import math

import torch


def edit_embeddings(encoder, tokenizer, pairs, *, learning_rate=0.001,
                    iterations=100, factor=0.3, joint=False):
    """Adam + MSE, masking gradients to source token rows.

    Single edits recompute detached target features on every iteration, as in
    the source scripts. Joint edits freeze target features before optimization.
    The stop metric uses pre-update features, matching the source scripts.
    """
    if not pairs:
        raise ValueError("Provide at least one edit.")
    if not joint and len(pairs) != 1:
        raise ValueError("Pass one pair per independent/sequential edit.")
    if iterations < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Iterations and learning rate must be positive.")
    if not math.isfinite(factor) or factor < 0:
        raise ValueError("Stopping factor must be finite and nonnegative.")
    weight = encoder.text_model.embeddings.token_embedding.weight
    if weight.dtype != torch.float32:
        raise ValueError("Use float32 text encoders for Adam embedding editing.")
    length = encoder.config.max_position_embeddings
    token_ids = []
    for old, new in pairs:
        for text in (old, new):
            ids = tokenizer.encode(text, add_special_tokens=True)
            if len(ids) <= 2 or len(ids) > length:
                raise ValueError("Edit text is empty or exceeds the encoder token limit.")
        token_ids.extend(tokenizer.encode(old, add_special_tokens=True)[1:-1])
    token_ids = sorted(set(token_ids))

    def embed(texts):
        ids = tokenizer(texts, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=length).input_ids.to(weight.device)
        return encoder(ids)[0]

    old_texts, new_texts = map(list, zip(*pairs))
    parameters = list(encoder.parameters())
    original_flags = [p.requires_grad for p in parameters]
    training = encoder.training
    encoder.eval()
    encoder.requires_grad_(False)
    weight.requires_grad_(True)
    optimizer = torch.optim.Adam([weight], lr=learning_rate)
    mse = torch.nn.MSELoss()
    history = []
    try:
        with torch.no_grad():
            initial_source, target = embed(old_texts), embed(new_texts)
            initial = mse(initial_source, target).item()
        threshold = initial * factor
        for step in range(iterations):
            source = embed(old_texts)
            if not joint:
                with torch.no_grad():
                    target = embed(new_texts)
            optimizer.zero_grad(set_to_none=True)
            loss = mse(source, target)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite MSE; no further updates were applied.")
            loss.backward()
            rows = weight.grad[token_ids].clone()
            weight.grad.zero_()
            weight.grad[token_ids] = rows
            optimizer.step()
            score = loss.item()
            history.append({"step": step + 1, "mse": loss.item(), "stop_metric": score})
            minimum_steps = 1 if joint else 2
            if step + 1 >= minimum_steps and score <= threshold:
                break
    finally:
        weight.grad = None
        for parameter, flag in zip(parameters, original_flags):
            parameter.requires_grad_(flag)
        encoder.train(training)
    return {"token_ids": token_ids, "initial_metric": initial,
            "threshold": threshold, "history": history}
