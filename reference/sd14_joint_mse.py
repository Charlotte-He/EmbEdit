# Original MSE function excerpt; see reference/README.md.
# SPDX-License-Identifier: MIT
import torch
import torch.nn as nn
device = torch.device("cpu")  # Set to the encoder device if using this reference.

def multi_change_emd(text_encoder, tokenizer, pairs,
                     lr=1e-3, iters=100, mse_factor=0.2):
    device = text_encoder.device
    mse_loss = nn.MSELoss()
    emb = text_encoder.text_model.embeddings.token_embedding.weight
    emb.requires_grad_(True)
    optimizer = torch.optim.Adam([emb], lr=lr)

    def embed(txt_list):
        ids = tokenizer(txt_list, return_tensors="pt",
                        padding="max_length", truncation=True,
                        max_length=77).input_ids.to(device)
        return text_encoder(ids)[0]

    tgt, src_ids = [], []
    for old, new in pairs:
        old_tok = tokenizer.encode(old, add_special_tokens=True)[1:-1]
        if not old_tok:
            raise ValueError(f"Tokenizer 无法解析 “{old}”")
        src_ids.extend(old_tok)
        # -------- 目标向量无需梯度 ----------
        with torch.no_grad():
            tgt.append(embed([new]))
    tgt = torch.cat(tgt, dim=0)

    # -------- 基准 MSE 也无需梯度 ----------
    with torch.no_grad():
        mse_thres = mse_loss(embed([p[0] for p in pairs]), tgt).item() * mse_factor
    print(f"### Batch-edit {len(pairs)} pairs, early-stop MSE {mse_thres:.6f}")

    for step in range(1, iters + 1):
        loss = mse_loss(embed([p[0] for p in pairs]), tgt)
        optimizer.zero_grad(); loss.backward()

        # 只更新相关 token 行
        grad_copy = emb.grad[src_ids].clone()
        emb.grad.zero_(); emb.grad[src_ids] = grad_copy
        optimizer.step()

        if loss.item() <= mse_thres:
            print(f"  ✓ stop at iter {step}, MSE={loss.item():.6f}")
            break
    print("### Batch-edit finished\n")
