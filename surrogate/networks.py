"""
Networks: the training loop shared by all neural networks of the study.
"""

import numpy as np


def set_seed(seed, threads):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)


def fit_early_stopping(model, x_tr, y_tr, x_va, y_va, loss_fn, lr, weight_decay, batch, max_epochs, patience,
                       seed, squeeze=False):
    """AdamW training with early stopping on the validation loss; the weights of the best epoch are kept
    (epoch 0 = starting weights). Used by every network of the study."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    as_tensor = lambda a: a if torch.is_tensor(a) else torch.from_numpy(a)
    x_tr, y_tr, x_va, y_va = map(as_tensor, (x_tr, y_tr, x_va, y_va))
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    forward = (lambda x: model(x).squeeze(-1)) if squeeze else model

    def val_loss():
        model.eval()
        with torch.no_grad():
            return float(loss_fn(forward(x_va), y_va))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_epoch, bad = val_loss(), 0, 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(forward(xb), yb).backward()
            optimizer.step()
        current = val_loss()
        if current < best:
            best, best_epoch, bad = current, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return best, best_epoch, best_state
