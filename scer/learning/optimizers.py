import torch
from transformers import AdamW


def get_bert_optim(network, lr, weight_decay):
    no_decay = ["bias", "LayerNorm.weight"]
    decay_params = []
    no_decay_params = []
    for n, p in network.named_parameters():
        if any(nd in n for nd in no_decay):
            decay_params.append(p)
        else:
            no_decay_params.append(p)
        
    if hasattr(network, "sigma_matrix") and network.sigma_matrix.requires_grad:
        print("✅ Adding sigma_matrix to SGD optimizer.")  
        params.append(network.sigma_matrix)

    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        eps=1e-8)
    return optimizer


def get_sgd_optim(network, lr, weight_decay):

    params = list(network.parameters())  
    if hasattr(network, "sigma_matrix") and network.sigma_matrix.requires_grad:
        print("✅ Adding sigma_matrix to SGD optimizer.")  
        params.append(network.sigma_matrix)
    
    return torch.optim.SGD(
        params,
        lr=lr,
        weight_decay=weight_decay,
        momentum=0.9)


get_optimizers = {
    "sgd": get_sgd_optim,
    "adamw": get_bert_optim
}
