import numpy as np

data_embed_collect = []
label_collect = []

def save_data():
    import torch  # Import torch here to avoid circular import
    data_embed_npy = torch.cat(data_embed_collect, axis=0).cpu().numpy()
    label_npy = torch.cat(label_collect, axis=0).cpu().numpy()

    np.save("data_embed_npy.npy", data_embed_npy)
    np.save("label_npy.npy", label_npy)