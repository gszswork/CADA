import torch

if __name__ == '__main__':
    batch_data = torch.rand([6, 128])
    batch = torch.tensor([0, 1, 0, 0, 1, 1])

    # Get the indices of 0s
    zeros_indices = torch.where(batch == 0)

    # Get the indices of 1s
    ones_indices = torch.where(batch == 1)

    concate_back = torch.zeros([6, 128])
    concate_back[zeros_indices] = batch_data[zeros_indices]