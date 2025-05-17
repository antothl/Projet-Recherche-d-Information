import torch.nn as nn


class shared_Linear(nn.Module):
    """
    """
    def __init__(self, d_in, d_out, branches):
        """
        """
        super(shared_Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.d_in, self.d_out = d_in, d_out
        self.branches = branches

    
    def forward(self, sequences):
        """
        """
        sequences = sequences.reshape(sequences.shape[0], self.branches, self.d_in)
        output = self.linear(sequences)
        return output.reshape(output.shape[0], output.shape[1]*output.shape[2])