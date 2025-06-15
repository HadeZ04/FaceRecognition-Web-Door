import torch
import torch.nn as nn
import torch.nn.functional as F
# Define the wrapper class again as it's needed for loading
class FaceEmbeddingModel(nn.Module):
    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model

    def forward(self, x):
        emb = self.embedding_model(x)
        # Ensure normalization if the saved model was trained with it
        return F.normalize(emb, p=2, dim=1)
