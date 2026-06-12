"""
Module containing PyTorch deep learning architectures for financial forecasting.
Implements multi-layer perceptrons (MLP), 1D convolutional neural networks (CNN),
long short-term memory networks (LSTM), hybrid CNN-LSTMs, multi-head attention blocks,
and Transformer encoders. All models predict a single-valued forward return target.
"""

import torch.nn as nn
import torch.nn.functional as F
import math
import torch

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for cross-sectional stock return prediction.

    Architecture:
    - Dynamic number of hidden linear layers defined by `hidden_layers`.
    - Choice of activation functions (ReLU, Tanh, GELU).
    - Kaiming uniform weight initialization for ReLU/Leaky ReLU activations, and
      Xavier uniform initialization for other activations.
    - Dropout layers applied to intermediate hidden layers for regularization.
    - Output layer with 1 unit and Xavier uniform initialization, predicting y.
    """


    def __init__(self,input_size,hidden_layers,activation,lr,wd,dropout,batch_size):
        super().__init__()
        self.lr = lr
        self.wd = wd
        self.batch_size = batch_size
        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU
        }
        self.activation = activation
        layers = []
        in_features = input_size

        for i, hidden_size in enumerate(hidden_layers):
            layer = nn.Linear(in_features, hidden_size)
            if activation == "relu":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            elif activation == "leaky_relu":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="leaky_relu")
            else:
                nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

            layers.append(layer)
            layers.append(activation_map[activation]())

            if dropout > 0 and i < len(hidden_layers) - 1:
                layers.append(nn.Dropout(dropout))

            in_features = hidden_size

        output_layer = nn.Linear(in_features, 1)
        nn.init.xavier_uniform_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)

        self.model = nn.Sequential(*layers)

    def forward(self,x):
        return self.model(x).squeeze(-1)
    
class CNNmodel(nn.Module):
    """
    1D Convolutional Neural Network (CNN) for extracting features from sequential inputs.

    Architecture:
    - Two Conv1d layers with kernel size 3, stride 1, and padding 1 to process temporal features.
    - Batch normalization and GELU activation after each convolutional layer.
    - Adaptive average pooling layer to squeeze sequence length dimension to 1.
    - Dropout layer for regularization.
    - Fully connected network (FC1 -> GELU -> FC2) mapping features to a single return prediction.
    - Kaiming normal initialization for convolutional and linear weights.
    """


    def __init__(self, input_features, out_channels_l1, out_channels_l2, dropout, lr, wd, fc_dim, batch_size):
        super().__init__()

        self.lr = lr
        self.wd = wd
        self.batch_size = batch_size

        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=out_channels_l1, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels_l1)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(in_channels=out_channels_l1, out_channels=out_channels_l2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels_l2)
        self.act2 = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(out_channels_l2, fc_dim)
        self.act3 = nn.GELU()
        self.fc2 = nn.Linear(fc_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv2.bias)

        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)
        nn.init.ones_(self.bn2.weight)
        nn.init.zeros_(self.bn2.bias)

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        return x.squeeze(-1)

    
class LSTMmodel(nn.Module):
    """
    Long Short-Term Memory (LSTM) network for capturing sequential temporal dependencies.

    Architecture:
    - Single-layer LSTM mapping input sequences to hidden dimensions.
    - Slices the final sequence step's hidden state.
    - Batch normalization and Dropout applied to the extracted sequence representation.
    - Fully connected network (FC1 -> GELU -> FC2) mapping output to a single return prediction.
    - Xavier uniform initialization for LSTM weights and Kaiming normal for linear weights.
    """


    def __init__(self, input_features, hidden_size, dropout, fc_dim, lr, wd, batch_size):
        super().__init__()

        self.lr = lr
        self.wd = wd
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size=input_features, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, fc_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(fc_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0)

        nn.init.zeros_(self.lstm.bias_ih_l0)
        nn.init.zeros_(self.lstm.bias_hh_l0)

        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc2.bias)

        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x.squeeze(-1)
    
class CNN_LSTM(nn.Module):
    """
    Hybrid CNN-LSTM network combining local feature extraction and temporal modeling.

    Architecture:
    - 1D Convolutional layer to extract local patterns across feature dimensions,
      followed by Batch Normalization and a GELU activation.
    - Single-layer LSTM to model temporal sequence dependencies from the CNN features.
    - Extracts the final LSTM sequence step, applies Dropout, and passes it through
      a final linear layer to predict target returns.
    - Kaiming normal initialization for convolutional and linear weights; orthogonal
      initialization for LSTM recurrent weights.
    """


    def __init__(self, input_features, cnn_out_channels, lstm_hidden_dim, dropout, lr, wd, batch_size):
        super().__init__()

        self.lr = lr
        self.wd = wd
        self.batch_size = batch_size

        self.cnn = nn.Conv1d(in_channels=input_features, out_channels=cnn_out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm1d(cnn_out_channels)
        self.act1 = nn.GELU()
        self.lstm = nn.LSTM(input_size=cnn_out_channels, hidden_size=lstm_hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.cnn.weight, nonlinearity="relu")
        nn.init.zeros_(self.cnn.bias)
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc.bias)

        for name, parameter in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(parameter)
            elif "weight_hh" in name:
                nn.init.orthogonal_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = self.bn(x)
        x = self.act1(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze(-1)

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention layer.

    Computes scaled dot-product attention over input sequences by projecting them
    into Query, Key, and Value spaces across multiple heads.

    Architecture:
    - Multi-head projections for Q, K, and V.
    - Computes attention scores with scaling by `sqrt(head_dim)`.
    - Softmax activation and dropout on attention weights.
    - Linear projection of concatenated attention output back to original embedding space.
    - Xavier uniform initialization for all linear projections.
    """


    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.Q = nn.Linear(embed_dim, embed_dim)
        self.K = nn.Linear(embed_dim, embed_dim)
        self.V = nn.Linear(embed_dim, embed_dim)

        self.out_projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for layer in [self.Q, self.K, self.V, self.out_projection]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):

        batch_size, seq_len, _ = x.shape

        q = self.Q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.K(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.V(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores /= math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        return self.out_projection(context)
    
class Transformer(nn.Module):
    """
    Transformer encoder network for sequential prediction.

    Architecture:
    - Linear embedding layer mapping input features to the internal embedding dimension.
    - Learned positional embeddings added to input representations.
    - Single encoder layer containing:
        - Multi-head attention block with a residual connection and layer normalization (Norm1).
        - Feed-forward expansion network (Linear -> GELU -> Dropout -> Linear) with residual
          connection and layer normalization (Norm2).
    - Global average pooling across the sequence dimension (mean pooling).
    - Output linear layer predicting target returns.
    - Xavier uniform initialization for embedding, projection, and feed-forward weights.
    """


    def __init__(self, input_features, embed_dim, num_heads, dropout, fc_dim, lr, wd, batch_size):
        super().__init__()

        self.lr = lr
        self.wd = wd
        self.batch_size = batch_size

        self.embedding = nn.Linear(input_features, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 12, embed_dim))

        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, fc_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, embed_dim)
        )

        self.linear_out = nn.Linear(embed_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.embedding.bias)

        nn.init.ones_(self.norm1.weight)
        nn.init.zeros_(self.norm1.bias)
        nn.init.ones_(self.norm2.weight)
        nn.init.zeros_(self.norm2.bias)

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.linear_out.weight)
        nn.init.zeros_(self.linear_out.bias)

        nn.init.zeros_(self.pos_embedding)

    def forward(self, x):

        x = self.embedding(x) + self.pos_embedding
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.fc(x))
        x = torch.mean(x, dim=1)

        return self.linear_out(x).squeeze(-1)