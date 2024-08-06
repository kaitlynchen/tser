import torch
import torch.nn as nn
import torch.nn.functional as F


def model_factory(config, data):
    task = config['task']
    feat_dim = data.feature_df.shape[1]  # dimensionality of data features
    # data windowing is used when samples don't have a predefined length or the length is too long
    max_seq_len = config['data_window_len'] if config['data_window_len'] is not None else config['max_seq_len']
    if max_seq_len is None:
        try:
            max_seq_len = data.max_seq_len
        except AttributeError as x:
            print("Data class does not define a maximum sequence length, so it must be defined with the script argument `max_seq_len`")
            raise x

    if (task == "imputation") or (task == "transduction"):
        raise NotImplementedError()
    if (task == "classification") or (task == "regression"):
        # dimensionality of labels
        num_labels = len(
            data.class_names) if task == "classification" else data.labels_df.shape[1]
        if config['model'] == 'local_cnn':
            return LocalCNN(in_channels=feat_dim, max_len=max_seq_len, num_classes=num_labels, final_emb_dim=config['d_model'],
                            conv_type=config['conv_type'], pool=config['pool'])
        else:
            raise ValueError("Unsupported model", config['model'])
    else:
        raise ValueError("Model class for task '{}' does not exist".format(task))

class LocalCNN(nn.Module):
    def __init__(self, in_channels, max_len=144, num_classes=1, final_emb_dim=512, conv_type="hierarchical", pool="seqpool"):
        """
        conv_type can be hierarchical or local

        pool can be seqpool (attention pooling in Compact Convolutional Transformer paper),
        average, or linear.
        """
        super().__init__()
        self.in_channels = in_channels
        self.max_len = max_len
        self.num_classes = num_classes
        self.final_emb_dim = final_emb_dim
        self.conv_type = conv_type
        self.pool = pool

        if self.conv_type == "hierarchical":  # Gradually reduces number of timesteps
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=5, stride=1),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=2, stride=2),
                nn.Conv1d(64, 128, 3, 1),
                nn.ReLU(),
                nn.AvgPool1d(2, 2),
                nn.Conv1d(128, 256, 3, 1),
                nn.ReLU(),
                nn.AvgPool1d(2, 2),
                nn.Conv1d(256, final_emb_dim, 3, 1),
                nn.AvgPool1d(2, 2),
            )
        elif self.conv_type == "local":  # No reducing number of timesteps
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=5, stride=1, padding='same'),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 128, 5, 1, padding='same'),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 256, 5, 1, dilation=3, padding='same'),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, final_emb_dim, 5, 1, dilation=5, padding='same'),
            )
        else:
            raise ValueError("invalid conv_type")

        if self.pool == "seqpool":
            self.attention_pool = nn.Linear(final_emb_dim, 1)
            self.fc = nn.Linear(final_emb_dim, num_classes)
        elif self.pool == "average":
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(final_emb_dim, num_classes)
            )
        elif self.pool == "linear":
            # Hack to fetch dimensions of CNN's output
            # TODO I recall there is a better way to do this, but could not find yet
            input = torch.randn((1, in_channels, max_len))  # [batch, channel, time]. Conv expects this order
            output = self.conv(input)  # [batch, channel, time]
            self.fc = nn.Sequential(
                nn.Flatten(),  # Default converts to [batch, channel*time]
                nn.Linear(output.shape[1]*output.shape[2], num_classes)
            )
        else:
            raise ValueError("invalid pool")

    def forward(self, x, plot_dir=None):
        """
        x should have shape [batch, time, channel]
        """
        x = x.permute((0, 2, 1))  # Convert to [batch, channel, time]
        x = self.conv(x)   # Convert to [batch, channel, time']  (may be fewer timesteps)
        if self.pool == "seqpool":
            x = x.permute((0, 2, 1))  # Convert back to [batch, time, channel]. For each example & timestep, use all channels to predict an attention score

            # Code from https://github.com/SHI-Labs/Compact-Transformers/blob/main/src/utils/transformers.py#L208
            # attention_pool outputs [batch, time, 1].
            # Softmax normalizes it so that sum across the time dimension (for each example) is 1.
            # Transpose it to [batch, 1, time], and then multiply with [batch, time, channel] -> [batch, 1, channel].
            # Squeeze out the 1 to get [batch, channel].
            # TODO add a positional encoding
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
            x = self.fc(x)
        elif self.pool in ["average", "linear"]:
            x = self.fc(x)
        else:
            raise ValueError("Invalid pool")
        return x

