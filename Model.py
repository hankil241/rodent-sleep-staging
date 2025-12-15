import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import math
import copy
from copy import deepcopy


########################################################################################

class CNN1D(Dataset):
    def __init__(self, eeg_data, emg_data, labels):
        # Ensure the inputs have matching dimensions for stacking
        assert eeg_data.shape == emg_data.shape, "EEG and EMG data must have the same shape"
        
        # Store data as PyTorch tensors
        self.eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
        self.emg_data = torch.tensor(emg_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Get the EEG and EMG samples for this index
        eeg_sample = self.eeg_data[idx]
        emg_sample = self.emg_data[idx]
        
        # Stack EEG and EMG to form a two-channel input
        combined_sample = torch.stack((eeg_sample, emg_sample), dim=0)
        
        # Get the label
        label = self.labels[idx]
        
        return combined_sample, label
    
class CNNBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pool_size=None, pool_step=None):
        super(CNNBlock1, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=7)
        # self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_step) if pool_size else None

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        x = F.relu(x)
        if self.pool:
            x = self.pool(x)

        return x
    
class FeatureExtractionCNN(nn.Module):
    def __init__(self):
        super(FeatureExtractionCNN, self).__init__()
        
        # Left Path
        self.left_conv1 = CNNBlock1(2, 64, 5, 3)
        self.left_conv2 = CNNBlock1(64, 128, 5, 1, 2, 2)
        self.left_dropout1 = nn.Dropout(0.2)
        self.left_conv3 = CNNBlock1(128, 128, 13, 1)
        self.left_conv4 = CNNBlock1(128, 256, 7, 1, 2, 2)
        self.left_conv5 = CNNBlock1(256, 256, 7, 1)
        self.left_conv6 = CNNBlock1(256, 64, 4, 1, 2, 2)
        self.left_conv7 = CNNBlock1(64, 32, 3, 1)
        self.left_conv8 = CNNBlock1(32, 64, 6, 1, 2, 2)
        self.left_conv9 = CNNBlock1(64, 8, 5, 1)
        self.left_conv10 = CNNBlock1(8, 8, 2, 1, 2, 2)
        

    def forward(self, x):
        # Left Path
        x_left = self.left_conv1(x)
        x_left = self.left_conv2(x_left)
        x_left = self.left_dropout1(x_left)
        x_left = self.left_conv3(x_left)
        x_left = self.left_conv4(x_left)
        x_left = self.left_conv5(x_left)
        x_left = self.left_conv6(x_left)
        x_left = self.left_conv7(x_left)
        x_left = self.left_conv8(x_left)
        x_left = self.left_conv9(x_left)
        x_left = self.left_conv10(x_left)
        
        return x_left

        return x_emg
    
class CNN1DModel(nn.Module):
    def __init__(self):
        super(CNN1DModel, self).__init__()

        # Feature extraction
        self.feature_extractor = FeatureExtractionCNN()
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(264, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        # Feature extraction
        batch_size, channels, _, time_steps = x.size()
        x = x.view(batch_size, channels, time_steps)  # Flatten stack for CNN processing

        features_left = self.feature_extractor(x)
        cnn_out = self.dropout(features_left)
        cnn_out = cnn_out.view(batch_size, -1)
        
        # Fully connected layer focusing on the target epoch
        fc_out = self.fc1(cnn_out)

        x = self.dropout(fc_out)
        x = torch.softmax(self.fc2(x), dim=1)

        return x
    

########################################################################################


class SSDataset(Dataset):
    def __init__(self, data, labels, transform=None):
      self.data = data
      self.labels = labels
      self.transform = transform

    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        # day_idx = idx // self.data.shape[1]
        # sample_idx = idx % self.data.shape[1]
        sample = self.data[idx]
        label = self.labels[idx]
        # sample = torch.tensor(sample, dtype=torch.float32)
        # label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            sample = self.transform(sample)

        return (sample, label)
    
class SSANN(nn.Module):
    def __init__(self):
        super(SSANN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,1))
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2,1), stride=2)
        self.fc1 = nn.Linear(32 * 5 * 1, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = x.view(-1, 32 * 5 * 1)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x


########################################################################################


class CNNBiLSTM(torch.utils.data.Dataset):
    def __init__(self, eeg_data, emg_data, labels):
        # Ensure the inputs have matching dimensions for stacking
        assert eeg_data.shape == emg_data.shape, "EEG and EMG data must have the same shape"
        
        # Store data as PyTorch tensors
        self.eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
        self.emg_data = torch.tensor(emg_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Get the EEG and EMG samples for this index
        eeg_sample = self.eeg_data[idx]
        emg_sample = self.emg_data[idx]
        
        # Stack EEG and EMG to form a two-channel input
        combined_sample = torch.stack((eeg_sample, emg_sample), dim=0)
        
        # Get the label
        label = self.labels[idx]
        
        return combined_sample, label
    
# Authors: Pierre Guetschel
#          Maciej Sliwowski
#
# License: BSD-3

import warnings
from typing import Dict, Iterable, List, Optional, Tuple

from collections import OrderedDict

import numpy as np
import torch
from docstring_inheritance import NumpyDocstringInheritanceInitMeta
from torchinfo import ModelStatistics, summary


def deprecated_args(obj, *old_new_args):
    out_args = []
    for old_name, new_name, old_val, new_val in old_new_args:
        if old_val is None:
            out_args.append(new_val)
        else:
            warnings.warn(
                f'{obj.__class__.__name__}: {old_name!r} is depreciated. Use {new_name!r} instead.'
            )
            if new_val is not None:
                raise ValueError(
                    f'{obj.__class__.__name__}: Both {old_name!r} and {new_name!r} were specified.'
                )
            out_args.append(old_val)
    return out_args


class EEGModuleMixin(metaclass=NumpyDocstringInheritanceInitMeta):
    """
    Mixin class for all EEG models in braindecode.

    Parameters
    ----------
    n_outputs : int
        Number of outputs of the model. This is the number of classes
        in the case of classification.
    n_chans : int
        Number of EEG channels.
    chs_info : list of dict
        Information about each individual EEG channel. This should be filled with
        ``info["chs"]``. Refer to :class:`mne.Info` for more details.
    n_times : int
        Number of time samples of the input window.
    input_window_seconds : float
        Length of the input window in seconds.
    sfreq : float
        Sampling frequency of the EEG recordings.
    add_log_softmax: bool
        Whether to use log-softmax non-linearity as the output function.
        LogSoftmax final layer will be removed in the future.
        Please adjust your loss function accordingly (e.g. CrossEntropyLoss)!
        Check the documentation of the torch.nn loss functions:
        https://pytorch.org/docs/stable/nn.html#loss-functions.

    Raises
    ------
    ValueError: If some input signal-related parameters are not specified
                and can not be inferred.

    FutureWarning: If add_log_softmax is True, since LogSoftmax final layer
                   will be removed in the future.

    Notes
    -----
    If some input signal-related parameters are not specified,
    there will be an attempt to infer them from the other parameters.
    """

    def __init__(
            self,
            n_outputs: Optional[int] = None,
            n_chans: Optional[int] = None,
            chs_info: Optional[List[Dict]] = None,
            n_times: Optional[int] = None,
            input_window_seconds: Optional[float] = None,
            sfreq: Optional[float] = None,
            add_log_softmax: Optional[bool] = False,
    ):
        if (
                n_chans is not None and
                chs_info is not None and
                len(chs_info) != n_chans
        ):
            raise ValueError(f'{n_chans=} different from {chs_info=} length')
        if (
                n_times is not None and
                input_window_seconds is not None and
                sfreq is not None and
                n_times != int(input_window_seconds * sfreq)
        ):
            raise ValueError(
                f'{n_times=} different from '
                f'{input_window_seconds=} * {sfreq=}'
            )
        self._n_outputs = n_outputs
        self._n_chans = n_chans
        self._chs_info = chs_info
        self._n_times = n_times
        self._input_window_seconds = input_window_seconds
        self._sfreq = sfreq
        self._add_log_softmax = add_log_softmax
        super().__init__()

    @property
    def n_outputs(self):
        if self._n_outputs is None:
            raise ValueError('n_outputs not specified.')
        return self._n_outputs

    @property
    def n_chans(self):
        if self._n_chans is None and self._chs_info is not None:
            return len(self._chs_info)
        elif self._n_chans is None:
            raise ValueError(
                'n_chans could not be inferred. Either specify n_chans or chs_info.'
            )
        return self._n_chans

    @property
    def chs_info(self):
        if self._chs_info is None:
            raise ValueError('chs_info not specified.')
        return self._chs_info

    @property
    def n_times(self):
        if (
                self._n_times is None and
                self._input_window_seconds is not None and
                self._sfreq is not None
        ):
            return int(self._input_window_seconds * self._sfreq)
        elif self._n_times is None:
            raise ValueError(
                'n_times could not be inferred. '
                'Either specify n_times or input_window_seconds and sfreq.'
            )
        return self._n_times

    @property
    def input_window_seconds(self):
        if (
                self._input_window_seconds is None and
                self._n_times is not None and
                self._sfreq is not None
        ):
            return self._n_times / self._sfreq
        elif self._input_window_seconds is None:
            raise ValueError(
                'input_window_seconds could not be inferred. '
                'Either specify input_window_seconds or n_times and sfreq.'
            )
        return self._input_window_seconds

    @property
    def sfreq(self):
        if (
                self._sfreq is None and
                self._input_window_seconds is not None and
                self._n_times is not None
        ):
            return self._n_times / self._input_window_seconds
        elif self._sfreq is None:
            raise ValueError(
                'sfreq could not be inferred. '
                'Either specify sfreq or input_window_seconds and n_times.'
            )
        return self._sfreq

    @property
    def add_log_softmax(self):
        if self._add_log_softmax:
            warnings.warn("LogSoftmax final layer will be removed! " +
                          "Please adjust your loss function accordingly (e.g. CrossEntropyLoss)!")
        return self._add_log_softmax

    @property
    def input_shape(self) -> Tuple[int]:
        """Input data shape."""
        return (1, self.n_chans, self.n_times)

    def get_output_shape(self) -> Tuple[int]:
        """Returns shape of neural network output for batch size equal 1.

        Returns
        -------
        output_shape: Tuple[int]
            shape of the network output for `batch_size==1` (1, ...)
    """
        with torch.inference_mode():
            try:
                return tuple(self.forward(
                    torch.zeros(
                        self.input_shape,
                        dtype=next(self.parameters()).dtype,
                        device=next(self.parameters()).device
                    )).shape)
            except RuntimeError as exc:
                if str(exc).endswith(
                        ("Output size is too small",
                         "Kernel size can't be greater than actual input size")
                ):
                    msg = (
                        "During model prediction RuntimeError was thrown showing that at some "
                        f"layer `{str(exc).split('.')[-1]}` (see above in the stacktrace). This "
                        "could be caused by providing too small `n_times`/`input_window_seconds`. "
                        "Model may require longer chunks of signal in the input than "
                        f"{self.input_shape}."
                    )
                    raise ValueError(msg) from exc
                raise exc

    mapping = None

    def load_state_dict(self, state_dict, *args, **kwargs):

        mapping = self.mapping if self.mapping else {}
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k in mapping:
                new_state_dict[mapping[k]] = v
            else:
                new_state_dict[k] = v

        return super().load_state_dict(new_state_dict, *args, **kwargs)

    def to_dense_prediction_model(self, axis: Tuple[int] = (2, 3)) -> None:
        """
        Transform a sequential model with strides to a model that outputs
        dense predictions by removing the strides and instead inserting dilations.
        Modifies model in-place.

        Parameters
        ----------
        axis: int or (int,int)
            Axis to transform (in terms of intermediate output axes)
            can either be 2, 3, or (2,3).

        Notes
        -----
        Does not yet work correctly for average pooling.
        Prior to version 0.1.7, there had been a bug that could move strides
        backwards one layer.

        """
        if not hasattr(axis, "__len__"):
            axis = [axis]
        assert all([ax in [2, 3] for ax in axis]), "Only 2 and 3 allowed for axis"
        axis = np.array(axis) - 2
        stride_so_far = np.array([1, 1])
        for module in self.modules():
            if hasattr(module, "dilation"):
                assert module.dilation == 1 or (module.dilation == (1, 1)), (
                    "Dilation should equal 1 before conversion, maybe the model is "
                    "already converted?"
                )
                new_dilation = [1, 1]
                for ax in axis:
                    new_dilation[ax] = int(stride_so_far[ax])
                module.dilation = tuple(new_dilation)
            if hasattr(module, "stride"):
                if not hasattr(module.stride, "__len__"):
                    module.stride = (module.stride, module.stride)
                stride_so_far *= np.array(module.stride)
                new_stride = list(module.stride)
                for ax in axis:
                    new_stride[ax] = 1
                module.stride = tuple(new_stride)

    def get_torchinfo_statistics(
            self,
            col_names: Optional[Iterable[str]] = (
                    "input_size",
                    "output_size",
                    "num_params",
                    "kernel_size",
            ),
            row_settings: Optional[Iterable[str]] = ("var_names", "depth"),
    ) -> ModelStatistics:
        """Generate table describing the model using torchinfo.summary.

        Parameters
        ----------
        col_names : tuple, optional
            Specify which columns to show in the output, see torchinfo for details, by default
            ("input_size", "output_size", "num_params", "kernel_size")
        row_settings : tuple, optional
             Specify which features to show in a row, see torchinfo for details, by default
             ("var_names", "depth")

        Returns
        -------
        torchinfo.ModelStatistics
            ModelStatistics generated by torchinfo.summary.
        """
        return summary(
            self,
            input_size=(1, self.n_chans, self.n_times),
            col_names=col_names,
            row_settings=row_settings,
            verbose=0,
        )

    def __str__(self) -> str:
        return str(self.get_torchinfo_statistics())

class _SmallCNN(nn.Module):  # smaller filter sizes to learn temporal information
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=64,
                kernel_size=(1, 50),
                stride=(1, 6),
                padding=(0, 22),
                bias=False,
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 8), stride=(1, 8), padding=(0, 2))
        self.dropout = nn.Dropout(p=0.5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(1, 8),
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(1, 8),
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(1, 8),
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4), padding=(0, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(self.pool1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        return x


class _LargeCNN(nn.Module):  # larger filter sizes to learn frequency information
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=64,
                kernel_size=(1, 400),
                stride=(1, 50),
                padding=(0, 175),
                bias=False,
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.dropout = nn.Dropout(p=0.5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(1, 6),
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(1, 6),
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(1, 6),
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(self.pool1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        return x


class _BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(_BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
        )

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(
            self.num_layers * 2, x.size(0), self.hidden_size
        ).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        return out


class DeepSleepNet(EEGModuleMixin, nn.Module):
    """Sleep staging architecture from Supratak et al 2017.

    Convolutional neural network and bidirectional-Long Short-Term
    for single channels sleep staging described in [Supratak2017]_.

    Parameters
    ----------
    return_feats : bool
        If True, return the features, i.e. the output of the feature extractor
        (before the final linear layer). If False, pass the features through
        the final linear layer.
    n_classes :
        Alias for n_outputs.

    References
    ----------
    .. [Supratak2017] Supratak, A., Dong, H., Wu, C., & Guo, Y. (2017).
       DeepSleepNet: A model for automatic sleep stage scoring based
       on raw single-channel EEG. IEEE Transactions on Neural Systems
       and Rehabilitation Engineering, 25(11), 1998-2008.
    """

    def __init__(
            self,
            n_outputs=5,
            return_feats=False,
            n_chans=None,
            chs_info=None,
            n_times=None,
            input_window_seconds=None,
            sfreq=None,
            n_classes=None,
    ):
        n_outputs, = deprecated_args(
            self,
            ('n_classes', 'n_outputs', n_classes, n_outputs),
        )
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        del n_classes
        self.cnn1 = _SmallCNN()
        self.cnn2 = _LargeCNN()
        self.dropout = nn.Dropout(0.5)
        self.bilstm = _BiLSTM(input_size=1408, hidden_size=512, num_layers=2)
        self.fc = nn.Sequential(nn.Linear(1408, 1024, bias=False),
                                nn.BatchNorm1d(num_features=1024))

        self.features_extractor = nn.Identity()
        self.len_last_layer = 1024
        self.return_feats = return_feats

        # TODO: Add new way to handle return_features == True
        if not return_feats:
            self.final_layer = nn.Linear(1024, self.n_outputs)
        else:
            self.final_layer = nn.Identity()

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """

        if x.ndim == 3:
            x = x.unsqueeze(1)

        x1 = self.cnn1(x)
        x1 = x1.flatten(start_dim=1)

        x2 = self.cnn2(x)
        x2 = x2.flatten(start_dim=1)

        x = torch.cat((x1, x2), dim=1)
        x = self.dropout(x)
        temp = x.clone()
        temp = self.fc(temp)
        x = x.unsqueeze(1)
        x = self.bilstm(x)
        x = x.squeeze()
        x = torch.add(x, temp)
        x = self.dropout(x)

        feats = self.features_extractor(x)

        if self.return_feats:
            return feats
        else:
            return self.final_layer(feats)



########################################################################################


class SlumberDataset(Dataset):
    def __init__(self, eeg_data, emg_data, labels, transform=None):
        self.eeg_data = eeg_data
        self.emg_data = emg_data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.eeg_data.shape[0]

    def __getitem__(self, idx):
        # Get the EEG and EMG samples
        eeg_sample = self.eeg_data[idx]
        emg_sample = self.emg_data[idx]
        label = self.labels[idx]

        # Convert numpy arrays to tensors
        eeg_sample = torch.tensor(eeg_sample, dtype=torch.float32)
        emg_sample = torch.tensor(emg_sample, dtype=torch.float32)

        if self.transform:
            eeg_sample = self.transform(eeg_sample)
            emg_sample = self.transform(emg_sample)

        # Stack EEG and EMG along the second dimension (axis=1) and add a channel dimension
        combined_sample = torch.stack([eeg_sample, emg_sample], dim=1)

        return (combined_sample, label)
    
class ResNetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_y, kernel_expansion_fct, strides, dropout_rate=0.0):
        super(ResNetBlock2D, self).__init__()
        
        # First convolutional block
        self.conv_x = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_y, 8*kernel_expansion_fct), stride=strides, padding='same')
        self.bn_x = nn.BatchNorm2d(out_channels)
        self.dropout_x = nn.Dropout2d(dropout_rate)
        
        # Second convolutional block
        self.conv_y = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_y, 5*kernel_expansion_fct), stride=strides, padding='same')
        self.bn_y = nn.BatchNorm2d(out_channels)
        self.dropout_y = nn.Dropout2d(dropout_rate)
        
        # Third convolutional block
        self.conv_z = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_y, 3*kernel_expansion_fct), stride=strides, padding='same')
        self.bn_z = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection for residual
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_y, 1), stride=strides, padding='same')
        self.bn_shortcut = nn.BatchNorm2d(out_channels)
        self.dropout_last = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        # First layer
        out = F.relu(self.dropout_x(self.bn_x(self.conv_x(x))))
        
        # Second layer
        out = F.relu(self.dropout_y(self.bn_y(self.conv_y(out))))
        
        # Third layer
        out = self.bn_z(self.conv_z(out))
        
        # Shortcut connection
        shortcut = self.bn_shortcut(self.shortcut(x))
        
        # Adding the shortcut to the output
        out += shortcut
        out = F.relu(self.dropout_last(out))
        
        return out
    
class ResNet2D(nn.Module):
    def __init__(self, input_shape, n_resnet_blocks, n_feature_maps, kernel_expansion_fct, kernel_y, strides, dropout_rate=0.0, nb_classes=3):
        super(ResNet2D, self).__init__()
        self.input_channels = input_shape[-1]
        
        # Create the ResNet blocks
        self.resnet_blocks = nn.ModuleList()
        for i in range(n_resnet_blocks - 1):
            self.resnet_blocks.append(ResNetBlock2D(self.input_channels if i == 0 else n_feature_maps * (2 ** (i-1)), 
                                                    n_feature_maps * (2 ** i), kernel_y, kernel_expansion_fct, strides, dropout_rate))
        
        # Final block (no shortcut needed)
        self.final_conv_x = nn.Conv2d(n_feature_maps * (2 ** (n_resnet_blocks - 2)), 
                                      n_feature_maps * (2 ** (n_resnet_blocks - 2)), 
                                      kernel_size=(kernel_y, 8*kernel_expansion_fct), 
                                      stride=strides, 
                                      padding='same')
        self.bn_final_x = nn.BatchNorm2d(n_feature_maps * (2 ** (n_resnet_blocks - 2)))
        
        self.final_conv_y = nn.Conv2d(n_feature_maps * (2 ** (n_resnet_blocks - 2)), 
                                      n_feature_maps * (2 ** (n_resnet_blocks - 2)), 
                                      kernel_size=(kernel_y, 8*kernel_expansion_fct), 
                                      stride=strides, 
                                      padding='same')
        self.bn_final_y = nn.BatchNorm2d(n_feature_maps * (2 ** (n_resnet_blocks - 2)))
        
        self.final_conv_z = nn.Conv2d(n_feature_maps * (2 ** (n_resnet_blocks - 2)), 
                                      n_feature_maps * (2 ** (n_resnet_blocks - 2)), 
                                      kernel_size=(kernel_y, 8*kernel_expansion_fct), 
                                      stride=strides, 
                                      padding='same')
        self.bn_final_z = nn.BatchNorm2d(n_feature_maps * (2 ** (n_resnet_blocks - 2)))

        self.bn_final_shortcut = nn.BatchNorm2d(n_feature_maps * (2 ** (n_resnet_blocks - 2)))
        
        self.dropout_final = nn.Dropout2d(dropout_rate)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(n_feature_maps * (2 ** (n_resnet_blocks - 2)), nb_classes)
    
    def forward(self, x):
        # Pass through ResNet blocks
        for block in self.resnet_blocks:
            x = block(x)
        
        # Final block without shortcut
        out = F.relu(self.dropout_final(self.bn_final_x(self.final_conv_x(x))))
        out = F.relu(self.dropout_final(self.bn_final_y(self.final_conv_y(x))))
        out = self.bn_final_z(self.final_conv_z(x))
        
        # Shortcut
        shortcut_out = self.bn_final_shortcut(x)

        out += shortcut_out

        # Global average pooling (instead of Flatten)
        x = F.relu(self.dropout_final(out))
        x = F.adaptive_avg_pool2d(out, (1, 1))
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.fc(x)
        
        # Softmax for classification
        x = F.softmax(x, dim=1)
        
        return x


########################################################################################


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, classes_weights, device):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.cr = nn.CrossEntropyLoss(weight=torch.tensor(classes_weights, dtype=torch.float32).to(device))

    def forward(self, output, target):
        return self.cr(output, target)

class AttnDataset(Dataset):
    def __init__(self, eeg_data, emg_data, labels, transform=None):
        self.eeg_data = eeg_data
        self.emg_data = emg_data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.eeg_data.shape[0]

    def __getitem__(self, idx):
        # Get the EEG and EMG samples
        eeg_sample = self.eeg_data[idx]
        emg_sample = self.emg_data[idx]
        label = self.labels[idx]

        # Convert numpy arrays to tensors
        eeg_sample = torch.tensor(eeg_sample, dtype=torch.float32)
        emg_sample = torch.tensor(emg_sample, dtype=torch.float32)

        if self.transform:
            eeg_sample = self.transform(eeg_sample)
            emg_sample = self.transform(emg_sample)

        # Stack EEG and EMG along the second dimension (axis=1) and add a channel dimension
        combined_sample = torch.stack([eeg_sample, emg_sample], dim=1)

        return (eeg_sample, label)


########################################################################################


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class GELU(nn.Module):
    # for older versions of PyTorch.  For new versions you can use nn.GELU() instead.
    def __init__(self):
        super(GELU, self).__init__()
        
    def forward(self, x):
        x = torch.nn.functional.gelu(x)
        return x
        
        
class MRCNN(nn.Module):
    def __init__(self, afr_reduced_cnn_size):
        super(MRCNN, self).__init__()
        drate = 0.5
        self.GELU = GELU()  # for older versions of PyTorch.  For new versions use nn.GELU() instead.
        self.features1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )

        self.features2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, 1)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat)
        x_concat = self.AFR(x_concat)
        return x_concat

##########################################################################################


def attention(query, key, value, dropout=None):
    "Implementation of Scaled dot product attention"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, afr_reduced_cnn_size, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.convs = clones(CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7, stride=1), 3)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        "Implements Multi-head attention"
        nbatches = query.size(0)

        query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key   = self.convs[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.convs[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        x, self.attn = attention(query, key, value, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linear(x)


class LayerNorm(nn.Module):
    "Construct a layer normalization module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerOutput(nn.Module):
    '''
    A residual connection followed by a layer norm.
    '''

    def __init__(self, size, dropout):
        super(SublayerOutput, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TCE(nn.Module):
    '''
    Transformer Encoder

    It is a stack of N layers.
    '''

    def __init__(self, layer, N):
        super(TCE, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class EncoderLayer(nn.Module):
    '''
    An encoder layer

    Made up of self-attention and a feed forward layer.
    Each of these sublayers have residual and layer norm, implemented by SublayerOutput.
    '''
    def __init__(self, size, self_attn, feed_forward, afr_reduced_cnn_size, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_output = clones(SublayerOutput(size, dropout), 2)
        self.size = size
        self.conv = CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7, stride=1, dilation=1)


    def forward(self, x_in):
        "Transformer Encoder"
        query = self.conv(x_in)
        x = self.sublayer_output[0](query, lambda x: self.self_attn(query, x_in, x_in))  # Encoder self-attention
        return self.sublayer_output[1](x, self.feed_forward)


class PositionwiseFeedForward(nn.Module):
    "Positionwise feed-forward network."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        "Implements FFN equation."
        return self.w_2(self.dropout(F.relu(self.w_1(x))))



class AttnSleep(nn.Module):
    def __init__(self):
        super(AttnSleep, self).__init__()

        N = 2  # number of TCE clones
        d_model = 36  # set to be 100 for SHHS dataset
        d_ff = 120   # dimension of feed forward
        h = 4  # number of attention heads
        dropout = 0.1
        num_classes = 3
        afr_reduced_cnn_size = 30

        self.mrcnn = MRCNN(afr_reduced_cnn_size) # use MRCNN_SHHS for SHHS dataset

        attn = MultiHeadedAttention(h, d_model, afr_reduced_cnn_size)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.tce = TCE(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size, dropout), N)

        self.fc = nn.Linear(d_model * afr_reduced_cnn_size, num_classes)

    def forward(self, x):
        x_feat = self.mrcnn(x)
        encoded_features = self.tce(x_feat)
        encoded_features = encoded_features.contiguous().view(encoded_features.shape[0], -1)
        final_output = self.fc(encoded_features)
        return final_output

######################################################################

class MRCNN_SHHS(nn.Module):
    def __init__(self, afr_reduced_cnn_size):
        super(MRCNN_SHHS, self).__init__()
        drate = 0.5
        self.GELU = GELU()  # for older versions of PyTorch.  For new versions use nn.GELU() instead.
        self.features1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=512/2, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )

        self.features2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=512*4, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=6, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=6, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, 1)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat)
        x_concat = self.AFR(x_concat)
        return x_concat