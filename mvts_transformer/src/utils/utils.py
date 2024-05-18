import json
import os
import sys
import builtins
import functools
import time
import ipdb
from copy import deepcopy

import numpy as np
import torch
import xlrd
import xlwt
from xlutils.copy import copy
from sklearn.neighbors import KernelDensity
from dtaidistance import dtw

import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

############################################################################################
# C-Mixup code taken from https://github.com/huaxiuyao/C-Mixup/blob/main/src/algorithm.py
############################################################################################

def stats_values(targets):  # from utils.py file in C-Mixup repo
    mean = np.mean(targets)
    min = np.min(targets)
    max = np.max(targets)
    std = np.std(targets)
    print(f'y stats: mean = {mean}, max = {max}, min = {min}, std = {std}')
    return mean, min, max, std


def get_mixup_sample_rate(args, data_packet, device='cuda', use_kde = False):

    mix_idx = []
    x_list, y_list = data_packet['x_train'], data_packet['y_train']
    is_np = isinstance(y_list,np.ndarray)
    if is_np:
        data_list = torch.tensor(y_list, dtype=torch.float32)
    else:
        data_list = y_list

    N = len(data_list)

    ######## use kde rate or uniform rate #######
    if args["mixtype"] == "dtw":
        x_numpy = x_list.cpu().detach().numpy().astype(np.float64)  # [batch, timesteps, channel]

        # For each channel, compute DTW distance between each pair of datapoints
        distance_matrices = []
        logging.disable(logging.INFO)
        for i in range(x_numpy.shape[2]):  # Loop through channels
            distance_matrix = dtw.distance_matrix_fast(x_numpy[:, :, i])  # [batch, batch]
            distance_matrices.append(distance_matrix)
        logging.disable(logging.NOTSET)
        distance_matrix = np.stack(distance_matrices, axis=0).mean(axis=0)  # [batch, batch]
        # print("AVG Distance matrix", distance_matrix[0:2, :])

        # Compute distance to similarity, and normalize
        each_rate = np.exp(-distance_matrix / (2 * (args["kde_bandwidth"]**2)))  # similarity to all other datapoints
        # np.fill_diagonal(each_rate, 0)  # Set similarity to oneself as 0
        # print("Unnormalized similarity", each_rate[0:2, :])
        each_rate /= np.sum(each_rate, axis=1, keepdims=True)
        # print("Normalized", each_rate[0:2, :])
        # print("Row sums", each_rate.sum(axis=1))
        mix_idx = each_rate
    else:
        for i in range(N):
            if args["mixtype"] == 'kde' or use_kde: # kde
                data_i = data_list[i]
                data_i = data_i.reshape(-1,data_i.shape[0]) # get 2D

                if args["show_process"]:
                    if i % (N // 10) == 0:
                        print('Mixup sample prepare {:.2f}%'.format(i * 100.0 / N ))
                    # if i == 0: print(f'data_list.shape = {data_list.shape}, std(data_list) = {torch.std(data_list)}')#, data_i = {data_i}' + f'data_i.shape = {data_i.shape}')

                ######### get kde sample rate ##########
                kd = KernelDensity(kernel=args["kde_type"], bandwidth=args["kde_bandwidth"]).fit(data_i)  # should be 2D
                each_rate = np.exp(kd.score_samples(data_list))
                each_rate /= np.sum(each_rate)  # norm

            else:
                each_rate = np.ones(y_list.shape[0]) * 1.0 / y_list.shape[0]

            ####### visualization: observe relative rate distribution shot #######
            if args["show_process"] and i == 0:
                    print(f'bw = {args["kde_bandwidth"]}')
                    print(f'each_rate[:10] = {each_rate[:10]}')
                    stats_values(each_rate)

            mix_idx.append(each_rate)

    mix_idx = np.array(mix_idx)
    # print("mixup_sample_rate", mix_idx.shape)
    # print(mix_idx[0])
    # print(y_list)

    self_rate = [mix_idx[i][i] for i in range(len(mix_idx))]

    if args["show_process"]:
        print(f'len(y_list) = {len(y_list)}, len(mix_idx) = {len(mix_idx)}, np.mean(self_rate) = {np.mean(self_rate)}, np.std(self_rate) = {np.std(self_rate)},  np.min(self_rate) = {np.min(self_rate)}, np.max(self_rate) = {np.max(self_rate)}')

    return mix_idx



def get_batch_kde_mixup_idx(args, Batch_X, Batch_Y, device):
    # assert Batch_X.shape[0] % 2 == 0
    Batch_packet = {}
    Batch_packet['x_train'] = Batch_X.cpu()
    Batch_packet['y_train'] = Batch_Y.cpu()

    Batch_rate = get_mixup_sample_rate(args, Batch_packet, device, use_kde=True) # batch -> kde
    if args["show_process"]:
        stats_values(Batch_rate[0])
        # print(f'Batch_rate[0][:20] = {Batch_rate[0][:20]}')
    idx2 = [np.random.choice(np.arange(Batch_X.shape[0]), p=Batch_rate[sel_idx])
            for sel_idx in np.arange(Batch_X.shape[0])]  # @joshuafan changed, used to be Batch_X.shape // 2
    return idx2

def generate_mixup_data(args, X1, Y1, device):
    lambd = np.random.beta(args["mix_alpha"], args["mix_alpha"])
    idx2 = get_batch_kde_mixup_idx(args, X1, Y1, device)
    X2 = X1[idx2]
    Y2 = Y1[idx2]
    mixup_X = X1 * lambd + X2 * (1 - lambd)
    mixup_Y = Y1 * lambd + Y2 * (1 - lambd)
    return mixup_X, mixup_Y

############################################################################################
# End of C-Mixup code
############################################################################################



def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"Finished {func.__name__!r} in {run_time} secs")
        return value

    return wrapper_timer


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {"epoch": epoch, "state_dict": state_dict}
    if not (optimizer is None):
        data["optimizer"] = optimizer.state_dict()
    torch.save(data, path)


def load_model(
    model,
    model_path,
    optimizer=None,
    resume=False,
    change_output=False,
    lr=None,
    lr_step=None,
    lr_factor=None,
):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict = deepcopy(checkpoint["state_dict"])
    if change_output:
        for key, val in checkpoint["state_dict"].items():
            if key.startswith("output_layer"):
                state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    print("Loaded model from {}. Epoch: {}".format(model_path, checkpoint["epoch"]))

    # resume optimizer parameters
    if optimizer is not None and resume:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"]
            start_lr = lr
            for i in range(len(lr_step)):
                if start_epoch >= lr_step[i]:
                    start_lr *= lr_factor[i]
            for param_group in optimizer.param_groups:
                param_group["lr"] = start_lr
            print("Resumed optimizer with start lr", start_lr)
        else:
            print("No optimizer parameters in checkpoint.")
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def load_config(config_filepath):
    """
    Using a json file with the master configuration (config file for each part of the pipeline),
    return a dictionary containing the entire configuration settings in a hierarchical fashion.
    """

    with open(config_filepath) as cnfg:
        config = json.load(cnfg)

    return config


def create_dirs(dirs):
    """
    Input:
        dirs: a list of directories to create, in case these directories are not found
    Returns:
        exit_code: 0 if success, -1 if failure
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def export_performance_metrics(
    filepath, metrics_table, header, book=None, sheet_name="metrics"
):
    """Exports performance metrics on the validation set for all epochs to an excel file"""

    if book is None:
        book = xlwt.Workbook()  # new excel work book

    book = write_table_to_sheet([header] + metrics_table, book, sheet_name=sheet_name)

    book.save(filepath)
    logger.info("Exported per epoch performance metrics in '{}'".format(filepath))

    return book


def write_row(sheet, row_ind, data_list):
    """Write a list to row_ind row of an excel sheet"""

    row = sheet.row(row_ind)
    for col_ind, col_value in enumerate(data_list):
        row.write(col_ind, col_value)
    return


def write_table_to_sheet(table, work_book, sheet_name=None):
    """Writes a table implemented as a list of lists to an excel sheet in the given work book object"""

    sheet = work_book.add_sheet(sheet_name)

    for row_ind, row_list in enumerate(table):
        write_row(sheet, row_ind, row_list)

    return work_book


def export_record(filepath, values):
    """Adds a list of values as a bottom row of a table in a given excel file"""

    read_book = xlrd.open_workbook(filepath, formatting_info=True)
    read_sheet = read_book.sheet_by_index(0)
    last_row = read_sheet.nrows

    work_book = copy(read_book)
    sheet = work_book.get_sheet(0)
    write_row(sheet, last_row, values)
    work_book.save(filepath)


def register_record(
    filepath, timestamp, experiment_name, best_metrics, final_metrics=None, test_metrics=None, comment=""
):
    """
    Adds the best and final metrics of a given experiment as a record in an excel sheet with other experiment records.
    Creates excel sheet if it doesn't exist.
    Args:
        filepath: path of excel file keeping records
        timestamp: string
        experiment_name: string
        best_metrics: dict of metrics at best epoch {metric_name: metric_value}. Includes "epoch" as first key
        final_metrics: dict of metrics at final epoch {metric_name: metric_value}. Includes "epoch" as first key
        test_metrics: dict of TEST metrics at final epoch {metric_name: metric_value}. Includes "epoch" as first key
        comment: optional description
    """
    metrics_names, metrics_values = zip(*best_metrics.items())
    row_values = [timestamp, experiment_name, comment] + list(metrics_values)
    if final_metrics is not None:
        final_metrics_names, final_metrics_values = zip(*final_metrics.items())
        row_values += list(final_metrics_values)
    if test_metrics is not None:
        test_metrics_names, test_metrics_values = zip(*test_metrics.items())
        row_values += list(test_metrics_values)
    if not os.path.exists(filepath):  # Create a records file for the first time
        logger.warning(
            "Records file '{}' does not exist! Creating new file ...".format(filepath)
        )
        directory = os.path.dirname(filepath)
        if len(directory) and not os.path.exists(directory):
            os.makedirs(directory)
        header = ["Timestamp", "Name", "Comment"] + ["Best " + m for m in metrics_names]
        if final_metrics is not None:
            header += ["Final " + m for m in final_metrics_names]
        if test_metrics is not None:
            header += ["Test " + m for m in test_metrics_names]
        book = xlwt.Workbook()  # excel work book
        book = write_table_to_sheet([header, row_values], book, sheet_name="records")
        book.save(filepath)
    else:
        try:
            export_record(filepath, row_values)
        except Exception as x:
            alt_path = os.path.join(
                os.path.dirname(filepath), "record_" + experiment_name
            )
            logger.error(
                "Failed saving in: '{}'! Will save here instead: {}".format(
                    filepath, alt_path
                )
            )
            export_record(alt_path, row_values)
            filepath = alt_path

    logger.info("Exported performance record to '{}'".format(filepath))


class Printer(object):
    """Class for printing output by refreshing the same line in the console, e.g. for indicating progress of a process"""

    def __init__(self, console=True):
        if console:
            self.print = self.dyn_print
        else:
            self.print = builtins.print

    @staticmethod
    def dyn_print(data):
        """Print things to stdout on one line, refreshing it dynamically"""
        sys.stdout.write("\r\x1b[K" + data.__str__())
        sys.stdout.flush()


def readable_time(time_difference):
    """Convert a float measuring time difference in seconds into a tuple of (hours, minutes, seconds)"""

    hours = time_difference // 3600
    minutes = (time_difference // 60) % 60
    seconds = time_difference % 60

    return hours, minutes, seconds


# def check_model1(model, verbose=False, stop_on_error=False):
#     status_ok = True
#     for name, param in model.named_parameters():
#         nan_grads = torch.isnan(param.grad)
#         nan_params = torch.isnan(param)
#         if nan_grads.any() or nan_params.any():
#             status_ok = False
#             print("Param {}: {}/{} nan".format(name, torch.sum(nan_params), param.numel()))
#             if verbose:
#                 print(param)
#             print("Grad {}: {}/{} nan".format(name, torch.sum(nan_grads), param.grad.numel()))
#             if verbose:
#                 print(param.grad)
#             if stop_on_error:
#                 ipdb.set_trace()
#     if status_ok:
#         print("Model Check: OK")
#     else:
#         print("Model Check: PROBLEM")


def check_model(
    model, verbose=False, zero_thresh=1e-8, inf_thresh=1e6, stop_on_error=False
):
    status_ok = True
    for name, param in model.named_parameters():
        param_ok = check_tensor(
            param, verbose=verbose, zero_thresh=zero_thresh, inf_thresh=inf_thresh
        )
        if not param_ok:
            status_ok = False
            print("Parameter '{}' PROBLEM".format(name))
        grad_ok = True
        if param.grad is not None:
            grad_ok = check_tensor(
                param.grad,
                verbose=verbose,
                zero_thresh=zero_thresh,
                inf_thresh=inf_thresh,
            )
        if not grad_ok:
            status_ok = False
            print("Gradient of parameter '{}' PROBLEM".format(name))
        if stop_on_error and not (param_ok and grad_ok):
            ipdb.set_trace()

    if status_ok:
        print("Model Check: OK")
    else:
        print("Model Check: PROBLEM")


def check_tensor(X, verbose=True, zero_thresh=1e-8, inf_thresh=1e6):
    is_nan = torch.isnan(X)
    if is_nan.any():
        print("{}/{} nan".format(torch.sum(is_nan), X.numel()))
        return False

    num_small = torch.sum(torch.abs(X) < zero_thresh)
    num_large = torch.sum(torch.abs(X) > inf_thresh)

    if verbose:
        print("Shape: {}, {} elements".format(X.shape, X.numel()))
        print("No 'nan' values")
        print("Min: {}".format(torch.min(X)))
        print("Median: {}".format(torch.median(X)))
        print("Max: {}".format(torch.max(X)))

        print("Histogram of values:")
        values = X.view(-1).detach().numpy()
        hist, binedges = np.histogram(values, bins=20)
        for b in range(len(binedges) - 1):
            print("[{}, {}): {}".format(binedges[b], binedges[b + 1], hist[b]))

        print("{}/{} abs. values < {}".format(num_small, X.numel(), zero_thresh))
        print("{}/{} abs. values > {}".format(num_large, X.numel(), inf_thresh))

    if num_large:
        print("{}/{} abs. values > {}".format(num_large, X.numel(), inf_thresh))
        return False

    return True


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def recursively_hook(model, hook_fn):
    for name, module in model.named_children():  # model._modules.items():
        if len(list(module.children())) > 0:  # if not leaf node
            for submodule in module.children():
                recursively_hook(submodule, hook_fn)
        else:
            module.register_forward_hook(hook_fn)


def compute_loss(
    net: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device = "cpu",
) -> torch.Tensor:
    """Compute the loss of a network on a given dataset.

    Does not compute gradient.

    Parameters
    ----------
    net:
        Network to evaluate.
    dataloader:
        Iterator on the dataset.
    loss_function:
        Loss function to compute.
    device:
        Torch device, or :py:class:`str`.

    Returns
    -------
    Loss as a tensor with no grad.
    """
    running_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            netout = net(x.to(device)).cpu()
            running_loss += loss_function(y, netout)

    return running_loss / len(dataloader)


# from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
