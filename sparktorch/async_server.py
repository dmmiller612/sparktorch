"""
The MIT License

Copyright 2019 Derek Miller

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from sparktorch.util import handle_features, load_torch_model, DataObj
from sparktorch.early_stopper import EarlyStopping
from pyspark.rdd import RDD
from typing import Dict, List
from uuid import uuid4
import numpy as np
from pyspark.rdd import PipelinedRDD
import torch
from torch.tensor import Tensor
import torch.distributed as dist
from torch.multiprocessing import Process
import os
from datetime import timedelta


def mapPartitionsWithIndex(rdd, f, preservesPartitioning=False):
    """
    Temporary function for barrier map partitions.
    """
    return PipelinedRDD(rdd, f, preservesPartitioning, isFromBarrier=True)


def process_generic_model(params: List[Tensor], iters: int, should_stop: Tensor, has_early_stop: bool = False):
    """
    Runs a mock training with zero grads. This is due to a bug where the connection gets reset with custom new groups.
    :param params: The params of the model
    :param iters: Iterations.
    """
    # Hopefully this function can go away in newer versions.
    for i in range(iters):
        for p in params:
            z = torch.zeros_like(p)
            dist.all_reduce(z, op=torch.distributed.ReduceOp.SUM)

        if has_early_stop:
            dist.all_reduce(should_stop, op=torch.distributed.ReduceOp.SUM)
            if should_stop.item() > 0:
                break
        else:
            print("STOPPING")


def handle_model(
    index: int,
    data: List[DataObj],
    torch_obj: str,
    master_url: str = '127.0.0.1',
    iters: int = 1000,
    world_size: int = 2,
    early_stop_patience: int = -1,
    verbose: int = 1,
    mini_batch: int = -1,
    validation_pct: float = 0,
    device: str = 'cpu'
) -> List[Dict]:
    """
    Runs the training of pytorch model, utilizing the distributed package.

    :param index: Partition index. Used for registering.
    :param data: The data from the partition
    :param torch_obj: The torch object string. Needs serialized
    :param master_url: The master url for the service.
    :param iters: The iterations for training
    :param world_size: The amount of partitions. Typically partitions + 1 for the driver
    :param verbose: whether to log the loss or not.
    :param mini_batch: Mini batch for training
    :param validation_pct: Validation percentage.
    :param device: The pytorch device to use for training. cpu/cuda
    :param early_stop_patience: Amount of patient for early stopping. -1 means don't use early stopping.

    :return: A list of the model state dictionary.
    """

    # Def Load model
    torch_obj = load_torch_model(torch_obj)
    model = torch_obj.model.to(device)
    model.train()

    criterion = torch_obj.criterion
    optimizer = torch_obj.optimizer

    # Initialize zero params for -1 index to fake train.
    params = [torch.zeros_like(p) for p in model.parameters()]

    if dist.is_initialized():
        dist.destroy_process_group()

    # Set up the distributed server.
    os.environ['MASTER_ADDR'] = master_url
    os.environ['MASTER_PORT'] = '5000'
    dist.init_process_group('gloo', rank=index + 1, world_size=world_size, timeout=timedelta(seconds=60))

    es = EarlyStopping(patience=early_stop_patience)
    should_stop = torch.zeros(1)
    has_early_stop = early_stop_patience > 0

    partition_id = str(uuid4())

    # If data is none, we still need to mock it out, since it is apart of the world.
    if data is None:
        process_generic_model(params, iters, should_stop, has_early_stop)
        return []

    # Process the data. Converts to x_train, y_train, x_val, y_val
    data_obj = handle_features(data, validation_pct)
    if data_obj.x_train is None:
        process_generic_model(params, iters, should_stop, has_early_stop)
        return []

    x_train = data_obj.x_train.to(device)
    y_train = data_obj.y_train.to(device) if data_obj.y_train is not None else x_train
    x_val = data_obj.x_val.to(device) if data_obj.x_val is not None else None
    y_val = data_obj.y_val.to(device) if data_obj.y_val is not None else x_val

    for i in range(iters):

        optimizer.zero_grad()

        # utilize minibatch
        if 0 < mini_batch < len(data_obj.x_train):
            idxs = np.random.choice(len(data_obj.x_train), mini_batch, replace=False).tolist()
            x_train = data_obj.x_train[idxs]
            y_train = data_obj.y_train[idxs]

        y_pred = model(x_train)

        try:
            loss = criterion(y_pred, y_train)
        except RuntimeError as e:
            # utilized when loss need a long label
            y_train = torch.flatten(y_train.long())
            loss = criterion(y_pred, y_train)

        loss_v = loss.item()

        # Process validation loss
        val_loss = None
        if x_val is not None:
            pred_val = model(x_val)

            try:
                val_loss = criterion(pred_val, y_val)
            except RuntimeError as e:
                y_val = torch.flatten(y_val.long())
                val_loss = criterion(pred_val, y_val)

            val_loss = val_loss.item()

        # Calculate gradients
        loss.backward()

        # Distributed part of training.
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
            param.grad.data /= (world_size-1)

        if has_early_stop:
            loss_to_use = val_loss if val_loss is not None else loss_v
            stop = es.step(loss_to_use)
            if stop:
                should_stop = should_stop + 1.0

            dist.all_reduce(should_stop, op=torch.distributed.ReduceOp.SUM)
            if should_stop.item() > 0:
                break

        optimizer.step()

        if verbose:
            print(f"Partition: {partition_id}. Iteration: {i}. Loss: {loss_v}, Val Loss: {val_loss}")

    return [model.state_dict()]


def train_async(
    rdd: RDD,
    torch_obj: str,
    master_url: str,
    iters: int = 10,
    partition_shuffles: int = 1,
    verbose: int = 1,
    mini_batch: int = -1,
    validation_pct: float = 0.0,
    world_size: int = 2,
    device: str = 'cpu',
    early_stop_patience: int = -1
) -> Dict:
    """
    Entry point to asynchronously train the model.

    :param rdd: The rdd of data to run on the model.
    :param torch_obj: The torch object as a string that includes the model.
    :param master_url: The main url for the driver.
    :param iters: Number of iterations for training.
    :param partition_shuffles: Number of partition shuffles (Need to implement)
    :param verbose: Verbosity of logs
    :param mini_batch: Mini batch for each iteration of training.
    :param validation_pct: How many items to validate
    :param world_size: number of partitions.
    :param device: pytorch device
    :return: The train dict.
    """

    # Start the driver process.
    p = Process(target=handle_model,
                args=(-1, None, torch_obj, master_url, iters, world_size, early_stop_patience))
    p.start()

    try:
        state_dict = None

        for i in range(partition_shuffles):
            # Run model with barrier execution mode.
            state_dict = mapPartitionsWithIndex(
                rdd, lambda i, x: handle_model(
                    i,
                    x,
                    torch_obj=torch_obj,
                    master_url=master_url,
                    iters=iters,
                    verbose=verbose,
                    mini_batch=mini_batch,
                    validation_pct=validation_pct,
                    world_size=world_size,
                    device=device,
                    early_stop_patience=int(early_stop_patience+0)
                )
            ).collect()

            if partition_shuffles - i > 1:
                num_partitions = rdd.getNumPartitions()
                rdd = rdd.repartition(num_partitions)

        return state_dict[0]

    finally:
        p.terminate()
        p.join()
