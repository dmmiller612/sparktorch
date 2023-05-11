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

import os
from typing import Dict, List, Union, Callable
from uuid import uuid4

import numpy as np
import torch
import torch.distributed as dist
from pyspark.rdd import PipelinedRDD
from pyspark.rdd import RDD
from pyspark.taskcontext import BarrierTaskContext

from sparktorch.early_stopper import EarlyStopping
from sparktorch.util import handle_features, load_torch_model, DataObj, load_base_torch


def get_available_port_port(address: str, context: BarrierTaskContext) -> int:
    port = ""
    if context.partitionId() == 0:
        try:
            import socket

            sock = socket.socket()
            sock.bind((address, 0))
            port = sock.getsockname()[1]
        except socket.error:
            pass
    available_port = context.allGather(str(port))[0]
    if not available_port:
        raise RuntimeError("Failed to find free port for distributed training.")

    return int(available_port)


def mapPartitionsWithIndex(rdd: RDD, f: Callable, preservesPartitioning: bool = False) -> PipelinedRDD:
    """
    Maps Partitions based off of the index.

    :param rdd: A valid rdd.
    :param f: Callable function with iterable.
    :param preservesPartitioning: Whether or not to preserve partitioning

    :return: A pipeline rdd.
    """
    return PipelinedRDD(rdd, f, preservesPartitioning, isFromBarrier=True)


def handle_model(
    index: int,
    data: List[DataObj],
    torch_obj: Union[str, List],
    iters: int = 1000,
    early_stop_patience: int = -1,
    verbose: int = 1,
    mini_batch: int = -1,
    validation_pct: float = 0,
    device: str = 'cpu',
    compile_mode: str = None,
) -> List[Dict]:
    """
    Runs the training of pytorch model, utilizing the distributed package.

    :param index: Partition index. Used for registering.
    :param data: The data from the partition
    :param torch_obj: The torch object string. Needs serialized
    :param iters: The iterations for training
    :param verbose: whether to log the loss or not.
    :param mini_batch: Mini batch for training
    :param validation_pct: Validation percentage.
    :param device: The pytorch device to use for training. cpu/cuda
    :param early_stop_patience: Amount of patient for early stopping. -1 means don't use early stopping.
    :param compile_mode: Torch compile option for 2.0 only.

    :return: A list of the model state dictionary.
    """
    # If a process has already been setup on the machine, kill it.
    if dist.is_initialized():
        dist.destroy_process_group()

    context = BarrierTaskContext.get()

    addrs = [e.address.split(":")[0] for e in context.getTaskInfos()]
    world_size = len(addrs)

    # Set up the distributed server.
    os.environ['MASTER_ADDR'] = str(addrs[0])
    os.environ['MASTER_PORT'] = str(get_available_port_port(addrs[0], context))
    os.environ['WORLD_SIZE'] = str(len(addrs))
    os.environ['NODE_RANK'] = str(context.partitionId())
    os.environ['RANK'] = str(context.partitionId())

    dist.init_process_group('gloo', rank=index, world_size=world_size)

    torch_obj = load_torch_model(torch_obj)

    # Loaded the model
    model = torch_obj.model.to(device)

    if compile_mode is not None:
        torch.compile(compile_mode)

    model.train()
    criterion = torch_obj.criterion
    optimizer = torch_obj.optimizer

    # Set up early stopping
    es = EarlyStopping(patience=early_stop_patience)
    should_stop = torch.zeros(1)
    has_early_stop = early_stop_patience > 0

    partition_id = str(uuid4())

    # Process the data. Converts to x_train, y_train, x_val, y_val
    data_obj = handle_features(data, validation_pct)

    # Passes all of the data
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
        val_loss_v = None
        if x_val is not None:
            pred_val = model(x_val)

            try:
                val_loss = criterion(pred_val, y_val)
            except RuntimeError as e:
                y_val = torch.flatten(y_val.long())
                val_loss = criterion(pred_val, y_val)

            val_loss_v = val_loss.item()

        # Calculate gradients
        loss.backward()

        # Distributed part of training.
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
            param.grad.data /= world_size

        # Processes the early stop work
        loss_distributed = None
        if has_early_stop:
            loss_to_use = val_loss if val_loss is not None else loss

            dist.all_reduce(loss_to_use, op=torch.distributed.ReduceOp.SUM)
            loss_distributed = loss_to_use.item() / world_size
            stop = es.step(loss_distributed)
            if stop:
                should_stop = should_stop + 1.0

            dist.all_reduce(should_stop, op=torch.distributed.ReduceOp.SUM)
            if should_stop.item() > 0:
                break

        optimizer.step()

        if verbose:
            print(f"Partition: {partition_id}. Iteration: {i}. Distributed Loss: {loss_distributed} "
                  f"Partition Training Loss: {loss_v}, "
                  f"Partition Validation Loss: {val_loss_v}")

    return [model.state_dict()]


def train_distributed(
    rdd: RDD,
    torch_obj: str,
    iters: int = 10,
    partition_shuffles: int = 1,
    verbose: int = 1,
    mini_batch: int = -1,
    validation_pct: float = 0.0,
    device: str = 'cpu',
    early_stop_patience: int = -1,
    compile_mode: str = None,
) -> Dict:
    """
    Entry point to train the model in distributed fashion.

    :param rdd: The rdd of data to run on the model.
    :param torch_obj: The torch object as a string that includes the model and param shapes.
    :param iters: Number of iterations for training.
    :param partition_shuffles: Number of partition shuffles.
    :param verbose: Verbosity of logs.
    :param mini_batch: Mini batch for each iteration of training.
    :param validation_pct: How many items to validate.
    :param device: pytorch device.
    :param early_stop_patience: amount of patience for early stopping.
    :param compile_mode: torch 2.0 compile mode.

    :return: The train dict.
    """
    torch_loaded, params = load_base_torch(torch_obj)

    state_dict = None
    for i in range(partition_shuffles):

        # Run model with barrier execution mode.
        state_dict = mapPartitionsWithIndex(
            rdd, lambda i, x: handle_model(
                i,
                x,
                torch_obj=torch_loaded,
                iters=iters,
                verbose=verbose,
                mini_batch=mini_batch,
                validation_pct=validation_pct,
                device=device,
                early_stop_patience=int(early_stop_patience + 0),
                compile_mode=compile_mode,
            )
        ).collect()

        if partition_shuffles - i > 1:
            num_partitions = rdd.getNumPartitions()
            rdd = rdd.repartition(num_partitions)

    return state_dict[0]
