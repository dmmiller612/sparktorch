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
from pyspark.rdd import RDD
from typing import Dict, List
from uuid import uuid4
import numpy as np
from pyspark.rdd import PipelinedRDD
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import os
from datetime import timedelta


def mapPartitionsWithIndex(rdd, f, preservesPartitioning=False):
    return PipelinedRDD(rdd, f, preservesPartitioning, isFromBarrier=True)


def process_generic_model(params, iters):
    # Hopefully this function can go away in newer versions.
    for i in range(iters):
        for p in params:
            z = torch.zeros_like(p)
            dist.all_reduce(z, op=torch.distributed.ReduceOp.SUM)


def handle_model(
    index: int,
    data: List[DataObj],
    torch_obj: str,
    master_url: str = '127.0.0.1',
    iters: int = 1000,
    world_size: int = 2,
    verbose: int = 1,
    mini_batch: int = -1,
    validation_pct: float = 0
):

    # Def Load model
    torch_obj = load_torch_model(torch_obj)
    model = torch_obj.model
    model.train()

    criterion = torch_obj.criterion
    optimizer = torch_obj.optimizer

    # Initialize zero params for -1 index
    params = [torch.zeros_like(p) for p in model.parameters()]

    if dist.is_initialized():
        dist.destroy_process_group()

    os.environ['MASTER_ADDR'] = master_url
    os.environ['MASTER_PORT'] = '5000'
    dist.init_process_group('gloo', rank=index + 1, world_size=world_size, timeout=timedelta(seconds=60))
    partition_id = str(uuid4())

    if data is None:
        process_generic_model(params, iters)
        return []

    data_obj = handle_features(data, validation_pct)
    if data_obj.x_train is None:
        process_generic_model(params, iters)
        return []

    x_train = data_obj.x_train
    y_train = data_obj.y_train if data_obj.y_train is not None else x_train
    x_val = data_obj.x_val
    y_val = data_obj.y_val if data_obj.y_val is not None else x_val

    for i in range(iters):
        optimizer.zero_grad()

        if 0 < mini_batch < len(data_obj.x_train):
            idxs = np.random.choice(len(data_obj.x_train), mini_batch, replace=False).tolist()
            x_train = data_obj.x_train[idxs]
            y_train = data_obj.y_train[idxs]

        y_pred = model(x_train)

        try:
            loss = criterion(y_pred, y_train)
        except RuntimeError as e:
            y_train = torch.flatten(y_train.long())
            loss = criterion(y_pred, y_train)

        loss.backward()

        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
            param.grad.data /= (world_size-1)

        optimizer.step()

        val_loss = None
        if x_val is not None:
            pred_val = model(x_val)
            try:
                val_loss = criterion(pred_val, y_val)
            except RuntimeError as e:
                y_val = torch.flatten(y_val.long())
                val_loss = criterion(pred_val, y_val)
            val_loss = val_loss.item()

        loss_v = loss.item()
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
    world_size: int = 2
) -> Dict:

    p = Process(target=handle_model,
                args=(-1, None, torch_obj, master_url, iters, world_size))
    p.start()

    try:
        state_dict = None
        for i in range(partition_shuffles):
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
                    world_size=world_size
                )
            ).collect()

            if partition_shuffles - i > 1:
                num_partitions = rdd.getNumPartitions()
                rdd = rdd.repartition(num_partitions)

        return state_dict[0]

    finally:
        p.terminate()
        p.join()
