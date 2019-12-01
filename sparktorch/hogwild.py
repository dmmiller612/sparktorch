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
from sparktorch.server import Server
from pyspark.rdd import RDD
import requests
import dill
from typing import Dict, List
from uuid import uuid4
import torch
import numpy as np


def get_state_dict(master_url: str = 'localhost:3000', retry=True) -> Dict:
    try:
        r = requests.get('http://{0}/parameters'.format(master_url), timeout=10)
    except Exception as e:
        if retry:
            r = requests.get('http://{0}/parameters'.format(master_url), timeout=10)
        else:
            raise e

    state_dict = dill.loads(r.content)
    return state_dict


def put_deltas_to_server(delta: List, master_url: str = 'localhost:3000', retry=True):
    try:
        requests.post('http://{0}/update'.format(master_url), data=dill.dumps(delta), timeout=10)
    except Exception as e:
        if retry:
            requests.post('http://{0}/update'.format(master_url), data=dill.dumps(delta), timeout=10)


def put_early_stop(loss, master_url: str = 'localhost:3000', retry=True):
    try:
        return requests.post('http://{0}/losses'.format(master_url), json={'loss': loss}, timeout=10).json()
    except Exception as e:
        if retry:
            return requests.post('http://{0}/losses'.format(master_url), json={'loss': loss}, timeout=10).json()


def get_main(master_url: str = 'localhost:3000') -> str:
    r = requests.get('http://{0}/'.format(master_url), timeout=3)
    return r.text


def handle_model(
    data: List[DataObj],
    torch_obj: str,
    master_url: str = 'localhost:3000',
    iters: int = 1000,
    verbose: int = 1,
    early_stop_patience: int = -1,
    mini_batch: int = -1,
    validation_pct: float = 0
):

    partition_id = str(uuid4())
    if data is None:
        return 'finished'

    data_obj = handle_features(data, validation_pct)
    if data_obj.x_train is None:
        return 'finished'

    x_train = data_obj.x_train
    y_train = data_obj.y_train if data_obj.y_train is not None else x_train
    x_val = data_obj.x_val
    y_val = data_obj.y_val if data_obj.y_val is not None else x_val

    torch_obj = load_torch_model(torch_obj)

    model = torch_obj.model
    model.train()

    criterion = torch_obj.criterion

    for i in range(iters):

        if 0 < mini_batch < len(data_obj.x_train):
            idxs = np.random.choice(len(data_obj.x_train), mini_batch, replace=False).tolist()
            x_train = data_obj.x_train[idxs]
            y_train = data_obj.y_train[idxs]

        state_dict = get_state_dict(master_url)
        model.load_state_dict(state_dict)

        y_pred = model(x_train)

        try:
            loss = criterion(y_pred, y_train)
        except RuntimeError as e:
            y_train = torch.flatten(y_train.long())
            loss = criterion(y_pred, y_train)

        loss.backward()

        val_loss = None
        if x_val is not None:
            pred_val = model(x_val)
            try:
                val_loss = criterion(pred_val, y_val)
            except RuntimeError as e:
                y_val = torch.flatten(y_val.long())
                val_loss = criterion(pred_val, y_val)
            val_loss = val_loss.item()

        gradients = []
        for param in model.parameters():
            gradients.append(param.grad)

        put_deltas_to_server(gradients, master_url)

        loss_v = loss.item()
        if verbose:
            print(f"Partition: {partition_id}. Iteration: {i}. Loss: {loss_v}, Val Loss: {val_loss}")

        if early_stop_patience > 0:
            loss_to_use = val_loss if val_loss is not None else loss_v
            should_stop = put_early_stop(loss_to_use, master_url)
            if should_stop['stop']:
                break

    return "finished"


def train(
    rdd: RDD,
    torch_obj: str,
    server: Server,
    iters: int = 10,
    partition_shuffles: int = 1,
    verbose: int = 1,
    early_stop_patience: int = -1,
    mini_batch: int = -1,
    validation_pct: float = 0.0
) -> Dict:
    try:
        master_url = str(server.master_url)

        for i in range(partition_shuffles):
            rdd.mapPartitions(
                lambda x: handle_model(
                    x,
                    torch_obj=torch_obj,
                    master_url=master_url,
                    iters=iters,
                    verbose=verbose,
                    early_stop_patience=early_stop_patience,
                    mini_batch=mini_batch,
                    validation_pct=validation_pct
                )
            ).foreach(lambda x: x)

            if partition_shuffles - i > 1:
                num_partitions = rdd.getNumPartitions()
                rdd = rdd.repartition(num_partitions)

        state_dict = get_state_dict(master_url)
        server.stop_server()

        return state_dict

    except Exception as e:
        server.stop_server()
        raise e

