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

import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import json
import dill
from typing import Any, Dict, List, Type, Tuple
import collections
import codecs

TorchObj = collections.namedtuple(
    'TorchObj', ['model', 'criterion', 'optimizer', 'optimizer_params', 'is_lazy', 'model_parameters']
)

DataObj = collections.namedtuple('DataObj', ['x_train', 'y_train', 'x_val', 'y_val'])


def torch_encoder(obj) -> str:
    """
    Encodes Torch Object or anything related
    :param obj: any object
    :return: decoded object
    """
    return codecs.encode(
        dill.dumps(obj), "base64"
    ).decode()


def torch_decoder(model_ser: str) -> Any:
    """
    Decodes the torch object or model
    :param model_ser: Serialized object
    :return: the object
    """
    return dill.loads(codecs.decode(model_ser.encode(), "base64"))


def handle_features(data: List[DataObj], validation_pct: float = 0.0) -> DataObj:
    """
    Processes features and converts them to torch tensors.

    :param data: The initial data is numpy vectors
    :param validation_pct: Percentage to use for a validation set
    :return: A DataObj with tensors

    """
    x_train = []
    y_train = []

    for feature in data:
        if feature.y_train is not None:
            if type(feature.y_train) is int or type(feature.y_train) is float:
                y_train.append([feature.y_train])
            else:
                y_train.append(feature.y_train)

        x_train.append(feature.x_train)

    if len(x_train) == 0:
        return DataObj(x_train=None, y_train=None, x_val=None, y_val=None)

    if validation_pct > 0:
        val_amt = int(len(x_train) * validation_pct)
        val_idxs = np.random.choice(len(x_train), val_amt, replace=False).tolist()
        train_idxs = list(set(val_idxs).symmetric_difference(set([i for i in range(len(x_train))])))

        full_x = torch.from_numpy(np.stack(x_train)).float()
        full_y = torch.from_numpy(np.asarray(y_train)).float() if y_train else None

        x_val = full_x[val_idxs]
        y_val = full_y[val_idxs] if full_y is not None else None

        x_train = full_x[train_idxs]
        y_train = full_y[train_idxs] if full_y is not None else None

        return DataObj(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

    x_train = torch.from_numpy(np.stack(x_train)).float()
    y_train = torch.from_numpy(np.asarray(y_train)).float() if y_train else None

    return DataObj(x_train=x_train, y_train=y_train, x_val=None, y_val=None)


def load_base_torch(torch_obj: str) -> Tuple:
    """
    Loads the base torch object from json.
    :param torch_obj: The json torch object
    :return: a tuple of the torch object and the shapes of the parameters.
    """
    obj = json.loads(torch_obj)
    return obj['torch_obj'], obj['shapes']


def load_torch_model(torch_obj: str, from_json: bool = False) -> TorchObj:
    """
    Loads the torch object. If it is from json, it will load the json body before handling the request.

    :param torch_obj: The torch object as a serialized string
    :param from_json: Determines whether we should load from json
    :return: A loaded model torch object
    """
    if from_json:
        torch_obj, _ = load_base_torch(torch_obj)

    loaded = torch_decoder(torch_obj)
    if loaded.is_lazy:
        model = loaded.model(**loaded.model_parameters) if loaded.model_parameters else loaded.model()
        return TorchObj(
            model=model,
            criterion=loaded.criterion(),
            optimizer=load_optimizer(loaded.optimizer, loaded.optimizer_params, model),
            optimizer_params=loaded.optimizer_params,
            is_lazy=False,
            model_parameters=None
        )

    model = loaded.model
    optimizer = load_optimizer(loaded.optimizer, loaded.optimizer_params, model)
    return TorchObj(
        model=model,
        criterion=loaded.criterion,
        optimizer=optimizer,
        optimizer_params=loaded.optimizer_params,
        is_lazy=False,
        model_parameters=None
    )


def serialize_torch_obj_lazy(
    model: Type[nn.Module],
    criterion: Type[Any],
    optimizer: Type[Optimizer],
    optimizer_params: Dict = None,
    model_parameters: Dict = None
) -> str:
    """
    Lazily serializes a torch object.
    :param model: The network's class that you want to run.
    :param criterion: The class of the criterion.
    :param optimizer: The class of the optimizer.
    :param optimizer_params: The optimizer parameters as a dictionary.
    :param model_parameters: The model parameters as a dictionary.
    :return: a serialized json string of the torch model parameters.
    """
    tmp_mod = model(**model_parameters) if model_parameters else model()
    params = [list(ps.shape) for ps in tmp_mod.parameters()]
    model_encoded = torch_encoder(
        TorchObj(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            is_lazy=True,
            model_parameters=model_parameters
        )
    )
    return json.dumps({
        'torch_obj': model_encoded,
        'shapes': params
    })


def serialize_torch_obj(
        model: nn.Module,
        criterion: Any,
        optimizer: Type[Optimizer],
        **kwargs
) -> str:
    model_encoded = torch_encoder(
        TorchObj(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            optimizer_params=kwargs,
            is_lazy=False,
            model_parameters=None
        )
    )
    return json.dumps({
        'torch_obj': model_encoded,
        'shapes': [list(ps.shape) for ps in model.parameters()]
    })


def load_optimizer(optimizer: Type[Optimizer], params: Dict, model: nn.Module) -> Optimizer:
    if params:
        return optimizer(model.parameters(), **params)

    return optimizer(model.parameters())
