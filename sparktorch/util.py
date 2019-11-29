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
import dill
from typing import Any, Dict, List, Type
import collections
import codecs

TorchObj = collections.namedtuple('TorchObj', ['model', 'criterion', 'optimizer', 'optimizer_params'])

DataObj = collections.namedtuple('DataObj', ['x_train', 'y_train', 'x_val', 'y_val'])


def torch_encoder(obj):
    """
    Encodes Torch Object or anything related
    :param obj: any object
    :return: decoded object
    """
    return codecs.encode(
        dill.dumps(obj), "base64"
    ).decode()


def torch_decoder(model_ser):
    return dill.loads(codecs.decode(model_ser.encode(), "base64"))


def handle_features(data: List[DataObj]) -> DataObj:
    x_train = []
    y_train = []

    for feature in data:

        if feature.y_train is not None:
            if type(feature.y_train) is int or type(feature.y_train) is float:
                y_train.append([feature.y_train])
            else:
                y_train.append(feature.y_train)

        x_train.append(feature.x_train)

    x_train = torch.from_numpy(np.stack(x_train)).float()
    y_train = torch.from_numpy(np.asarray(y_train)).float() if y_train else None

    return DataObj(x_train=x_train, y_train=y_train, x_val=None, y_val=None)


def load_torch_model(torch_obj: str) -> TorchObj:
    loaded = torch_decoder(torch_obj)
    model = loaded.model
    optimizer = load_optimizer(loaded.optimizer, loaded.optimizer_params, model)
    return TorchObj(
        model=model,
        criterion=loaded.criterion,
        optimizer=optimizer,
        optimizer_params=loaded.optimizer_params
    )


def serialize_torch_obj(
        model: nn.Module,
        criterion: Any,
        optimizer: Type[Optimizer],
        **kwargs
) -> str:
    return torch_encoder(
        TorchObj(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            optimizer_params=kwargs
        )
    )


def load_optimizer(optimizer: Type[Optimizer], params: Dict, model: nn.Module) -> Optimizer:
    if params:
        return optimizer(model.parameters(), **params)

    return optimizer(model.parameters())
