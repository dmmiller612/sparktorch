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

from sparktorch.util import handle_features, load_torch_model, TorchObj, DataObj
from sparktorch.server import Server
import requests
import dill
from typing import Dict, List
from uuid import uuid4


def get_state_dict(master_url: str = 'localhost:3000') -> Dict:
    r = requests.get('http://{0}/parameters'.format(master_url))
    state_dict = dill.loads(r.content)
    return state_dict


def put_deltas_to_server(delta: List, master_url: str = 'localhost:3000'):
    requests.post('http://{0}/update'.format(master_url), data=dill.dumps(delta))


def get_main(master_url: str = 'localhost:3000') -> str:
    r = requests.get('http://{0}/'.format(master_url))
    return r.text


def handle_model(
    data: List[DataObj],
    torch_obj: TorchObj,
    master_url: str = 'localhost:3000',
    iters: int = 1000,
    verbose: int = 1
):

    partition_id = str(uuid4())
    data = list(data)
    if data is None or len(data) == 0:
        return

    data_obj = handle_features(data)

    x_train = data_obj.x_train
    y_train = data_obj.y_train

    torch_obj = load_torch_model(torch_obj)

    model = torch_obj.model
    model.train()

    criterion = torch_obj.criterion

    for i in range(iters):
        state_dict = get_state_dict(master_url)
        model.load_state_dict(state_dict)

        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()

        gradients = []
        for param in model.parameters():
            gradients.append(param.grad)

        put_deltas_to_server(gradients)

        if verbose:
            print(f"Partition: {partition_id}. Iteration: {i}. Loss: {loss.item()}")


def train(
    rdd,
    torch_obj: TorchObj,
    server: Server,
    iters: int = 10,
    partition_shuffles: int = 1,
    verbose: int = 1
) -> Dict:
    try:
        master_url = str(server.master_url)

        for i in range(partition_shuffles):
            rdd.foreachPartition(
                lambda x: handle_model(x, torch_obj=torch_obj, master_url=master_url, iters=iters, verbose=verbose)
            )

            if partition_shuffles - i > 1:
                num_partitions = rdd.getNumPartitions()
                rdd = rdd.repartition(num_partitions)

        state_dict = get_state_dict(master_url)
        server.stop_server()

        return state_dict

    except Exception as e:
        server.stop_server()
        raise e

