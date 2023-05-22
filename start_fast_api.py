import logging
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from config.config import Config
from src_code.agent.agent import AlphaZeroChess

NUM_WORKERS = 16
# to run:
# gunicorn -w <NUM_WORKERS>  -k uvicorn.workers.UvicornWorker -b 0.0.0.0 start_fast_api:app
# uvicorn start_fast_api:app --host 0.0.0.0 --port 8000 --reload
tf.get_logger().setLevel('ERROR')
physical_devices = tf.config.list_physical_devices('GPU')
if len(tf.config.list_physical_devices('GPU')) > 0:
    gpu_idx = 0  # Set the index of the GPU you want to use
    # Get the GPU device
    gpu_device = physical_devices[gpu_idx]
    # Set the GPU memory growth
    tf.config.experimental.set_memory_growth(gpu_device, True)
else:
    print('No GPUs available')


# Define your request body schema as a Pydantic BaseModel
class ChessData(BaseModel):
    state: List[List[List[float]]]  # The exact structure depends on the input shape of your model


app = FastAPI()
config = Config(num_iterations=0, verbosity=True)
agent_api = AlphaZeroChess(config)


@app.post('/predict')
async def predict(data: ChessData):
    state = np.array(data.state)
    state_expanded = np.expand_dims(state, axis=0)
    # state_ds = tf.data.Dataset.from_tensor_slices(state_expanded)
    # state_ds_batched = state_ds.batch(1)

    policy_preds, value_preds = agent_api.network.predict(state_expanded, verbose=0)
    return {"policy_preds": policy_preds.tolist(), "value_preds": value_preds.tolist()}
