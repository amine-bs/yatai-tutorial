import bentoml
from bentoml.io import NumpyNdarray

model_runner = bentoml.sklearn.get("regressor:latest").to_runner()

svc = bentoml.Service("regression", runners=[model_runner])

input_spec = NumpyNdarray(dtype="int", shape=(-1, 5))


@svc.api(input=input_spec, output=NumpyNdarray())
async def predict(input_arr):
    return await model_runner.predict.async_run(input_arr)