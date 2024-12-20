import os
from typing import List

import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import ORJSONResponse
from starlette.responses import Response, StreamingResponse

from api import depthpro, file_uploader
from middleware import LimitRequestSizeMiddleware, lifespan
from models import PredictArgs, PredictResponse

app = FastAPI(lifespan=lifespan)
app.add_middleware(LimitRequestSizeMiddleware)


def __return_response(request: PredictArgs, stream=False, background_tasks: BackgroundTasks = None):
    if request.video is not None and request.images is not None:
        return Response(
            "Cannot use both images and video in the same request", status_code=400
        )
    if request.video is not None and not stream:
        return Response(
            "Cannot use both video and without streaming", status_code=400
        )
    model = depthpro()
    response = model.call_model(images=request.images,
                                video=request.video,
                                box_objects=request.boxObjects,
                                scale_factor=request.scale_factor,
                                start_second=request.start_second,
                                end_second=request.end_second,
                                focal_length_px=request.focal_length_px)
    try:
        if stream:
            background_tasks.add_task(model.unload_model_after_stream)
            return StreamingResponse((resp.model_dump_json() + '\n' for resp in response),
                                     media_type="application/json")
        else:
            response_dict = {}
            for resp in response:
                response_dict.update(resp.response)
            model.unload_model_after_stream()
            return PredictResponse(response=response_dict)
    except Exception as e:
        model.unload_model_after_stream()
        raise e


@app.post("/v1/predict", response_model=PredictResponse, response_class=ORJSONResponse)
async def predict(request: PredictArgs):
    return __return_response(request)


@app.post("/v1/predict_async")
async def predict_async(request: PredictArgs, background_tasks: BackgroundTasks):
    return __return_response(request, stream=True, background_tasks=background_tasks)


@app.put("/v1/asset")
async def asset(files: List[UploadFile] = File(...)):
    return file_uploader().upload_batch(files)


if __name__ == "__main__":
    port = os.environ.get('PORT', '8002')
    uvicorn.run(app, host="0.0.0.0", port=int(port), log_config='conf/log.ini')
