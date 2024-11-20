from typing import Optional, List, Tuple, Union

from pydantic import BaseModel, Field


class BoxObjects(BaseModel):
    frame: int = Field(..., description="frame index")
    object_id: int = Field(..., description="object id to be annotated")
    bbox: Optional[Tuple[float, float, float, float]] = Field((0.0, 0.0, 0.0, 0.0),
                                                              description="The bounding box in x1,y1,x2,y2 format")


class PredictArgs(BaseModel):
    images: Optional[List[str]] = Field(...,
                                        description="The images to predict in base64 or the path of the images to load")
    video: Optional[str] = Field(..., description="The path of the video to predict")
    boxObjects: Optional[List[BoxObjects]] = Field(...,
                                                   description="Boxes to be assigned in the video or photo. Must be at least one")
    scale_factor: Optional[float] = Field(1, description="The scale factor of the media to reduce the memory")
    start_second: Optional[int] = Field(0, description="The starting frame for the prediction")
    end_second: Optional[int] = Field(None, description="The end frame for the prediction")
    focal_length_px: Optional[float] = Field(None, description="The focal length in pixels for the prediction")


class DepthResponseWithBox(BaseModel):
    object_id: int = Field(..., description="object id")
    mean_distance: float = Field(...,
                                 description="The mean distance between camera and object in meters")
    focal_length_px: float = Field(..., description="The focal length in pixels for the prediction")


class DepthResponseImage(BaseModel):
    dept_matrix: List = Field(..., description="Depth matrix")
    focal_length_px: float = Field(..., description="The focal length in pixels for the prediction")


class PredictResponse(BaseModel):
    response: dict[int, List[Union[DepthResponseWithBox, DepthResponseImage]]] = Field(...,
                                                                                       description="The output masks from Depth-pro with frame: result or the full matrix")
