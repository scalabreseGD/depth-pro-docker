import os.path
from functools import lru_cache

import toml

from .depth_pro import DepthPro
from .storage import FileUploader

__depth_pro_serve = None
__file_uploader = None


def depthpro() -> DepthPro:
    global __depth_pro_serve
    conf = read_conf()
    checkpoints_path = conf['app']['checkpoints_path']
    __checkpoints_exists(checkpoints_path)
    if __depth_pro_serve is None:
        __depth_pro_serve = DepthPro(checkpoints_path=checkpoints_path)
    return __depth_pro_serve


def __checkpoints_exists(checkpoints_path):
    if not os.path.exists(checkpoints_path):
        raise FileNotFoundError(f"{checkpoints_path} file not found")


def file_uploader() -> FileUploader:
    global __file_uploader
    if __file_uploader is None:
        conf = read_conf()
        __file_uploader = FileUploader(base_path=conf['file_uploader']['base_path'])
    return __file_uploader


@lru_cache(maxsize=1)
def read_conf():
    with open("conf/conf.toml", "r") as f:
        data = toml.load(f)
    return data
