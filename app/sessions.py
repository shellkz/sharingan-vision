"""ONNX session 的 dependency injection。session 本身由 main.py 的 lifespan 建立、
存進 app.state,這裡只是給路由用的取用捷徑,不負責建立或快取。"""
from fastapi import Request


def get_objectness_session(request: Request):
    return request.app.state.fastsam_session


def get_embedding_session(request: Request):
    return request.app.state.dinov2_session
