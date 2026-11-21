"""scripts/api_docs.py: the API page's examples match the router's real endpoints and request models."""
import pytest

from scripts.api_docs import OPS, add_examples
from src.service.router import ModelReq, SearchReq, WriteReq, create_app

BODY = {"/search": SearchReq, "/upsert": WriteReq, "/delete": WriteReq, "/admin/migrate": ModelReq,
        "/admin/cutover": ModelReq, "/admin/finalize": ModelReq}


@pytest.fixture(scope="module")
def schema():
    return add_examples(create_app()).openapi()


def test_every_router_endpoint_is_documented_and_every_documented_one_exists(schema):
    routes = {(p, m) for p, item in schema["paths"].items() for m in item}
    assert routes == set(OPS)


@pytest.mark.parametrize("path", sorted(BODY))
def test_examples_are_valid_request_bodies(schema, path):
    examples = schema["paths"][path]["post"]["requestBody"]["content"]["application/json"]["examples"]
    assert examples
    for name, ex in examples.items():
        BODY[path].model_validate(ex["value"])   # raises if an example drifts from the router's model
