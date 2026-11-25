"""scripts/demo_call.py, the colourful HTTP helper of docs/guides/demo_security_a2a_memory.md.

pytest.ini skips doctests under scripts/, so this runs the module's own examples (how a response is coloured,
how picks are labelled, how long values are shortened).
"""
import doctest

import scripts.demo_call as demo_call


def test_demo_call_doctests():
    result = doctest.testmod(demo_call, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert result.attempted >= 5 and result.failed == 0


def test_a_refused_request_is_red_even_when_the_body_looks_fine():
    assert demo_call.classify(403, {"results": []})[0] == "red"
    assert demo_call.classify(404, {"detail": "no such entry"}) == ("red", "not found")
    assert demo_call.classify(200, {"results": [], "security": {"findings": [{"mode": "log"}]}})[0] == "yellow"
