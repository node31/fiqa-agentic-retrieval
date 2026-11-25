"""scripts/llm_check.py: the settings check and the one-call preflight that agent_eval.py and the A2A server run."""
from scripts.llm_check import problems, ping
from src.agent.llm import LLM, FakeLLM


def test_problems_names_placeholders_and_a_bad_endpoint():
    good = {"AZURE_OPENAI_ENDPOINT": "https://r.example/openai/v1/", "AZURE_OPENAI_DEPLOYMENT": "gpt-x",
            "AZURE_OPENAI_API_KEY": "k" * 40}
    assert problems(good) == []
    assert problems({**good, "AZURE_OPENAI_ENDPOINT": "https://r.example/"})[0].startswith("AZURE_OPENAI_ENDPOINT should end")
    assert problems({**good, "AZURE_OPENAI_API_KEY": "<your-key>"}) == ["AZURE_OPENAI_API_KEY still looks like a placeholder: <your-key>"]


def test_ping_bypasses_the_cache_and_explains_a_401():
    ok, msg = ping(FakeLLM(lambda s, u: "OK"))
    assert ok and msg.startswith("OK:")

    class Unauthorized(Exception):
        status_code = 401

    class Broken(LLM):
        backend, model = "azure", "gpt-x"

        def _call(self, system, user, max_output_tokens):
            raise Unauthorized("Access denied")

    broken = Broken(use_cache=True)
    ok, msg = ping(broken)
    assert not ok and "likely cause: the key is wrong" in msg
    assert broken.use_cache and broken.requests_sent == 1 and broken.cache_hits == 0   # cache restored, request counted
