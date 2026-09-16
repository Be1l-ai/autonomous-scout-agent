"""llama-cpp-python is a heavy compiled dependency, so CI stubs it out. These
tests cover the plumbing (queue, dedupe, scope, HTML parsing), not inference."""

import sys
import types

if "llama_cpp" not in sys.modules:
    stub = types.ModuleType("llama_cpp")

    class Llama:  # pragma: no cover - test double
        def __init__(self, *args, **kwargs):
            self.args = kwargs

        def create_chat_completion(self, *args, **kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"relevant": true, "needs_worker": false, '
                            '"confidence": 0.5, "reason": "stub", "next_actions": []}'
                        }
                    }
                ]
            }

    stub.Llama = Llama
    sys.modules["llama_cpp"] = stub
