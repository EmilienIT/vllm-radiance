#!/usr/bin/env python3
"""CPU-only contracts; no vLLM, model weights or GPU imports."""
import ast
import copy
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import patch_qwen_tool_grammar as patch


def main():
    namespace = {"_QWEN_XML_STRUCTURAL_TAG_MODELS": {"qwen_3", "qwen_3_5", "qwen_3_coder"}}
    exec(patch.HELPERS, namespace)
    normalize = namespace["_radiance_qwen_integer_bounds"]
    auto = namespace["_radiance_qwen_auto_grammar"]
    for key in ("RADIANCE_QWEN_AUTO_TOOL_GRAMMAR", "RADIANCE_QWEN_INT64_BOUND_COMPAT"):
        os.environ.pop(key, None)
    assert not auto("qwen_3_coder")
    os.environ["RADIANCE_QWEN_AUTO_TOOL_GRAMMAR"] = "1"
    assert auto("qwen_3_coder") and not auto("llama")
    os.environ["RADIANCE_QWEN_AUTO_TOOL_GRAMMAR"] = "true"
    assert not auto("qwen_3_coder")
    exact = {"type": ["integer", "null"], "minimum": float(-(2**63)), "maximum": 42.0}
    normalize(exact)
    assert type(exact["minimum"]) is int and exact["minimum"] == -(2**63)
    assert type(exact["maximum"]) is int and exact["maximum"] == 42
    for bound in (2**63 - 1, -(2**63)):
        schema = {"type": "integer", "maximum": bound}
        normalize(schema)
        assert schema["maximum"] == bound
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        try:
            normalize({"type": "integer", key: float(2**63)})
            raise AssertionError("out-of-range bound silently accepted")
        except ValueError as exc:
            assert "Export exact integer bounds" in str(exc)
    os.environ["RADIANCE_QWEN_INT64_BOUND_COMPAT"] = "1"
    compat = {"type": "integer", "maximum": float(2**63)}
    normalize(compat)
    assert compat["maximum"] == 2**63 - 1
    for key in ("minimum", "exclusiveMaximum", "exclusiveMinimum"):
        try:
            normalize({"type": "integer", key: float(2**63)})
            raise AssertionError("unsupported compatibility bound accepted")
        except ValueError:
            pass
    untouched = {"type": "number", "maximum": float(2**63), "default": compat,
                 "examples": [compat], "enum": [compat]}
    expected = copy.deepcopy(untouched)
    normalize(untouched)
    assert untouched == expected
    nested = {"properties": {"maximum": {"type": "integer", "maximum": 32.0}},
              "anyOf": [{"items": {"type": "integer", "maximum": 33.0}}]}
    normalize(nested)
    assert type(nested["properties"]["maximum"]["maximum"]) is int
    assert type(nested["anyOf"][0]["items"]["maximum"]) is int

    # Execute every guarded replacement against a minimal matching source,
    # prove repeat application is a NOOP, and fail on drift rather than guessing.
    anchors = []
    class Capture:
        def __call__(self, path, anchor, new, sentinel, label):
            anchors.append((path.name, anchor, new, sentinel, label))
    real_apply = patch.apply
    patch.apply = Capture()
    patch.main()
    patch.apply = real_apply
    with tempfile.TemporaryDirectory(prefix="radiance-qwen-contract-") as temp:
        for name in ("structural_tag_registry.py", "sampling_params.py"):
            parts = [entry for entry in anchors if entry[0] == name]
            # Independent enclosing functions make source anchors syntactically valid.
            source = ""
            for i, (_, anchor, _, _, _) in enumerate(parts):
                if anchor.startswith("def "):
                    source += anchor + "    tools,\n) -> bool:\n    pass\n"
                elif "except VLLMValidationError" in anchor:
                    source += "def fallback():\n        if True:\n            try:\n                pass\n" + anchor + "                pass\n"
                else:
                    source += f"def check_{i}():\n" + anchor + ("        pass\n" if anchor.rstrip().endswith(":") else "")
            target = Path(temp) / name
            target.write_text(source)
            for _, anchor, new, sentinel, label in parts:
                real_apply(target, anchor, new, sentinel, label)
            once = target.read_text()
            ast.parse(once)
            for _, anchor, new, sentinel, label in parts:
                real_apply(target, anchor, new, sentinel, label)
            assert target.read_text() == once
            try:
                real_apply(target, "not a matching source", "", "absent sentinel", "drift")
                raise AssertionError("source drift accepted")
            except SystemExit:
                pass
    print("Qwen tool grammar CPU contracts: PASS")


if __name__ == "__main__":
    main()
