"""Unit tests for the dependency-light parts of the factor-mining package.

These avoid importing qlib / alphalens / openai (heavy optional deps), exercising
expression validation, code conversion and LLM-response parsing only.
"""

from __future__ import annotations

from quantify.factor.llm import parse_factor_response
from quantify.factor.qlib_data import qlib_to_ts_code, ts_code_to_qlib
from quantify.factor.validator import validate_expression


class TestValidator:
    def test_valid_expression(self):
        result = validate_expression("Mean($close, 5) / Mean($close, 20)")
        assert result.ok
        assert "close" in result.fields
        assert "Mean" in result.operators

    def test_empty_expression(self):
        assert not validate_expression("   ").ok

    def test_unbalanced_parens(self):
        assert not validate_expression("Mean($close, 5").ok
        assert not validate_expression("Mean($close, 5))").ok

    def test_unknown_field(self):
        result = validate_expression("Mean($bogus, 5)")
        assert not result.ok
        assert "未知字段" in result.error

    def test_unknown_operator(self):
        result = validate_expression("Frobnicate($close, 5)")
        assert not result.ok
        assert "未知算子" in result.error

    def test_requires_a_field(self):
        assert not validate_expression("5 + 3").ok

    def test_illegal_characters(self):
        assert not validate_expression("$close; import os").ok


class TestCodeConversion:
    def test_roundtrip_sh(self):
        assert ts_code_to_qlib("600000.SH") == "SH600000"
        assert qlib_to_ts_code("SH600000") == "600000.SH"

    def test_roundtrip_sz(self):
        assert ts_code_to_qlib("000001.SZ") == "SZ000001"
        assert qlib_to_ts_code("SZ000001") == "000001.SZ"

    def test_roundtrip_bj(self):
        assert ts_code_to_qlib("830799.BJ") == "BJ830799"
        assert qlib_to_ts_code("BJ830799") == "830799.BJ"


class TestParseFactorResponse:
    def test_plain_json(self):
        content = (
            '{"factors": [{"name": "mom20", "expression": "Delta($close, 20)", '
            '"hypothesis": "momentum", "category": "momentum"}]}'
        )
        out = parse_factor_response(content)
        assert len(out) == 1
        assert out[0].name == "mom20"
        assert out[0].expression == "Delta($close, 20)"

    def test_json_with_code_fence(self):
        content = '```json\n{"factors": [{"name": "x", "expression": "$close"}]}\n```'
        out = parse_factor_response(content)
        assert len(out) == 1
        assert out[0].expression == "$close"

    def test_bare_list(self):
        content = '[{"name": "x", "expression": "Mean($volume, 5)"}]'
        out = parse_factor_response(content)
        assert len(out) == 1

    def test_skips_items_without_expression(self):
        content = '{"factors": [{"name": "x"}, {"name": "y", "expression": "$close"}]}'
        out = parse_factor_response(content)
        assert len(out) == 1
        assert out[0].expression == "$close"

    def test_garbage_returns_empty(self):
        assert parse_factor_response("not json at all") == []
        assert parse_factor_response("") == []

    def test_nested_data_factors(self):
        content = '{"data": {"factors": [{"name": "x", "expression": "Mean($close, 5)"}]}}'
        out = parse_factor_response(content)
        assert len(out) == 1
        assert out[0].expression == "Mean($close, 5)"

    def test_alias_key(self):
        content = '{"results": [{"name": "x", "expression": "$volume"}]}'
        out = parse_factor_response(content)
        assert len(out) == 1
        assert out[0].expression == "$volume"

    def test_empty_factors_array(self):
        assert parse_factor_response('{"factors": []}') == []
