"""Tests for patterns module."""

from ql_jax.patterns.observable import Settings
from ql_jax.patterns.lazy import lazy
from ql_jax.patterns.visitor import make_visitor
from ql_jax.time.date import Date


class TestSettings:
    def test_default_evaluation_date(self):
        Settings.set_evaluation_date(None)
        assert Settings.evaluation_date() is None

    def test_set_evaluation_date(self):
        d = Date(15, 6, 2023)
        Settings.set_evaluation_date(d)
        assert Settings.evaluation_date() == d

    def test_recalculate(self):
        d = Date(1, 1, 2024)
        Settings.set_evaluation_date(d)
        assert Settings.evaluation_date().serial == d.serial


class TestLazyCompute:
    def test_cached_call(self):
        call_count = 0

        @lazy
        def expensive(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        assert expensive(5) == 10
        assert expensive(5) == 10
        assert call_count == 1  # cached

    def test_different_args(self):
        call_count = 0

        @lazy
        def f(x):
            nonlocal call_count
            call_count += 1
            return x + 1

        assert f(1) == 2
        assert f(2) == 3
        assert call_count == 2


class TestVisitor:
    def test_singledispatch_visitor(self):
        visitor = make_visitor()

        @visitor.register(int)
        def visit_int(x):
            return f"int:{x}"

        @visitor.register(str)
        def visit_str(x):
            return f"str:{x}"

        assert visitor(42) == "int:42"
        assert visitor("hello") == "str:hello"
