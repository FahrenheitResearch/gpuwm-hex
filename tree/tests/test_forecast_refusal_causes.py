"""Stopped forecasts preserve the concrete cause behind transaction wrappers."""
from hexcore.drivers import run_cuda_v841_forecast as forecast


def test_transaction_refusal_retains_the_chained_numerical_failure() -> None:
    try:
        try:
            raise FloatingPointError("column 7 produced a nonfinite tendency")
        except FloatingPointError as cause:
            raise forecast.proof.CompositeTransactionError("step aborted without publication") from cause
    except forecast.proof.CompositeTransactionError as error:
        details = forecast.refusal_exception_details(error)
    assert details["exception"] == "CompositeTransactionError"
    assert details["message"] == "step aborted without publication"
    assert details["exception_chain"] == [
        {"exception": "CompositeTransactionError", "message": "step aborted without publication", "relation": "raised"},
        {"exception": "FloatingPointError", "message": "column 7 produced a nonfinite tendency", "relation": "cause"},
    ]
    assert "FloatingPointError: column 7" in details["traceback"]
    assert "direct cause" in details["traceback"]


def test_implicit_context_is_retained_but_explicit_suppression_is_respected() -> None:
    try:
        try:
            raise ValueError("original input refusal")
        except ValueError:
            raise RuntimeError("wrapper")
    except RuntimeError as error:
        details = forecast.refusal_exception_details(error)
    assert details["exception_chain"][1]["relation"] == "context"
    try:
        try:
            raise ValueError("suppressed context")
        except ValueError:
            raise RuntimeError("intentional standalone refusal") from None
    except RuntimeError as error:
        details = forecast.refusal_exception_details(error)
    assert len(details["exception_chain"]) == 1
    assert "suppressed context" not in details["traceback"]


def test_exception_cycle_does_not_loop() -> None:
    error = RuntimeError("cycle")
    error.__cause__ = error
    details = forecast.refusal_exception_details(error)
    assert len(details["exception_chain"]) == 1
    assert "RuntimeError: cycle" in details["traceback"]
