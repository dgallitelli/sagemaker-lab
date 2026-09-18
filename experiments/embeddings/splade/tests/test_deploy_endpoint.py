import pytest

from deploy_endpoint import _ndcg_at_k


def test_ndcg_at_k_uses_global_math_module() -> None:
    result = _ndcg_at_k(
        ranked_pids=["relevant", "less-relevant"],
        qrels={"relevant": 1.0, "less-relevant": 0.5},
        k=2,
    )

    assert result == pytest.approx(1.0)
