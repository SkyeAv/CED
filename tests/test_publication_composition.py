"""Publication projections preserve the underlying semantic provider contract."""

import torch

from ced.model.training import PublicationFeatureProvider, RandomProjectionProvider


def test_publication_provider_preserves_base_features_and_reconstruction() -> None:
    class SemanticBase(RandomProjectionProvider):
        def reconstruction_loss(self, relation_id: torch.Tensor, statement_id: torch.Tensor) -> torch.Tensor:
            return relation_id.float().sum() + statement_id.float().sum()

    base = SemanticBase(8, 4, 1, 7)
    provider = PublicationFeatureProvider(8, 4, 1, 7, torch.zeros(2, 3), torch.zeros(2), base=base)
    relations = torch.tensor([0, 1])
    statements = torch.tensor([0, 1])
    assert torch.equal(provider.node_features(statements), base.node_features(statements))
    assert torch.equal(
        provider.statement_features(relations, statements), base.statement_features(relations, statements)
    )
    assert provider.reconstruction_loss(relations, statements) == 2
    assert "base.node_vectors" in provider.state_dict()
