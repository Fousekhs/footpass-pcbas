"""Tests for ``pcspot.models.graph``."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcspot.models.graph import (
    DEFAULT_RADIUS,
    EDGE_FEATURE_DIM,
    GraphEdges,
    HGTEncoder,
    PLAYER_EDGE_TYPES,
    PlayerHGTBlock,
    PlayerNodeEmbedder,
    ZONE_GRID,
    build_player_graphs,
    build_zone_assignment,
    compute_edge_features,
    zone_adjacency,
)


def _toy_inputs(B: int = 2, T: int = 4, P: int = 6, D: int = 16) -> tuple:
    torch.manual_seed(0)
    pitch_xy = torch.rand(B, T, P, 2)
    velocity = torch.zeros(B, T, P, 2)
    bbox = torch.zeros(B, T, P, 4)
    roles = torch.randint(1, 14, (B, T, P), dtype=torch.long)
    teams = torch.tensor([[0, 0, 0, 1, 1, 1]] * B, dtype=torch.long)
    valid = torch.ones(B, T, P, dtype=torch.bool)
    valid[:, :, -1] = False  # last player padded out for half the steps
    return pitch_xy, velocity, bbox, roles, teams, valid, D


class BuildGraphsTests(unittest.TestCase):
    def test_masks_are_consistent(self) -> None:
        pitch_xy, _, _, _, teams, valid, _ = _toy_inputs()
        edges = build_player_graphs(teams, pitch_xy, valid, knn=2)
        # Required edge types are present.
        for et in ("same_team", "opponent", "near", "self"):
            self.assertIn(et, edges.masks)
        # No self-loops in same_team / opponent.
        eye = torch.eye(pitch_xy.shape[2], dtype=torch.bool)
        for et in ("same_team", "opponent"):
            m = edges.masks[et]
            self.assertFalse(bool(m[..., eye].any()))
        # Near respects kNN cap.
        near = edges.masks["near"]
        # Each valid source row has at most knn=2 outgoing True entries
        # over its dst row (note: matrix is dst-major in our convention;
        # the `topk` was per-dst across src so we check per-row counts).
        per_row = near.sum(dim=-1)
        # Where the destination is valid, we expect <= knn neighbours.
        valid_dst = valid
        violations = ((per_row > 2) & valid_dst).sum().item()
        self.assertEqual(violations, 0)

    def test_invalid_players_have_no_edges(self) -> None:
        pitch_xy, _, _, _, teams, valid, _ = _toy_inputs()
        edges = build_player_graphs(teams, pitch_xy, valid, knn=3)
        # Padded players must have no incoming or outgoing same-team edge.
        same = edges.masks["same_team"]
        self.assertFalse(bool(same[:, :, -1, :].any()))
        self.assertFalse(bool(same[:, :, :, -1].any()))


class PlayerHGTBlockTests(unittest.TestCase):
    def test_forward_shapes_and_grad(self) -> None:
        pitch_xy, velocity, bbox, roles, teams, valid, D = _toy_inputs()
        embed = PlayerNodeEmbedder(hidden_dim=D)
        h = embed(pitch_xy, velocity, bbox, roles, teams)
        edges = build_player_graphs(teams, pitch_xy, valid, knn=3)
        block = PlayerHGTBlock(hidden_dim=D, num_heads=4)
        out, z_out = block(h, edges, teams)
        self.assertEqual(out.shape, h.shape)
        self.assertIsNone(z_out)  # zones not configured
        # Padded slots must be zeroed.
        self.assertTrue(torch.allclose(out[:, :, -1, :], torch.zeros_like(out[:, :, -1, :])))
        # Gradient flows.
        loss = (out * valid.unsqueeze(-1).float()).sum()
        loss.backward()
        # At least one trainable param has a non-zero grad.
        any_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.parameters())
        self.assertTrue(any_grad)

    def test_handles_fully_masked_edge_type(self) -> None:
        # Construct a mini batch where all valid masks for one type are False.
        B, T, P, D = 1, 2, 3, 8
        h = torch.randn(B, T, P, D, requires_grad=True)
        teams = torch.tensor([[0, 0, 0]], dtype=torch.long)
        valid = torch.ones(B, T, P, dtype=torch.bool)
        edges = GraphEdges(
            masks={
                "self": torch.eye(P, dtype=torch.bool).expand(B, T, P, P),
                "near": torch.zeros(B, T, P, P, dtype=torch.bool),  # empty
                "same_team": torch.ones(B, T, P, P, dtype=torch.bool)
                & ~torch.eye(P, dtype=torch.bool),
                "opponent": torch.zeros(B, T, P, P, dtype=torch.bool),
            },
            valid_player=valid,
        )
        block = PlayerHGTBlock(hidden_dim=D, num_heads=2)
        out, _ = block(h, edges, teams)
        self.assertFalse(torch.isnan(out).any())


class HGTEncoderTests(unittest.TestCase):
    def test_stack_runs(self) -> None:
        pitch_xy, velocity, bbox, roles, teams, valid, D = _toy_inputs()
        embed = PlayerNodeEmbedder(hidden_dim=D)
        h = embed(pitch_xy, velocity, bbox, roles, teams)
        edges = build_player_graphs(teams, pitch_xy, valid, knn=3)
        enc = HGTEncoder(hidden_dim=D, num_layers=3, num_heads=4)
        out = enc(h, edges, teams)
        self.assertEqual(out.shape, h.shape)


class EdgeFeatureTests(unittest.TestCase):
    def test_compute_edge_features_shape_and_zero_on_invalid(self) -> None:
        pitch_xy, velocity, _, _, teams, valid, _ = _toy_inputs()
        feats = compute_edge_features(
            teams=teams, pitch_xy=pitch_xy, valid=valid, velocity=velocity
        )
        B, T, P, _ = pitch_xy.shape
        self.assertEqual(feats.shape, (B, T, P, P, EDGE_FEATURE_DIM))
        # All channels for edges touching the padded player (last column)
        # must be zero by construction.
        self.assertEqual(float(feats[:, :, -1, :].abs().sum()), 0.0)
        self.assertEqual(float(feats[:, :, :, -1].abs().sum()), 0.0)

    def test_distance_channel_is_symmetric(self) -> None:
        pitch_xy, velocity, _, _, teams, valid, _ = _toy_inputs()
        feats = compute_edge_features(
            teams=teams, pitch_xy=pitch_xy, valid=valid, velocity=velocity
        )
        dist = feats[..., 0]
        # dist[b, t, i, j] should equal dist[b, t, j, i] on valid edges.
        valid_pair = (valid.unsqueeze(2) & valid.unsqueeze(3))
        self.assertTrue(torch.allclose(dist * valid_pair, dist.transpose(-1, -2) * valid_pair, atol=1e-5))

    def test_attention_uses_edge_features(self) -> None:
        torch.manual_seed(0)
        pitch_xy, velocity, bbox, roles, teams, valid, D = _toy_inputs()
        embed = PlayerNodeEmbedder(hidden_dim=D, use_acceleration=False)
        h = embed(pitch_xy, velocity, bbox, roles, teams)
        edges = build_player_graphs(
            teams,
            pitch_xy,
            valid,
            knn=3,
            velocity=velocity,
            compute_edge_features_flag=True,
        )
        block = PlayerHGTBlock(
            hidden_dim=D, num_heads=4, edge_feat_dim=EDGE_FEATURE_DIM
        )
        out, _ = block(h, edges, teams)
        self.assertEqual(out.shape, h.shape)
        # Gradient flow should also reach the edge_proj parameters.
        loss = (out * valid.unsqueeze(-1).float()).sum()
        loss.backward()
        any_edge_grad = False
        for et, attn in block.attentions.items():
            if attn.edge_proj is not None and attn.edge_proj.weight.grad is not None:
                if float(attn.edge_proj.weight.grad.abs().sum()) > 0.0:
                    any_edge_grad = True
                    break
        self.assertTrue(any_edge_grad)


class PlayerNodeEmbedderTests(unittest.TestCase):
    def test_acceleration_changes_output(self) -> None:
        torch.manual_seed(0)
        pitch_xy, velocity, bbox, roles, teams, _, D = _toy_inputs()
        embed = PlayerNodeEmbedder(hidden_dim=D, use_acceleration=True)
        h_zero = embed(pitch_xy, velocity, bbox, roles, teams, acceleration=None)
        accel = torch.ones_like(velocity)
        h_one = embed(pitch_xy, velocity, bbox, roles, teams, acceleration=accel)
        # The two forwards differ because non-zero acceleration shifts the
        # post-layernorm features.
        self.assertGreater(float((h_zero - h_one).abs().sum()), 0.0)

    def test_time_features_propagate(self) -> None:
        torch.manual_seed(0)
        pitch_xy, velocity, bbox, roles, teams, _, D = _toy_inputs()
        time_dim = 6
        embed = PlayerNodeEmbedder(
            hidden_dim=D, use_acceleration=False, time_dim=time_dim
        )
        time_a = torch.zeros(pitch_xy.shape[0], pitch_xy.shape[1], time_dim)
        time_b = torch.ones_like(time_a)
        h_a = embed(pitch_xy, velocity, bbox, roles, teams, time_features=time_a)
        h_b = embed(pitch_xy, velocity, bbox, roles, teams, time_features=time_b)
        self.assertGreater(float((h_a - h_b).abs().sum()), 0.0)

    def test_jersey_branch_changes_output(self) -> None:
        torch.manual_seed(0)
        pitch_xy, velocity, bbox, roles, teams, _, D = _toy_inputs()
        embed = PlayerNodeEmbedder(
            hidden_dim=D, use_acceleration=False, num_jerseys=20, jersey_dim=4
        )
        B, T, P, _ = pitch_xy.shape
        shirts_a = torch.zeros(B, P, dtype=torch.long)
        shirts_b = torch.full((B, P), 7, dtype=torch.long)
        h_a = embed(pitch_xy, velocity, bbox, roles, teams, shirt_numbers=shirts_a)
        h_b = embed(pitch_xy, velocity, bbox, roles, teams, shirt_numbers=shirts_b)
        self.assertGreater(float((h_a - h_b).abs().sum()), 0.0)

    def test_jersey_out_of_range_does_not_crash(self) -> None:
        torch.manual_seed(0)
        pitch_xy, velocity, bbox, roles, teams, _, D = _toy_inputs()
        embed = PlayerNodeEmbedder(
            hidden_dim=D, use_acceleration=False, num_jerseys=10, jersey_dim=4
        )
        B, T, P, _ = pitch_xy.shape
        # Sentinels and out-of-range get remapped to slot 0.
        shirts = torch.tensor([[-1, 0, 9, 999, -7, 50]] * B, dtype=torch.long)
        out = embed(pitch_xy, velocity, bbox, roles, teams, shirt_numbers=shirts)
        self.assertFalse(torch.isnan(out).any())

    def test_extra_scalar_branch_propagates(self) -> None:
        # Use non-constant per-channel scalars: a constant-along-the-last-dim
        # tensor would collapse to zero under LayerNorm (mean == value,
        # var == 0), so we'd see no difference even with the branch wired in.
        torch.manual_seed(0)
        pitch_xy, velocity, bbox, roles, teams, _, D = _toy_inputs()
        embed = PlayerNodeEmbedder(
            hidden_dim=D, use_acceleration=False, extra_scalar_dim=5
        )
        B, T, P, _ = pitch_xy.shape
        base = torch.arange(5, dtype=torch.float32).view(1, 1, 1, 5).expand(B, T, P, 5)
        extra_a = base.clone()
        extra_b = base + torch.randn(B, T, P, 5) * 0.5
        h_a = embed(pitch_xy, velocity, bbox, roles, teams, extra_scalars=extra_a)
        h_b = embed(pitch_xy, velocity, bbox, roles, teams, extra_scalars=extra_b)
        self.assertGreater(float((h_a - h_b).abs().sum()), 0.0)


class ZoneBuilderTests(unittest.TestCase):
    def test_assignment_rows_sum_to_one_for_valid(self) -> None:
        torch.manual_seed(0)
        B, T, P = 2, 3, 5
        pitch_xy = torch.rand(B, T, P, 2)
        valid = torch.ones(B, T, P, dtype=torch.bool)
        valid[:, :, -1] = False
        A = build_zone_assignment(pitch_xy, valid, grid=(6, 4))
        # Valid rows sum to 1 (within fp tolerance).
        row_sum = A.sum(dim=-1)
        self.assertTrue(
            torch.allclose(row_sum[valid], torch.ones_like(row_sum[valid]), atol=1e-5)
        )
        # Padded rows are all-zero.
        self.assertEqual(float(A[~valid].abs().sum()), 0.0)

    def test_assignment_is_local(self) -> None:
        # Place a single player at the center of a known cell and check
        # the bulk of the weight falls on that cell.
        pitch_xy = torch.zeros(1, 1, 1, 2)
        # Grid (6, 4): cell (3, 2) center is at ((3+0.5)/6, (2+0.5)/4)
        pitch_xy[0, 0, 0, 0] = (3 + 0.5) / 6
        pitch_xy[0, 0, 0, 1] = (2 + 0.5) / 4
        valid = torch.ones(1, 1, 1, dtype=torch.bool)
        A = build_zone_assignment(pitch_xy, valid, grid=(6, 4))
        z = 2 * 6 + 3  # j * Gx + i
        self.assertAlmostEqual(float(A[0, 0, 0, z]), 1.0, places=5)

    def test_soft_splat_distributes_across_neighbors(self) -> None:
        # Player exactly on a cell boundary in x should split 50/50.
        pitch_xy = torch.zeros(1, 1, 1, 2)
        # Halfway between cell (3, 2) and (4, 2): u = 3.5 -> x = 4/6.
        pitch_xy[0, 0, 0, 0] = 4.0 / 6.0
        pitch_xy[0, 0, 0, 1] = (2 + 0.5) / 4
        valid = torch.ones(1, 1, 1, dtype=torch.bool)
        A = build_zone_assignment(pitch_xy, valid, grid=(6, 4))
        z_left = 2 * 6 + 3
        z_right = 2 * 6 + 4
        self.assertAlmostEqual(float(A[0, 0, 0, z_left]), 0.5, places=5)
        self.assertAlmostEqual(float(A[0, 0, 0, z_right]), 0.5, places=5)

    def test_adjacency_is_row_normalized(self) -> None:
        adj = zone_adjacency((6, 4))
        Z = 6 * 4
        self.assertEqual(adj.shape, (Z, Z))
        self.assertTrue(
            torch.allclose(adj.sum(dim=-1), torch.ones(Z), atol=1e-6)
        )
        # Corner zone (0, 0) has 4 neighbors including self.
        self.assertGreater(float(adj[0].sum()), 0.0)


class RadiusAndDegreeTests(unittest.TestCase):
    def test_radius_edges_present_when_enabled(self) -> None:
        pitch_xy, velocity, _, _, teams, valid, _ = _toy_inputs()
        edges = build_player_graphs(
            teams,
            pitch_xy,
            valid,
            knn=2,
            use_radius=True,
            radius=DEFAULT_RADIUS,
            compute_degree_counts=True,
        )
        self.assertIn("radius", edges.masks)
        self.assertIsNotNone(edges.degree_counts)
        self.assertEqual(edges.degree_counts.shape[-1], 2)

    def test_degree_counts_match_known_layout(self) -> None:
        # Place 3 same-team players in a tight cluster + 1 opp far away,
        # plus 1 same-team a bit far. Check counts at one of the cluster
        # members.
        B, T, P = 1, 1, 5
        pitch_xy = torch.zeros(B, T, P, 2)
        # cluster members at (0, 0), (0.01, 0), (0, 0.01)
        pitch_xy[0, 0, 0] = torch.tensor([0.0, 0.0])
        pitch_xy[0, 0, 1] = torch.tensor([0.01, 0.0])
        pitch_xy[0, 0, 2] = torch.tensor([0.0, 0.01])
        # same-team far away
        pitch_xy[0, 0, 3] = torch.tensor([0.9, 0.9])
        # opponent inside the radius
        pitch_xy[0, 0, 4] = torch.tensor([0.02, 0.02])
        teams = torch.tensor([[0, 0, 0, 0, 1]], dtype=torch.long)
        valid = torch.ones(B, T, P, dtype=torch.bool)
        edges = build_player_graphs(
            teams,
            pitch_xy,
            valid,
            knn=2,
            use_radius=True,
            radius=0.05,
            compute_degree_counts=True,
        )
        dc = edges.degree_counts[0, 0]  # (P, 2)
        # Player 0: same-team within 0.05 = {1, 2} -> 2; opp = {4} -> 1.
        self.assertAlmostEqual(float(dc[0, 0]), 2.0)
        self.assertAlmostEqual(float(dc[0, 1]), 1.0)


class ZoneBlockTests(unittest.TestCase):
    def test_zone_block_runs_and_grad_flows(self) -> None:
        pitch_xy, velocity, bbox, roles, teams, valid, D = _toy_inputs()
        B, T, P, _ = pitch_xy.shape
        embed = PlayerNodeEmbedder(hidden_dim=D)
        h = embed(pitch_xy, velocity, bbox, roles, teams)
        edges = build_player_graphs(teams, pitch_xy, valid, knn=3)
        edges.zone_assignment = build_zone_assignment(
            pitch_xy, valid, grid=ZONE_GRID
        )
        edges.zone_adjacency = zone_adjacency(ZONE_GRID)
        Z = ZONE_GRID[0] * ZONE_GRID[1]
        block = PlayerHGTBlock(hidden_dim=D, num_heads=4, num_zones=Z)
        out, z_out = block(h, edges, teams)
        self.assertEqual(out.shape, h.shape)
        self.assertEqual(z_out.shape, (B, T, Z, D))
        # Padded slots zeroed.
        self.assertTrue(
            torch.allclose(out[:, :, -1, :], torch.zeros_like(out[:, :, -1, :]))
        )
        loss = (out * valid.unsqueeze(-1).float()).sum()
        loss.backward()
        # Zone-specific parameters get gradients.
        self.assertIsNotNone(block.zone_value.weight.grad)
        self.assertGreater(float(block.zone_value.weight.grad.abs().sum()), 0.0)

    def test_zone_counts_more_with_more_players(self) -> None:
        # Build two batches: one with 2 players stacked in a single zone,
        # one with 8 players in the same zone. The zone's per-team SUM
        # count must scale linearly with the number of players. We place
        # them at an actual cell center (not a 4-corner like (0.5, 0.5))
        # so the soft bilinear splat fully concentrates each player's
        # contribution into one cell instead of spreading 0.25 to each
        # of 4 neighbors.
        D = 8
        Gx, Gy = ZONE_GRID
        Z = Gx * Gy
        # Center of cell (i=2, j=1) for grid (6, 4).
        cell_center_x = (2 + 0.5) / Gx
        cell_center_y = (1 + 0.5) / Gy
        block = PlayerHGTBlock(hidden_dim=D, num_heads=2, num_zones=Z)

        def _zone_count(num_players: int) -> float:
            B, T = 1, 1
            P = num_players
            pitch_xy = torch.empty(B, T, P, 2)
            pitch_xy[..., 0] = cell_center_x
            pitch_xy[..., 1] = cell_center_y
            valid = torch.ones(B, T, P, dtype=torch.bool)
            teams = torch.zeros(B, P, dtype=torch.long)  # all team 0
            h = torch.zeros(B, T, P, D)  # value contribution = 0
            A = build_zone_assignment(pitch_xy, valid, grid=ZONE_GRID)
            with torch.no_grad():
                _, cnt0, _ = block._update_zone(
                    h=h,
                    z_prev=None,
                    A=A,
                    adj=zone_adjacency(ZONE_GRID),
                    teams=teams,
                    valid=valid,
                )
            return float(cnt0.max())

        c2 = _zone_count(2)
        c8 = _zone_count(8)
        # The whole point of SUM aggregation: more players -> more count.
        self.assertGreater(c8, c2)
        self.assertAlmostEqual(c2, 2.0, places=4)
        self.assertAlmostEqual(c8, 8.0, places=4)


if __name__ == "__main__":
    unittest.main()
