"""Player-Centric soccer action spotting package.

Combines a Heterogeneous Graph Transformer (HGT) over per-frame
multi-agent graphs with an MS-TCN++ temporal bridge over per-player
sequences and a player-aware adaptation of the SoccerNet CALF loss.

The package targets the FOOTPASS / SoccerNet PCBAS-2026 layout already
mirrored under ``data/pcbas_one_match/`` and consumed by
``scripts/pcbas_data.py``. See ``docs/player_centric_hgt_mstcn_calf_design.md``
for the full design.
"""

__version__ = "0.1.0"
