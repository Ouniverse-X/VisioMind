# Real Isaac Sim run evidence

Strict success requires a verified action-free terminal placement, guarded release, post-settle container containment, and requested-cell 3-D AABB containment. The broader environment `task_success` flag is recorded but is not used as the acceptance criterion.

Generated from 4 selected engineering runs: 0 strict successes and 4 strict failures. This is a reproducibility audit, not a statistically powered benchmark.

| Run | Strict | Control / env steps | Navigation | Drop | Released / contained | Video |
|---|---:|---:|---:|---:|---:|---:|
| half_apple_to_packing_box_place_inside_i10_20260823_190553 | FAIL | 873 / 1296 | 0.791 m | 0.190 m | True / True | 5.3 MB |
| half_apple_to_packing_box_place_inside_i10_20260823_191650 | FAIL | - / 1269 | - m | - m | - / - | 5.2 MB |
| half_apple_to_packing_box_place_inside_i10_20260823_193048 | FAIL | 872 / 1282 | 0.791 m | 0.167 m | True / True | 5.3 MB |
| plier_to_toolbox_cell3_industrial_i00_20260824_200729 | FAIL | - / 4764 | - m | - m | - / - | 15.3 MB |

The JSON companion contains SHA-256 hashes, exact geometry, negative-run failure codes, and all terminal gate fields.
