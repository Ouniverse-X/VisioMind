# Real Isaac Sim run evidence

Strict success requires a verified action-free terminal placement, guarded release, and post-settle 3-D AABB containment. The broader environment `task_success` flag is recorded but is not used as the acceptance criterion.

Generated from 3 selected engineering runs: 2 strict successes and 1 strict failure. This is a reproducibility audit, not a statistically powered benchmark.

| Run | Strict | Control / env steps | Navigation | Drop | Released / contained | Video |
|---|---:|---:|---:|---:|---:|---:|
| half_apple_to_packing_box_place_inside_i10_20260823_190553 | PASS | 873 / 1296 | 0.791 m | 0.190 m | True / True | 5.3 MB |
| half_apple_to_packing_box_place_inside_i10_20260823_191650 | FAIL | - / 1269 | - m | - m | - / - | 5.2 MB |
| half_apple_to_packing_box_place_inside_i10_20260823_193048 | PASS | 872 / 1282 | 0.791 m | 0.167 m | True / True | 5.3 MB |

The JSON companion contains SHA-256 hashes, exact geometry, negative-run failure codes, and all terminal gate fields.
