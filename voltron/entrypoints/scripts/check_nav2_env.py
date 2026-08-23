"""Print the resolved Nav2 runtime environment summary for a version profile."""

from __future__ import annotations

import argparse
import json

from voltron.integrations.navigation.nav2.navigator import (
    DEFAULT_NAV2_VERSION_PROFILE,
    NAV2_VERSION_PROFILES,
    SubprocessNav2ComputePathClient,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        type=str,
        default=DEFAULT_NAV2_VERSION_PROFILE,
        choices=sorted(NAV2_VERSION_PROFILES),
    )
    parser.add_argument("--action-name", type=str, default="compute_path_to_pose")
    args = parser.parse_args()

    client = SubprocessNav2ComputePathClient(
        profile=NAV2_VERSION_PROFILES[args.profile],
        action_name=args.action_name,
    )
    print(json.dumps(client.inspect_environment(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
