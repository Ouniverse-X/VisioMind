"""CLI wrapper for generating BEHAVIOR variant traversability maps."""

from voltron.integrations.simulator.behavior.tools.variant_trav_map import (
    build_parser,
    generate_variant_trav_map,
    main,
)

__all__ = ["build_parser", "generate_variant_trav_map", "main"]


if __name__ == "__main__":
    main()
