"""CLI wrapper for exporting BEHAVIOR scene graphs."""

from voltron.integrations.simulator.behavior.tools.scene_graph_export import build_parser, export_behavior_scene_graph, main

__all__ = ["build_parser", "export_behavior_scene_graph", "main"]


if __name__ == "__main__":
    main()
