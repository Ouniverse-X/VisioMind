from __future__ import annotations

import argparse
import json

from visiomind.action.config_loader import add_config_argument
from visiomind.action.integrations.navigation.nav2.navigator import DEFAULT_NAV2_VERSION_PROFILE


def _add_task_args(parser: argparse.ArgumentParser) -> None:
    add_config_argument(parser)
    parser.add_argument("--task-id", type=str, default="task_behavior_001")
    parser.add_argument("--task-desc", type=str, default="把红色杯子从厨房拿到客厅")
    parser.add_argument(
        "--task-type",
        choices=["manipulation", "navigation", "interaction", "observation"],
        default="manipulation",
    )
    parser.add_argument(
        "--planner-mode",
        choices=["auto", "scripted", "benchmark"],
        default="auto",
        help="Execution planning policy. auto: runtime-state-driven mixed-agent planning; scripted: fixed templates; benchmark: deterministic constrained planning.",
    )


def _add_behavior_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env-id", type=str, default="sim_behavior_r1_pro/tidying_bedroom")
    parser.add_argument("--env-idx", type=int, default=0)
    parser.add_argument("--total-n-envs", type=int, default=1)
    parser.add_argument(
        "--behavior-scene-file",
        type=str,
        default=None,
        help="Absolute or relative path to a BEHAVIOR scene JSON file. When set, overrides the default cached task template.",
    )
    parser.add_argument(
        "--behavior-tro-state-file",
        type=str,
        default=None,
        help="Absolute or relative path to a BEHAVIOR TRO state JSON file. When set, overrides the default task-instance state.",
    )
    parser.add_argument(
        "--behavior-task-instance-id",
        type=int,
        default=None,
        help="Optional BEHAVIOR task instance id to force for this run.",
    )
    parser.add_argument(
        "--behavior-scene-state-include-aabb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include live object/door axis-aligned bounding boxes in sampled scene state.",
    )
    parser.add_argument(
        "--behavior-robot-start-position",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional BEHAVIOR robot start position override applied after scene-file pose loading.",
    )
    parser.add_argument(
        "--behavior-robot-start-orientation",
        type=float,
        nargs=4,
        default=None,
        metavar=("QX", "QY", "QZ", "QW"),
        help="Optional BEHAVIOR robot start orientation override applied after scene-file pose loading.",
    )
    parser.add_argument(
        "--behavior-post-reset-robot-position",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional test-only robot position applied after env.reset() succeeds.",
    )
    parser.add_argument(
        "--behavior-post-reset-robot-orientation",
        type=float,
        nargs=4,
        default=None,
        metavar=("QX", "QY", "QZ", "QW"),
        help="Optional test-only robot orientation applied after env.reset() succeeds.",
    )
    parser.add_argument(
        "--behavior-post-reset-object-states",
        type=json.loads,
        default=None,
        metavar="JSON",
        help="Exact object pose and named-state overrides keyed by BEHAVIOR object name.",
    )
    parser.add_argument(
        "--behavior-post-reset-robot-joint-positions",
        type=float,
        nargs="+",
        default=None,
        metavar="QPOS",
        help="Exact robot joint positions applied after env.reset().",
    )
    parser.add_argument(
        "--behavior-post-reset-robot-joint-velocities",
        type=float,
        nargs="+",
        default=None,
        metavar="QVEL",
        help="Exact robot joint velocities applied after env.reset().",
    )
    parser.add_argument(
        "--behavior-post-reset-refresh-observation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refresh the post-reset observation by rendering without stepping physics.",
    )
    parser.add_argument(
        "--behavior-post-reset-settle-steps",
        type=int,
        default=5,
        help="Number of zero-action env steps after post-reset robot pose override.",
    )
    parser.add_argument(
        "--behavior-builtin-vlm-detector-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the BEHAVIOR/GR00T built-in asynchronous VLM detector. "
            "Disabled by default; VisioMindAction Vision completion remains the primary completion path."
        ),
    )
    parser.add_argument("--embodiment", type=str, default="behavior_r1_pro")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-control-steps", type=int, default=120)
    parser.add_argument(
        "--runtime-termination-use-environment-success-signal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow environment predicate success to terminate task/subtask execution.",
    )
    parser.add_argument(
        "--runtime-termination-use-brain-completion-signal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow Brain/Vision completion verdicts to terminate task/subtask execution.",
    )
    parser.add_argument(
        "--runtime-termination-environment-signal-policy",
        choices=["allow_early_success", "evidence_only"],
        default="allow_early_success",
        help="Use environment success for early termination or only as completion evidence.",
    )
    parser.add_argument(
        "--action-verify-every-control-steps",
        type=int,
        default=400,
        help="Run Action internal-step VLM verification every N successful control steps.",
    )
    parser.add_argument(
        "--action-max-unverified-internal-step-control-steps",
        type=int,
        default=5000,
        help="Maximum control steps for each unverified Action internal step before the runtime gives up.",
    )
    parser.add_argument(
        "--action-control-mode",
        choices=["whole_body_local", "manipulation_only"],
        default="whole_body_local",
        help="Control mode injected into ACTION subtasks.",
    )
    parser.add_argument(
        "--action-allow-base-motion",
        action="store_true",
        help="Allow local base commands during ACTION subtasks for whole-body policies.",
    )
    parser.add_argument(
        "--progress-log-every",
        type=int,
        default=20,
        help="Emit closed-loop progress logs every N control steps. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--recording-video-scale",
        type=float,
        default=1.0,
        help="Scale BEHAVIOR recording frames before writing video. Use 0.5 for roughly half-size output.",
    )
    parser.add_argument(
        "--logging-verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write detailed process logs. Use --no-logging-verbose to suppress high-frequency action and pose records.",
    )
    parser.add_argument(
        "--logging-memory-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write memory diagnostic snapshots to process logs. Disabled by default because each snapshot is verbose.",
    )
    parser.add_argument(
        "--logging-nav2-path-snapshots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write one Nav2 path snapshot whenever the planned path changes.",
    )
    parser.add_argument(
        "--log-navigation-candidates",
        action="store_true",
        help="Emit one compact Navigation candidate-point snapshot per unique selection for map debugging.",
    )
    parser.add_argument(
        "--no-auto-register",
        action="store_true",
        help="Disable BEHAVIOR env auto-registration and assume env id is already registered.",
    )


def _add_service_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gr00t-host", type=str, default="127.0.0.1")
    parser.add_argument("--gr00t-port", type=int, default=5555)
    parser.add_argument(
        "--policy-backend",
        choices=["groot", "pi05", "openpi_comet"],
        default="groot",
        help="groot: GR00T via ZeroMQ; pi05: Pi0.5 via WebSocket; openpi_comet: official OpenPI Comet WebSocket",
    )
    parser.add_argument("--pi05-endpoint", type=str, default="ws://127.0.0.1:9000")
    parser.add_argument("--pi05-timeout-s", type=float, default=15.0)
    parser.add_argument(
        "--pi05-task-id",
        type=int,
        default=None,
        help="BEHAVIOR task ID for Pi0.5 checkpoint switching and stage tracking.",
    )
    parser.add_argument("--openpi-comet-endpoint", type=str, default="ws://127.0.0.1:9000")
    parser.add_argument("--openpi-comet-timeout-s", type=float, default=60.0)
    parser.add_argument("--openpi-comet-task-name", type=str, default=None)
    parser.add_argument("--openpi-comet-task-id", type=int, default=None)
    parser.add_argument("--openpi-comet-prompt", type=str, default=None)
    parser.add_argument(
        "--openpi-comet-action-mode",
        choices=["raw", "dict"],
        default="raw",
        help="raw: pass native robot_r1 23D actions; dict: split into VisioMindAction action.* keys.",
    )
    parser.add_argument("--vision-endpoint", type=str, default="http://127.0.0.1:8081/process")
    parser.add_argument("--vision-timeout-s", type=float, default=60.0)
    parser.add_argument("--vision-max-retries", type=int, default=0)
    parser.add_argument("--vision-retry-backoff-s", type=float, default=1.0)
    parser.add_argument(
        "--vision-completion-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Vision-backed completion verdicts.",
    )
    parser.add_argument("--vision-completion-positive-streak", type=int, default=3)
    parser.add_argument("--vision-completion-stability-steps", type=int, default=5)
    parser.add_argument("--vision-completion-min-confidence", type=float, default=0.75)
    parser.add_argument("--vision-completion-action-delta-threshold", type=float, default=0.03)
    parser.add_argument(
        "--vision-completion-check-interval-steps",
        type=int,
        default=200,
        help="Run Brain/Vision completion checks every N control steps. Environment failures still terminate immediately.",
    )
    parser.add_argument(
        "--vision-completion-agent-scope",
        nargs="+",
        default=["ACTION"],
        help="Agent names whose subtask completion should use Brain/Vision checks. Default: ACTION.",
    )
    parser.add_argument(
        "--vision-completion-use-memory-guidance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--vision-completion-include-third-person",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the recording third-person camera in Vision completion composites when available.",
    )
    parser.add_argument(
        "--vision-completion-max-images",
        type=int,
        default=4,
        help="Maximum number of camera views to stitch into one Vision completion image.",
    )
    parser.add_argument("--vision-completion-max-image-side-px", type=int, default=1024)
    parser.add_argument("--vision-completion-jpeg-quality", type=int, default=90)
    parser.add_argument("--vision-completion-max-image-b64-chars", type=int, default=900000)
    parser.add_argument(
        "--vision-completion-image-detail",
        choices=["low", "high", "auto"],
        default="high",
        help="OpenAI image detail setting for completion monitor images.",
    )
    parser.add_argument(
        "--vision-heartbeat-interval-steps",
        type=int,
        default=200,
        help="Run task-level asynchronous Vision environment heartbeat every N environment steps. Set <= 0 to disable.",
    )
    parser.add_argument("--memory-agent-endpoint", type=str, default="http://127.0.0.1:8070/rpc")
    parser.add_argument(
        "--memory-mode",
        choices=["agent", "local"],
        default="agent",
        help="agent: MemoryAgentClient; local: local MemoryAgent over HEMSAdapter",
    )
    parser.add_argument(
        "--memory-agent-enabled", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--memory-llm-backend", type=str, default=None)
    parser.add_argument("--memory-llm-base-url", type=str, default=None)
    parser.add_argument("--memory-llm-model", type=str, default=None)
    parser.add_argument("--memory-llm-api-key", type=str, default=None)
    parser.add_argument("--memory-llm-api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--memory-llm-timeout-s", type=float, default=30.0)
    parser.add_argument("--memory-llm-temperature", type=float, default=0.0)
    parser.add_argument("--memory-llm-max-retries", type=int, default=0)
    parser.add_argument("--memory-llm-retry-backoff-s", type=float, default=1.0)
    parser.add_argument(
        "--memory-experience-extraction-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--memory-experience-extraction-min-confidence-to-write", type=float, default=0.4
    )
    parser.add_argument(
        "--memory-experience-extraction-min-confidence-to-promote", type=float, default=0.7
    )
    parser.add_argument(
        "--memory-experience-extraction-extract-completion-criteria",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--memory-experience-extraction-extract-clarification-answers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )


def _add_llm_and_action_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--brain-planner",
        choices=["openai", "rule"],
        default="openai",
        help="openai: use an OpenAI-compatible gateway; rule: use local RuleBasedPlanner",
    )
    parser.add_argument("--brain-base-url", type=str, default=None)
    parser.add_argument("--brain-model", type=str, default=None)
    parser.add_argument("--brain-api-key", type=str, default=None)
    parser.add_argument("--brain-api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--brain-timeout-s", type=float, default=30.0)
    parser.add_argument("--brain-temperature", type=float, default=0.1)
    parser.add_argument("--brain-max-retries", type=int, default=0)
    parser.add_argument("--brain-retry-backoff-s", type=float, default=1.0)
    parser.add_argument(
        "--brain-interactive-planning-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draft, clarify, and confirm a text plan before executable planning.",
    )
    parser.add_argument(
        "--brain-interactive-planning-require-user-confirmation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--brain-interactive-planning-ask-when-uncertain",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--brain-interactive-planning-max-questions", type=int, default=5)
    parser.add_argument(
        "--brain-interactive-planning-reuse-memory-criteria-min-confidence",
        type=float,
        default=0.8,
    )
    parser.add_argument("--action-selector", choices=["openai", "heuristic"], default="openai")
    parser.add_argument("--action-base-url", type=str, default=None)
    parser.add_argument("--action-model", type=str, default=None)
    parser.add_argument("--action-api-key", type=str, default=None)
    parser.add_argument("--action-api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--action-timeout-s", type=float, default=10.0)
    parser.add_argument("--action-temperature", type=float, default=0.0)
    parser.add_argument("--action-max-retries", type=int, default=0)
    parser.add_argument("--action-retry-backoff-s", type=float, default=1.0)
    parser.add_argument(
        "--action-internal-planning-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Action Agent internal subtask decomposition before VLA execution.",
    )
    parser.add_argument(
        "--action-internal-step-completion-use-vision-completion-monitor",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--action-internal-step-completion-require-verified-completion",
        action=argparse.BooleanOptionalAction,
        default=True,
    )


def _add_navigation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--navigation-backend",
        choices=["nav2"],
        default="nav2",
        help="VLN backend. Only Nav2 is supported.",
    )
    parser.add_argument("--navigation-base-url", type=str, default=None)
    parser.add_argument("--navigation-model", type=str, default=None)
    parser.add_argument("--navigation-api-key", type=str, default=None)
    parser.add_argument("--navigation-api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--navigation-timeout-s", type=float, default=30.0)
    parser.add_argument("--navigation-temperature", type=float, default=0.1)
    parser.add_argument("--navigation-max-retries", type=int, default=0)
    parser.add_argument("--navigation-retry-backoff-s", type=float, default=1.0)
    parser.add_argument(
        "--hovsg-graph-root",
        type=str,
        default=None,
        help="Root directory containing <scene_id>/graph exports from HOV-SG.",
    )
    parser.add_argument(
        "--hovsg-scene-id",
        type=str,
        default=None,
        help="Scene id used to resolve HOV-SG graph assets and stored into task/runtime metadata.",
    )
    parser.add_argument(
        "--hovsg-graph-path",
        type=str,
        default=None,
        help="Direct path to one exported HOV-SG graph directory. Overrides --hovsg-graph-root for the selected scene.",
    )
    parser.add_argument(
        "--hovsg-scene-map",
        type=str,
        default=None,
        help="Path to a JSON map: {env_id: {scene_id, graph_root|graph_path}}.",
    )
    parser.add_argument(
        "--hovsg-nav-graph-type",
        type=str,
        default=None,
        help="Preferred HOV-SG navigation graph type, e.g. voronoi_graph or global_room_graph.",
    )
    parser.add_argument(
        "--hovsg-direct-room-transition-max-gap-m",
        type=float,
        default=0.25,
        help="Maximum cross-room gap treated as a strong doorway transition when generating HOV-SG portal waypoints.",
    )
    parser.add_argument(
        "--hovsg-direct-room-transition-min-span-m",
        type=float,
        default=1.0,
        help="Minimum doorway span treated as a strong doorway transition when generating HOV-SG portal waypoints.",
    )
    parser.add_argument(
        "--hovsg-object-approach-min-portal-stance-clearance-m",
        type=float,
        default=0.45,
        help="Minimum allowed distance between an object-approach final stance and door-like objects.",
    )
    parser.add_argument(
        "--nav2-version-profile",
        type=str,
        default=DEFAULT_NAV2_VERSION_PROFILE,
        help="Pinned Nav2 runtime profile. Use this instead of mixing ROS packages into the VisioMindAction env.",
    )
    parser.add_argument(
        "--nav2-action-name",
        type=str,
        default="compute_path_to_pose",
        help="Nav2 action name used by the ROS subprocess worker.",
    )
    parser.add_argument(
        "--nav2-planner-id",
        type=str,
        default=None,
        help="Optional Nav2 planner id passed to ComputePathToPose.",
    )
    parser.add_argument(
        "--nav2-frame-id",
        type=str,
        default="map",
        help="Map frame used for Nav2 ComputePathToPose requests.",
    )
    parser.add_argument(
        "--nav2-timeout-s",
        type=float,
        default=8.0,
        help="Timeout for Nav2 compute-path requests.",
    )
    parser.add_argument(
        "--nav2-strict",
        action="store_true",
        help="Fail fast when Nav2 is unavailable instead of falling back to semantic HOV-SG waypoints.",
    )
    parser.add_argument(
        "--nav2-trav-map-filename",
        type=str,
        default=None,
        help="Optional traversability map filename to use instead of inferring from the scene filename.",
    )
    parser.add_argument(
        "--nav2-portal-analysis-map-resolution",
        type=float,
        default=0.05,
        help="Resolution used when checking doorway traversability for portal refinement.",
    )
    parser.add_argument(
        "--nav2-portal-clearance-radius-m",
        type=float,
        default=0.35,
        help="Circular clearance radius used when validating portal crossings against the traversability map.",
    )
    parser.add_argument(
        "--nav2-portal-corridor-standoff-m",
        type=float,
        default=0.18,
        help="Minimum standoff distance used to build a pre/post-portal transition corridor.",
    )
    parser.add_argument(
        "--nav2-portal-sampling-step-m",
        type=float,
        default=0.05,
        help="Sampling interval along the portal span when searching for a traversable crossing.",
    )
    parser.add_argument(
        "--nav2-local-path-clearance-radius-m",
        type=float,
        default=0.0,
        help="Minimum obstacle clearance radius enforced for Nav2 local-path planning and post-processing.",
    )
    parser.add_argument(
        "--nav2-local-path-waypoint-spacing-m",
        type=float,
        default=0.35,
        help="Spacing used when resampling Nav2 local paths into controller waypoints.",
    )
    parser.add_argument(
        "--navigation-prefer-forward-facing-motion",
        action="store_true",
        help="Prefer turning the base to face the travel direction instead of using lateral holonomic motion during waypoint tracking.",
    )
    parser.add_argument(
        "--navigation-portal-alignment-distance-threshold",
        type=float,
        default=1.2,
        help="Only enter explicit portal-alignment control when the active portal waypoint is within this distance.",
    )
    parser.add_argument(
        "--navigation-portal-prealign-distance-threshold-m",
        type=float,
        default=1.2,
        help="Distance to doorway midpoint at which local-path tracking starts blending its heading toward the portal normal.",
    )
    parser.add_argument(
        "--navigation-portal-alignment-footprint-width-m",
        type=float,
        default=0.72,
        help="Effective base width used to tighten portal centering tolerance for narrow doors.",
    )
    parser.add_argument(
        "--navigation-portal-alignment-min-lateral-deadband-m",
        type=float,
        default=0.01,
        help="Minimum doorway-centering tolerance allowed after width-based tightening.",
    )
    parser.add_argument(
        "--navigation-portal-alignment-wide-clearance-margin-m",
        type=float,
        default=0.4,
        help="Skip explicit portal alignment when portal width exceeds base width by at least this total margin.",
    )
    parser.add_argument(
        "--navigation-max-linear-velocity",
        type=float,
        default=0.60,
        help="Maximum forward linear velocity used by the waypoint fallback controller.",
    )
    parser.add_argument(
        "--navigation-linear-gain",
        type=float,
        default=0.45,
        help="Base linear gain used by the waypoint fallback controller.",
    )
    parser.add_argument(
        "--navigation-local-path-linear-gain",
        type=float,
        default=0.75,
        help="Linear gain used while following Nav2 local-path waypoints.",
    )
    parser.add_argument(
        "--navigation-local-path-max-linear-velocity",
        type=float,
        default=0.85,
        help="Maximum forward linear velocity used while following Nav2 local-path waypoints.",
    )
    parser.add_argument(
        "--navigation-portal-alignment-max-linear-velocity",
        type=float,
        default=0.28,
        help="Maximum forward linear velocity allowed during portal alignment.",
    )
    parser.add_argument(
        "--navigation-object-approach-final-waypoint-tolerance-m",
        type=float,
        default=0.45,
        help="Final waypoint radius required before handing object_approach goals to manipulation.",
    )
    parser.add_argument(
        "--navigation-max-angular-velocity",
        type=float,
        default=0.8,
        help="Maximum angular velocity used by the waypoint fallback controller.",
    )
    parser.add_argument(
        "--navigation-local-path-angular-gain-scale",
        type=float,
        default=0.7,
        help="Angular-gain scale applied while following Nav2 local-path waypoints.",
    )


def build_closed_loop_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _add_task_args(parser)
    _add_behavior_runtime_args(parser)
    _add_service_args(parser)
    _add_llm_and_action_args(parser)
    _add_navigation_args(parser)
    return parser


__all__ = ["build_closed_loop_parser"]
