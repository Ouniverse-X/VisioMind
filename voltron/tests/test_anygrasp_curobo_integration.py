from __future__ import annotations

# Inject the competition overlay modules so the tests resolve the updated
# classes (like GraspExecutor with begin_place_inside) instead of the original repo.
import importlib.util
import sys
from pathlib import Path

COMP_ROOT = Path(__file__).resolve().parents[2] / "voltron"
if COMP_ROOT.exists():
    def _install_comp_overlay(module_name: str, relative_path: str, *, load_as: str | None = None) -> None:
        path = COMP_ROOT / relative_path
        load_name = load_as or module_name
        spec = importlib.util.spec_from_file_location(load_name, path)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            sys.modules[load_name] = module
            spec.loader.exec_module(module)
            if load_as:
                sys.modules[module_name] = module
            if "." in module_name:
                parent_name, attribute = module_name.rsplit(".", 1)
                try:
                    parent = importlib.import_module(parent_name)
                    setattr(parent, attribute, module)
                except ImportError:
                    pass

    _install_comp_overlay("voltron.config_loader", "config_loader.py")
    _install_comp_overlay("voltron.agents.action.body.skill_selection", "agents/action/body/skill_selection.py")
    try:
        importlib.import_module("voltron.integrations.manipulation.anygrasp.frame_adapter")
    except ImportError:
        pass
    _install_comp_overlay(
        "voltron.integrations.manipulation.anygrasp.grasp_executor",
        "integrations/manipulation/anygrasp/grasp_executor.py",
        load_as="voltron.integrations.manipulation.anygrasp.grasp_executor_competition_overlay",
    )
    _install_comp_overlay(
        "voltron.agents.action.skills.execution.anygrasp_skill",
        "agents/action/skills/execution/anygrasp_skill.py",
    )
    _install_comp_overlay("voltron.agents.action.skills.registry", "agents/action/skills/registry.py")

import base64
import json
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from voltron.agents.action.skills.execution.anygrasp_skill import (
    AnyGraspSkill,
    _open_jaw_clearance_passes,
    _world_vertical_grasp_rotation,
)
from voltron.agents.action.tools.action_projection import ActionProjection
from voltron.integrations.manipulation.anygrasp.detector import (
    AnyGraspDetector,
    GraspCandidate,
    filter_candidates_by_approach,
    validate_detection_inputs,
    workspace_limits,
)
from voltron.integrations.manipulation.anygrasp.frame_adapter import (
    ANYGRASP_TO_EEF_ROTATION,
    AnyGraspFrameAdapter,
)
from voltron.integrations.manipulation.anygrasp.grasp_executor import (
    GraspExecution,
    GraspExecutor,
    GraspResult,
    GripperGeometryAdapter,
    _pose_to_matrix,
)
from voltron.integrations.manipulation.anygrasp.observation import (
    GraspObservation,
    capture_grasp_observation,
    optical_camera_pose_world,
    rgbd_to_points,
    target_mask_from_segmentation,
)
from voltron.runtime.orchestrator.closed_loop.completion_monitor import CompletionMonitor
from voltron.runtime.testing import MockMemoryAdapter, MockPolicyAdapter
from voltron.shared.context import ExecutionContext, LocalSkillSelection, Subtask, TaskRequest
from voltron.shared.enums import AgentName, AgentStatus, TaskType
from voltron.shared.models import RuntimeFeedback, SubtaskStepOutcome


@dataclass
class FakeCandidate:
    score: float = 0.9
    translation: np.ndarray = field(default_factory=lambda: np.array([0.1, 0.0, 0.5], dtype=np.float32))
    rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float32))
    width: float = 0.05
    depth: float = 0.02
    height: float = 0.03

    @property
    def approach_direction(self) -> np.ndarray:
        return self.rotation_matrix[:, 0]


def make_subtask() -> Subtask:
    return Subtask(
        subtask_id="st_grasp",
        agent=AgentName.ACTION,
        action="pick_up",
        target={"object": "radio_89"},
        parameters={"observation": {"annotation.human.coarse_action": ("pick up radio",)}},
    )


def make_context() -> ExecutionContext:
    return ExecutionContext(
        trace_id="trace",
        task_request=TaskRequest(
            task_id="task",
            description="pick up radio",
            task_type=TaskType.INTERACTION,
        ),
    )


def make_selection() -> LocalSkillSelection:
    return LocalSkillSelection(
        skill_id="anygrasp_manipulation_skill",
        confidence=0.95,
        reason="test",
        source="test",
    )


class FakeGroup:
    def __init__(self, scores: list[float] | None = None) -> None:
        values = scores or [0.9, 0.7]
        self.scores = np.asarray(values, dtype=np.float32)
        self.translations = np.asarray([[0.1 + i, 0.0, 0.5] for i in range(len(values))])
        self.rotation_matrices = np.repeat(np.eye(3)[None], len(values), axis=0)
        self.widths = np.full(len(values), 0.05)
        self.depths = np.full(len(values), 0.02)
        self.heights = np.full(len(values), 0.03)

    def __len__(self) -> int:
        return len(self.scores)

    def nms(self) -> "FakeGroup":
        return self

    def sort_by_score(self) -> "FakeGroup":
        return self


class TestObservationPreparation:
    def test_rgbd_projection_keeps_color_and_mask_alignment(self) -> None:
        depth = np.array([[1.0, 0.0], [2.0, 0.5]], dtype=np.float32)
        colors = np.array(
            [[[255, 0, 0], [0, 0, 0]], [[0, 255, 0], [0, 0, 255]]],
            dtype=np.uint8,
        )
        mask = np.array([[True, False], [False, True]])
        K = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
        points, aligned_colors, aligned_mask = rgbd_to_points(
            depth, colors, K, depth_trunc=1.5, target_mask=mask
        )
        np.testing.assert_allclose(points, [[0.0, 0.0, 1.0], [0.25, 0.25, 0.5]])
        np.testing.assert_allclose(aligned_colors, [[1, 0, 0], [0, 0, 1]])
        assert aligned_mask.tolist() == [True, True]

    def test_target_mask_uses_instance_label_mapping(self) -> None:
        seg = np.array([[0, 5], [7, 5]])
        mask = target_mask_from_segmentation(seg, {5: "radio_89", 7: "table_1"}, "radio_89")
        assert mask.tolist() == [[False, True], [False, True]]

    def test_target_mask_matches_bddl_category_alias(self) -> None:
        seg = np.array([[0, 5], [5, 0]])
        labels = {5: "radio_receiver.n.01_1"}
        mask = target_mask_from_segmentation(seg, labels, "radio_89")
        assert mask.tolist() == [[False, True], [True, False]]

    def test_head_cam_alias_resolves_r1_zed_sensor(self) -> None:
        from voltron.integrations.manipulation.anygrasp.observation import _find_sensor

        expected = object()
        robot = SimpleNamespace(sensors={"robot_r1:zed_link:Camera:0": expected})
        assert _find_sensor(robot, "head_cam") is expected

    def test_dynamic_modalities_are_rendered_and_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = {"renders": 0, "reads": 0}

        class FakeSensor:
            def __init__(self) -> None:
                self.modalities = {"rgb"}
                self.intrinsic_matrix = np.eye(3, dtype=np.float32)

            def add_modality(self, modality: str) -> None:
                self.modalities.add(modality)

            def get_obs(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
                state["reads"] += 1
                if state["renders"] < 2:
                    raise RuntimeError("annotator frame is not ready")
                return (
                    {
                        "rgb": np.full((2, 2, 3), 128, dtype=np.uint8),
                        "depth_linear": np.ones((2, 2), dtype=np.float32),
                        "seg_instance": np.array([[5, 5], [0, 0]], dtype=np.int32),
                    },
                    {"seg_instance": {5: "radio_89"}},
                )

            def get_position_orientation(self) -> tuple[np.ndarray, np.ndarray]:
                return np.zeros(3), np.array([0, 0, 0, 1])

        fake_sim = SimpleNamespace(render=lambda: state.__setitem__("renders", state["renders"] + 1))
        monkeypatch.setitem(sys.modules, "omnigibson", SimpleNamespace(sim=fake_sim))
        sensor = FakeSensor()
        packet = capture_grasp_observation(
            SimpleNamespace(sensors={"head_cam": sensor}),
            sensor_name="head_cam",
            target_name="radio_89",
            mask_dilation_px=0,
            min_target_points=1,
            sensor_warmup_frames=1,
            sensor_read_retries=2,
        )
        assert sensor.modalities == {"rgb", "depth_linear", "seg_instance"}
        assert state == {"renders": 2, "reads": 2}
        assert packet.region_mask is not None and int(packet.region_mask.sum()) == 2

    def test_usd_to_optical_pose_flips_y_and_z_axes(self) -> None:
        pose = optical_camera_pose_world([1, 2, 3], [0, 0, 0, 1])
        np.testing.assert_allclose(pose[:3, :3], np.diag([1, -1, -1]))
        np.testing.assert_allclose(pose[:3, 3], [1, 2, 3])

    def test_invalid_intrinsics_retry_after_sensor_warmup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = {"renders": 0, "intrinsic_reads": 0}

        class FakeSensor:
            def __init__(self) -> None:
                self.modalities = {"rgb", "depth_linear", "seg_instance"}

            @property
            def intrinsic_matrix(self) -> np.ndarray:
                state["intrinsic_reads"] += 1
                if state["intrinsic_reads"] < 3:
                    return np.zeros((3, 3), dtype=np.float32)
                return np.array(
                    [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                )

            def get_obs(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
                return (
                    {
                        "rgb": np.full((2, 2, 3), 128, dtype=np.uint8),
                        "depth_linear": np.ones((2, 2), dtype=np.float32),
                        "seg_instance": np.array([[5, 5], [0, 0]], dtype=np.int32),
                    },
                    {"seg_instance": {5: "radio_89"}},
                )

            def get_position_orientation(self) -> tuple[np.ndarray, np.ndarray]:
                return np.zeros(3), np.array([0, 0, 0, 1])

        monkeypatch.setitem(
            sys.modules,
            "omnigibson",
            SimpleNamespace(
                sim=SimpleNamespace(
                    render=lambda: state.__setitem__("renders", state["renders"] + 1)
                )
            ),
        )
        packet = capture_grasp_observation(
            SimpleNamespace(sensors={"head_cam": FakeSensor()}),
            sensor_name="head_cam",
            target_name="radio_89",
            mask_dilation_px=0,
            min_target_points=1,
            sensor_warmup_frames=1,
            sensor_read_retries=3,
        )
        assert packet.points.shape == (4, 3)
        assert state["intrinsic_reads"] >= 3
        assert state["renders"] == 2

    def test_permanently_invalid_intrinsics_fail_closed(self) -> None:
        class FakeSensor:
            modalities = {"rgb", "depth_linear", "seg_instance"}
            intrinsic_matrix = np.zeros((3, 3), dtype=np.float32)

            def get_obs(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
                return (
                    {
                        "rgb": np.full((2, 2, 3), 128, dtype=np.uint8),
                        "depth_linear": np.ones((2, 2), dtype=np.float32),
                        "seg_instance": np.ones((2, 2), dtype=np.int32),
                    },
                    {"seg_instance": {1: "radio_89"}},
                )

            def get_position_orientation(self) -> tuple[np.ndarray, np.ndarray]:
                return np.zeros(3), np.array([0, 0, 0, 1])

        with pytest.raises(RuntimeError, match="intrinsics"):
            capture_grasp_observation(
                SimpleNamespace(sensors={"head_cam": FakeSensor()}),
                sensor_name="head_cam",
                target_name="radio_89",
                mask_dilation_px=0,
                min_target_points=1,
                sensor_warmup_frames=0,
                sensor_read_retries=2,
            )


class TestFrameAdapter:
    def test_basis_mapping_is_right_handed_and_explicit(self) -> None:
        adapter = AnyGraspFrameAdapter()
        mapping = adapter.validate_basis_mapping()
        np.testing.assert_allclose(
            ANYGRASP_TO_EEF_ROTATION.T @ ANYGRASP_TO_EEF_ROTATION,
            np.eye(3),
        )
        assert mapping["determinant"] == pytest.approx(1.0)
        assert mapping["anygrasp_approach_to_eef_z"] == [1.0, 0.0, 0.0]
        assert mapping["anygrasp_jaw_to_eef_y"] == [0.0, 1.0, 0.0]

    def test_camera_candidate_to_world_preserves_origin_and_axes(self) -> None:
        adapter = AnyGraspFrameAdapter()
        pose = adapter.camera_candidate_to_world(
            [0.1, 0.2, 0.8],
            np.eye(3),
            np.eye(4),
        )
        np.testing.assert_allclose(pose.canonical_origin_world, [0.1, 0.2, 0.8])
        np.testing.assert_allclose(pose.approach_world, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(pose.jaw_world, [0.0, 1.0, 0.0])
        np.testing.assert_allclose(
            pose.eef_rotation_world,
            ANYGRASP_TO_EEF_ROTATION,
        )
        assert pose.eef_quaternion_xyzw.shape == (4,)
        assert np.isfinite(pose.eef_quaternion_xyzw).all()


class TestDetectorContract:
    def test_open_jaw_clearance_rejects_physically_brittle_fit(self) -> None:
        evidence = {
            "available": True,
            "open_jaw_continuous_cross_section_intersects": True,
            "target_between_open_fingers": True,
            "open_jaw_continuous_inner_clearance_m": 0.0006,
        }
        assert _open_jaw_clearance_passes(evidence, 0.0) is True
        assert _open_jaw_clearance_passes(evidence, 0.004) is False

    def test_open_jaw_clearance_accepts_usable_margin(self) -> None:
        evidence = {
            "available": True,
            "open_jaw_continuous_cross_section_intersects": True,
            "target_between_open_fingers": True,
            "open_jaw_continuous_inner_clearance_m": 0.006,
        }
        assert _open_jaw_clearance_passes(evidence, 0.004) is True

    def test_validation_rejects_misaligned_colors_and_mask(self) -> None:
        points = np.zeros((3, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="colors must match"):
            validate_detection_inputs(points, np.zeros((2, 3)), None)
        with pytest.raises(ValueError, match="region_mask"):
            validate_detection_inputs(points, points.copy(), np.ones(2, dtype=bool))

    def test_workspace_limits_are_built_from_target_region(self) -> None:
        points = np.array([[0, 0, 1], [1, 2, 3], [2, 4, 6]], dtype=np.float32)
        limits = workspace_limits(points, np.array([False, True, False]), 0.1)
        np.testing.assert_allclose(limits, [0.9, 1.1, 1.9, 2.1, 2.9, 3.1])

    def test_approach_filter_is_applied(self) -> None:
        forward = GraspCandidate(1, np.zeros(3), np.eye(3), 0.05, 0.02, 0.03)
        reverse_rotation = np.diag([-1.0, -1.0, 1.0])
        reverse = GraspCandidate(0.5, np.zeros(3), reverse_rotation, 0.05, 0.02, 0.03)
        kept = filter_candidates_by_approach([forward, reverse], [1, 0, 0], 0.2)
        assert kept == [forward]

    def test_remote_ping_requires_requested_top_down_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Response:
            ok = True

            def json(self) -> dict[str, Any]:
                return {"detector_loaded": True, "top_down_grasp": False}

        monkeypatch.setitem(
            sys.modules,
            "requests",
            SimpleNamespace(get=lambda *_args, **_kwargs: Response()),
        )
        detector = AnyGraspDetector(
            {"endpoint": "http://127.0.0.1:18090", "top_down_grasp": True}
        )
        assert detector.ping() is False

    def test_remote_ping_rejects_unrequested_top_down_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Response:
            ok = True

            def json(self) -> dict[str, Any]:
                return {"detector_loaded": True, "top_down_grasp": True}

        monkeypatch.setitem(
            sys.modules,
            "requests",
            SimpleNamespace(get=lambda *_args, **_kwargs: Response()),
        )
        detector = AnyGraspDetector(
            {"endpoint": "http://127.0.0.1:18091", "top_down_grasp": False}
        )
        assert detector.ping() is False

    def test_remote_request_sends_colors_mask_and_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        class Response:
            ok = True
            def raise_for_status(self) -> None:
                return None
            def json(self) -> dict[str, Any]:
                return {"candidates": []}

        def post(url: str, *, json: dict[str, Any], timeout: float) -> Response:
            captured.update(url=url, payload=json, timeout=timeout)
            return Response()

        import requests
        monkeypatch.setattr(requests, "post", post)
        detector = AnyGraspDetector(
            {"endpoint": "http://localhost:8090", "request_timeout_s": 12.5}
        )
        detector.detect(
            np.ones((4, 3), dtype=np.float32),
            np.full((4, 3), 0.5, dtype=np.float32),
            region_mask=np.array([True, True, False, False]),
        )
        assert captured["timeout"] == 12.5
        assert "colors_b64" in captured["payload"]
        assert "region_mask_b64" in captured["payload"]


    def test_local_mode_uses_real_sdk_signature(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        detection_dir = tmp_path / "grasp_detection"
        checkpoint = detection_dir / "log" / "checkpoint_detection.tar"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        calls: dict[str, Any] = {}

        class FakeAnyGrasp:
            def __init__(self, config: Any) -> None:
                calls["config"] = config
            def load_net(self) -> None:
                calls["loaded"] = True
            def get_grasp(self, points: Any, colors: Any, **kwargs: Any) -> tuple[Any, None]:
                calls.update(points=points, colors=colors, kwargs=kwargs)
                return FakeGroup([0.8]), None

        monkeypatch.setitem(sys.modules, "gsnet", SimpleNamespace(AnyGrasp=FakeAnyGrasp))
        detector = AnyGraspDetector({"sdk_root": str(tmp_path), "top_k": 1})
        result = detector.detect(
            np.ones((5, 3), dtype=np.float32),
            np.full((5, 3), 0.25, dtype=np.float32),
            region_mask=np.array([True, True, False, False, False]),
        )
        assert calls["loaded"] is True
        assert calls["colors"].shape == (5, 3)
        assert calls["kwargs"]["lims"] is not None
        assert result[0].score == pytest.approx(0.8)


class TestServerContract:
    def test_server_applies_target_workspace_and_approach_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from voltron.integrations.manipulation.anygrasp import server

        calls: dict[str, Any] = {}

        class NativeDetector:
            def get_grasp(self, points: Any, colors: Any, **kwargs: Any) -> tuple[Any, None]:
                calls.update(points=points, colors=colors, kwargs=kwargs)
                return FakeGroup([0.9, 0.8]), None

        monkeypatch.setattr(server, "_detector", NativeDetector())
        points = np.array([[0, 0, 1], [0.1, 0, 1], [1, 1, 1]], dtype=np.float32)
        colors = np.full_like(points, 0.5)
        mask = np.array([True, True, False])
        request = server.DetectRequest(
            points_b64=base64.b64encode(points.tobytes()).decode(),
            points_shape=list(points.shape),
            colors_b64=base64.b64encode(colors.tobytes()).decode(),
            colors_shape=list(colors.shape),
            region_mask_b64=base64.b64encode(mask.tobytes()).decode(),
            region_mask_shape=list(mask.shape),
            approach_direction=[1, 0, 0],
            approach_thresh=0.2,
        )
        response = server.detect(request)
        assert calls["kwargs"]["lims"] is not None
        assert len(response.candidates) == 2

    def test_server_releases_inference_cuda_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from voltron.integrations.manipulation.anygrasp import server

        released: list[bool] = []

        class NativeDetector:
            def get_grasp(self, points: Any, colors: Any, **kwargs: Any) -> tuple[Any, None]:
                return FakeGroup([0.9]), None

        monkeypatch.setattr(server, "_detector", NativeDetector())
        monkeypatch.setattr(server, "_release_cuda_cache", lambda: released.append(True))
        points = np.ones((2, 3), dtype=np.float32)
        request = server.DetectRequest(
            points_b64=base64.b64encode(points.tobytes()).decode(),
            points_shape=[2, 3],
            colors_b64=base64.b64encode(points.tobytes()).decode(),
            colors_shape=[2, 3],
        )
        server.detect(request)
        assert released == [True]

    def test_server_rejects_missing_colors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi import HTTPException
        from voltron.integrations.manipulation.anygrasp import server
        monkeypatch.setattr(server, "_detector", object())
        points = np.ones((2, 3), dtype=np.float32)
        request = server.DetectRequest(
            points_b64=base64.b64encode(points.tobytes()).decode(),
            points_shape=[2, 3],
        )
        with pytest.raises(HTTPException) as error:
            server.detect(request)
        assert error.value.status_code == 422


class TestExecutor:
    def test_place_inside_releases_object_and_returns_physical_evidence(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import torch
        from omnigibson import object_states
        from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection

        state: dict[str, Any] = {
            "held": SimpleNamespace(
                name="half_apple_213",
                aabb=(np.array([0.10, 0.10, 0.10]), np.array([0.20, 0.20, 0.20])),
            )
        }
        planner_calls = []
        navigation_calls = []

        class Primitives:
            @staticmethod
            def _navigate_to_pose_direct(pose, *, low_precision):
                assert low_precision is True
                navigation_calls.append(pose.cpu().numpy())
                yield np.ones(23, dtype=np.float32)

            @staticmethod
            def _sample_pose_with_object_and_predicate(
                predicate, held, target, *, world_aligned
            ):
                assert predicate == object_states.Inside
                assert held.name == "half_apple_213"
                assert target.name == "parts_bin"
                assert world_aligned is True
                return (
                    torch.tensor([0.2, 0.2, 0.2]),
                    torch.tensor([0.0, 0.0, 0.0, 1.0]),
                )

            @staticmethod
            def _get_hand_pose_for_object_pose(desired_pose):
                assert np.allclose(desired_pose[0].numpy(), [0.2, 0.2, 0.2])
                return (
                    torch.tensor([0.3, 0.3, 0.3]),
                    torch.tensor([0.0, 0.0, 0.0, 1.0]),
                )

            @staticmethod
            def _plan_joint_motion(**kwargs):
                assert kwargs["ignore_objects"][-1].name == "parts_bin"
                planner_calls.append(kwargs)
                return [np.zeros(23, dtype=np.float32)]

            @staticmethod
            def _execute_motion_plan(trajectory):
                yield from trajectory

            @staticmethod
            def _execute_release():
                yield np.zeros(23, dtype=np.float32)
                state["held"] = None

            @staticmethod
            def _settle_robot():
                yield np.zeros(23, dtype=np.float32)

            @staticmethod
            def _get_obj_in_hand():
                return state["held"]

        def get_shortest_path(
            floor, source_world, target_world, *, entire_path, robot
        ):
            assert floor == 0
            assert entire_path is True
            assert np.allclose(source_world, [1.25, 0.25])
            if np.allclose(target_world, [-0.2, 0.25]):
                return (
                    torch.tensor(
                        [[1.25, 0.25], [0.8, 0.25], [0.3, 0.25], [-0.2, 0.25]]
                    ),
                    torch.tensor(1.45),
                )
            return None, None

        robot = SimpleNamespace(
            default_arm="left",
            eef_link_names={"left": "left_eef"},
            get_position_orientation=lambda: (
                torch.tensor([1.25, 0.25, 0.0]),
                torch.tensor([0.0, 0.0, 0.0, 1.0]),
            ),
            scene=SimpleNamespace(objects=[], get_shortest_path=get_shortest_path),
        )
        execution = GraspExecutor(
            robot=robot,
            primitives=Primitives(),
        ).begin_place_inside(
            SimpleNamespace(
                name="parts_bin",
                aabb=(np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.5])),
            )
        )
        actions = []
        outcome = None
        while outcome is None:
            action, outcome = execution.advance()
            if action is not None:
                actions.append(action)
        assert len(actions) == 7
        assert outcome.success is True
        assert outcome.physical_evidence["released"] is True
        assert outcome.physical_evidence["containment_check_available"] is True
        assert outcome.physical_evidence["aabb_contained"] is True
        assert outcome.physical_evidence["placement_mode"] == "place_inside"
        assert outcome.physical_evidence["destination_object"] == "parts_bin"

    def test_camera_to_world_uses_capture_time_optical_pose(self) -> None:
        robot = SimpleNamespace(default_arm="right")
        executor = GraspExecutor(robot=robot, primitives=object())
        camera_pose = np.eye(4, dtype=np.float32)
        camera_pose[:3, 3] = [1, 2, 3]
        position, quaternion = executor.camera_to_world(
            np.array([0.1, 0.2, 0.3]),
            np.eye(3),
            camera_pose_world=camera_pose,
        )
        np.testing.assert_allclose(position, [1.1, 2.2, 3.3])
        rotation = _pose_to_matrix(position, quaternion)[:3, :3]
        np.testing.assert_allclose(rotation[:, 2], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(rotation[:, 1], [0.0, 1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(rotation[:, 0], [0.0, 0.0, -1.0], atol=1e-6)
        assert np.linalg.det(rotation) == pytest.approx(1.0)

    def test_execution_yields_one_native_action_per_advance(self) -> None:
        outcome = GraspResult(
            True,
            "radio_89",
            np.zeros(3),
            np.array([0, 0, 0, 1]),
            0.9,
            2,
        )

        def generator():
            yield np.zeros(23, dtype=np.float32)
            yield np.ones(23, dtype=np.float32)
            return outcome

        execution = GraspExecution(generator())
        first, result = execution.advance()
        assert result is None and first["robot_r1"].shape == (23,)
        second, result = execution.advance()
        assert result is None and second["robot_r1"].sum() == 23
        action, result = execution.advance()
        assert action is None and result is outcome and execution.done

    def test_release_planner_memory_drops_owned_curobo_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import torch

        motion_generator = object()
        primitives = SimpleNamespace(_motion_generator=motion_generator)
        executor = GraspExecutor(robot=SimpleNamespace(default_arm="left"), primitives=primitives)
        executor._owns_primitives = True
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        executor.release_planner_memory()

        assert executor._primitives is None
        assert primitives._motion_generator is None
        assert executor._primitives_init_failed is False

    def test_controller_compatibility_supports_smooth_grippers(self) -> None:
        import torch as th

        class JointController:
            use_delta_commands = False
            dof_idx = np.array([0, 1])

            @staticmethod
            def _reverse_preprocess_command(command: Any) -> Any:
                return command

        class MultiFingerGripperController:
            @staticmethod
            def compute_no_op_action(control_dict: Any) -> Any:
                return th.tensor([0.25])

        robot = SimpleNamespace(
            default_arm="left",
            action_dim=3,
            controllers={
                "arm_left": JointController(),
                "gripper_left": MultiFingerGripperController(),
            },
            controller_action_idx={"arm_left": np.s_[0:2], "gripper_left": np.s_[2:3]},
            get_control_dict=lambda: {},
            get_joint_positions=lambda: th.zeros(3),
        )

        class Primitives:
            @staticmethod
            def _empty_action(follow_arm_targets: bool = True) -> Any:
                return th.zeros(3)

            @staticmethod
            def _postprocess_action(action: Any) -> Any:
                return action

            @staticmethod
            def _get_obj_in_hand() -> None:
                return None

        primitives = Primitives()
        executor = GraspExecutor(robot=robot, primitives=primitives)
        executor._install_controller_compatibility(primitives)

        action = robot.q_to_action(th.tensor([0.4, 0.5]))
        th.testing.assert_close(action, th.tensor([0.4, 0.5, 0.25]))
        close_action = next(iter(primitives._move_fingers_to_limit("lower")))
        assert close_action[2].item() == -1.0

    def test_controller_compatibility_preserves_velocity_base_motion(self) -> None:
        import torch as th

        class HolonomicBaseJointController:
            dof_idx = np.array([0, 1, 2])
            motor_type = "velocity"
            control_freq = 30.0

            @staticmethod
            def _reverse_preprocess_command(command: Any) -> Any:
                return command

        class JointController:
            dof_idx = np.array([3, 4])
            use_delta_commands = False

            @staticmethod
            def _reverse_preprocess_command(command: Any) -> Any:
                return command

        class MultiFingerGripperController:
            @staticmethod
            def compute_no_op_action(control_dict: Any) -> Any:
                return th.tensor([0.0])

        robot = SimpleNamespace(
            default_arm="left",
            action_dim=6,
            controllers={
                "base": HolonomicBaseJointController(),
                "arm_left": JointController(),
                "gripper_left": MultiFingerGripperController(),
            },
            controller_action_idx={
                "base": np.s_[0:3],
                "arm_left": np.s_[3:5],
                "gripper_left": np.s_[5:6],
            },
            get_control_dict=lambda: {},
            get_joint_positions=lambda: th.zeros(5),
        )
        primitives = SimpleNamespace()
        executor = GraspExecutor(robot=robot, primitives=primitives)
        executor._install_controller_compatibility(primitives)

        action = robot.q_to_action(th.tensor([0.1, 0.0, 0.1, 0.4, 0.5]))
        assert action[0].item() > 0.0
        assert action[0].item() == pytest.approx(0.3, abs=1e-6)
        assert action[2].item() > 0.0
        assert action[2].item() == pytest.approx(0.2, abs=1e-6)
        th.testing.assert_close(action[3:5], th.tensor([0.4, 0.5]))
        assert action[5].item() == 0.0

    def test_physical_staged_close_plan_targets_object_span_without_lower_limit(self) -> None:
        plan = GraspExecutor.physical_staged_close_plan(
            open_qpos=[0.049, 0.049],
            lower_qpos=[0.0, 0.0],
            open_gap_m=0.098,
            target_y_bounds_m=[-0.032, 0.032],
            compression_m=0.004,
            stage_count=6,
        )

        assert plan["desired_gap_m"] == pytest.approx(0.060)
        assert plan["achieved_gap_m"] == pytest.approx(0.060)
        np.testing.assert_allclose(plan["target_qpos"], [0.030, 0.030])
        assert len(plan["stage_qpos"]) == 6
        assert np.all(np.asarray(plan["target_qpos"]) > 0.0)
        np.testing.assert_allclose(plan["stage_qpos"][-1], plan["target_qpos"])

    def test_physical_staged_close_plan_clamps_tiny_object_above_lower_limit(self) -> None:
        plan = GraspExecutor.physical_staged_close_plan(
            open_qpos=[0.049, 0.049],
            lower_qpos=[0.0, 0.0],
            open_gap_m=0.098,
            target_y_bounds_m=[-0.001, 0.001],
            compression_m=0.004,
            stage_count=4,
        )

        assert plan["clamped_above_lower_limit"] is True
        np.testing.assert_allclose(plan["target_qpos"], [0.001, 0.001])
        assert plan["achieved_gap_m"] == pytest.approx(0.002)

    def test_physical_staged_close_stops_on_bilateral_contact(self) -> None:
        assert GraspExecutor.physical_staged_close_should_stop(
            {"bilateral_finger_contact": True, "grasp_state_passed": False},
            target_displacement_m=0.003,
            displacement_tolerance_m=0.008,
            stage_index=3,
        )

    def test_post_lift_yaw_sequence_alternates_and_returns_neutral(self) -> None:
        assert GraspExecutor.post_lift_yaw_sequence(35.0, 2) == [
            35.0,
            -35.0,
            35.0,
            -35.0,
            0.0,
        ]
        assert GraspExecutor.post_lift_yaw_sequence(0.0, 2) == []

    def test_place_back_distances_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError, match="place_back_clearance_m"):
            GraspExecutor(
                robot=SimpleNamespace(default_arm="left"),
                primitives=SimpleNamespace(),
                place_back_clearance_m=-0.001,
            )

    def test_physical_staged_close_aborts_on_target_displacement(self) -> None:
        with pytest.raises(RuntimeError, match="0.0090 m at stage 2"):
            GraspExecutor.physical_staged_close_should_stop(
                {"bilateral_finger_contact": False, "grasp_state_passed": False},
                target_displacement_m=0.009,
                displacement_tolerance_m=0.008,
                stage_index=2,
            )

    def test_physical_staged_close_aborts_on_moving_unilateral_contact(self) -> None:
        with pytest.raises(RuntimeError, match="single-finger contact pushed target"):
            GraspExecutor.physical_staged_close_should_stop(
                {
                    "target_finger_contact_count": 1,
                    "bilateral_finger_contact": False,
                    "grasp_state_passed": False,
                },
                target_displacement_m=0.003,
                displacement_tolerance_m=0.008,
                unilateral_contact_displacement_tolerance_m=0.002,
                stage_index=2,
            )

    def test_grasp_uses_safe_standoff_precise_pregrasp_and_constrained_approach(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import torch as th

        class FakeEmbodimentSelection:
            DEFAULT = "default"

        monkeypatch.setitem(
            sys.modules,
            "omnigibson.action_primitives.curobo",
            SimpleNamespace(CuRoboEmbodimentSelection=FakeEmbodimentSelection),
        )
        events: list[str] = []
        state: dict[str, Any] = {
            "held": None,
            "moves": [],
            "whole_body_plans": [],
            "whole_body_executions": [],
            "target_z": 0.5,
            "eef_z": 0.0,
        }
        target = SimpleNamespace(
            name="radio_89",
            get_position_orientation=lambda: (
                th.tensor([0.1, 0.0, state["target_z"]]),
                th.tensor([0.0, 0.0, 0.0, 1.0]),
            ),
        )

        class Primitives:
            @staticmethod
            def _execute_release():
                events.append("release")
                yield th.zeros(6)

            @staticmethod
            def _plan_joint_motion(**kwargs: Any) -> Any:
                assert set(kwargs["target_pos"]) == {"left_eef"}
                assert set(kwargs["target_quat"]) == {"left_eef"}
                assert kwargs["embodiment_selection"] == FakeEmbodimentSelection.DEFAULT
                state["whole_body_plans"].append(
                    (kwargs["target_pos"]["left_eef"].clone(), kwargs)
                )
                events.append("whole_body_plan")
                return th.zeros((1, 6))

            @staticmethod
            def _execute_motion_plan(trajectory: Any, **kwargs: Any):
                assert tuple(trajectory.shape) == (1, 6)
                state["whole_body_executions"].append(kwargs)
                events.append("whole_body_execute")
                yield th.zeros(6)

            @staticmethod
            def _move_hand(pose: Any, **kwargs: Any):
                state["moves"].append((pose[0].clone(), kwargs))
                if len(state["moves"]) == 2:
                    state["target_z"] = 0.65
                    state["eef_z"] = 0.15
                events.append("move")
                yield th.zeros(6)

            @staticmethod
            def _execute_grasp():
                events.append("close")
                state["held"] = target
                yield th.zeros(6)

            @staticmethod
            def _settle_robot():
                events.append("settle")
                yield th.zeros(6)

            @staticmethod
            def _get_obj_in_hand() -> Any:
                return state["held"]

        robot = SimpleNamespace(
            default_arm="left",
            grasping_mode="assisted",
            eef_link_names={"left": "left_eef"},
            eef_to_fingertip_lengths={"left": {"finger_a": 0.1, "finger_b": 0.1}},
            get_eef_pose=lambda arm: (
                th.tensor([0.0, 0.0, state["eef_z"]]),
                th.tensor([0.0, 0.0, 0.0, 1.0]),
            ),
            get_position_orientation=lambda: (
                th.zeros(3),
                th.tensor([0.0, 0.0, 0.0, 1.0]),
            ),
        )
        candidate = FakeCandidate(depth=0.02)
        executor = GraspExecutor(
            robot=robot,
            primitives=Primitives(),
            verification_require_attachment_valid=False,
        )
        execution = executor.begin_grasp(
            candidate,
            camera_pose_world=np.eye(4, dtype=np.float32),
            target_obj=target,
        )
        outcome = None
        while outcome is None:
            _action, outcome = execution.advance()

        assert events[:5] == [
            "release",
            "whole_body_plan",
            "whole_body_execute",
            "whole_body_plan",
            "whole_body_execute",
        ]
        whole_body_pos, standoff_plan_kwargs = state["whole_body_plans"][0]
        precise_pregrasp_pos, precise_plan_kwargs = state["whole_body_plans"][1]
        final_grasp_pos, approach_kwargs = state["moves"][0]
        _lift_pos, lift_kwargs = state["moves"][1]

        np.testing.assert_allclose(final_grasp_pos.numpy(), [0.02, 0.0, 0.5], atol=1e-6)
        np.testing.assert_allclose(precise_pregrasp_pos.numpy(), [-0.06, 0.0, 0.5], atol=1e-6)
        np.testing.assert_allclose(whole_body_pos.numpy(), [-0.33, 0.0, 0.5], atol=1e-6)
        assert "ignore_objects" not in standoff_plan_kwargs
        assert precise_plan_kwargs["ignore_objects"] == [target]
        assert state["whole_body_executions"] == [
            {"low_precision": True, "stop_on_contact": False},
            {"low_precision": False, "stop_on_contact": True},
        ]
        assert approach_kwargs == {
            "motion_constraint": [1, 1, 1, 1, 1, 0],
            "ignore_objects": [target],
        }
        assert lift_kwargs == {"ignore_objects": [target]}
        assert outcome.success is True
        assert outcome.object_in_hand == "radio_89"

        adapter = GripperGeometryAdapter.from_robot(robot, "left")
        assert adapter.fingertip_depth_m == pytest.approx(0.1)
        calibrated = GripperGeometryAdapter.from_robot(
            robot,
            "left",
            fingertip_depth_override_m=0.018,
        )
        assert calibrated.fingertip_depth_m == pytest.approx(0.018)
        assert calibrated.eef_approach_offset_m == pytest.approx(0.0)
        assert calibrated.source == "config_override"
        shifted = GripperGeometryAdapter.from_robot(
            robot,
            "left",
            fingertip_depth_override_m=0.018,
            eef_approach_offset_m=0.01,
        )
        np.testing.assert_allclose(
            shifted.eef_position(np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.03),
            [0.0, 0.0, 0.022],
        )
        assert shifted.eef_origin_candidate_x(0.03) == pytest.approx(0.022)

    def test_precise_pregrasp_aborts_after_first_target_displacement(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import torch as th

        monkeypatch.setitem(
            sys.modules,
            "omnigibson.action_primitives.curobo",
            SimpleNamespace(CuRoboEmbodimentSelection=SimpleNamespace(DEFAULT="default")),
        )
        state = {"target_pos": th.tensor([0.1, 0.0, 0.5])}
        target = SimpleNamespace(
            name="radio_89",
            get_position_orientation=lambda: (state["target_pos"], th.zeros(4)),
        )

        class Primitives:
            @staticmethod
            def _execute_release():
                yield th.zeros(6)

            @staticmethod
            def _plan_joint_motion(**kwargs: Any) -> Any:
                return th.zeros((1, 6))

            @staticmethod
            def _execute_motion_plan(trajectory: Any, **kwargs: Any):
                yield th.zeros(6)

            @staticmethod
            def _move_hand(pose: Any, **kwargs: Any):
                yield th.zeros(6)
                yield th.zeros(6)

            @staticmethod
            def _get_obj_in_hand() -> None:
                return None

        robot = SimpleNamespace(
            default_arm="left",
            grasping_mode="assisted",
            eef_link_names={"left": "left_eef"},
            eef_to_fingertip_lengths={"left": {"finger_a": 0.1, "finger_b": 0.1}},
            get_eef_pose=lambda arm: (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])),
            get_position_orientation=lambda: (
                th.zeros(3),
                th.tensor([0.0, 0.0, 0.0, 1.0]),
            ),
        )
        execution = GraspExecutor(robot=robot, primitives=Primitives()).begin_grasp(
            FakeCandidate(),
            camera_pose_world=np.eye(4, dtype=np.float32),
            target_obj=target,
        )
        assert execution.advance()[1] is None  # release
        assert execution.advance()[1] is None  # whole-body standoff
        assert execution.advance()[1] is None  # first precise-pregrasp action
        state["target_pos"] = th.tensor([0.106, 0.0, 0.5])

        _action, outcome = execution.advance()

        assert outcome is not None
        assert outcome.success is False
        assert outcome.failure_phase == "precise_pregrasp"
        assert outcome.scene_changed is True
        assert outcome.total_sim_steps == 3
        assert "target moved during contact-guarded motion" in str(outcome.error)

    def test_lift_failure_is_not_reported_as_grasp_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import torch as th

        class FakeEmbodimentSelection:
            DEFAULT = "default"

        monkeypatch.setitem(
            sys.modules,
            "omnigibson.action_primitives.curobo",
            SimpleNamespace(CuRoboEmbodimentSelection=FakeEmbodimentSelection),
        )
        target = SimpleNamespace(
            name="radio_89",
            get_position_orientation=lambda: (th.zeros(3), th.zeros(4)),
        )
        state: dict[str, Any] = {"held": None, "move_calls": 0}

        class Primitives:
            @staticmethod
            def _execute_release():
                yield th.zeros(6)

            @staticmethod
            def _plan_joint_motion(**kwargs: Any) -> Any:
                return th.zeros((1, 6))

            @staticmethod
            def _execute_motion_plan(trajectory: Any, **kwargs: Any):
                yield th.zeros(6)

            @staticmethod
            def _move_hand(pose: Any, **kwargs: Any):
                state["move_calls"] += 1
                if state["move_calls"] == 2:
                    raise RuntimeError("lift planning failed")
                yield th.zeros(6)

            @staticmethod
            def _execute_grasp():
                state["held"] = target
                yield th.zeros(6)

            @staticmethod
            def _settle_robot():
                yield th.zeros(6)

            @staticmethod
            def _get_obj_in_hand() -> Any:
                return state["held"]

        robot = SimpleNamespace(
            default_arm="left",
            grasping_mode="assisted",
            eef_link_names={"left": "left_eef"},
            eef_to_fingertip_lengths={"left": {"finger_a": 0.1, "finger_b": 0.1}},
            get_eef_pose=lambda arm: (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])),
            get_position_orientation=lambda: (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])),
        )
        execution = GraspExecutor(robot=robot, primitives=Primitives()).begin_grasp(
            FakeCandidate(),
            camera_pose_world=np.eye(4, dtype=np.float32),
            target_obj=target,
        )
        outcome = None
        while outcome is None:
            _action, outcome = execution.advance()

        assert outcome.success is False
        assert outcome.object_in_hand == "radio_89"
        assert outcome.failure_phase == "lift"
        assert outcome.scene_changed is True
        assert "lift planning failed" in str(outcome.error)


class FakeDetector:
    def ping(self) -> bool:
        return True
    def detect(self, points: Any, colors: Any, **kwargs: Any) -> list[FakeCandidate]:
        assert points.shape == colors.shape
        assert kwargs["region_mask"].shape == (len(points),)
        return [FakeCandidate()]


class FakeExecutor:
    def begin_grasp(self, candidate: Any, *, camera_pose_world: Any, target_obj: Any) -> GraspExecution:
        outcome = GraspResult(
            True,
            target_obj.name,
            np.array([1, 2, 3]),
            np.array([0, 0, 0, 1]),
            candidate.score,
            1,
        )
        def generator():
            yield np.zeros(23, dtype=np.float32)
            return outcome
        return GraspExecution(generator())

    def begin_grasp_by_object(self, target_obj: Any) -> GraspExecution:
        raise AssertionError("built-in fallback should not be used")


class TestAnyGraspSkill:
    def make_skill(self, *, allow_fallback: bool = False) -> AnyGraspSkill:
        return AnyGraspSkill(
            memory=MockMemoryAdapter(),
            policy=MockPolicyAdapter(),
            projector=ActionProjection.from_embodiment("behavior_r1_pro"),
            anygrasp_config={
                "endpoint": "http://unused",
                "allow_fallback": allow_fallback,
                "target_anchor_tolerance_m": 2.0,
            },
        )

    def test_target_mask_anchor_filter_rejects_off_object_candidate(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class MixedDetector(FakeDetector):
            def detect(self, points: Any, colors: Any, **kwargs: Any) -> list[FakeCandidate]:
                return [
                    FakeCandidate(translation=np.array([0.01, 0.0, 1.0], dtype=np.float32)),
                    FakeCandidate(translation=np.array([0.5, 0.0, 1.0], dtype=np.float32)),
                ]

        skill = self.make_skill()
        skill._anygrasp_config["target_anchor_tolerance_m"] = 0.04
        skill._detector = MixedDetector()
        packet = GraspObservation(
            points=np.array(
                [[0.0, 0.0, 1.0], [0.02, 0.0, 1.0], [0.5, 0.0, 1.0]],
                dtype=np.float32,
            ),
            colors=np.full((3, 3), 0.5, dtype=np.float32),
            region_mask=np.array([True, True, False]),
            camera_pose_world=np.eye(4, dtype=np.float32),
            camera_sensor="head_cam",
        )
        monkeypatch.setattr(skill, "_capture_observation", lambda subtask: packet)

        candidates, captured = skill._detect_candidates(make_subtask())

        assert captured is packet
        assert len(candidates) == 1
        np.testing.assert_allclose(candidates[0].translation, [0.01, 0.0, 1.0])

    def test_world_vertical_orientation_uses_target_footprint_minor_axis(self) -> None:
        points = np.array(
            [
                [-0.10, -0.01, 0.5],
                [-0.10, 0.01, 0.5],
                [0.10, -0.01, 0.5],
                [0.10, 0.01, 0.5],
            ],
            dtype=np.float64,
        )

        rotation, audit = _world_vertical_grasp_rotation(
            points,
            np.eye(4),
            np.eye(3),
            jaw_axis="minor",
        )

        np.testing.assert_allclose(rotation[:, 0], [0.0, 0.0, -1.0], atol=1e-7)
        assert abs(float(rotation[1, 1])) == pytest.approx(1.0)
        assert np.linalg.det(rotation) == pytest.approx(1.0)
        assert audit["jaw_axis"] == "minor"

    def test_world_vertical_mode_applies_world_gate_after_detection(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class CapturingDetector(FakeDetector):
            def __init__(self) -> None:
                self.kwargs: dict[str, Any] = {}

            def detect(self, points: Any, colors: Any, **kwargs: Any) -> list[FakeCandidate]:
                self.kwargs = kwargs
                return [
                    FakeCandidate(
                        translation=np.array([0.0, 0.0, 0.5], dtype=np.float32)
                    )
                ]

        skill = self.make_skill()
        skill._anygrasp_config.update(
            {
                "candidate_force_world_vertical_approach": True,
                "candidate_world_vertical_jaw_axis": "minor",
                "candidate_max_world_approach_z": -0.95,
                "target_anchor_tolerance_m": 0.2,
            }
        )
        detector = CapturingDetector()
        skill._detector = detector
        points = np.array(
            [
                [-0.10, -0.01, 0.5],
                [-0.10, 0.01, 0.5],
                [0.10, -0.01, 0.5],
                [0.10, 0.01, 0.5],
            ],
            dtype=np.float32,
        )
        packet = GraspObservation(
            points=points,
            colors=np.full_like(points, 0.5),
            region_mask=np.ones(len(points), dtype=bool),
            camera_pose_world=np.eye(4, dtype=np.float32),
            camera_sensor="head_cam",
        )
        monkeypatch.setattr(skill, "_capture_observation", lambda subtask: packet)

        candidates, _ = skill._detect_candidates(make_subtask())

        assert detector.kwargs["approach_direction"] is None
        assert len(candidates) == 1
        np.testing.assert_allclose(
            candidates[0].approach_direction, [0.0, 0.0, -1.0], atol=1e-7
        )

    def test_successful_execution_is_advanced_by_normal_control_steps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        skill = self.make_skill()
        skill._detector = FakeDetector()
        skill._executor = FakeExecutor()
        packet = GraspObservation(
            points=np.ones((10, 3), dtype=np.float32),
            colors=np.full((10, 3), 0.5, dtype=np.float32),
            region_mask=np.ones(10, dtype=bool),
            camera_pose_world=np.eye(4, dtype=np.float32),
            camera_sensor="head_cam",
        )
        monkeypatch.setattr(skill, "_capture_observation", lambda subtask: packet)
        monkeypatch.setattr(skill, "_find_target_object", lambda subtask: SimpleNamespace(name="radio_89"))
        subtask = make_subtask()

        first = skill.execute(subtask, make_context(), make_selection())
        assert first.status == AgentStatus.SUCCESS
        assert first.result["grasp_plan_completed"] is False
        assert first.runtime_artifacts["projected_action"]["robot_r1"].shape == (23,)

        final = skill.execute(subtask, make_context(), make_selection())
        assert final.status == AgentStatus.SUCCESS
        assert final.result["grasp_plan_completed"] is True
        assert final.result["grasp_success"] is True
        assert final.result["object_in_hand"] == "radio_89"
        assert final.result["skill_source"] == "anygrasp_curobo"

    def test_physical_failure_recaptures_and_redetects_before_retry(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class CountingDetector(FakeDetector):
            def __init__(self) -> None:
                self.calls = 0

            def detect(self, points: Any, colors: Any, **kwargs: Any) -> list[FakeCandidate]:
                self.calls += 1
                return [
                    FakeCandidate(
                        translation=np.array([0.1 * self.calls, 0.0, 0.5], dtype=np.float32)
                    )
                ]

        class RetryExecutor(FakeExecutor):
            def __init__(self) -> None:
                self.calls = 0
                self.planner_release_calls = 0
                self.camera_poses: list[np.ndarray] = []

            def release_planner_memory(self) -> None:
                self.planner_release_calls += 1

            def begin_grasp(
                self,
                candidate: Any,
                *,
                camera_pose_world: Any,
                target_obj: Any,
            ) -> GraspExecution:
                self.calls += 1
                self.camera_poses.append(np.asarray(camera_pose_world).copy())
                success = self.calls == 2
                outcome = GraspResult(
                    success,
                    target_obj.name if success else None,
                    np.array([1, 2, 3]),
                    np.array([0, 0, 0, 1]),
                    candidate.score,
                    1,
                    None if success else "first candidate moved the scene",
                    failure_phase=None if success else "assisted_approach",
                    scene_changed=not success,
                )

                def generator():
                    yield np.zeros(23, dtype=np.float32)
                    return outcome

                return GraspExecution(generator())

        skill = self.make_skill()
        detector = CountingDetector()
        executor = RetryExecutor()
        skill._detector = detector
        skill._executor = executor
        camera_poses = [np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)]
        camera_poses[1][0, 3] = 1.0
        capture_calls = 0

        def capture(_subtask: Any) -> GraspObservation:
            nonlocal capture_calls
            pose = camera_poses[capture_calls]
            capture_calls += 1
            return GraspObservation(
                points=np.ones((10, 3), dtype=np.float32),
                colors=np.full((10, 3), 0.5, dtype=np.float32),
                region_mask=np.ones(10, dtype=bool),
                camera_pose_world=pose,
                camera_sensor="head_cam",
            )

        monkeypatch.setattr(skill, "_capture_observation", capture)
        monkeypatch.setattr(
            skill,
            "_find_target_object",
            lambda subtask: SimpleNamespace(name="radio_89"),
        )
        subtask = make_subtask()

        first = skill.execute(subtask, make_context(), make_selection())
        second = skill.execute(subtask, make_context(), make_selection())
        final = skill.execute(subtask, make_context(), make_selection())

        assert first.result["grasp_attempt"] == 1
        assert second.result["grasp_attempt"] == 2
        assert final.status == AgentStatus.SUCCESS
        assert final.result["grasp_success"] is True
        assert detector.calls == 2
        assert capture_calls == 2
        assert executor.calls == 2
        assert executor.planner_release_calls == 2
        np.testing.assert_allclose(executor.camera_poses[1], camera_poses[1])

    def test_scene_target_lookup_uses_objects_and_category(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        skill = self.make_skill()
        radio = SimpleNamespace(
            name="radio_receiver_fictional_model_0",
            category="radio_receiver",
            model="fictional_model",
            prim_path="/World/radio_receiver",
        )
        table = SimpleNamespace(
            name="table_model_0",
            category="table",
            model="table_model",
            prim_path="/World/table",
        )
        scene = SimpleNamespace(objects=[radio, table])
        monkeypatch.setitem(
            sys.modules,
            "omnigibson",
            SimpleNamespace(sim=SimpleNamespace(scenes=[scene])),
        )
        assert skill._find_target_object(make_subtask()) is radio

    def test_strict_mode_never_hides_detector_failure_with_fallback(self) -> None:
        skill = self.make_skill(allow_fallback=False)
        skill._detector_init_failed = True
        skill._executor_init_failed = True
        result = skill.execute(make_subtask(), make_context(), make_selection())
        assert result.status == AgentStatus.FAILURE
        assert result.error_code == "ANYGRASP_FAILED"


class TestCompletionAndConfig:
    def test_verified_grasp_completes_action_subtask(self) -> None:
        monitor = CompletionMonitor(use_environment_success_signal=False)
        subtask = make_subtask()
        result = SimpleNamespace(
            result={
                "skill_id": "anygrasp_manipulation_skill",
                "grasp_plan_completed": True,
                "grasp_success": True,
                "physical_grasp_verified": True,
                "physical_evidence": {
                    "passed": True,
                    "target_z_rise_passed": True,
                    "relative_pose_stable": True,
                    "object_identity_matches": True,
                    "attachment_passed": True,
                    "sample_count": 5,
                    "required_sample_count": 5,
                },
                "object_in_hand": "radio_89",
                "target_object": "radio_89",
                "skill_source": "anygrasp_curobo",
            }
        )
        decision = monitor.evaluate_subtask_step(
            subtask=subtask,
            context=make_context(),
            result=result,
            environment_outcome=SubtaskStepOutcome(
                done=True,
                success=True,
                feedback=RuntimeFeedback(step_count=3),
            ),
            control_step=3,
        )
        assert decision.done is True and decision.success is True
        assert decision.verdict.source == "verified_physical_grasp_and_environment"

    def test_example_config_is_strict_and_selects_pick_up(self) -> None:
        path = "/mnt/data/huangyixuan/isaac/voltron/configs/radio_pick_up_anygrasp_i00.json"
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
        assert config["action_subtask_action"] == "pick_up"
        assert config["action_target_object"] == "radio_89"
        assert config["anygrasp"]["allow_fallback"] is False
        assert config["anygrasp"]["camera_sensor"] == "robot_r1:zed_link:Camera:0"

    def test_frame_debug_config_uses_explicit_tool_calibration_fields(self) -> None:
        path = "/mnt/data/huangyixuan/isaac/voltron/configs/radio_pick_up_anygrasp_frame_debug_i00.json"
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
        anygrasp = config["anygrasp"]
        assert config["action_target_object"] == "radio_89"
        assert anygrasp["top_k"] == 100
        assert anygrasp["max_attempts"] == 1
        assert anygrasp["candidate_detection_refreshes"] == 1
        assert anygrasp["candidate_detection_only"] is False
        assert anygrasp["fingertip_depth_override_m"] is None
        assert anygrasp["eef_approach_offset_m"] == 0.0
        assert anygrasp["candidate_target_centroid_tolerance_m"] == 0.06
        assert anygrasp["target_depth_outlier_m"] == 0.10
        assert anygrasp["grasping_mode_override"] == "physical"
        assert anygrasp["allow_fallback"] is False

    def test_single_candidate_debug_config_is_conservative(self) -> None:
        path = "/mnt/data/huangyixuan/isaac/voltron/configs/half_apple_pick_up_anygrasp_debug_single_candidate_i10.json"
        with open(path, encoding="utf-8") as handle:
            config = json.load(handle)
        anygrasp = config["anygrasp"]
        assert config["action_subtask_action"] == "pick_up"
        assert anygrasp["max_attempts"] == 1
        assert anygrasp["candidate_detection_refreshes"] == 1
        assert anygrasp["top_k"] == 1
        assert anygrasp["allow_fallback"] is False
        assert anygrasp["candidate_force_world_vertical_approach"] is False
        assert anygrasp["candidate_target_collision_geometry_enabled"] is False
        assert anygrasp["candidate_inner_line_gate_enabled"] is False
        assert anygrasp["candidate_fit_depth_to_robot_inner_line"] is False
        assert anygrasp["candidate_recenter_to_target_centroid"] is False
        assert anygrasp["physical_staged_close_enabled"] is False
