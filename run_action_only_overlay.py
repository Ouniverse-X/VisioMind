from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "voltron"
PROJECT_ROOT = ROOT.parent
os.environ["VOLTRON_HOME"] = str(ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install(module_name: str, relative_path: str, *, load_as: str | None = None) -> None:
    path = ROOT / relative_path
    load_name = load_as or module_name
    spec = importlib.util.spec_from_file_location(load_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load competition overlay module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[load_name] = module
    spec.loader.exec_module(module)
    if load_as:
        sys.modules[module_name] = module
    if "." in module_name:
        parent_name, attribute = module_name.rsplit(".", 1)
        parent = importlib.import_module(parent_name)
        setattr(parent, attribute, module)


def main() -> None:
    _install("voltron.config_loader", "config_loader.py")
    _install("voltron.agents.action.body.skill_selection", "agents/action/body/skill_selection.py")
    _install(
        "voltron.integrations.simulator.behavior.execution.action_stepper",
        "integrations/simulator/behavior/execution/action_stepper.py",
    )
    importlib.import_module("voltron.integrations.manipulation.anygrasp.frame_adapter")
    _install(
        "voltron.integrations.manipulation.anygrasp.grasp_executor",
        "integrations/manipulation/anygrasp/grasp_executor.py",
        load_as="voltron.integrations.manipulation.anygrasp.grasp_executor_competition_overlay",
    )
    _install(
        "voltron.agents.action.skills.execution.anygrasp_skill",
        "agents/action/skills/execution/anygrasp_skill.py",
    )
    _install("voltron.agents.action.skills.registry", "agents/action/skills/registry.py")
    action_only_path = ROOT / "entrypoints/examples/closed_loop/action_only.py"
    spec = importlib.util.spec_from_file_location(
        "xh_competition_action_only_runtime", action_only_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load competition entrypoint: {action_only_path}")
    action_only = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = action_only
    spec.loader.exec_module(action_only)
    action_only.main()


if __name__ == "__main__":
    main()
