from __future__ import annotations

import argparse
import json
from pathlib import Path

from .instruction_model import IndustrialInstructionModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("instruction")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).parents[2] / "models" / "industrial_instruction.joblib",
    )
    args = parser.parse_args()
    plan = IndustrialInstructionModel(args.model).parse(args.instruction)
    print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
