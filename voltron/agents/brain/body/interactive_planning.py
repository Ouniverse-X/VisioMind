from __future__ import annotations

from uuid import uuid4

from typing import Any

from voltron.agents.brain.contracts import (
    BrainPlanningSession,
    ClarificationQuestion,
    PlanConfirmation,
    UserAnswer,
)
from voltron.agents.brain.skills.planning.interactive_skill import BrainInteractivePlanningSkill
from voltron.shared.context import Plan, TaskRequest


class BrainInteractivePlanningController:
    def __init__(
        self,
        *,
        memory: Any,
        skill: BrainInteractivePlanningSkill | None = None,
    ) -> None:
        self.memory = memory
        self.skill = skill or BrainInteractivePlanningSkill()

    def begin(
        self,
        request: TaskRequest,
        planning_context: dict[str, Any],
        *,
        provisional_plan: Plan | None = None,
    ) -> BrainPlanningSession:
        draft = self.skill.draft_text_plan(
            request.description,
            planning_context,
            provisional_plan=provisional_plan,
        )
        session = BrainPlanningSession(
            session_id=f"brain_plan_{uuid4().hex[:12]}",
            task_id=request.task_id,
            user_instruction=request.description,
            draft=draft,
        )
        required_items = [item for item in draft.uncertainties if item.get("required", True)]
        optional_items = [item for item in draft.uncertainties if not item.get("required", True)]
        for item in [*required_items, *optional_items[: self.skill.max_questions]]:
            session.add_question(
                ClarificationQuestion(
                    question_id=str(
                        item.get("uncertainty_id") or f"q_{len(session.dialogue) + 1:02d}"
                    ),
                    question=str(
                        item.get("question") or "Please clarify the planning requirement."
                    ),
                    reason=str(item.get("reason") or ""),
                    applies_to=str(item.get("applies_to") or "task"),
                    options=list(item.get("options") or []),
                    required=bool(item.get("required", True)),
                    subtask_id=item.get("subtask_id"),
                    step_id=item.get("step_id"),
                    agent=item.get("agent"),
                    intent=item.get("intent"),
                )
            )
        if not session.dialogue:
            session.status = "awaiting_confirmation"
        self._mirror_session(session)
        self._record_observation("brain.plan_draft", session)
        return session

    def answer(self, session: BrainPlanningSession, answer: UserAnswer) -> BrainPlanningSession:
        session.add_answer(answer)
        self._mirror_session(session)
        self._record_observation(
            "brain.user_clarification", session, answer=answer.to_dialogue_item()
        )
        return session

    def confirm(
        self,
        session: BrainPlanningSession,
        confirmation: PlanConfirmation,
    ) -> BrainPlanningSession:
        session.set_confirmation(confirmation)
        self._mirror_session(session)
        self._record_observation(
            "brain.plan_confirmation", session, confirmation=confirmation.to_dict()
        )
        return session

    def _mirror_session(self, session: BrainPlanningSession) -> None:
        self.memory.update_task_context({"interactive_planning": session.to_task_context()})

    def _record_observation(
        self,
        observation_type: str,
        session: BrainPlanningSession,
        **payload: Any,
    ) -> None:
        record = getattr(self.memory, "record_working_observation", None)
        if not callable(record):
            return
        observation = {
            "source": "brain",
            "observation_type": observation_type,
            "session_id": session.session_id,
            "task_id": session.task_id,
            "status": session.status,
            "interactive_planning": session.to_task_context(),
            **payload,
        }
        record(observation)


__all__ = ["BrainInteractivePlanningController"]
