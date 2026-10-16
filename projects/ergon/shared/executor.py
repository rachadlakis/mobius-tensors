"""Shared A2A ``AgentExecutor`` for every worker agent.

One copy of the A2A executor logic; each employee's ``agent_executor.py`` is a
three-line subclass.

Responsibilities:

* unpack the incoming A2A message, run the ADK agent, stream ``working`` status
  updates, and ``complete`` the task with the final answer;
* harvest any files the agent's tools saved as artifacts *during this turn*
  (diffed against a pre-run snapshot) and staple them to the A2A task, so the
  Chief of Staff actually receives them;
* ``_prepare`` hook - with ``save_incoming_files=True`` (employees with
  ``run_python``) it persists forwarded uploads as named artifacts the tools can
  load by name; otherwise a no-op.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Mapping
from contextlib import aclosing

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState, UnsupportedOperationError
from a2a.utils.errors import ServerError
from google.adk import Runner
from google.adk.events import Event
from google.genai import types

from shared.a2a_files import (
    convert_a2a_parts_to_genai,
    convert_genai_parts_to_a2a,
    file_part,
)
from shared.config import USER_ID

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BaseA2AExecutor(AgentExecutor):
    """Runs an ADK agent behind the A2A protocol. Subclass and (optionally)
    override ``_prepare``."""

    def __init__(self, runner: Runner, save_incoming_files: bool = False):
        self.runner = runner
        self._running_sessions: dict = {}
        # Employees whose tools open files by name (run_python) need the files
        # the Chief of Staff forwarded saved as artifacts before the turn runs.
        self._save_incoming = save_incoming_files

    # ------------------------------------------------------------------ #
    # Hook
    # ------------------------------------------------------------------ #
    async def _prepare(self, new_message: types.Content, session_id: str) -> None:
        """Runs once the session exists, before the agent runs: saves forwarded
        files as artifacts when the employee asked for it, otherwise a no-op."""
        if self._save_incoming:
            await self._save_incoming_files(new_message, session_id)

    # ------------------------------------------------------------------ #
    # A2A entry points
    # ------------------------------------------------------------------ #
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if not context.task_id or not context.context_id:
            raise ValueError("RequestContext must have task_id and context_id")
        if not context.message:
            raise ValueError("RequestContext must have a message")

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            await updater.submit()
        await updater.start_work()
        await self._process_request(
            types.UserContent(parts=convert_a2a_parts_to_genai(context.message.parts)),
            context.context_id,
            updater,
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())

    # ------------------------------------------------------------------ #
    # Turn processing
    # ------------------------------------------------------------------ #
    def _run_agent(
        self, session_id, new_message: types.Content
    ) -> AsyncGenerator[Event, None]:
        return self.runner.run_async(
            session_id=session_id, user_id=USER_ID, new_message=new_message
        )

    async def _process_request(
        self, new_message: types.Content, session_id: str, task_updater: TaskUpdater
    ) -> None:
        session_obj = await self._upsert_session(session_id)
        session_id = session_obj.id

        await self._prepare(new_message, session_id)

        # Snapshot existing artifacts so we only forward what THIS turn produced.
        known_artifacts = await self._artifact_versions(session_id)

        # Drain the ADK runner generator to completion rather than `break`ing on
        # the final event: an early break leaves it to be GC-closed later on a
        # different asyncio context, which trips ADK's OpenTelemetry span
        # teardown ("Token was created in a different Context") and logs the
        # invocation as cancelled on every request. `aclosing` is the belt-and-
        # braces close for the exception path.
        completed = False
        async with aclosing(self._run_agent(session_id, new_message)) as events:
            async for event in events:
                if completed:
                    continue  # final response already sent; let the generator finish
                if event.is_final_response():
                    parts = convert_genai_parts_to_a2a(
                        event.content.parts if event.content and event.content.parts else []
                    )
                    logger.debug("Yielding final response: %s", parts)
                    if parts:
                        await task_updater.add_artifact(parts)
                    for name, fpart in await self._new_file_artifacts(session_id, known_artifacts):
                        logger.debug("Attaching produced file artifact: %s", name)
                        await task_updater.add_artifact([fpart], name=name)
                    await task_updater.complete()
                    completed = True
                    continue
                if not event.get_function_calls():
                    logger.debug("Yielding update response")
                    await task_updater.update_status(
                        TaskState.working,
                        message=task_updater.new_agent_message(
                            convert_genai_parts_to_a2a(
                                event.content.parts
                                if event.content and event.content.parts
                                else []
                            ),
                        ),
                    )
                else:
                    logger.debug("Skipping event")

    # ------------------------------------------------------------------ #
    # Artifact plumbing
    # ------------------------------------------------------------------ #
    async def _save_incoming_files(
        self, new_message: types.Content, session_id: str
    ) -> None:
        """Persist every inline file blob on the user turn as a session artifact,
        keyed by its display name, so tools can load it by name."""
        svc = getattr(self.runner, "artifact_service", None)
        if svc is None:
            return
        for i, part in enumerate(getattr(new_message, "parts", None) or []):
            blob = getattr(part, "inline_data", None)
            data = getattr(blob, "data", None) if blob else None
            if not isinstance(data, (bytes, bytearray)):
                continue
            name = (
                getattr(blob, "display_name", None)
                or getattr(part, "display_name", None)
                or f"upload_{i + 1}"
            )
            try:
                result = svc.save_artifact(
                    app_name=self.runner.app_name, user_id=USER_ID,
                    session_id=session_id, filename=str(name),
                    artifact=types.Part(inline_data=types.Blob(
                        data=bytes(data),
                        mime_type=getattr(blob, "mime_type", None) or "application/octet-stream",
                    )),
                )
                if hasattr(result, "__await__"):
                    await result
                logger.debug("saved incoming file artifact: %s", name)
            except Exception as e:  # artifact backend unavailable - not fatal
                logger.warning("could not save incoming file %r: %s", name, e)

    async def _artifact_keys(self, session_id: str) -> set[str]:
        svc = getattr(self.runner, "artifact_service", None)
        if svc is None:
            return set()
        try:
            return set(await svc.list_artifact_keys(
                app_name=self.runner.app_name, user_id=USER_ID, session_id=session_id
            ))
        except Exception as e:  # artifact backend unavailable - not fatal
            logger.warning("could not list artifacts: %s", e)
            return set()

    async def _artifact_versions(self, session_id: str) -> dict[str, int]:
        """``{filename: latest version}`` for the session.

        Snapshotting names alone is not enough. Saving an artifact under a name
        that already exists creates a new *version* of that name, not a new key,
        so a tool that always writes to a fixed filename (``report.pdf``)
        looks unchanged on the second turn of a
        conversation and its output is silently never forwarded. Comparing
        versions catches the overwrite.

        ``-1`` means "version unknown" - a backend without version support
        degrades to the old name-only behaviour rather than re-sending files.
        """
        svc = getattr(self.runner, "artifact_service", None)
        if svc is None:
            return {}
        out: dict[str, int] = {}
        for name in await self._artifact_keys(session_id):
            try:
                versions = await svc.list_versions(
                    app_name=self.runner.app_name, user_id=USER_ID,
                    session_id=session_id, filename=name,
                )
            except Exception as e:  # backend without versioning - not fatal
                logger.warning("could not list versions of %s: %s", name, e)
                versions = None
            out[name] = max(versions) if versions else -1
        return out

    async def _new_file_artifacts(self, session_id: str, known: Mapping[str, int]):
        """(name, A2A file Part) for every artifact this turn created *or* rewrote.

        ``known`` is the pre-run snapshot from :meth:`_artifact_versions`.
        """
        svc = getattr(self.runner, "artifact_service", None)
        if svc is None:
            return []
        out = []
        # sorted() so the order files are stapled to the task is deterministic.
        for name, version in sorted((await self._artifact_versions(session_id)).items()):
            if name in known and version <= known[name]:
                continue
            try:
                part = await svc.load_artifact(
                    app_name=self.runner.app_name, user_id=USER_ID,
                    session_id=session_id, filename=name,
                )
            except Exception as e:
                logger.warning("could not load artifact %s: %s", name, e)
                continue
            if part and part.inline_data and part.inline_data.data:
                out.append((name, file_part(
                    name, part.inline_data.data, part.inline_data.mime_type
                )))
        return out

    async def _upsert_session(self, session_id: str):
        session = await self.runner.session_service.get_session(
            app_name=self.runner.app_name, user_id=USER_ID, session_id=session_id
        )
        if session is None:
            session = await self.runner.session_service.create_session(
                app_name=self.runner.app_name,
                user_id=USER_ID,
                session_id=session_id,
            )
        if session is None:
            raise RuntimeError(f"Failed to get or create session: {session_id}")
        return session
