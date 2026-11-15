"""FastAPI app: thin JSON layer over SessionStore. Game-agnostic via GameAdapter."""

from __future__ import annotations

import os
import random
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, FastAPI, HTTPException, Query

from server.adapters import default_adapters
from server.adapters.base import GameAdapter
from server.reports import DEFAULT_REPORTS_DIR, write_report
from server.schemas import (
    ActionRequest,
    BatchActionRequest,
    BoardGeometry,
    CreateGameRequest,
    ErrorResponse,
    GameExport,
    GameParams,
    GameView,
    NamedScenario,
    ReportRequest,
    ReportResponse,
    RewindRequest,
    ScenarioSummary,
)
from server.sessions import (
    CannotGoBack,
    IllegalAction,
    Session,
    SessionError,
    SessionStore,
    UnknownSession,
)

_ERR = {"model": ErrorResponse}


def _to_adapter_params(p: GameParams) -> dict[str, Any]:
    """Fields the client set (plus defaults); the adapter maps them to game parameters."""
    return p.model_dump(exclude_none=True)


def _from_adapter_params(session: Session) -> GameParams:
    return GameParams.model_validate({**session.params, "num_players": session.game.num_players()})


def create_app(
    dev_mode: bool = False,
    reports_dir: Path | str | None = None,
    inline_bots: bool | None = None,
    adapters: Mapping[str, GameAdapter] | None = None,
) -> FastAPI:
    # Production default: bots think on a background thread. BUS_BOTS_INLINE=1 (or
    # inline_bots=True) plays them synchronously inside the request, for tests and tools.
    if inline_bots is None:
        inline_bots = os.environ.get("BUS_BOTS_INLINE", "") not in ("", "0")
    reports_path = Path(reports_dir or os.environ.get("BUS_REPORTS_DIR") or DEFAULT_REPORTS_DIR)
    store = SessionStore(
        default_adapters() if adapters is None else adapters, inline_bots=inline_bots
    )
    api = APIRouter(prefix="/api")

    def lookup(game_id: str) -> Session:
        try:
            return store.get(game_id)
        except UnknownSession:
            raise HTTPException(404, f"unknown game {game_id!r}") from None

    def render(s: Session, seat: int | None = None, at: int | None = None) -> GameView:
        with s.lock:
            if at is not None and not 0 <= at <= len(s.history):
                raise HTTPException(409, f"at={at} out of range 0..{len(s.history)}")
            frame = at is not None and at != len(s.history)
            state = s.state_at(at)
            if seat is None:
                seat = 0 if state.is_terminal() else max(state.current_player(), 0)
            if not 0 <= seat < len(s.seats):
                raise HTTPException(422, f"seat {seat} out of range")
            seats: list[str] = list(s.seats)
            v = s.view(seat, at)
            if frame:
                # A past frame: "thinking" = a bot was to move there (playback shows it next).
                cp = None if state.is_terminal() else state.current_player()
                thinking = cp is not None and cp >= 0 and not s.is_human(cp)
                error = None
            else:
                thinking, error = s.bot_thinking, s.bot_error
            return GameView.model_validate(
                {
                    "id": s.id,
                    "game": s.game_name,
                    "seats": seats,
                    "seat_types": seats,
                    "bot_thinking": thinking,
                    "bot_error": error,
                    "seat": seat,
                    "history_length": len(s.history) if at is None else at,
                    "history": s.history_entries(seat, at),
                    "dev_mode": dev_mode,
                    "current_player": v["current_player"],
                    "is_terminal": v["is_terminal"],
                    "legal_actions": v["legal_actions"],
                    "returns": v.get("returns"),
                    "winners": v.get("winners"),
                    "view": v,
                }
            )

    def make(
        game: str,
        params: GameParams,
        seats: list[str],
        seed: int | None,
        history: list[int],
        scenario: dict[str, Any] | None = None,
    ):
        # An explicit seed means deterministic play (tests/tools, salt ""); otherwise the seed is
        # random and a secret salt keeps bot/chance picks unpredictable. Neither is ever sent out.
        salt = "" if seed is not None else secrets.token_hex(16)
        if seed is None:
            seed = random.randrange(2**31)
        adapter_params = _to_adapter_params(params)
        if scenario is not None:
            if game not in store.adapters:
                raise HTTPException(400, f"unknown game {game!r}")
            try:
                fixed = store.adapters[game].scenario_params(scenario)
            except (ValueError, TypeError, AttributeError, KeyError) as e:
                raise HTTPException(422, f"invalid scenario: {e}") from None
            # The scenario fixes these; a field the client set explicitly must agree with it.
            given = {k for k in params.model_fields_set if getattr(params, k, None) is not None}
            for k, v in fixed.items():
                if k in given and adapter_params.get(k) != v:
                    raise HTTPException(
                        422, f"{k} {adapter_params.get(k)} does not match scenario ({v})"
                    )
                adapter_params[k] = v
        try:
            return store.create(game, adapter_params, seats, seed, history, scenario, salt)
        except IllegalAction as e:
            raise HTTPException(400, f"invalid history: {e}") from None
        except (SessionError, ValueError) as e:
            raise HTTPException(400, str(e)) from None

    @api.post("/games", operation_id="create_game", responses={400: _ERR, 422: _ERR})
    def create_game(req: CreateGameRequest) -> GameView:
        return render(make(req.game, req.params, list(req.seats), req.seed, [], req.scenario))

    @api.post("/games/import", operation_id="import_game", responses={400: _ERR, 422: _ERR})
    def import_game(body: GameExport) -> GameView:
        return render(
            make(body.game, body.params, list(body.seats), None, body.history, body.scenario)
        )

    def scenarios_of(game: str) -> list[dict[str, Any]]:
        adapter = store.adapters.get(game)
        if adapter is None:
            raise HTTPException(404, f"unknown game {game!r}")
        return [{**sc, "game": game} for sc in adapter.named_scenarios()]

    @api.get("/scenarios", operation_id="list_scenarios", responses={404: _ERR})
    def list_scenarios(game: Annotated[str, Query()] = "bus") -> list[ScenarioSummary]:
        return [ScenarioSummary.model_validate(sc) for sc in scenarios_of(game)]

    @api.get("/scenarios/{scenario_id}", operation_id="get_scenario", responses={404: _ERR})
    def get_scenario(scenario_id: str, game: Annotated[str, Query()] = "bus") -> NamedScenario:
        for sc in scenarios_of(game):
            if sc["id"] == scenario_id:
                return NamedScenario.model_validate(sc)
        raise HTTPException(404, f"unknown scenario {scenario_id!r}")

    @api.get("/boards/{game}", operation_id="get_board", responses={404: _ERR})
    def get_board(game: str) -> BoardGeometry:
        adapter = store.adapters.get(game)
        geometry = None if adapter is None else adapter.board_geometry()
        if geometry is None:
            raise HTTPException(404, f"no board geometry for game {game!r}")
        return BoardGeometry(game=game, geometry=geometry)

    @api.get("/games/{game_id}", operation_id="get_game", responses={404: _ERR, 409: _ERR})
    def get_game(
        game_id: str,
        seat: Annotated[int | None, Query()] = None,
        at: Annotated[
            int | None,
            Query(
                description="Render the state after the first `at` history entries (a read-only "
                "frame for client playback of bot turns); omitted = the current state."
            ),
        ] = None,
    ) -> GameView:
        return render(lookup(game_id), seat, at)

    @api.post(
        "/games/{game_id}/actions",
        operation_id="post_action",
        responses={404: _ERR, 409: _ERR},
    )
    def post_action(
        game_id: str, req: ActionRequest, seat: Annotated[int | None, Query()] = None
    ) -> GameView:
        s = lookup(game_id)
        with s.lock:
            try:
                s.apply(req.action)
            except IllegalAction as e:
                raise HTTPException(409, str(e)) from None
            return render(s, seat)

    @api.post(
        "/games/{game_id}/actions/batch",
        operation_id="post_action_batch",
        responses={404: _ERR, 409: _ERR},
    )
    def post_action_batch(
        game_id: str, req: BatchActionRequest, seat: Annotated[int | None, Query()] = None
    ) -> GameView:
        s = lookup(game_id)
        with s.lock:
            try:
                s.apply_batch(req.action_ids, seat)
            except IllegalAction as e:
                raise HTTPException(409, str(e)) from None
            return render(s, seat)

    @api.post("/games/{game_id}/back", operation_id="go_back", responses={404: _ERR, 409: _ERR})
    def go_back(game_id: str, seat: Annotated[int | None, Query()] = None) -> GameView:
        s = lookup(game_id)
        with s.lock:
            try:
                s.back()
            except CannotGoBack as e:
                raise HTTPException(409, str(e)) from None
            return render(s, seat)

    @api.post(
        "/games/{game_id}/bots/retry",
        operation_id="retry_bots",
        responses={404: _ERR, 409: _ERR},
    )
    def retry_bots(game_id: str, seat: Annotated[int | None, Query()] = None) -> GameView:
        s = lookup(game_id)
        with s.lock:
            try:
                s.retry_bots()
            except IllegalAction as e:
                raise HTTPException(409, str(e)) from None
            return render(s, seat)

    @api.post(
        "/games/{game_id}/rewind",
        operation_id="rewind_game",
        responses={400: _ERR, 403: _ERR, 404: _ERR},
    )
    def rewind_game(
        game_id: str, req: RewindRequest, seat: Annotated[int | None, Query()] = None
    ) -> GameView:
        if not dev_mode:
            raise HTTPException(403, "rewind requires dev mode")
        s = lookup(game_id)
        with s.lock:
            try:
                s.rewind(req.player_actions)
            except SessionError as e:
                raise HTTPException(400, str(e)) from None
            return render(s, seat)

    @api.get(
        "/games/{game_id}/export",
        operation_id="export_game",
        responses={404: _ERR, 409: _ERR},
        response_model_exclude_unset=True,
    )
    def export_game(game_id: str) -> GameExport:
        s = lookup(game_id)
        with s.lock:
            blocked = s.adapter.export_block(s.state)
            if blocked is not None:
                raise HTTPException(409, blocked)
            return build_export(s)

    def build_export(s: Session) -> GameExport:
        with s.lock:
            seats: list[str] = list(s.seats)
            data: dict[str, Any] = {
                "game": s.game_name,
                "params": _from_adapter_params(s),
                "seats": seats,
                "history": list(s.history),
            }
            if s.scenario is not None:  # omitted for plain games (keeps old exports unchanged)
                data["scenario"] = s.scenario
            return GameExport.model_validate(data)

    @api.post("/games/{game_id}/report", operation_id="report_game", responses={404: _ERR})
    def report_game(game_id: str, req: ReportRequest) -> ReportResponse:
        s = lookup(game_id)
        with s.lock:
            export = build_export(s)  # server-side report: not subject to export_block
            report_id, path = write_report(
                reports_path,
                {
                    "note": req.note,
                    "category": req.category,
                    "session_id": s.id,
                    "game": export.game,
                    "params": export.params.model_dump(),
                    "seats": list(export.seats),
                    "seed": s.seed,
                    "salt": s.salt,
                    "export": export.model_dump(exclude_unset=True),
                    "view": render(s).model_dump(),
                    "history": list(s.history),
                },
            )
            return ReportResponse(report_id=report_id, path=str(path))

    app = FastAPI(title="Bus server")
    app.include_router(api)
    return app


app = create_app(dev_mode=bool(os.environ.get("BUS_DEV")))
