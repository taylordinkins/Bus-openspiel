"""Request/response envelopes for the HTTP API (OpenAPI source for generated web types)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator

from bots.registry import seat_type_error
from server.bots import SEAT_TYPE_PATTERN


def _check_seat_type(v: str) -> str:
    err = seat_type_error(v)
    if err is not None:
        raise ValueError(err)
    return v


SeatType = Annotated[
    str,
    Field(
        pattern=SEAT_TYPE_PATTERN,
        description="human, or a bot seat type: kind or kind:param (random).",
    ),
    AfterValidator(_check_seat_type),
]


class GameParams(BaseModel):
    """Creation parameters for a game; adapters read the fields they know (extras are kept)."""

    model_config = ConfigDict(extra="allow")

    num_players: int | None = Field(
        default=None,
        ge=2,
        le=5,
        description="Number of players (the adapter enforces its own range).",
    )
    returns: Literal["win_loss", "margin"] = Field(
        default="win_loss", description="Terminal returns mode."
    )


class CreateGameRequest(BaseModel):
    game: str = "bus"
    params: GameParams = Field(default_factory=GameParams)
    seats: list[SeatType]
    seed: int | None = None
    scenario: dict[str, Any] | None = Field(
        default=None,
        description="Optional scenario JSON (docs/SCENARIOS.md); the game starts from it.",
    )
    bot_think_time_s: float | None = Field(
        default=None,
        ge=0.1,
        le=60,
        description="Per-move think-time cap in seconds for search bots (mcts seats); default 5.",
    )


class LegalAction(BaseModel):
    """One action the seat to move may take, as described by the adapter."""

    id: int
    label: str
    kind: str | None = None
    params: dict[str, Any] | None = None


class Event(BaseModel):
    """One automatic step the engine ran after an action; `type` selects the other fields."""

    model_config = ConfigDict(extra="allow")

    type: str


class HistoryEntry(BaseModel):
    """One past action; `actor` is null for chance outcomes. Hidden picks have a null id."""

    index: int
    actor: int | None
    id: int | None = Field(description="Action id; null when hidden from the rendered seat.")
    label: str
    events: list[Event] = Field(
        default_factory=lambda: [],
        description="Automatic steps the engine ran as a result of this action (adapter-defined).",
    )


class BoardGeometry(BaseModel):
    """Static board data for the renderer, served once per game type."""

    game: str
    geometry: dict[str, Any] = Field(description="Adapter-defined geometry payload.")


class GameView(BaseModel):
    """A game as seen by one seat."""

    id: str
    game: str
    seats: list[SeatType]
    seat_types: list[SeatType]
    bot_thinking: bool = Field(
        description="A bot seat is to move and has not failed (false in the error state)."
    )
    bot_error: str | None = Field(
        default=None, description="Set when the bot worker failed; POST bots/retry to retry."
    )
    bot_think_time_s: float | None = Field(
        default=None, description="Think-time cap in seconds for search bot seats."
    )
    seat: int
    history_length: int
    history: list[HistoryEntry] = Field(description="Labelled action history, oldest first.")
    dev_mode: bool = Field(description="Server allows dev-only operations (rewind).")
    current_player: int | None = Field(description="Seat to move; null at chance or terminal.")
    is_terminal: bool
    legal_actions: list[LegalAction] = Field(description="Actions for current_player.")
    returns: list[float] | None = Field(default=None, description="Final returns when terminal.")
    winners: list[int] | None = Field(default=None, description="Winning seats when terminal.")
    view: dict[str, Any] = Field(description="Adapter-defined view payload.")


class ActionRequest(BaseModel):
    action: int


class BatchActionRequest(BaseModel):
    action_ids: list[int] = Field(
        min_length=1, max_length=32, description="Actions applied in order, all or nothing."
    )


class RewindRequest(BaseModel):
    player_actions: int


class GameExport(BaseModel):
    """Replay file: everything needed to rebuild a game by replaying `history`."""

    game: str
    params: GameParams
    seats: list[SeatType]
    history: list[int]
    scenario: dict[str, Any] | None = Field(
        default=None, description="Scenario the game started from, if any."
    )
    bot_think_time_s: float | None = Field(
        default=None,
        ge=0.1,
        le=60,
        description="Per-move think-time cap in seconds for search bots (mcts seats); default 5.",
    )


class BotSearchOption(BaseModel):
    """A net-guided MCTS seat offered for a model, e.g. `mcts_net:50-<name>`."""

    seat_type: str
    label: str


class BotModel(BaseModel):
    """A trained net offered as a `net:<name>` seat (models/manifest.yaml)."""

    seat_type: str = Field(description="Seat type to post, `net:<name>`.")
    label: str
    description: str = ""
    players: list[int] = Field(description="Player counts the model may be seated in.")
    search: list[BotSearchOption] = Field(
        default_factory=lambda: [], description="Net-guided MCTS seats (`mcts_net:<sims>-<name>`)."
    )


class ScenarioSummary(BaseModel):
    """A named built-in scenario offered in the lobby (UI_DESIGN 6.9)."""

    id: str
    game: str
    title: str
    description: str = ""
    players: int
    returns: Literal["win_loss", "margin"]
    round: int
    phase: str = Field(description="Phase code the position rests at (ACTIONS 3).")


class NamedScenario(ScenarioSummary):
    scenario: dict[str, Any] = Field(
        description="Scenario JSON (docs/SCENARIOS.md) to post as CreateGameRequest.scenario."
    )


class ReportRequest(BaseModel):
    note: str = Field(min_length=1, description="What went wrong (required).")
    category: str | None = Field(default=None, description='e.g. "engine", "ui", "other".')

    @field_validator("note")
    @classmethod
    def _note_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("note must not be blank")
        return v


class ReportResponse(BaseModel):
    report_id: str
    path: str


class ErrorResponse(BaseModel):
    detail: str
