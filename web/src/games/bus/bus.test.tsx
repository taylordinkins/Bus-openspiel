import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, vi } from "vitest";
import GameScreen from "../../shell/GameScreen";
import type { GameView } from "../../shell/useGame";
import { getPlugin } from "../registry";

const GET = vi.fn();
const POST = vi.fn();
vi.mock("../../api/client", () => ({
  api: { GET: (...a: unknown[]) => GET(...a), POST: (...a: unknown[]) => POST(...a) },
}));

function view(over: Partial<GameView> = {}): GameView {
  return {
    id: "b1",
    game: "bus",
    seats: ["human", "random", "random"],
    seat_types: ["human", "random", "random"],
    bot_thinking: false,
    bot_error: null,
    seat: 0,
    history_length: 0,
    history: [],
    dev_mode: false,
    current_player: 0,
    is_terminal: false,
    legal_actions: [
      { id: 11, label: "Extend line at A" },
      { id: 12, label: "Pass" },
    ],
    returns: null,
    winners: null,
    view: { prompt: { phase: "LN", actor: 0, units_left: 2 } },
    ...over,
  };
}

beforeEach(() => {
  GET.mockReset();
  POST.mockReset();
});

describe("Bus plugin on the action-list fallback", () => {
  it("is registered for the bus game", () => {
    expect(getPlugin("bus").id).toBe("bus");
  });

  it("shows the prompt with P0 for seat 0 and posts the clicked action", async () => {
    GET.mockResolvedValue({ data: view() });
    POST.mockResolvedValue({ data: view({ legal_actions: [{ id: 13, label: "Next" }] }) });
    render(<GameScreen gameId="b1" pollMs={{ fast: 10000, slow: 10000 }} />);
    expect(await screen.findByTestId("bus-prompt")).toHaveTextContent("P0 to act");
    expect(screen.queryByText(/Player 1/)).toBeNull();
    expect(screen.getByText(/P0 human, P1 random, P2 random/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Extend line at A" }));
    await waitFor(() =>
      expect(POST).toHaveBeenCalledWith("/api/games/{game_id}/actions", {
        params: { path: { game_id: "b1" }, query: { seat: 0 } },
        body: { action: 11 },
      }),
    );
  });
});
