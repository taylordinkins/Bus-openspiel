import { render, screen } from "@testing-library/react";
import { beforeEach, vi } from "vitest";
import type { GameView } from "../../shell/useGame";
import { getPlugin } from "../registry";
import midView from "./fixtures/mid.json";
import geometry from "./fixtures/geometry.json";
import { resetGeometryCache } from "./geometry";

const GET = vi.fn();
vi.mock("../../api/client", () => ({ api: { GET: (...a: unknown[]) => GET(...a) } }));

beforeEach(() => {
  GET.mockReset();
  resetGeometryCache();
});

describe("Bus plugin board", () => {
  it("fetches the geometry once and draws the board from the server's nested view", async () => {
    GET.mockResolvedValue({ data: { game: "bus", geometry: geometry.geometry } });
    // the server's GameView carries the adapter's four keys one level down (GameView.view.view)
    const game = { ...midView, id: "g", game: "bus", seats: [], view: midView } as unknown as GameView;
    const plugin = getPlugin("bus");
    const { rerender, container } = render(<>{plugin.renderBoard(game)}</>);
    expect(await screen.findByTestId("bus-svg")).toBeInTheDocument();
    expect(container.querySelectorAll(".building").length).toBeGreaterThan(0);
    expect(screen.getByTestId("bus-prompt")).toHaveTextContent("P2 to act");
    rerender(<>{plugin.renderBoard({ ...game })}</>);
    expect(GET).toHaveBeenCalledTimes(1);
    expect(GET).toHaveBeenCalledWith("/api/boards/{game}", { params: { path: { game: "bus" } } });
  });

  it("keeps the raw view available", () => {
    const game = { ...midView, id: "g", game: "bus", seats: [], view: midView } as unknown as GameView;
    GET.mockResolvedValue({ data: {} });
    render(<>{getPlugin("bus").renderBoard(game)}</>);
    expect(screen.getByLabelText("Game state", { selector: "pre" })).toBeInTheDocument();
  });
});
