import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, vi } from "vitest";
import Lobby from "./Lobby";
import { parseStartDoc } from "./startDoc";

const GET = vi.fn();
const POST = vi.fn();
vi.mock("../api/client", () => ({
  api: { GET: (...a: unknown[]) => GET(...a), POST: (...a: unknown[]) => POST(...a) },
}));

const NAMED = [
  { id: "fresh_4p", game: "bus", title: "Fresh game (4 players)", description: "Fresh.", players: 4, returns: "win_loss", round: 0, phase: "SB" },
  { id: "three", game: "bus", title: "Three seats", description: "", players: 3, returns: "margin", round: 1, phase: "SEL" },
];
const THREE = { format: 1, parameters: { players: 3, returns: "margin" }, round: 1, description: "Three-seat fixture" };

beforeEach(() => {
  GET.mockReset();
  POST.mockReset();
  GET.mockImplementation((path: string) =>
    Promise.resolve(
      path === "/api/scenarios"
        ? { data: NAMED }
        : { data: { ...NAMED[1], scenario: THREE } },
    ),
  );
});

function scenarioMode() {
  render(<Lobby onCreated={onCreated} />);
  fireEvent.click(screen.getByLabelText("Scenario"));
}
const onCreated = vi.fn();
const json = () => screen.getByLabelText("Scenario JSON") as HTMLTextAreaElement;

describe("parseStartDoc", () => {
  it("classifies scenarios, replays and junk", () => {
    expect(parseStartDoc("  ").kind).toBe("empty");
    expect(parseStartDoc("{nope").kind).toBe("error");
    expect(parseStartDoc("[1]").kind).toBe("error");
    expect(parseStartDoc('{"a": 1}').kind).toBe("error");
    const s = parseStartDoc('{"format": 1}');
    expect(s).toMatchObject({ kind: "scenario", players: 4, returns: "win_loss" });
    expect(parseStartDoc('{"format": 1, "parameters": {"players": 7}}')).toMatchObject({ players: null });
    const r = parseStartDoc(
      JSON.stringify({ game: "bus", params: { num_players: 3, returns: "margin" }, seats: ["human", "random", "human"], history: [1, 2] }),
    );
    expect(r).toMatchObject({ kind: "replay", players: 3, returns: "margin", actions: 2, fromScenario: false });
  });
});

describe("Lobby scenarios", () => {
  it("does not fetch the named list until Scenario is chosen", () => {
    render(<Lobby onCreated={vi.fn()} />);
    expect(GET).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Players")).toBeInTheDocument();
  });

  it("starts a game from pasted scenario JSON with the scenario's seat count", async () => {
    POST.mockResolvedValue({ data: { id: "s1" } });
    scenarioMode();
    fireEvent.change(json(), { target: { value: JSON.stringify(THREE) } });
    expect(screen.getByRole("status")).toHaveTextContent("Scenario · 3 players · Margin");
    expect(screen.queryByLabelText("Players")).toBeNull();
    expect(screen.getByLabelText("Seat P2")).toBeInTheDocument();
    expect(screen.queryByLabelText("Seat P3")).toBeNull();
    fireEvent.change(screen.getByLabelText("Seat P2"), { target: { value: "human" } });
    fireEvent.click(screen.getByRole("button", { name: "Create game" }));
    await waitFor(() => expect(onCreated).toHaveBeenCalledWith("s1"));
    expect(POST).toHaveBeenCalledWith("/api/games", {
      body: { game: "bus", seats: ["human", "random", "human"], scenario: THREE },
    });
  });

  it("lists named scenarios and loads the picked one into the editor", async () => {
    POST.mockResolvedValue({ data: { id: "n1" } });
    scenarioMode();
    const pick = await screen.findByLabelText("Named scenario");
    await waitFor(() => expect(pick).toBeEnabled());
    expect(screen.getByRole("option", { name: "Three seats · 3p" })).toBeInTheDocument();
    fireEvent.change(pick, { target: { value: "three" } });
    await waitFor(() => expect(JSON.parse(json().value)).toEqual(THREE));
    expect(GET).toHaveBeenCalledWith("/api/scenarios/{scenario_id}", {
      params: { path: { scenario_id: "three" }, query: { game: "bus" } },
    });
    fireEvent.click(screen.getByRole("button", { name: "Create game" }));
    await waitFor(() => expect(POST).toHaveBeenCalled());
    expect(POST.mock.calls[0]?.[1]).toMatchObject({ body: { scenario: THREE } });
  });

  it("loads an uploaded file and imports a replay with the chosen seats", async () => {
    POST.mockResolvedValue({ data: { id: "r1" } });
    scenarioMode();
    const replay = { game: "bus", params: { num_players: 3, returns: "win_loss" }, seats: ["human", "human", "random"], history: [4, 5] };
    const file = new File([JSON.stringify(replay)], "game.json", { type: "application/json" });
    Object.defineProperty(file, "text", { value: () => Promise.resolve(JSON.stringify(replay)) });
    fireEvent.change(screen.getByLabelText("Upload scenario file"), { target: { files: [file] } });
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("Replay · 3 players"));
    expect((screen.getByLabelText("Seat P1") as HTMLSelectElement).value).toBe("human");
    expect(screen.queryByLabelText(/Seed/)).toBeNull();
    fireEvent.change(screen.getByLabelText("Seat P2"), { target: { value: "human" } });
    fireEvent.click(screen.getByRole("button", { name: "Create game" }));
    await waitFor(() => expect(onCreated).toHaveBeenCalledWith("r1"));
    expect(POST).toHaveBeenCalledWith("/api/games/import", {
      body: { ...replay, seats: ["human", "human", "human"] },
    });
  });

  it("shows invalid JSON in an alert without posting", async () => {
    scenarioMode();
    fireEvent.change(json(), { target: { value: "{ not json" } });
    expect(screen.getByRole("status")).toHaveTextContent("Not valid JSON");
    fireEvent.click(screen.getByRole("button", { name: "Create game" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Not valid JSON");
    expect(POST).not.toHaveBeenCalled();
  });

  it("asks for a document when the editor is empty", async () => {
    scenarioMode();
    fireEvent.click(screen.getByRole("button", { name: "Create game" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("No scenario yet");
  });

  it("shows the server's rejection in the themed alert", async () => {
    POST.mockResolvedValue({ error: { detail: "invalid scenario: S-PHASE: LN is not terminal but has no legal action" } });
    scenarioMode();
    fireEvent.change(json(), { target: { value: '{"format": 1}' } });
    fireEvent.click(screen.getByRole("button", { name: "Create game" }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveClass("alert");
    expect(alert).toHaveTextContent("S-PHASE");
  });

  it("reports a failed scenario list", async () => {
    GET.mockResolvedValue({ error: { detail: "boom" } });
    scenarioMode();
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not list scenarios: boom");
  });

  it("is keyboard reachable: mode radios, picker, upload and editor are focusable controls", async () => {
    scenarioMode();
    for (const el of [
      screen.getByLabelText("Fresh setup"),
      screen.getByLabelText("Scenario"),
      screen.getByLabelText("Named scenario"),
      screen.getByLabelText("Upload scenario file"),
      json(),
    ]) {
      expect(el).not.toHaveAttribute("tabindex", "-1");
      expect(el.tagName).toMatch(/INPUT|SELECT|TEXTAREA/);
    }
    await waitFor(() => expect(screen.getByLabelText("Named scenario")).toBeEnabled());
  });
});
