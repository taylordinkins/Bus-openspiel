import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import Lobby from "./Lobby";

const POST = vi.fn();
vi.mock("../api/client", () => ({
  api: { GET: vi.fn(), POST: (...a: unknown[]) => POST(...a) },
}));

beforeEach(() => {
  POST.mockReset();
  POST.mockResolvedValue({ data: { id: "g1" } });
});

const seat = (i: number) => screen.getByLabelText(`Seat P${i}`) as HTMLSelectElement;
const think = () => screen.queryByLabelText(/Think time/);

describe("Lobby seat picker", () => {
  it("defaults to a human at P0 and random bots, without a think-time field", () => {
    render(<Lobby onCreated={vi.fn()} />);
    expect([0, 1, 2, 3].map((i) => seat(i).value)).toEqual(["human", "random", "random", "random"]);
    expect(think()).toBeNull();
    expect(screen.queryByText(/Player 1/)).toBeNull();
  });

  it("offers the bot kinds and MCTS strengths, mapped to seat type strings", () => {
    render(<Lobby onCreated={vi.fn()} />);
    const opts = Array.from(seat(1).options).map((o) => [o.value, o.textContent]);
    expect(opts).toEqual([
      ["human", "Human"],
      ["random", "Random bot"],
      ["heuristic", "Heuristic bot"],
      ["mcts:50", "MCTS · Quick (50 sims)"],
      ["mcts:200", "MCTS · Normal (200 sims)"],
      ["mcts:1000", "MCTS · Strong (1000 sims)"],
    ]);
  });

  it("shows think time only while an MCTS seat is present", () => {
    render(<Lobby onCreated={vi.fn()} />);
    fireEvent.change(seat(2), { target: { value: "mcts:200" } });
    expect(think()).toHaveValue(5);
    fireEvent.change(seat(2), { target: { value: "heuristic" } });
    expect(think()).toBeNull();
  });

  it("posts heuristic seats without a think time", async () => {
    render(<Lobby onCreated={vi.fn()} />);
    fireEvent.change(seat(1), { target: { value: "heuristic" } });
    fireEvent.click(screen.getByRole("button", { name: "Create game" }));
    await waitFor(() => expect(POST).toHaveBeenCalled());
    const body = POST.mock.calls[0]![1].body;
    expect(body.seats).toEqual(["human", "heuristic", "random", "random"]);
    expect(body).not.toHaveProperty("bot_think_time_s");
  });

  it("posts MCTS seats with the chosen think time and blocks an invalid one", async () => {
    render(<Lobby onCreated={vi.fn()} />);
    fireEvent.change(seat(3), { target: { value: "mcts:1000" } });
    fireEvent.change(think()!, { target: { value: "0" } });
    expect(screen.getByRole("button", { name: "Create game" })).toBeDisabled();
    fireEvent.change(think()!, { target: { value: "2.5" } });
    fireEvent.click(screen.getByRole("button", { name: "Create game" }));
    await waitFor(() => expect(POST).toHaveBeenCalled());
    const body = POST.mock.calls[0]![1].body;
    expect(body.seats).toEqual(["human", "random", "random", "mcts:1000"]);
    expect(body.bot_think_time_s).toBe(2.5);
  });
});
