/**
 * Classify the JSON pasted or uploaded in the lobby (UI_DESIGN §6.9): a scenario
 * (docs/SCENARIOS.md, has `format`) or a replay export (GET /api/games/{id}/export).
 * Only the shape is read here; the server's strict loader is the validator.
 */

export type Returns = "win_loss" | "margin";

export type StartDoc =
  | { kind: "empty" }
  | { kind: "error"; message: string }
  | {
      kind: "scenario";
      doc: Record<string, unknown>;
      /** Seat count for the seat cards; null if the document's count is not 3..5. */
      players: number | null;
      returns: Returns;
      description: string;
      phase: string | null;
    }
  | {
      kind: "replay";
      doc: Record<string, unknown>;
      players: number | null;
      returns: Returns;
      actions: number;
      seats: string[];
      fromScenario: boolean;
    };

const isObj = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

const seatCount = (n: unknown): number | null =>
  typeof n === "number" && Number.isInteger(n) && n >= 3 && n <= 5 ? n : null;

const returnsOf = (v: unknown): Returns => (v === "margin" ? "margin" : "win_loss");

/** Players a scenario declares: `parameters.players`, else the seat list, else the default 4. */
function scenarioPlayers(doc: Record<string, unknown>): number | null {
  const params = isObj(doc.parameters) ? doc.parameters : {};
  if (params.players !== undefined) return seatCount(params.players);
  if (Array.isArray(doc.seats)) return seatCount(doc.seats.length);
  if (isObj(doc.seats)) return seatCount(Object.keys(doc.seats).length);
  return 4;
}

export function parseStartDoc(text: string): StartDoc {
  if (text.trim() === "") return { kind: "empty" };
  let doc: unknown;
  try {
    doc = JSON.parse(text);
  } catch (e) {
    return { kind: "error", message: `Not valid JSON: ${e instanceof Error ? e.message : String(e)}` };
  }
  if (!isObj(doc)) return { kind: "error", message: "Expected a JSON object." };
  if ("format" in doc) {
    const params = isObj(doc.parameters) ? doc.parameters : {};
    return {
      kind: "scenario",
      doc,
      players: scenarioPlayers(doc),
      returns: returnsOf(params.returns),
      description: typeof doc.description === "string" ? doc.description : "",
      phase: typeof doc.phase === "string" ? doc.phase : null,
    };
  }
  if (Array.isArray(doc.history) && Array.isArray(doc.seats)) {
    const params = isObj(doc.params) ? doc.params : {};
    const seats = doc.seats.filter((s): s is string => typeof s === "string");
    return {
      kind: "replay",
      doc,
      players: seatCount(doc.seats.length),
      returns: returnsOf(params.returns),
      actions: doc.history.length,
      seats,
      fromScenario: doc.scenario != null,
    };
  }
  return {
    kind: "error",
    message: 'Not a scenario (it needs "format": 1) or an exported replay (it needs "seats" and "history").',
  };
}
