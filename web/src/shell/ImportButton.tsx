import { useState, type ChangeEvent } from "react";
import { api } from "../api/client";
import { IconUpload } from "../theme/icons";
import { errorText } from "./useGame";

interface Props {
  onImported: (gameId: string) => void;
  /** Icon-only look for the game toolbar (the label stays "Import replay" for assistive tech). */
  compact?: boolean;
}

/** File input that posts a replay export to /api/games/import, styled as a button. */
export default function ImportButton({ onImported, compact = false }: Props) {
  const [error, setError] = useState<string | null>(null);

  async function pick(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setError(null);
    try {
      const body = JSON.parse(await file.text());
      const { data, error: err } = await api.POST("/api/games/import", { body });
      if (data) onImported(data.id);
      else setError(errorText(err));
    } catch (ex) {
      setError(ex instanceof Error ? ex.message : "Import failed");
    }
  }

  return (
    <span className={`import${compact ? " import-compact" : ""}`}>
      <label className={`btn import-btn${compact ? " btn-sm btn-ghost btn-icon" : ""}`} title="Import replay">
        <IconUpload />
        <span className={compact ? "sr-only" : undefined}>Import replay</span>{" "}
        <input
          className="sr-only"
          type="file"
          accept="application/json,.json"
          onChange={(e) => void pick(e)}
        />
      </label>
      {error && (
        <p role="alert" className="alert import-error">
          Import error: {error}
        </p>
      )}
    </span>
  );
}
