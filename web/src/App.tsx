import { useEffect, useState } from "react";
import GameScreen from "./shell/GameScreen";
import ImportButton from "./shell/ImportButton";
import Lobby from "./shell/Lobby";
import { Wordmark } from "./theme/icons";
import "./shell/layout.css";

const parseHash = (): string | null => {
  const m = /^#\/game\/(.+)$/.exec(window.location.hash);
  return m?.[1] ? decodeURIComponent(m[1]) : null;
};

const goTo = (id: string) => {
  window.location.hash = `#/game/${encodeURIComponent(id)}`;
};

export default function App() {
  const [gameId, setGameId] = useState<string | null>(parseHash);

  useEffect(() => {
    const onHash = () => setGameId(parseHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  if (gameId) {
    return (
      <main className="app app-game">
        <h1 className="app-title">
          <Wordmark height={34} />
          <span className="sr-only">Bus</span>
        </h1>
        <GameScreen
          key={gameId}
          gameId={gameId}
          onExit={() => {
            window.location.hash = "";
          }}
          onImported={goTo}
        />
      </main>
    );
  }

  return (
    <main className="app app-lobby">
      <header className="lobby-hero">
        <h1 className="app-title">
          <Wordmark height={112} />
          <span className="sr-only">Bus</span>
        </h1>
        <div className="lobby-hero-text">
          <p className="eyebrow">Splotter&rsquo;s Bus · Complete Edition rules · reworked map</p>
          <p className="lobby-tagline">Build bus lines, carry the city&rsquo;s passengers, mind the clock.</p>
          <p className="lobby-sub">3 to 5 seats, any mix of people at this screen and bots.</p>
        </div>
      </header>
      <div className="lobby-grid">
        <Lobby onCreated={goTo} />
        <aside className="lobby-aside">
          <figure className="lobby-map" aria-hidden="true">
            <img src="/bus-map.png" alt="" width={1991} height={1991} decoding="async" />
          </figure>
          <section className="card lobby-card" aria-label="Load a replay">
            <h2 className="section-title">
              <span className="num-tile">2</span> Continue a game
            </h2>
            <p className="lobby-hint">Load a replay exported from an earlier game; play resumes where it stopped. To change its seats, use New game → Scenario.</p>
            <ImportButton onImported={goTo} />
          </section>
          <section className="card lobby-card lobby-tips" aria-label="How to play here">
            <h2 className="section-title">
              <span className="num-tile">3</span> At the table
            </h2>
            <ul>
              <li>
                <b>Click what glows.</b> Streets, sites, stations, action spaces and the clock are the controls.
              </li>
              <li>
                <b>Hover</b> any place or panel to see who is there and what it does.
              </li>
              <li>
                <b>Back</b> undoes your last decision; nothing in Bus is hidden or random.
              </li>
            </ul>
          </section>
        </aside>
      </div>
    </main>
  );
}
