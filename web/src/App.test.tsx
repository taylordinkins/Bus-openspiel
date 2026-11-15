import { render, screen } from "@testing-library/react";
import App from "./App";

describe("App", () => {
  beforeEach(() => {
    window.location.hash = "";
  });

  it("renders the heading and lobby", () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: "Bus" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Create game" })).toBeInTheDocument();
  });
});
