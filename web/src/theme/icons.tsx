import type { ReactNode, SVGProps } from "react";

/** One line-icon set for the app chrome: 20 px box, 2 px round ink strokes (UI_DESIGN §3a). */
function Svg({ children, ...rest }: SVGProps<SVGSVGElement> & { children: ReactNode }) {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      className="icon"
      {...rest}
    >
      {children}
    </svg>
  );
}

export const IconUndo = () => (
  <Svg>
    <path d="M7 4 3 8l4 4" />
    <path d="M3 8h9a5 5 0 0 1 0 10H8" />
  </Svg>
);
export const IconDownload = () => (
  <Svg>
    <path d="M10 3v10M6 9l4 4 4-4" />
    <path d="M3 15v2h14v-2" />
  </Svg>
);
export const IconUpload = () => (
  <Svg>
    <path d="M10 13V3M6 7l4-4 4 4" />
    <path d="M3 15v2h14v-2" />
  </Svg>
);
export const IconFlag = () => (
  <Svg>
    <path d="M5 18V3" />
    <path d="M5 3h10l-2.5 4L15 11H5" />
  </Svg>
);
export const IconArrowLeft = () => (
  <Svg>
    <path d="M16 10H4M9 5l-5 5 5 5" />
  </Svg>
);
export const IconPlus = () => (
  <Svg>
    <path d="M10 4v12M4 10h12" />
  </Svg>
);
export const IconMinus = () => (
  <Svg>
    <path d="M4 10h12" />
  </Svg>
);
export const IconFit = () => (
  <Svg>
    <path d="M3 7V3h4M13 3h4v4M17 13v4h-4M7 17H3v-4" />
  </Svg>
);
export const IconRewind = () => (
  <Svg>
    <path d="M9 5 3 10l6 5zM17 5l-6 5 6 5z" />
  </Svg>
);
export const IconList = () => (
  <Svg>
    <path d="M7 5h10M7 10h10M7 15h10M3 5h.01M3 10h.01M3 15h.01" />
  </Svg>
);
/** Settings: three slider rails with knobs. */
export const IconSliders = () => (
  <Svg>
    <path d="M3 5h7M14 5h3M3 10h2M9 10h8M3 15h9M16 15h1" />
    <circle cx="12" cy="5" r="2" />
    <circle cx="7" cy="10" r="2" />
    <circle cx="14" cy="15" r="2" />
  </Svg>
);
export const IconSkip = () => (
  <Svg>
    <path d="M3 5l6 5-6 5zM10 5l6 5-6 5z" />
    <path d="M18 5v10" />
  </Svg>
);

/** The Bus wordmark after the map's logo: B red, U yellow, S blue, each with an ink keyline. */
export function Wordmark({ height = 40 }: { height?: number }) {
  return (
    <svg
      className="wordmark"
      height={height}
      viewBox="0 0 132 64"
      aria-hidden="true"
      focusable="false"
    >
      {/* route stripes behind the letters, like lines on the map */}
      <path d="M2 58 L130 58" stroke="var(--logo-red)" strokeWidth="4" />
      <path d="M2 52 L130 52" stroke="var(--logo-yellow)" strokeWidth="4" />
      <g
        fontFamily="var(--font-display)"
        fontWeight={800}
        fontSize="58"
        stroke="var(--ink)"
        strokeWidth="3"
        paintOrder="stroke"
        strokeLinejoin="round"
      >
        <text x="2" y="48" fill="var(--logo-red)">
          B
        </text>
        <text x="40" y="48" fill="var(--logo-yellow)">
          U
        </text>
        <text x="80" y="48" fill="var(--logo-blue)">
          S
        </text>
      </g>
    </svg>
  );
}
