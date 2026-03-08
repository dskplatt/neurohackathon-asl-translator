"use client";

export type SigningState = "waiting" | "reading" | "captured" | "resolved";

type SigningStatusProps = {
  state: SigningState;
  letter?: string;
};

const STATE_CONFIG: Record<
  SigningState,
  { dot: string; label: string }
> = {
  waiting: { dot: "#444444", label: "Waiting" },
  reading: { dot: "#00FF88", label: "Reading Sign" },
  captured: { dot: "#FFD700", label: "Captured" },
  resolved: { dot: "#00BFFF", label: "Word Resolved" },
};

export default function SigningStatus({ state, letter }: SigningStatusProps) {
  const { dot, label } = STATE_CONFIG[state];
  const displayLabel =
    state === "captured" ? `Captured: ${letter ?? ""}` : label;

  return (
    <div
      className="fixed top-4 left-1/2 -translate-x-1/2 z-50 flex items-center gap-2 rounded-full px-4 py-2"
      style={{ background: "rgba(0,0,0,0.6)", fontFamily: "'Space Mono', monospace" }}
    >
      <span
        className={state === "reading" ? "signing-status-pulse" : undefined}
        style={{
          display: "inline-block",
          width: 8,
          height: 8,
          borderRadius: "50%",
          backgroundColor: dot,
          transition: "background-color 200ms ease",
          flexShrink: 0,
        }}
      />
      <span
        className="text-sm text-stone-200 whitespace-nowrap"
        style={{ transition: "opacity 200ms ease" }}
      >
        {displayLabel}
      </span>

      {/* Scoped pulse keyframes — only active in "reading" state */}
      <style jsx>{`
        @keyframes signing-pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50%       { opacity: 0.5; transform: scale(0.85); }
        }
        .signing-status-pulse {
          animation: signing-pulse 1.2s ease-in-out infinite;
        }
      `}</style>
    </div>
  );
}
