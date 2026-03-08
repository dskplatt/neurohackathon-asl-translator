"use client";

import { useEffect, useRef, useState } from "react";

const WAVE_XS = Array.from({ length: 18 }, (_, i) => 40 + i * (30 / 17));

const GLYPHS: Record<string, string> = {
  A: "M 42 28 L 50 4 L 58 28 L 54 18 L 46 18 L 46 18 L 46 18 L 46 18 L 46 18 L 46 18 L 46 18 L 46 18 L 46 18 L 46 18 L 46 18 L 46 18 L 46 18 L 46 18",
  B: "M 42 4 L 42 28 L 50 28 L 55 25 L 55 20 L 50 16 L 55 12 L 55 7 L 50 4 L 42 4 L 42 4 L 42 4 L 42 4 L 42 4 L 42 4 L 42 4 L 42 4 L 42 4",
  C: "M 56 8 L 53 4 L 47 4 L 44 8 L 42 14 L 42 18 L 44 24 L 47 28 L 53 28 L 56 24 L 56 24 L 56 24 L 56 24 L 56 24 L 56 24 L 56 24 L 56 24 L 56 24",
  D: "M 42 4 L 42 28 L 49 28 L 54 24 L 56 20 L 56 16 L 56 12 L 54 8 L 49 4 L 42 4 L 42 4 L 42 4 L 42 4 L 42 4 L 42 4 L 42 4 L 42 4 L 42 4",
  E: "M 56 4 L 42 4 L 42 28 L 56 28 L 56 28 L 42 28 L 42 16 L 53 16 L 53 16 L 53 16 L 53 16 L 53 16 L 53 16 L 53 16 L 53 16 L 53 16 L 53 16 L 53 16",
  F: "M 56 4 L 42 4 L 42 28 L 42 16 L 53 16 L 53 16 L 53 16 L 53 16 L 53 16 L 53 16 L 53 16 L 53 16 L 53 16 L 53 16 L 53 16 L 53 16 L 53 16 L 53 16",
  G: "M 56 8 L 53 4 L 47 4 L 44 8 L 42 14 L 42 18 L 44 24 L 47 28 L 53 28 L 56 24 L 56 16 L 51 16 L 51 16 L 51 16 L 51 16 L 51 16 L 51 16 L 51 16",
  H: "M 42 4 L 42 28 L 42 16 L 58 16 L 58 4 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28",
  I: "M 44 4 L 56 4 L 50 4 L 50 28 L 44 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28",
  J: "M 44 4 L 56 4 L 50 4 L 50 24 L 48 28 L 46 28 L 44 25 L 44 25 L 44 25 L 44 25 L 44 25 L 44 25 L 44 25 L 44 25 L 44 25 L 44 25 L 44 25 L 44 25",
  K: "M 42 4 L 42 28 L 42 16 L 56 4 L 42 16 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28",
  L: "M 42 4 L 42 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28",
  M: "M 42 28 L 42 4 L 50 16 L 58 4 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28",
  N: "M 42 28 L 42 4 L 58 28 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4",
  O: "M 47 4 L 44 8 L 42 14 L 42 18 L 44 24 L 47 28 L 53 28 L 56 24 L 58 18 L 58 14 L 56 8 L 53 4 L 47 4 L 47 4 L 47 4 L 47 4 L 47 4 L 47 4",
  P: "M 42 28 L 42 4 L 50 4 L 55 8 L 55 14 L 50 16 L 42 16 L 42 16 L 42 16 L 42 16 L 42 16 L 42 16 L 42 16 L 42 16 L 42 16 L 42 16 L 42 16 L 42 16",
  Q: "M 47 4 L 44 8 L 42 14 L 42 18 L 44 24 L 47 28 L 53 28 L 56 24 L 58 18 L 58 14 L 56 8 L 53 4 L 47 4 L 53 24 L 56 30 L 56 30 L 56 30 L 56 30",
  R: "M 42 28 L 42 4 L 50 4 L 55 8 L 55 14 L 50 16 L 42 16 L 50 16 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28 L 56 28",
  S: "M 56 8 L 53 4 L 47 4 L 44 8 L 44 13 L 47 16 L 53 16 L 56 19 L 56 24 L 53 28 L 47 28 L 44 24 L 44 24 L 44 24 L 44 24 L 44 24 L 44 24 L 44 24",
  T: "M 42 4 L 58 4 L 50 4 L 50 28 L 50 28 L 50 28 L 50 28 L 50 28 L 50 28 L 50 28 L 50 28 L 50 28 L 50 28 L 50 28 L 50 28 L 50 28 L 50 28 L 50 28",
  U: "M 42 4 L 42 22 L 44 27 L 47 30 L 53 30 L 56 27 L 58 22 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4",
  V: "M 42 4 L 50 28 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4",
  W: "M 42 4 L 46 28 L 50 16 L 54 28 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4",
  X: "M 42 4 L 58 28 L 50 16 L 42 28 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4 L 58 4",
  Y: "M 42 4 L 50 16 L 58 4 L 50 16 L 50 28 L 50 28 L 50 28 L 50 28 L 50 28 L 50 28 L 50 28 L 50 28 L 50 28 L 50 28 L 50 28 L 50 28 L 50 28 L 50 28",
  Z: "M 42 4 L 58 4 L 42 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28 L 58 28",
};

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function parseCoords(d: string): number[] {
  return (d.match(/-?\d+\.?\d*/g) ?? []).map(Number);
}

function applyCoords(template: string, coords: number[]): string {
  let i = 0;
  return template.replace(/-?\d+\.?\d*/g, () => String(Math.round(coords[i++] * 10) / 10));
}

const X_OFFSET = 30;
const Y_OFFSET = 45;

function getCursorPos(index: number, maxCharsPerLine: number) {
  const line = Math.floor(index / maxCharsPerLine);
  const col = index % maxCharsPerLine;
  return { x: col * X_OFFSET, y: line * Y_OFFSET };
}

function makeWaveSegment(slotIndex: number, time: number, active: boolean): string {
  const t = time * 0.002;
  // Amplitude increases drastically when "signing" to match EMG bursts
  const baseAmp = active ? 10 : 1.5;
  
  const points = WAVE_XS.map((x) => {
    const screenX = x + slotIndex * X_OFFSET;
    
    // Complex signal mimicking muscle activation
    const carrier = 
      0.7 * Math.sin(screenX * 0.1 - t * 2) +
      0.4 * Math.sin(screenX * 0.2 + t * 3) +
      0.3 * Math.sin(screenX * 0.4 - t * 5);
      
    // Envelope for a traveling pulse
    const travelingX = screenX - t * 100;
    const envelope = Math.pow(Math.sin(travelingX * 0.02), 4) + 0.1;
    
    const y = 16 + baseAmp * carrier * envelope;
    return `${x.toFixed(2)} ${y.toFixed(2)}`;
  });
  return "M " + points.join(" L ");
}

function WaveSegment({ 
  slotIndex, pos, isActiveRef, opacity 
}: { 
  slotIndex: number, pos: {x: number, y: number}, isActiveRef: React.MutableRefObject<boolean>, opacity: number 
}) {
  const pathRef = useRef<SVGPathElement>(null);

  useEffect(() => {
    let frame: number;
    function animate() {
      if (pathRef.current) {
        pathRef.current.setAttribute("d", makeWaveSegment(slotIndex, performance.now(), isActiveRef.current));
      }
      frame = requestAnimationFrame(animate);
    }
    frame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frame);
  }, [slotIndex, isActiveRef]);

  return (
    <path 
      ref={pathRef}
      stroke="white"
      strokeWidth="1.5"
      fill="none"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ transform: `translate(${pos.x}px, ${pos.y}px)`, opacity }}
    />
  );
}

function LetterMorph({ 
  targetD, pos, slotIndex, isActiveRef 
}: { 
  targetD: string, pos: {x: number, y: number}, slotIndex: number, isActiveRef: React.MutableRefObject<boolean>
}) {
  const pathRef = useRef<SVGPathElement>(null);

  useEffect(() => {
    const start = performance.now();
    const duration = 400; // Animation duration in ms
    
    // Freeze the exact wave pattern at the time the letter is placed
    const initialD = makeWaveSegment(slotIndex, start, isActiveRef.current);
    const from = parseCoords(initialD);
    const to = parseCoords(targetD);
    let frame: number;

    function animate(now: number) {
      const t = Math.min((now - start) / duration, 1);
      const eased = easeInOutCubic(t);
      const current = from.map((f, i) => f + (to[i] - f) * eased);
      
      if (pathRef.current) {
        pathRef.current.setAttribute("d", applyCoords(initialD, current));
      }

      if (t < 1) {
        frame = requestAnimationFrame(animate);
      } else {
        if (pathRef.current) {
          pathRef.current.setAttribute("d", targetD);
        }
      }
    }
    frame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frame);
  }, [slotIndex, targetD, isActiveRef]);

  return (
    <path 
      ref={pathRef}
      stroke="white"
      strokeWidth="1.5"
      fill="none"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ transform: `translate(${pos.x}px, ${pos.y}px)` }}
    />
  );
}

type LetterData = { id: string; char: string; slotIndex: number; targetD: string; };

interface Props {
  wsUrl: string;
}

export default function ASLTranslator({ wsUrl }: Props) {
  const isActiveRef = useRef(false);
  const wsRef = useRef<WebSocket | null>(null);

  const [letters, setLetters] = useState<LetterData[]>([]);
  const [pendingChars, setPendingChars] = useState<string[]>([]);
  const [isMorphing, setIsMorphing] = useState(false);
  
  // Default to 20 until client renders
  const [maxCharsPerLine, setMaxCharsPerLine] = useState(20);

  useEffect(() => {
    function handleResize() {
      // With paddingLeft: 15vw and paddingRight: 15vw, available width is 70vw.
      const availableWidth = window.innerWidth * 0.70;
      const charWidth = X_OFFSET * 1.2; // Adjusted for scale(1.2)
      setMaxCharsPerLine(Math.max(10, Math.floor(availableWidth / charWidth)));
    }
    
    // Set initial
    handleResize();
    
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    let ws: WebSocket;
    let reconnectTimer: ReturnType<typeof setTimeout>;

    function connect() {
      try {
        ws = new WebSocket(wsUrl);
        wsRef.current = ws;
        
        ws.onmessage = (evt) => {
          try {
            const data = JSON.parse(evt.data as string);
            if (data.type === "signing_active") {
              isActiveRef.current = data.active;
            } else if (data.type === "word_resolved" && data.primary) {
              const wordChars = data.primary.toUpperCase().split("").filter((c: string) => GLYPHS[c]);
              if (wordChars.length > 0) {
                setPendingChars((prev) => [...prev, ...wordChars, " "]);
              }
            }
          } catch {
            // ignore
          }
        };
        
        ws.onclose = () => {
          reconnectTimer = setTimeout(connect, 2000);
        };
        
        ws.onerror = () => {
          ws.close();
        };
      } catch {
        reconnectTimer = setTimeout(connect, 2000);
      }
    }

    connect();
    return () => {
      clearTimeout(reconnectTimer);
      wsRef.current?.close();
    };
  }, [wsUrl]);

  // Sequentially process morph queue
  useEffect(() => {
    if (pendingChars.length === 0) return;
    if (isMorphing) return;

    setIsMorphing(true);
    const char = pendingChars[0];
    const slotIndex = letters.length;

    if (char === " ") {
      setLetters((prev) => [...prev, { id: Math.random().toString(36).slice(2), char, slotIndex, targetD: "" }]);
      setPendingChars((prev) => prev.slice(1));
      setTimeout(() => setIsMorphing(false), 200); // Quick pause for spaces
      return;
    }

    const targetD = GLYPHS[char];
    if (!targetD) {
      setPendingChars((prev) => prev.slice(1));
      setIsMorphing(false);
      return;
    }

    setLetters((prev) => [...prev, { id: Math.random().toString(36).slice(2), char, slotIndex, targetD }]);
    setPendingChars((prev) => prev.slice(1));
    
    // Wait for the animation to complete before starting the next one
    setTimeout(() => {
      setIsMorphing(false);
    }, 450); // 400ms duration + 50ms pause between letters
  }, [pendingChars, isMorphing, letters.length]);

  const currentLine = Math.floor(letters.length / maxCharsPerLine);
  const remainingInLine = (currentLine + 1) * maxCharsPerLine - letters.length;
  const waveSlots = Array.from({ length: remainingInLine }, (_, i) => letters.length + i);

  return (
    <div className="w-screen h-screen bg-black flex items-center overflow-hidden" style={{ paddingLeft: "15vw" }}>
      <svg
        style={{
          width: "1px",
          height: "1px",
          overflow: "visible",
          // We translate the entire SVG *up* by exactly how many lines have passed, 
          // keeping the current drawing line perfectly vertically centered.
          transform: `scale(1.2) translate(-40px, ${-currentLine * Y_OFFSET}px)`,
          transition: "transform 0.5s ease-in-out",
        }}
      >
        {/* Render fully printed text */}
        {letters.map((l) => {
          if (l.char === " ") return null;
          const pos = getCursorPos(l.slotIndex, maxCharsPerLine);
          return <LetterMorph key={l.id} targetD={l.targetD} pos={pos} slotIndex={l.slotIndex} isActiveRef={isActiveRef} />;
        })}
        
        {/* Render the continuous wave ahead of the text */}
        {waveSlots.map((slotIndex) => {
          const pos = getCursorPos(slotIndex, maxCharsPerLine);
          const col = slotIndex % maxCharsPerLine;
          const fadeStart = maxCharsPerLine - 6;
          const opacity = col > fadeStart ? Math.max(0, 1 - (col - fadeStart) / 6) : 1;
          
          return (
            <WaveSegment 
              key={`wave-${slotIndex}`} 
              slotIndex={slotIndex} 
              pos={pos} 
              isActiveRef={isActiveRef} 
              opacity={opacity} 
            />
          );
        })}
      </svg>
    </div>
  );
}
