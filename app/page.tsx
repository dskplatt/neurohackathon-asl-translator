import Link from "next/link";

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
  "(": "M 52 4 L 46 8 L 44 14 L 44 20 L 46 26 L 52 30 L 52 30 L 52 30 L 52 30 L 52 30 L 52 30 L 52 30 L 52 30 L 52 30 L 52 30 L 52 30 L 52 30 L 52 30",
  ")": "M 48 4 L 54 8 L 56 14 L 56 20 L 54 26 L 48 30 L 48 30 L 48 30 L 48 30 L 48 30 L 48 30 L 48 30 L 48 30 L 48 30 L 48 30 L 48 30 L 48 30 L 48 30",
  "Θ": "M 50 4 L 44 8 L 44 24 L 50 28 L 56 24 L 56 8 L 50 4 L 50 4 M 44 16 L 56 16 L 56 16 L 56 16 L 56 16 L 56 16 L 56 16 L 56 16 L 56 16 L 56 16",
};

function SvgTitle({ text }: { text: string }) {
  const X_OFFSET = 24;
  const letters = text.toUpperCase().split('');
  
  // Calculate total width needed based on character count and offset
  const totalWidth = letters.length > 0 ? (letters.length - 1) * X_OFFSET + 20 : 0;
  
  return (
    <svg 
      viewBox={`0 0 ${totalWidth} 40`} 
      className="w-full max-w-[95vw] md:max-w-[700px] mx-auto h-24 md:h-36 mb-8"
    >
      {letters.map((char, i) => {
        if (char === ' ') return null;
        const pathData = GLYPHS[char];
        if (!pathData) return null;
        
        return (
          <path
            key={i}
            d={pathData}
            stroke="#e7e5e4"
            strokeWidth="2.5"
            fill="none"
            strokeLinecap="round"
            strokeLinejoin="round"
            // The letters in GLYPHS are drawn starting at X=42.
            // By translating by (i * X_OFFSET - 40), the first letter starts at X=2.
            style={{ transform: `translate(${i * X_OFFSET - 40}px, 4px)` }}
          />
        );
      })}
    </svg>
  );
}

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-black text-stone-200 flex flex-col font-[family-name:var(--font-montserrat)] relative overflow-hidden">
      {/* Main Content */}
      <main className="flex-1 flex flex-col items-center justify-center relative z-10 px-6 text-center">
        
        {/* Title Section Using the Custom SVG Font */}
        <SvgTitle text="cosign" />
        
        <p className="text-xl md:text-2xl text-gray-400 max-w-2xl mb-12 font-light tracking-wide">
          Real-Time ASL Transcription
        </p>

        {/* Action Tabs / Buttons */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 w-full max-w-md">
          <Link
            href="/translator"
            className="w-full sm:w-auto px-8 py-3 bg-stone-200 text-black text-lg font-semibold rounded-full hover:bg-stone-300 transition-all duration-300 transform hover:scale-105 flex items-center justify-center gap-2 group shadow-[0_0_20px_rgba(231,229,228,0.2)]"
          >
            Transcribe
          </Link>
        </div>
      </main>

      {/* Footer */}
      <footer className="relative z-10 py-6 text-center text-sm text-gray-600">
        <p>Built for the Neurohackathon</p>
      </footer>
    </div>
  );
}
