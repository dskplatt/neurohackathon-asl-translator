import Link from "next/link";

interface Props {
  searchParams: { letters?: string };
}

export default function CalibrateSessionPage({ searchParams }: Props) {
  const lettersParam = searchParams.letters || "all";
  
  // Format the display string
  const displayLetters = lettersParam === "all" 
    ? "All Letters" 
    : lettersParam.split(",").join(", ");

  return (
    <div className="min-h-screen bg-black text-white flex flex-col font-[family-name:var(--font-montserrat)] relative overflow-hidden">
      <header className="w-full p-6 flex justify-between items-center z-10 relative">
        <Link 
          href="/calibrate" 
          className="text-gray-400 hover:text-white transition-colors flex items-center gap-2"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
          Back to Selection
        </Link>
      </header>

      <main className="flex-1 flex flex-col items-center justify-center relative z-10 px-6 text-center pb-20">
        <h1 className="text-4xl md:text-5xl font-bold mb-4 tracking-wider drop-shadow-[0_0_12px_rgba(255,255,255,0.2)]">
          Calibration Session
        </h1>
        
        <p className="text-xl text-gray-400 max-w-2xl mb-2 font-light tracking-wide">
          Follow the instructions below to train the model.
        </p>

        <p className="text-lg text-white font-semibold mb-10 tracking-wide bg-white/10 px-6 py-2 rounded-full border border-white/20">
          Calibrating: <span className="text-gray-300">{displayLetters}</span>
        </p>

        {/* Placeholder for the calibration UI */}
        <div className="w-full max-w-3xl aspect-video bg-white/5 border border-white/10 rounded-2xl flex flex-col items-center justify-center mb-10 shadow-[0_0_30px_rgba(0,0,0,0.5)] relative overflow-hidden">
          <svg className="w-16 h-16 text-gray-600 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
          </svg>
          <p className="text-gray-500 text-lg">Calibration Visualizer Placeholder</p>
          
          {/* Subtle scanning effect overlay for the placeholder */}
          <div className="absolute inset-0 bg-gradient-to-b from-transparent via-white/5 to-transparent w-full h-full animate-[scan_3s_ease-in-out_infinite]" />
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 w-full max-w-md">
          <button className="w-full sm:w-auto px-8 py-3 bg-white text-black text-base font-semibold rounded-full hover:bg-gray-200 transition-all duration-300 transform hover:scale-105 shadow-[0_0_40px_rgba(255,255,255,0.3)]">
            Start Calibration
          </button>
        </div>
      </main>

      {/* Footer */}
      <footer className="relative z-10 py-6 text-center text-sm text-gray-600">
        <p>Built for the Neurohackathon</p>
      </footer>

      {/* Add inline style for scan animation */}
      <style dangerouslySetInlineStyle={{
        __html: `
          @keyframes scan {
            0% { transform: translateY(-100%); }
            100% { transform: translateY(100%); }
          }
        `
      }} />
    </div>
  );
}
