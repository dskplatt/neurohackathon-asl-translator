"use client";

import Link from "next/link";
import { useState } from "react";
import { useRouter } from "next/navigation";

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

export default function CalibrateSelectionPage() {
  const router = useRouter();
  const [selectionMode, setSelectionMode] = useState<"all" | "custom">("all");
  const [selectedLetters, setSelectedLetters] = useState<Set<string>>(new Set());

  const toggleLetter = (letter: string) => {
    const newSelection = new Set(selectedLetters);
    if (newSelection.has(letter)) {
      newSelection.delete(letter);
    } else {
      newSelection.add(letter);
    }
    setSelectedLetters(newSelection);
  };

  const handleBegin = () => {
    if (selectionMode === "custom" && selectedLetters.size === 0) {
      alert("Please select at least one letter to calibrate.");
      return;
    }
    
    const lettersParam = selectionMode === "all" ? "all" : Array.from(selectedLetters).join(",");
    router.push(`/calibrate/session?letters=${encodeURIComponent(lettersParam)}`);
  };

  return (
    <div className="min-h-screen bg-black text-stone-200 flex flex-col font-[family-name:var(--font-montserrat)] relative overflow-hidden">
      <header className="w-full p-6 flex justify-between items-center z-10 relative">
        <Link 
          href="/" 
          className="text-gray-400 hover:text-stone-200 transition-colors flex items-center gap-2"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
          Back
        </Link>
      </header>

      <main className="flex-1 flex flex-col items-center relative z-10 px-6 text-center pb-20 max-w-4xl mx-auto w-full">
        <h1 className="text-4xl md:text-5xl font-bold mb-6 tracking-wider drop-shadow-[0_0_12px_rgba(255,255,255,0.2)] mt-8">
          Calibration Setup
        </h1>
        
        <p className="text-xl text-gray-400 max-w-2xl mb-10 font-light tracking-wide">
          Select which letters you would like to calibrate.
        </p>

        {/* Selection Mode Toggle */}
        <div className="flex bg-stone-200/10 p-1 rounded-full mb-10 border border-stone-200/20">
          <button
            onClick={() => setSelectionMode("all")}
            className={`px-6 py-2 rounded-full font-medium transition-all duration-300 ${
              selectionMode === "all" 
                ? "bg-stone-200 text-black" 
                : "text-gray-400 hover:text-stone-200"
            }`}
          >
            All Letters
          </button>
          <button
            onClick={() => setSelectionMode("custom")}
            className={`px-6 py-2 rounded-full font-medium transition-all duration-300 ${
              selectionMode === "custom" 
                ? "bg-stone-200 text-black" 
                : "text-gray-400 hover:text-stone-200"
            }`}
          >
            Custom Selection
          </button>
        </div>

        {/* Custom Letters Grid */}
        <div className={`w-full transition-all duration-500 overflow-hidden ${selectionMode === "custom" ? "max-h-[800px] opacity-100 mb-10" : "max-h-0 opacity-0 mb-0"}`}>
          <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-7 gap-3 md:gap-4 w-full p-4 border border-stone-200/10 rounded-2xl bg-stone-200/5 shadow-[0_0_30px_rgba(0,0,0,0.5)]">
            {ALPHABET.map((letter) => {
              const isSelected = selectedLetters.has(letter);
              return (
                <button
                  key={letter}
                  onClick={() => toggleLetter(letter)}
                  className={`aspect-square flex items-center justify-center text-xl font-semibold rounded-xl transition-all duration-200 border ${
                    isSelected
                      ? "bg-stone-200 text-black border-stone-200 transform scale-105"
                      : "bg-transparent text-gray-400 border-stone-200/20 hover:border-stone-200/50 hover:text-stone-200 hover:bg-stone-200/5"
                  }`}
                >
                  {letter}
                </button>
              );
            })}
          </div>
          <div className="flex justify-end mt-4">
            <button 
              onClick={() => setSelectedLetters(new Set())}
              className="text-sm text-gray-400 hover:text-stone-200 transition-colors"
            >
              Clear Selection
            </button>
          </div>
        </div>

        {/* Action Button */}
        <button 
          onClick={handleBegin}
          className="w-full sm:w-auto px-10 py-4 bg-stone-200 text-black text-lg font-bold rounded-full hover:bg-stone-300 transition-all duration-300 transform hover:scale-105 mt-auto"
        >
          Begin Calibrating
        </button>
      </main>

      {/* Footer */}
      <footer className="relative z-10 py-6 text-center text-sm text-gray-600">
        <p>Built for the Neurohackathon</p>
      </footer>
    </div>
  );
}
