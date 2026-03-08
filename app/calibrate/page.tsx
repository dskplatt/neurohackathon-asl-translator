"use client";

import Link from "next/link";
import Image from "next/image";
import { useState } from "react";

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

export default function CalibrateSessionPage() {
  const selectedLetters = ALPHABET;
  
  // Format the display string
  const displayLetters = "All Letters";

  const [isCalibrating, setIsCalibrating] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [currentLetterIndex, setCurrentLetterIndex] = useState(0);
  const [currentRound, setCurrentRound] = useState(0); // 0-4
  const [completedLetters, setCompletedLetters] = useState<Set<string>>(new Set());
  const [errorMessage, setErrorMessage] = useState("");
  const [trainingLog, setTrainingLog] = useState<string>("");

  const handleStartCalibration = async () => {
    setIsCalibrating(true);
    setIsTraining(false);
    setCurrentLetterIndex(0);
    setCurrentRound(0);
    setCompletedLetters(new Set());
    setErrorMessage("");
    setTrainingLog("");

    try {
      const res = await fetch("/api/calibrate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ letters: selectedLetters, samplesPerLetter: 5 })
      });

      if (!res.ok || !res.body) {
        console.error("Failed to start calibration");
        setIsCalibrating(false);
        setErrorMessage("ERROR: Could not connect to Myo armband. Please check connection.");
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      let currentLetIdx = 0;
      let curRound = 0;
      let isModelTraining = false;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'training_start') {
                 isModelTraining = true;
                 setIsTraining(true);
                 setTrainingLog("Starting personal model training...\n");
              } else if (data.type === 'stdout' && isModelTraining) {
                 const text = data.text;
                 // Keep the last few lines of training log
                 setTrainingLog(prev => {
                    const newLog = prev + text;
                    const logLines = newLog.split('\n');
                    return logLines.slice(Math.max(logLines.length - 10, 0)).join('\n');
                 });
              } else if (data.type === 'stdout' && !isModelTraining) {
                const text = data.text;

                // Handle Error
                if (text.includes('ERROR: Could not connect to Myo armband')) {
                   setErrorMessage("ERROR: Could not connect to Myo armband. Please check connection.");
                   setIsCalibrating(false);
                }
                
                // Handle Skipping
                const skipMatch = text.match(/\[([A-Z])\] Already has \d+ samples — skipping/);
                if (skipMatch) {
                   const letter = skipMatch[1];
                   setCompletedLetters(prev => new Set(prev).add(letter));
                   const idx = selectedLetters.indexOf(letter);
                   if (idx >= 0 && idx >= currentLetIdx) {
                      currentLetIdx = idx + 1;
                      setCurrentLetterIndex(currentLetIdx);
                   }
                }

                // Handle waiting for sample
                const waitMatch = text.match(/\[([A-Z])\] Waiting for sample \((\d+)\/\d+\)/);
                if (waitMatch) {
                   const letter = waitMatch[1];
                   const roundNum = parseInt(waitMatch[2], 10);
                   const idx = selectedLetters.indexOf(letter);
                   if (idx >= 0) {
                      currentLetIdx = idx;
                      setCurrentLetterIndex(idx);
                   }
                   curRound = roundNum - 1;
                   setCurrentRound(curRound);
                }

                // Handle Captured!
                if (text.includes('Captured!')) {
                   curRound += 1;
                   setCurrentRound(curRound);
                }

                // Handle done
                const doneMatch = text.match(/'([A-Z])' done/);
                if (doneMatch) {
                   const letter = doneMatch[1];
                   
                   // Ensure it flashes the full letter as green for a brief moment
                   setCurrentRound(5);
                   
                   setTimeout(() => {
                     setCompletedLetters(prev => new Set(prev).add(letter));
                     const idx = selectedLetters.indexOf(letter);
                     if (idx >= 0 && idx >= currentLetIdx) {
                        currentLetIdx = idx + 1;
                        setCurrentLetterIndex(currentLetIdx);
                        setCurrentRound(0);
                     }
                   }, 500); // 500ms delay so we can see the 5th dot and the letter turn green
                }
              } else if (data.type === 'stderr') {
                const text = data.text;
                if (text.includes('ERROR') || text.includes('Exception') || text.includes('Could not connect')) {
                   setErrorMessage("ERROR: Could not connect to Myo armband. Please check connection.");
                   setIsCalibrating(false);
                }
              } else if (data.type === 'done') {
                if (data.code !== 0 && data.code !== undefined && !isModelTraining) {
                  setErrorMessage("ERROR: Could not connect to Myo armband. Please check connection.");
                  setIsCalibrating(false);
                  setIsTraining(false);
                } else {
                  setTimeout(() => {
                    setIsCalibrating(false);
                    setIsTraining(false);
                  }, 5000);
                }
              }
            } catch (e) {
              // Ignore parse errors from partial chunks
            }
          }
        }
      }
    } catch (err) {
      console.error(err);
      setErrorMessage("ERROR: Could not connect to Myo armband. Please check connection.");
      setIsCalibrating(false);
    }
  };

  return (
    <div className="min-h-screen bg-black text-stone-200 flex flex-col font-[family-name:var(--font-montserrat)] relative overflow-hidden">
      <header className="w-full p-6 flex justify-between items-center z-10 relative">
        <Link 
          href="/translator" 
          className="text-gray-400 hover:text-stone-200 transition-colors flex items-center gap-2"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
          Back to Transcription
        </Link>
      </header>

      <main className="flex-1 flex flex-col items-center justify-center relative z-10 px-6 text-center pb-20">
        <h1 className="text-4xl md:text-5xl font-bold mb-4 tracking-wider drop-shadow-[0_0_12px_rgba(255,255,255,0.2)]">
          Calibration Session
        </h1>
        
        <p className="text-xl text-gray-400 max-w-2xl mb-2 font-light tracking-wide">
          Follow the instructions below to train the model.
        </p>

        <p className="text-lg text-stone-200 font-semibold mb-10 tracking-wide bg-stone-200/10 px-6 py-2 rounded-full border border-stone-200/20">
          Calibrating: <span className="text-stone-300">All Letters</span>
        </p>

        {errorMessage && (
          <div className="mb-6 text-red-600 font-semibold text-lg max-w-2xl">
            {errorMessage}
          </div>
        )}

        {/* Dynamic UI container */}
        <div className={`w-full max-w-5xl flex gap-8 transition-all duration-700 ease-in-out ${isCalibrating ? 'flex-row' : 'flex-col items-center'}`}>
          
          {/* ASL Image Container */}
          <div className={`relative flex flex-col items-center justify-center overflow-hidden p-4 transition-all duration-700
            ${isCalibrating ? 'w-1/2 aspect-[4/5]' : 'w-full max-w-3xl aspect-video mb-10'}
          `}>
            <Image 
              src="/asl_alphabet.png" 
              alt="ASL Alphabet Reference" 
              fill 
              className="object-contain z-10 p-4" 
              priority
            />
          </div>

          {/* Calibration Grid Container - Only shown during calibration */}
          <div className={`transition-all duration-700 overflow-hidden flex flex-col items-center justify-center
            ${isCalibrating ? 'w-1/2 opacity-100 scale-100' : 'w-0 opacity-0 scale-95 hidden'}
          `}>
            <div className="bg-stone-200/5 border border-stone-200/10 rounded-2xl p-6 shadow-[0_0_30px_rgba(0,0,0,0.5)] w-full">
              <h2 className="text-2xl font-semibold mb-6">Calibration Progress</h2>
              <div className="grid grid-cols-4 sm:grid-cols-5 md:grid-cols-6 gap-3">
                {selectedLetters.map((letter, index) => {
                  const isCompleted = completedLetters.has(letter);
                  const isCurrent = index === currentLetterIndex && isCalibrating;
                  
                  return (
                    <div key={letter} className="flex flex-col items-center gap-2">
                      {/* The main letter display */}
                      <div 
                        className={`w-full aspect-square flex items-center justify-center text-3xl font-bold rounded-xl transition-all duration-300
                          ${isCompleted ? 'text-green-600/80' : 
                            isCurrent && currentRound === 5 ? 'text-green-600/80 scale-110' :
                            isCurrent ? 'text-stone-200 scale-110' : 
                            'text-stone-600'
                          }
                        `}
                      >
                        {letter}
                      </div>

                      {/* 5 Dots Indicator */}
                      <div className="flex gap-1 justify-center">
                        {[0, 1, 2, 3, 4].map((dotIndex) => {
                          let dotColor = "bg-stone-700/50"; // default grey
                          
                          if (isCompleted) {
                            dotColor = "bg-stone-700/50";
                          } else if (isCurrent) {
                            // Light up green if this round is finished
                            if (dotIndex < currentRound) {
                              dotColor = "bg-green-600/80";
                            } 
                            // Current round being worked on
                            else if (dotIndex === currentRound && currentRound < 5) {
                              dotColor = "bg-green-400 animate-pulse";
                            }
                          }

                          return (
                            <div 
                              key={dotIndex} 
                              className={`w-2 h-2 rounded-full transition-all duration-300 ${dotColor}`}
                            />
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

        </div>

        {/* Action Buttons */}
        {!isCalibrating && !isTraining && (
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 w-full max-w-md mt-10">
            <button 
              onClick={handleStartCalibration}
              className="w-full sm:w-auto px-10 py-4 bg-stone-200 text-black text-lg font-bold rounded-full hover:bg-stone-300 transition-all duration-300 transform hover:scale-105 mt-auto"
            >
              Start Calibration
            </button>
          </div>
        )}
        
        {isCalibrating && currentLetterIndex >= selectedLetters.length && !isTraining && (
          <div className="mt-8 text-xl text-stone-400 font-bold animate-pulse">
            Calibration Complete! Waiting for training to start...
          </div>
        )}

        {isTraining && (
          <div className="mt-8 w-full max-w-3xl flex flex-col items-center">
            <h3 className="text-2xl text-green-600/80 font-bold mb-4 animate-pulse">Training Personal Model...</h3>
            <div className="bg-stone-900 border border-stone-800 rounded-lg p-4 font-mono text-sm text-stone-300 h-48 w-full overflow-y-auto whitespace-pre-wrap text-left shadow-inner">
              {trainingLog || "Initializing..."}
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="relative z-10 py-6 text-center text-sm text-gray-600">
        <p>Built for the Neurohackathon</p>
      </footer>
    </div>
  );
}
