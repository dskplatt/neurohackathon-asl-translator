const { WebSocketServer } = require("ws");

const wss = new WebSocketServer({ port: 8000, path: "/ws" });

wss.on("error", (err) => {
  if (err.code === "EADDRINUSE") {
    console.error("Port 8000 is already in use. Is the real backend or another mock already running?");
    console.error("Stop the existing process on port 8000 and retry: lsof -ti:8000 | xargs kill");
    process.exit(1);
  }
  throw err;
});

console.log("Mock ASL server running on ws://localhost:8000/ws");

wss.on("connection", (ws) => {
  console.log("Client connected");

  const sequence = [
    { ms: 1000,  msg: { type: "signing_active", active: true } },
    { ms: 2500,  msg: { type: "letter_captured", letter_index: 0, confidence: 0.91 } },
    { ms: 3500,  msg: { type: "letter_captured", letter_index: 1, confidence: 0.87 } },
    { ms: 4500,  msg: { type: "letter_captured", letter_index: 2, confidence: 0.95 } },
    { ms: 5500,  msg: { type: "letter_captured", letter_index: 3, confidence: 0.78 } },
    { ms: 6500,  msg: { type: "letter_captured", letter_index: 4, confidence: 0.88 } },
    {
      ms: 7500,
      msg: {
        type: "word_resolved",
        primary: "HELLO",
        score: 0.847,
        alternates: ["HELLS", "JELLO"],
      },
    },
    { ms: 13000, msg: { type: "signing_active", active: true } },
    { ms: 14000, msg: { type: "letter_captured", letter_index: 0, confidence: 0.95 } },
    { ms: 15000, msg: { type: "letter_captured", letter_index: 1, confidence: 0.82 } },
    { ms: 16000, msg: { type: "letter_captured", letter_index: 2, confidence: 0.90 } },
    {
      ms: 17500,
      msg: {
        type: "word_resolved",
        primary: "WORLD",
        score: 0.911,
        alternates: ["WORDY", "WONLD"],
      },
    },
  ];

  const timers = sequence.map(({ ms, msg }) =>
    setTimeout(() => {
      if (ws.readyState === ws.OPEN) {
        ws.send(JSON.stringify(msg));
        console.log(`→ ${ms}ms`, msg.type, msg.primary ?? "");
      }
    }, ms)
  );

  ws.on("close", () => {
    console.log("Client disconnected");
    timers.forEach(clearTimeout);
  });
});
