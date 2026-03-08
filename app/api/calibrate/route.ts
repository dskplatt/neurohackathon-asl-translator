import { NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const letters: string[] = body.letters || [];

    const lettersArg = letters.join(',').toLowerCase();
    const scriptPath = path.join(process.cwd(), 'scripts', 'collect_calibration_data.py');

    const encoder = new TextEncoder();

    const stream = new ReadableStream({
      start(controller) {
        // Use -u for unbuffered Python stdout so we get real-time lines
        const pythonProcess = spawn('python', [
          '-u', 
          scriptPath,
          '--samples-per-letter', '5',
          ...(lettersArg ? ['--letters', lettersArg] : [])
        ]);

        pythonProcess.stdout.on('data', (data) => {
          const text = data.toString();
          console.log(`Calibration Output: ${text}`);
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'stdout', text })}\n\n`));
        });

        pythonProcess.stderr.on('data', (data) => {
          const text = data.toString();
          console.error(`Calibration Error: ${text}`);
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'stderr', text })}\n\n`));
        });

        pythonProcess.on('close', (code) => {
          console.log(`Calibration script exited with code ${code}`);
          
          if (code === 0) {
            // Chain to training script
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'training_start' })}\n\n`));
            const trainScriptPath = path.join(process.cwd(), 'scripts', 'train_personal_model.py');
            const trainProcess = spawn('python', ['-u', trainScriptPath]);
            
            trainProcess.stdout.on('data', (data) => {
              const text = data.toString();
              console.log(`Train Output: ${text}`);
              controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'stdout', text })}\n\n`));
            });

            trainProcess.stderr.on('data', (data) => {
              const text = data.toString();
              console.error(`Train Error: ${text}`);
              controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'stderr', text })}\n\n`));
            });

            trainProcess.on('close', (trainCode) => {
              console.log(`Train script exited with code ${trainCode}`);
              controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'done', code: trainCode })}\n\n`));
              controller.close();
            });

            req.signal.addEventListener('abort', () => {
              console.log('Client aborted connection, killing train process');
              trainProcess.kill();
            });
          } else {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'done', code })}\n\n`));
            controller.close();
          }
        });

        req.signal.addEventListener('abort', () => {
          console.log('Client aborted connection, killing python process');
          pythonProcess.kill();
        });
      }
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    });
  } catch (error) {
    console.error('Error starting calibration:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to start calibration process' },
      { status: 500 }
    );
  }
}
