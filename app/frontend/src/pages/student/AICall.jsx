import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useAuthStore } from '../../store/authStore';
import './AICall.css';

const STATUS = {
    IDLE: 'idle',
    CONNECTING: 'connecting',  // WS opened, waiting for gRPC ready signal
    CALLING: 'calling',        // "Calling AI Tutor…" ring animation
    ACTIVE: 'active',
    ERROR: 'error',
};

// Which side is currently "talking"
const TURN = {
    NONE: 'none',
    AI: 'ai',
    STUDENT: 'student',
};

export default function AICall() {
    const { lessonId } = useParams();
    const navigate = useNavigate();
    const { token } = useAuthStore();

    const [status, setStatus] = useState(STATUS.IDLE);
    const [duration, setDuration] = useState(0);
    const [isMuted, setIsMuted] = useState(false);
    const [errorMsg, setErrorMsg] = useState('');
    const [audioLevel, setAudioLevel] = useState(0);
    const [turn, setTurn] = useState(TURN.NONE);
    const [aiText, setAiText] = useState('');          // current sentence AI is saying

    const wsRef = useRef(null);
    const mediaStreamRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const audioContextRef = useRef(null);
    const analyserRef = useRef(null);
    const animFrameRef = useRef(null);
    const timerRef = useRef(null);
    const canvasRef = useRef(null);
    const sourceNodeRef = useRef(null);

    // ─── Audio playback queue (sentence-level) ───
    const audioQueueRef = useRef([]);       // [{wavData, aiText}]
    const isPlayingRef = useRef(false);
    const currentAudioRef = useRef(null);   // currently playing Audio element
    const pendingAiTextRef = useRef('');    // text received before the next audio blob

    // ─── Speech-level interrupt detection ───
    // Instead of naïve chunk-size check we sample the mic RMS via AnalyserNode.
    // Only trigger interrupt when the student is genuinely speaking (RMS above
    // threshold for several consecutive frames) — prevents echo/noise triggers.
    const INTERRUPT_RMS_THRESHOLD = 0.035;   // mic RMS that counts as "speaking"
    const INTERRUPT_FRAMES_NEEDED = 3;       // consecutive frames above threshold
    const consecutiveSpeechRef = useRef(0);  // counter of consecutive loud frames
    const DUCK_DURATION_MS = 220;            // fade-out length for audio ducking

    // ─── Duration timer ───
    useEffect(() => {
        if (status === STATUS.ACTIVE) {
            timerRef.current = setInterval(() => setDuration(d => d + 1), 1000);
        } else {
            clearInterval(timerRef.current);
        }
        return () => clearInterval(timerRef.current);
    }, [status]);

    // ─── Audio visualizer ───
    const drawWaveform = useCallback(() => {
        const canvas = canvasRef.current;
        const analyser = analyserRef.current;
        if (!canvas || !analyser) return;

        const ctx = canvas.getContext('2d');
        const WIDTH = canvas.width;
        const HEIGHT = canvas.height;
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        const draw = () => {
            animFrameRef.current = requestAnimationFrame(draw);
            analyser.getByteTimeDomainData(dataArray);

            ctx.fillStyle = 'rgba(15, 23, 42, 0)';
            ctx.clearRect(0, 0, WIDTH, HEIGHT);

            ctx.lineWidth = 2.5;
            const gradient = ctx.createLinearGradient(0, 0, WIDTH, 0);
            gradient.addColorStop(0, 'rgba(99, 102, 241, 0.4)');
            gradient.addColorStop(0.5, 'rgba(99, 102, 241, 1)');
            gradient.addColorStop(1, 'rgba(99, 102, 241, 0.4)');
            ctx.strokeStyle = gradient;
            ctx.beginPath();

            const sliceWidth = WIDTH / bufferLength;
            let x = 0;
            let maxAmplitude = 0;

            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = (v * HEIGHT) / 2;
                maxAmplitude = Math.max(maxAmplitude, Math.abs(v - 1));
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
                x += sliceWidth;
            }
            ctx.lineTo(WIDTH, HEIGHT / 2);
            ctx.stroke();
            setAudioLevel(Math.min(maxAmplitude * 5, 1));
        };
        draw();
    }, []);

    // ─── Play queued WAV audio responses sequentially (sentence-by-sentence) ───
    const playNextInQueue = useCallback(() => {
        if (isPlayingRef.current || audioQueueRef.current.length === 0) {
            if (audioQueueRef.current.length === 0 && !isPlayingRef.current) {
                setTurn(TURN.NONE);
                setAiText('');
            }
            return;
        }

        isPlayingRef.current = true;
        setTurn(TURN.AI);
        const { wavData, aiText: sentenceText } = audioQueueRef.current.shift();
        if (sentenceText) setAiText(sentenceText);

        const blob = new Blob([wavData], { type: 'audio/wav' });
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        currentAudioRef.current = audio;

        audio.onended = () => {
            URL.revokeObjectURL(url);
            currentAudioRef.current = null;
            isPlayingRef.current = false;
            playNextInQueue();
        };
        audio.onerror = () => {
            URL.revokeObjectURL(url);
            currentAudioRef.current = null;
            isPlayingRef.current = false;
            playNextInQueue();
        };
        audio.play().catch(() => {
            currentAudioRef.current = null;
            isPlayingRef.current = false;
            playNextInQueue();
        });
    }, []);

    // ─── Send interrupt signal with audio ducking ───
    const sendInterrupt = useCallback(() => {
        // Audio ducking — smooth fade-out instead of abrupt stop
        const audio = currentAudioRef.current;
        if (audio) {
            const startVol = audio.volume;
            const steps = 10;
            const stepMs = DUCK_DURATION_MS / steps;
            let step = 0;
            const fade = setInterval(() => {
                step++;
                audio.volume = Math.max(0, startVol * (1 - step / steps));
                if (step >= steps) {
                    clearInterval(fade);
                    audio.pause();
                    audio.volume = startVol; // reset for GC
                    currentAudioRef.current = null;
                }
            }, stepMs);
        }
        // Clear queue
        audioQueueRef.current = [];
        isPlayingRef.current = false;
        consecutiveSpeechRef.current = 0;
        setTurn(TURN.STUDENT);
        setAiText('');

        // Notify server
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ signal: 'interrupt' }));
        }
    }, []);

    // ─── Start call ───
    const startCall = useCallback(async () => {
        setStatus(STATUS.CONNECTING);
        setErrorMsg('');
        setDuration(0);

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 48000,
                },
            });
            mediaStreamRef.current = stream;

            // Audio context for visualization
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            audioContextRef.current = audioCtx;
            const source = audioCtx.createMediaStreamSource(stream);
            const analyser = audioCtx.createAnalyser();
            analyser.fftSize = 2048;
            analyser.smoothingTimeConstant = 0.85;
            source.connect(analyser);
            analyserRef.current = analyser;
            sourceNodeRef.current = source;

            // Open WebSocket
            const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
            const ws = new WebSocket(
                `${wsProtocol}://${window.location.host}/api/v1/ws/audio/${lessonId}?token=${token}`
            );
            wsRef.current = ws;

            ws.onopen = () => {
                // WS is open but we wait for the "ready" signal before going ACTIVE
                setStatus(STATUS.CALLING);
            };

            ws.onmessage = (event) => {
                if (typeof event.data === 'string') {
                    // JSON control message
                    try {
                        const msg = JSON.parse(event.data);
                        if (msg.signal === 'ready') {
                            // Models loaded — transition from "calling" to "active"
                            setStatus(STATUS.ACTIVE);
                            drawWaveform();
                            // NOW start the MediaRecorder
                            _startRecording(stream, ws);
                        }
                        if (msg.signal === 'interrupt') {
                            // Fade out instead of hard-pausing to avoid mid-word cutoff
                            const interruptAudio = currentAudioRef.current;
                            if (interruptAudio && !interruptAudio.paused) {
                                const startVol = interruptAudio.volume;
                                const steps = 8;
                                const stepMs = DUCK_DURATION_MS / steps;
                                let step = 0;
                                const fade = setInterval(() => {
                                    step++;
                                    interruptAudio.volume = Math.max(0, startVol * (1 - step / steps));
                                    if (step >= steps) {
                                        clearInterval(fade);
                                        interruptAudio.pause();
                                        interruptAudio.volume = startVol;
                                        if (currentAudioRef.current === interruptAudio) {
                                            currentAudioRef.current = null;
                                        }
                                    }
                                }, stepMs);
                            }
                            audioQueueRef.current = [];
                            isPlayingRef.current = false;
                            pendingAiTextRef.current = '';
                            setTurn(TURN.STUDENT);
                            setAiText('');
                        }
                        if (msg.ai_text) {
                            pendingAiTextRef.current = msg.ai_text;
                        }
                    } catch { /* ignore malformed */ }
                } else {
                    // Binary — WAV audio for one sentence
                    const sentenceText = pendingAiTextRef.current || '';
                    pendingAiTextRef.current = '';
                    audioQueueRef.current.push({ wavData: event.data, aiText: sentenceText });
                    playNextInQueue();
                }
            };

            ws.onerror = () => {
                setErrorMsg('Connection error. Please try again.');
                setStatus(STATUS.ERROR);
            };

            ws.onclose = () => {
                if (status !== STATUS.IDLE) setStatus(STATUS.IDLE);
            };
        } catch (err) {
            if (err.name === 'NotAllowedError') {
                setErrorMsg('Microphone access denied. Please allow microphone access and try again.');
            } else {
                setErrorMsg(err.message || 'Failed to start call');
            }
            setStatus(STATUS.ERROR);
        }
    }, [lessonId, token, drawWaveform, playNextInQueue, status]);

    // ─── Measure mic RMS from AnalyserNode (client-side speech detection) ───
    const getMicRMS = useCallback(() => {
        const analyser = analyserRef.current;
        if (!analyser) return 0;
        const buf = new Float32Array(analyser.fftSize);
        analyser.getFloatTimeDomainData(buf);
        let sum = 0;
        for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
        return Math.sqrt(sum / buf.length);
    }, []);

    // ─── Start MediaRecorder (called only after "ready" signal) ───
    const _startRecording = useCallback((stream, ws) => {
        const recorder = new MediaRecorder(stream, {
            mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus'
                : 'audio/webm',
        });
        mediaRecorderRef.current = recorder;

        recorder.ondataavailable = (e) => {
            if (e.data.size > 0 && ws.readyState === WebSocket.OPEN) {
                // ── Speech-gated interrupt detection ──
                // Only fire interrupt when we detect genuine student speech
                // (mic RMS above threshold for N consecutive recorder ticks).
                if (isPlayingRef.current) {
                    const rms = getMicRMS();
                    if (rms > INTERRUPT_RMS_THRESHOLD) {
                        consecutiveSpeechRef.current += 1;
                        if (consecutiveSpeechRef.current >= INTERRUPT_FRAMES_NEEDED) {
                            sendInterrupt();
                        }
                    } else {
                        consecutiveSpeechRef.current = 0;
                    }
                } else {
                    consecutiveSpeechRef.current = 0;
                }
                e.data.arrayBuffer().then(buf => ws.send(buf));
            }
        };

        recorder.start(250);
    }, [sendInterrupt, getMicRMS]);

    // ─── End call ───
    const endCall = useCallback(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop();
        }
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        if (mediaStreamRef.current) {
            mediaStreamRef.current.getTracks().forEach(t => t.stop());
            mediaStreamRef.current = null;
        }
        if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
            audioContextRef.current.close();
        }
        // Stop any playing audio
        if (currentAudioRef.current) {
            currentAudioRef.current.pause();
            currentAudioRef.current = null;
        }
        audioQueueRef.current = [];
        isPlayingRef.current = false;

        setStatus(STATUS.IDLE);
        setAudioLevel(0);
        setTurn(TURN.NONE);
        setAiText('');
    }, []);

    // ─── Toggle mute ───
    const toggleMute = useCallback(() => {
        if (mediaStreamRef.current) {
            const audioTrack = mediaStreamRef.current.getAudioTracks()[0];
            if (audioTrack) {
                audioTrack.enabled = !audioTrack.enabled;
                setIsMuted(!audioTrack.enabled);
            }
        }
    }, []);

    useEffect(() => {
        return () => endCall();
    }, [endCall]);

    const formatTime = (seconds) => {
        const m = Math.floor(seconds / 60).toString().padStart(2, '0');
        const s = (seconds % 60).toString().padStart(2, '0');
        return `${m}:${s}`;
    };

    return (
        <div className="ai-call-page">
            {/* Animated background blobs */}
            <div className="ai-call-bg">
                <div className="blob blob-1" />
                <div className="blob blob-2" />
                <div className="blob blob-3" />
            </div>

            <div className="ai-call-container">
                {/* Back button */}
                <button className="ai-call-back" onClick={() => { endCall(); navigate(-1); }}>
                    <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                        <path d="M12.5 15L7.5 10l5-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                    Back
                </button>

                {/* AI Avatar */}
                <div className={`ai-avatar-wrapper ${status === STATUS.ACTIVE ? 'active' : ''} ${status === STATUS.CALLING ? 'calling' : ''} ${turn === TURN.AI ? 'speaking' : ''}`}>
                    <div className="ai-avatar-ring" style={{ transform: `scale(${1 + audioLevel * 0.3})` }} />
                    <div className="ai-avatar-ring ring-2" style={{ transform: `scale(${1 + audioLevel * 0.2})` }} />
                    <div className="ai-avatar">
                        <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                            <path d="M24 4C12.96 4 4 12.96 4 24s8.96 20 20 20 20-8.96 20-20S35.04 4 24 4zm0 6c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6 2.69-6 6-6zm0 28.4c-5 0-9.44-2.56-12.02-6.44.06-3.98 8.02-6.16 12.02-6.16s11.96 2.18 12.02 6.16A14.97 14.97 0 0124 38.4z" fill="white" />
                        </svg>
                    </div>
                </div>

                {/* Status text */}
                <h2 className="ai-call-title">
                    {status === STATUS.IDLE && 'AI Tutor'}
                    {status === STATUS.CONNECTING && 'Connecting...'}
                    {status === STATUS.CALLING && 'Calling AI Tutor…'}
                    {status === STATUS.ACTIVE && turn === TURN.AI && 'AI Tutor — Speaking'}
                    {status === STATUS.ACTIVE && turn === TURN.STUDENT && 'AI Tutor — Listening'}
                    {status === STATUS.ACTIVE && turn === TURN.NONE && 'AI Tutor — Ready'}
                    {status === STATUS.ERROR && 'Connection Failed'}
                </h2>

                {status === STATUS.ACTIVE && (
                    <span className="ai-call-timer">{formatTime(duration)}</span>
                )}

                {/* AI speech subtitle */}
                {status === STATUS.ACTIVE && turn === TURN.AI && aiText && (
                    <p className="ai-call-subtitle">{aiText}</p>
                )}

                {/* Waveform canvas */}
                <div className={`ai-waveform-container ${status === STATUS.ACTIVE ? 'visible' : ''}`}>
                    <canvas ref={canvasRef} width={320} height={80} className="ai-waveform-canvas" />
                </div>

                {/* Error message */}
                {status === STATUS.ERROR && errorMsg && (
                    <div className="ai-call-error">
                        <span>⚠️</span> {errorMsg}
                    </div>
                )}

                {/* Idle state description */}
                {status === STATUS.IDLE && (
                    <p className="ai-call-desc">
                        Start a voice session with your AI tutor. Speak naturally and the AI will respond to your questions in real-time.
                    </p>
                )}

                {/* Controls */}
                <div className="ai-call-controls">
                    {status === STATUS.IDLE || status === STATUS.ERROR ? (
                        <button className="ai-call-btn call-start" onClick={startCall}>
                            <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
                                <path d="M22 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07 19.5 19.5 0 01-6-6 19.79 19.79 0 01-3.07-8.67A2 2 0 014.11 2h3a2 2 0 012 1.72c.127.96.361 1.903.7 2.81a2 2 0 01-.45 2.11L8.09 9.91a16 16 0 006 6l1.27-1.27a2 2 0 012.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0122 16.92z" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                            </svg>
                            Start Session
                        </button>
                    ) : status === STATUS.CONNECTING ? (
                        <div className="ai-call-connecting">
                            <div className="spinner-ring" />
                            <span>Establishing connection…</span>
                        </div>
                    ) : status === STATUS.CALLING ? (
                        <div className="ai-call-active-controls">
                            <button className="ai-ctrl-btn end-call" onClick={endCall} title="Cancel call">
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                                    <path d="M10.68 13.31a16 16 0 003.41 2.6l1.27-1.27a2 2 0 012.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0122 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07 19.5 19.5 0 01-6-6 19.79 19.79 0 01-3.07-8.67A2 2 0 014.11 2h3a2 2 0 012 1.72c.127.96.361 1.903.7 2.81a2 2 0 01-.45 2.11L8.09 9.91" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                    <line x1="1" y1="1" x2="23" y2="23" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                                </svg>
                                <span className="ctrl-label">Cancel</span>
                            </button>
                        </div>
                    ) : (
                        <div className="ai-call-active-controls">
                            <button
                                className={`ai-ctrl-btn ${isMuted ? 'muted' : ''}`}
                                onClick={toggleMute}
                                title={isMuted ? 'Unmute' : 'Mute'}
                            >
                                {isMuted ? (
                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                                        <path d="M1 1l22 22M9 9v3a3 3 0 005.12 2.12M15 9.34V4a3 3 0 00-5.94-.6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                        <path d="M17 16.95A7 7 0 015 12m14 0a7 7 0 01-.11 1.23M12 19v4m-4 0h8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                    </svg>
                                ) : (
                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                                        <path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                        <path d="M19 10v2a7 7 0 01-14 0v-2M12 19v4m-4 0h8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                    </svg>
                                )}
                                <span className="ctrl-label">{isMuted ? 'Unmute' : 'Mute'}</span>
                            </button>

                            <button className="ai-ctrl-btn end-call" onClick={endCall} title="End call">
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                                    <path d="M10.68 13.31a16 16 0 003.41 2.6l1.27-1.27a2 2 0 012.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0122 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07 19.5 19.5 0 01-6-6 19.79 19.79 0 01-3.07-8.67A2 2 0 014.11 2h3a2 2 0 012 1.72c.127.96.361 1.903.7 2.81a2 2 0 01-.45 2.11L8.09 9.91" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                    <line x1="1" y1="1" x2="23" y2="23" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                                </svg>
                                <span className="ctrl-label">End</span>
                            </button>
                        </div>
                    )}
                </div>

                {/* Session info badge */}
                {(status === STATUS.ACTIVE || status === STATUS.CALLING) && (
                    <div className="ai-call-info-badge">
                        <span className="pulse-dot" />
                        {status === STATUS.CALLING ? 'Ringing…' : 'Live · AI Tutor Session'}
                    </div>
                )}
            </div>
        </div>
    );
}
