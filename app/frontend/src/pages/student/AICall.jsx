import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useAuthStore } from '../../store/authStore';
import './AICall.css';

const STATUS = {
    IDLE: 'idle',
    CONNECTING: 'connecting',
    ACTIVE: 'active',
    ERROR: 'error',
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

    const wsRef = useRef(null);
    const mediaStreamRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const audioContextRef = useRef(null);
    const analyserRef = useRef(null);
    const animFrameRef = useRef(null);
    const timerRef = useRef(null);
    const canvasRef = useRef(null);
    const sourceNodeRef = useRef(null);
    const audioQueueRef = useRef([]);
    const isPlayingRef = useRef(false);

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

            // Background
            ctx.fillStyle = 'rgba(15, 23, 42, 0)';
            ctx.clearRect(0, 0, WIDTH, HEIGHT);

            // Waveform
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

    // ─── Play queued WAV audio responses sequentially ───
    const playNextInQueue = useCallback(() => {
        if (isPlayingRef.current || audioQueueRef.current.length === 0) return;

        isPlayingRef.current = true;
        const wavData = audioQueueRef.current.shift();
        const blob = new Blob([wavData], { type: 'audio/wav' });
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.onended = () => {
            URL.revokeObjectURL(url);
            isPlayingRef.current = false;
            playNextInQueue(); // play next if queued
        };
        audio.onerror = () => {
            URL.revokeObjectURL(url);
            isPlayingRef.current = false;
            playNextInQueue();
        };
        audio.play().catch(() => {
            isPlayingRef.current = false;
            playNextInQueue();
        });
    }, []);

    // ─── Start call ───
    const startCall = useCallback(async () => {
        setStatus(STATUS.CONNECTING);
        setErrorMsg('');
        setDuration(0);

        try {
            // Request microphone access
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
            ws.binaryType = 'arraybuffer';
            wsRef.current = ws;

            ws.onopen = () => {
                setStatus(STATUS.ACTIVE);
                drawWaveform();

                // Start MediaRecorder to send audio chunks
                const recorder = new MediaRecorder(stream, {
                    mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                        ? 'audio/webm;codecs=opus'
                        : 'audio/webm',
                });
                mediaRecorderRef.current = recorder;

                recorder.ondataavailable = (e) => {
                    if (e.data.size > 0 && ws.readyState === WebSocket.OPEN) {
                        e.data.arrayBuffer().then(buf => ws.send(buf));
                    }
                };

                recorder.start(250); // send a chunk every 250ms
            };

            ws.onmessage = (event) => {
                // Received complete WAV audio for one conversational turn
                audioQueueRef.current.push(event.data);
                playNextInQueue();
            };

            ws.onerror = () => {
                setErrorMsg('Connection error. Please try again.');
                setStatus(STATUS.ERROR);
            };

            ws.onclose = (event) => {
                if (status !== STATUS.IDLE) {
                    setStatus(STATUS.IDLE);
                }
            };
        } catch (err) {
            if (err.name === 'NotAllowedError') {
                setErrorMsg('Microphone access denied. Please allow microphone access and try again.');
            } else {
                setErrorMsg(err.message || 'Failed to start call');
            }
            setStatus(STATUS.ERROR);
        }
    }, [lessonId, token, drawWaveform, playNextInQueue]);

    // ─── End call ───
    const endCall = useCallback(() => {
        // Stop MediaRecorder
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop();
        }
        // Close WebSocket
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        // Stop media tracks
        if (mediaStreamRef.current) {
            mediaStreamRef.current.getTracks().forEach(t => t.stop());
            mediaStreamRef.current = null;
        }
        // Stop visualization
        if (animFrameRef.current) {
            cancelAnimationFrame(animFrameRef.current);
        }
        // Close audio context
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
            audioContextRef.current.close();
        }
        // Clear audio playback queue
        audioQueueRef.current = [];
        isPlayingRef.current = false;

        setStatus(STATUS.IDLE);
        setAudioLevel(0);
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

    // Cleanup on unmount
    useEffect(() => {
        return () => endCall();
    }, [endCall]);

    // ─── Format duration ───
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
                <div className={`ai-avatar-wrapper ${status === STATUS.ACTIVE ? 'active' : ''}`}>
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
                    {status === STATUS.ACTIVE && 'AI Tutor — Listening'}
                    {status === STATUS.ERROR && 'Connection Failed'}
                </h2>

                {status === STATUS.ACTIVE && (
                    <span className="ai-call-timer">{formatTime(duration)}</span>
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
                {status === STATUS.ACTIVE && (
                    <div className="ai-call-info-badge">
                        <span className="pulse-dot" />
                        Live · AI Tutor Session
                    </div>
                )}
            </div>
        </div>
    );
}
