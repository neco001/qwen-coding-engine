import React, { useEffect, useState, useRef } from 'react';
import { Activity, ShieldAlert, Cpu, Eye, RadioTower, Zap, Terminal, Hash, Layers } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';

interface TokenCount { prompt: number; completion: number; }
interface TelemetryState {
    active_model: string;
    status: string; // "idle", "live", "processing"
    operation: string;
    progress_percent: number;
    request_tokens: TokenCount;
    session_tokens: TokenCount;
    loop_iteration: number;
    role_mapping: Record<string, string>;
    thinking: string;
    streaming_content: string;
    is_live: boolean;
    session_display_id?: number;
    session_display_name?: string;
}

const DEFAULT_STATE: TelemetryState = {
    active_model: 'Standby...',
    status: 'idle',
    operation: '',
    progress_percent: 0,
    request_tokens: { prompt: 0, completion: 0 },
    session_tokens: { prompt: 0, completion: 0 },
    loop_iteration: 0,
    role_mapping: {},
    thinking: '',
    streaming_content: '',
    is_live: false,
    session_display_id: 1,
    session_display_name: 'Sesja 1',
};

// Session type for multi-session support
interface Session {
    id: string;
    name: string;
    projectId: string;
    telemetry: TelemetryState;
    isConnected: boolean;
    ws: WebSocket | null;
}

function App() {
    const [sessions, setSessions] = useState<Session[]>([]);
    const [activeSessionId, setActiveSessionId] = useState<string>('');
    const streamRef = useRef<HTMLDivElement>(null);
    const fadeTimers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

    // Initialize sessions from VSCode extension or defaults
    useEffect(() => {
        // Try to get sessions from VSCode extension first
        const vscodeSessions = (window as any).INITIAL_SESSIONS;
        
        if (vscodeSessions && Array.isArray(vscodeSessions)) {
            // Use sessions from extension
            const initialSessions: Session[] = vscodeSessions.map((s: any) => ({
                id: s.id,
                name: s.name,
                projectId: s.projectId,
                telemetry: DEFAULT_STATE,
                isConnected: false,
                ws: null
            }));
            setSessions(initialSessions);
            if (initialSessions.length > 0) {
                setActiveSessionId(initialSessions[0].id);
            }
        } else {
            // Fallback: Default sessions with generic names
            const defaultSessions: Session[] = [
                { id: 'session_1', name: 'Sesja 1', projectId: 'default_1', telemetry: { ...DEFAULT_STATE, session_display_id: 1, session_display_name: 'Sesja 1' }, isConnected: false, ws: null },
                { id: 'session_2', name: 'Sesja 2', projectId: 'default_2', telemetry: { ...DEFAULT_STATE, session_display_id: 2, session_display_name: 'Sesja 2' }, isConnected: false, ws: null }
            ];
            setSessions(defaultSessions);
            setActiveSessionId('session_1');
        }
    }, []);

    // Connect WebSocket for each session
    useEffect(() => {
        sessions.forEach(session => {
            if (session.ws) return; // Already connected

            const ws = new WebSocket(`ws://127.0.0.1:8878/ws/telemetry?project_id=${session.projectId}`);
            
            ws.onopen = () => {
                setSessions(prev => prev.map(s =>
                    s.id === session.id ? { ...s, isConnected: true } : s
                ));
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'heartbeat') return;
                    
                    setSessions(prev => prev.map(s => {
                        if (s.id === session.id) {
                            const newTelemetry = { ...s.telemetry, ...data };
                            // Update is_live based on status
                            newTelemetry.is_live = data.status === 'live' || data.status === 'processing';
                            
                            // P2-1 FIX: Skip update if nothing meaningful changed (prevents UI flickering)
                            if (JSON.stringify(newTelemetry) === JSON.stringify(s.telemetry)) {
                                return s;
                            }
                            return { ...s, telemetry: newTelemetry };
                        }
                        return s;
                    }));

                    // Auto-clear streaming content after 12s of silence
                    if (data.streaming_content || data.thinking) {
                        if (fadeTimers.current.has(session.id)) {
                            clearTimeout(fadeTimers.current.get(session.id));
                        }
                        fadeTimers.current.set(session.id, setTimeout(() => {
                            setSessions(prev => prev.map(s =>
                                s.id === session.id
                                    ? { ...s, telemetry: { ...s.telemetry, streaming_content: '', thinking: '' } }
                                    : s
                            ));
                        }, 12000));
                    }
                } catch (e) {
                    console.error("Failed to parse telemetry");
                }
            };

            ws.onclose = () => {
                setSessions(prev => prev.map(s =>
                    s.id === session.id ? { ...s, isConnected: false, ws: null } : s
                ));
                // Attempt reconnect after 3 seconds
                setTimeout(() => {
                    setSessions(prev => prev.map(s =>
                        s.id === session.id ? { ...s, ws: null } : s
                    ));
                }, 3000);
            };

            ws.onerror = () => {
                ws.close();
            };

            // Update session with WebSocket
            setSessions(prev => prev.map(s =>
                s.id === session.id ? { ...s, ws } : s
            ));
        });

        return () => {
            sessions.forEach(s => s.ws?.close());
            // P2-2 FIX: Clear all timers and clear the map to prevent memory leak
            fadeTimers.current.forEach((timer) => clearTimeout(timer));
            fadeTimers.current.clear();
        };
    }, [sessions.length]);

    // Get active session
    const activeSession = sessions.find(s => s.id === activeSessionId);
    const telemetry = activeSession?.telemetry || DEFAULT_STATE;
    const isConnected = activeSession?.isConnected || false;

    // Auto-scroll logic: only when at bottom or small height
    useEffect(() => {
        if (streamRef.current) {
            const { scrollTop, scrollHeight, clientHeight } = streamRef.current;
            const isAtBottom = scrollHeight - scrollTop <= clientHeight + 50;
            if (isAtBottom) {
                streamRef.current.scrollTop = scrollHeight;
            }
        }
    }, [telemetry.streaming_content, telemetry.thinking]);

    const hasStream = (telemetry.streaming_content?.length > 0) || (telemetry.thinking?.length > 0);
    const hasRoles = Object.keys(telemetry.role_mapping).length > 0;
    const hasProgress = telemetry.status === 'processing' && telemetry.progress_percent > 0;

    return (
        <div className="app-container h-screen bg-[#0a0a0a] text-white/80 font-mono flex flex-col select-none text-[13px]">

            {/* 0. SESSION TABS (Multi-session support) */}
            <div className="flex items-center gap-1 px-2 py-1.5 border-b border-white/5 bg-white/[0.02] shrink-0">
                <Layers className="w-3.5 h-3.5 text-white/30 mr-1" />
                {sessions.map(session => (
                    <button
                        key={session.id}
                        onClick={() => setActiveSessionId(session.id)}
                        className={`px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider rounded-sm transition-all ${
                            activeSessionId === session.id
                                ? 'bg-[#facc15]/20 text-[#facc15] border border-[#facc15]/40'
                                : 'bg-white/5 text-white/40 border border-transparent hover:bg-white/10'
                        }`}
                    >
                        {/* P3-2 FIX: Compute display name dynamically from session_display_id */}
                        {session.telemetry.session_display_name ||
                         (session.telemetry.session_display_id ? `Sesja ${session.telemetry.session_display_id}` : session.name)}
                        {session.isConnected && (
                            <span className="ml-1.5 w-1.5 h-1.5 inline-block rounded-full bg-green-500 animate-pulse" />
                        )}
                    </button>
                ))}
            </div>

            {/* 1. TACTICAL HEADER */}
            <header className="flex justify-between items-center px-3 py-3 border-b border-white/5 shrink-0">
                <div className="flex items-center space-x-3">
                    <Eye className={`w-5 h-5 transition-all duration-500 ${hasStream ? 'text-[#facc15] drop-shadow-[0_0_8px_rgba(250,204,21,0.5)]' : 'text-white/20'}`} />
                    <h1 className="text-[11px] tracking-[0.3em] font-black text-white/90 uppercase">
                        SPECTER <span className="text-[#facc15]">QWEN</span> HUD
                    </h1>
                </div>
                <div className={`flex items-center space-x-1.5 px-2 py-0.5 rounded-sm border ${isConnected ? 'bg-green-500/10 border-green-500/40 text-green-400' : 'bg-red-500/10 border-red-500/40 text-red-400'} text-[8px] font-bold tracking-widest uppercase`}>
                    <RadioTower className={`w-3 h-3 ${isConnected ? 'animate-pulse' : ''}`} />
                    <span>{isConnected ? 'Uplink' : 'Signal Lost'}</span>
                </div>
            </header>

            <main className="flex-1 overflow-y-auto p-3 space-y-5">

                {/* 1.5 PROGRESS BAR (For long operations) */}
                {hasProgress && (
                    <section className="bg-white/[0.02] border border-white/10 p-2.5 rounded-sm">
                        <div className="flex items-center justify-between mb-1.5">
                            <div className="text-[9px] text-white/40 uppercase tracking-wider flex items-center gap-1">
                                <Zap className="w-3 h-3 animate-pulse" />
                                {telemetry.operation || 'Processing...'}
                            </div>
                            <div className="text-[10px] text-[#facc15] font-bold">
                                {telemetry.progress_percent}%
                            </div>
                        </div>
                        <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-gradient-to-r from-[#facc15] to-[#facc15]/50 transition-all duration-300"
                                style={{ width: `${telemetry.progress_percent}%` }}
                            />
                        </div>
                    </section>
                )}

                {/* 2. COMPUTE HUB (Active Model) */}
                <section className="bg-white/[0.03] border border-white/10 p-2.5 rounded-sm">
                    <div className="text-[10px] text-white/30 uppercase tracking-[0.2em] mb-1.5 flex items-center gap-1">
                        <Cpu className="w-3 h-3" /> Compute Hub
                    </div>
                    <div className="text-[12px] text-[#facc15]/90 font-bold truncate px-1">
                        {telemetry.active_model}
                    </div>
                </section>

                {/* 3. RESOURCE LOAD (Token Metrics) - THE MOST IMPORTANT */}
                <section className="space-y-4">
                    <div className="flex items-center justify-between px-1">
                        <div className="text-[10px] text-white/30 uppercase tracking-[0.2em]">Resource Load</div>
                        <div className="text-[9px] text-white/20 uppercase tracking-widest">↳ This Prompt</div>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                        {(['prompt', 'completion'] as const).map(type => (
                            <div key={type} className="bg-white/[0.02] border-t border-white/10 p-3 pt-2">
                                <div className="text-[9px] text-white/40 uppercase mb-1">{type === 'prompt' ? 'TX (Input)' : 'RX (Output)'}</div>
                                <div className="text-[22px] font-black text-[#facc15] tracking-tight tabular-nums">
                                    {telemetry.request_tokens[type].toLocaleString()}
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Session Summary */}
                    <div className="bg-white/[0.01] border border-white/5 p-2 flex justify-between items-center rounded-sm">
                        <div className="text-[10px] text-white/20 uppercase tracking-widest pl-1">↳ Session Total</div>
                        <div className="flex space-x-4 pr-1">
                            <span className="text-[13px] font-bold text-[#facc15]/60 tabular-nums">
                                <span className="opacity-30 mr-1 text-[9px]">TX</span> {telemetry.session_tokens.prompt.toLocaleString()}
                            </span>
                            <span className="text-[13px] font-bold text-[#facc15] tabular-nums">
                                <span className="opacity-30 mr-1 text-[9px]">RX</span> {telemetry.session_tokens.completion.toLocaleString()}
                            </span>
                        </div>
                    </div>
                </section>

                {/* 4. LOGIC MATRIX (Role Mapping) */}
                <section className="space-y-2">
                    <div className="text-[10px] text-white/30 uppercase tracking-[0.2em] px-1 flex items-center gap-1">
                        <Hash className="w-3 h-3" /> Logic Matrix
                        {(telemetry.is_live || telemetry.status === 'live' || telemetry.status === 'processing') && (
                            <span className="ml-2 px-1.5 py-0.5 bg-green-500/20 border border-green-500/40 text-green-400 text-[7px] font-black uppercase tracking-widest rounded-sm animate-pulse">
                                LIVE
                            </span>
                        )}
                    </div>
                    <div className="bg-white/[0.02] border border-white/10 rounded-sm divide-y divide-white/5">
                        {hasRoles ? (
                            Object.entries(telemetry.role_mapping).map(([role, model]) => (
                                <div key={role} className="p-2 flex justify-between items-center gap-3">
                                    <div className="text-[10px] uppercase text-[#facc15]/60 font-bold w-20 shrink-0">{role}</div>
                                    <div className="text-[11px] text-white/40 truncate italic flex-1 text-right">{model as string}</div>
                                </div>
                            ))
                        ) : (
                            <div className="text-[11px] text-white/10 italic p-3 text-center">
                                Awaiting Pro Mode assignment...
                            </div>
                        )}
                    </div>
                </section>

                {/* 5. LIVE STREAM (Now with Markdown and Correct Direction) */}
                <section className={`flex flex-col space-y-2 ${hasStream ? 'block' : 'hidden'}`}>
                    <div className="text-[10px] text-[#facc15]/50 uppercase tracking-[0.2em] px-1 flex items-center space-x-2">
                        <Zap className="w-3 h-3 animate-pulse" />
                        <span>Live Evolution</span>
                    </div>
                    <div
                        ref={streamRef}
                        className="bg-white/[0.02] border border-[#facc15]/10 rounded-sm p-4 max-h-[350px] overflow-y-auto scroll-smooth leading-relaxed"
                    >
                        <div className="prose prose-invert prose-sm max-w-none">
                            {telemetry.thinking && (
                                <div className="text-[#529b0d] italic mb-4 opacity-80 border-l-2 border-[#529b0d]/30 pl-3 py-1 font-medium text-[12px]">
                                    <ReactMarkdown>{telemetry.thinking}</ReactMarkdown>
                                </div>
                            )}
                            {telemetry.streaming_content && (
                                <div className="text-white/80 text-[12px] font-medium markdown-body">
                                    <ReactMarkdown>{telemetry.streaming_content}</ReactMarkdown>
                                </div>
                            )}
                        </div>
                    </div>
                </section>

                {/* 6. PRO LOOP (Demoted to bottom, minimized) */}
                <section className="border-t border-white/5 pt-4 pb-2 opacity-60">
                    <div className="flex items-center justify-between bg-white/[0.02] p-2 rounded-sm border border-white/5">
                        <div className="flex items-center space-x-3">
                            <Activity className="w-4 h-4 text-[#facc15]/40" />
                            <div>
                                <div className="text-[9px] text-white/30 uppercase tracking-widest">White Cell Iterations</div>
                                <div className="text-[15px] text-white font-black tabular-nums">
                                    {telemetry.loop_iteration.toString().padStart(2, '0')}
                                </div>
                            </div>
                        </div>
                        <AnimatePresence>
                            {telemetry.loop_iteration > 5 && (
                                <motion.div
                                    initial={{ opacity: 0, scale: 0.9 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    className="bg-red-500/20 text-red-400 px-2 py-0.5 rounded flex items-center space-x-1"
                                >
                                    <ShieldAlert className="w-3 h-3" />
                                    <span className="text-[9px] font-black">LOOP_BND</span>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                </section>

            </main>

            {/* FOOTER */}
            <footer className="px-4 py-3 border-t border-white/5 flex justify-between items-center text-[9px] uppercase tracking-[0.3em] text-white/10 shrink-0">
                <div className="flex items-center gap-3">
                    <span>Core.v1.1.3 - Multi-Session</span>
                    <span className="text-white/20">|</span>
                    <span className={isConnected ? 'text-green-400/60' : 'text-red-400/60'}>
                        {sessions.filter(s => s.isConnected).length}/{sessions.length} Sessions Active
                    </span>
                </div>
                <span className="text-[#facc15]/10">0X-SPECTER-LENS-HUD</span>
            </footer>

            <style dangerouslySetInnerHTML={{
                __html: `
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #facc1515; border-radius: 2px; }
        ::-webkit-scrollbar-thumb:hover { background: #facc1530; }
        
        .markdown-body p { margin-bottom: 0.75rem; }
        .markdown-body code { background: rgba(255,255,255,0.05); padding: 2px 4px; border-radius: 3px; font-size: 0.9em; }
        .markdown-body pre { background: rgba(0,0,0,0.3); padding: 10px; border-radius: 4px; overflow-x: auto; margin: 10px 0; border: 1px solid rgba(255,255,255,0.05); }
      `}} />
        </div>
    );
}

export default App;
