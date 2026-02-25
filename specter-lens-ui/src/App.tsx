import React, { useEffect, useState } from 'react';
import { Activity, ShieldAlert, Cpu, Eye, RadioTower } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

function App() {
    const [telemetry, setTelemetry] = useState({
        active_model: 'Standby...',
        session_tokens: { prompt: 0, completion: 0 },
        loop_iteration: 0,
        role_mapping: {}
    });
    const [pulse, setPulse] = useState(false);
    const [isConnected, setIsConnected] = useState(false);

    useEffect(() => {
        // Connect to port 8878
        const ws = new WebSocket('ws://localhost:8878/ws/telemetry');

        ws.onopen = () => {
            setIsConnected(true);
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                setTelemetry(t => ({ ...t, ...data }));
                setPulse(true);
                setTimeout(() => setPulse(false), 300);
            } catch (e) {
                console.error("Failed to parse telemetry");
            }
        };

        ws.onclose = () => {
            setIsConnected(false);
            // Reconnect after 3 seconds
            setTimeout(() => {
                window.location.reload();
            }, 3000);
        };

        return () => {
            ws.close();
        };
    }, []);

    return (
        <div className="app-container h-screen bg-[#0a0a0a] text-white/80 font-mono overflow-hidden flex flex-col select-none text-[12px] border-l border-white/5" style={{ paddingLeft: '4px', paddingRight: '4px' }}>
            {/* Tactical Header */}
            <header className="flex justify-between items-center border-b border-[#facc15]/30 pb-3 mb-6 shrink-0" style={{ marginLeft: '2px', marginRight: '2px' }}>
                <div className="flex items-center space-x-2">
                    <Eye className={`w-4 h-4 ${pulse ? 'text-[#facc15]' : 'text-white/40'} transition-opacity duration-300`} />
                    <h1 className="text-[11px] tracking-[0.25em] font-black text-white/90 uppercase">
                        SPECTER <span className="text-[#facc15]">QWEN</span> HUD
                    </h1>
                </div>

                <div className="flex items-center">
                    <div className={`flex items-center space-x-1.5 px-2 py-0.5 rounded-sm border ${isConnected ? 'bg-green-500/10 border-green-500/40 text-green-400' : 'bg-red-500/10 border-red-500/40 text-red-400'} text-[7px] font-bold tracking-widest uppercase`}>
                        <RadioTower className={`w-2.5 h-2.5 ${isConnected ? 'animate-pulse' : ''}`} />
                        <span>{isConnected ? 'Uplink' : 'Signal Lost'}</span>
                    </div>
                </div>
            </header>

            {/* Vertical HUD Content */}
            <main className="space-y-5">

                {/* ACTIVE NODE */}
                <section className="bg-white/[0.02] border border-white/10 p-3 rounded-sm relative group">
                    <div className="text-[7px] text-white/30 uppercase tracking-[0.3em] mb-2">Compute Hub</div>
                    <div className="flex items-center space-x-3">
                        <Cpu className="w-4 h-4 text-[#facc15]/80" />
                        <span className="text-[13px] text-white font-bold tracking-tight truncate">
                            {telemetry.active_model}
                        </span>
                    </div>
                    <div className="absolute top-0 left-0 w-full h-[1px] bg-[#facc15]/10 animate-[scan_5s_infinite]" />
                </section>

                {/* CONSUMPTION METRICS */}
                <section className="space-y-4">
                    <div className="text-[7px] text-white/30 uppercase tracking-[0.3em] px-1">Resource Load</div>

                    <div className="grid gap-4">
                        <div className="bg-white/[0.01] border-l-2 border-[#facc15]/40 p-2">
                            <div className="flex justify-between items-baseline mb-1">
                                <span className="text-[8px] text-white/40">TX_PROMPT</span>
                                <span className="text-[10px] font-bold text-[#facc15]">{telemetry.session_tokens.prompt.toLocaleString()}</span>
                            </div>
                            <div className="h-[1px] bg-white/5 overflow-hidden">
                                <motion.div
                                    className="h-full bg-[#facc15]/40"
                                    initial={{ width: 0 }}
                                    animate={{ width: isConnected ? "30%" : "0%" }}
                                />
                            </div>
                        </div>

                        <div className="bg-white/[0.01] border-l-2 border-[#facc15]/40 p-2">
                            <div className="flex justify-between items-baseline mb-1">
                                <span className="text-[8px] text-white/40">RX_COMPLETION</span>
                                <span className="text-[10px] font-bold text-[#facc15]">{telemetry.session_tokens.completion.toLocaleString()}</span>
                            </div>
                            <div className="h-[1px] bg-white/5 overflow-hidden">
                                <motion.div
                                    className="h-full bg-[#facc15]/40"
                                    initial={{ width: 0 }}
                                    animate={{ width: isConnected ? "50%" : "0%" }}
                                />
                            </div>
                        </div>
                    </div>
                </section>

                {/* ITERATION CYCLE */}
                <section className="bg-[#facc15]/5 border border-[#facc15]/20 p-4 rounded-sm flex items-center justify-between">
                    <div>
                        <div className="text-[7px] text-[#facc15]/60 uppercase tracking-[0.3em] mb-1">Cycle Sequence</div>
                        <div className="text-4xl text-white font-black tracking-tighter tabular-nums leading-none">
                            {telemetry.loop_iteration.toString().padStart(2, '0')}
                        </div>
                    </div>
                    <div className="flex flex-col items-end">
                        <Activity className="w-5 h-5 text-[#facc15]/40 mb-2" />
                        <AnimatePresence>
                            {telemetry.loop_iteration > 2 && (
                                <motion.div
                                    initial={{ opacity: 0, x: 10 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    className="text-red-500 flex items-center space-x-1"
                                >
                                    <ShieldAlert className="w-3.5 h-3.5" />
                                    <span className="text-[8px] font-bold">LOOP_BND</span>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                </section>

                {/* ROLE MAPPING */}
                <section className="space-y-3">
                    <div className="text-[7px] text-white/30 uppercase tracking-[0.3em] px-1">Logic Matrix</div>
                    <div className="bg-white/[0.02] border border-white/10 rounded-sm divide-y divide-white/5">
                        {Object.entries(telemetry.role_mapping).map(([role, model]) => (
                            <div key={role} className="p-2.5">
                                <div className="text-[8px] uppercase text-[#facc15]/60 mb-1">{role}</div>
                                <div className="text-[10px] font-medium text-white/60 truncate italic">{model as string}</div>
                            </div>
                        ))}
                        {Object.keys(telemetry.role_mapping).length === 0 && (
                            <div className="text-[9px] text-white/10 italic p-3 text-center">Standby for Matrix Assignment...</div>
                        )}
                    </div>
                </section>

            </main>

            {/* Tactical Footer */}
            <footer className="mt-10 border-t border-white/5 pt-4 flex justify-between items-center text-[7px] uppercase tracking-widest text-white/10">
                <span>Core.v6.Alpha</span>
                <span className="text-[#facc15]/20">0X-SPECTER-LENS</span>
            </footer>

            <style dangerouslySetInnerHTML={{
                __html: `
        @keyframes scan {
          0% { transform: translateY(-30px); opacity: 0; }
          50% { opacity: 0.2; }
          100% { transform: translateY(80px); opacity: 0; }
        }
        ::-webkit-scrollbar { width: 3px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #facc1515; }
        ::-webkit-scrollbar-thumb:hover { background: #facc1530; }
      `}} />
        </div>
    );
}

export default App;
