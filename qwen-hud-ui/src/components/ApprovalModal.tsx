import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, CheckCircle2, XCircle, FileCode, GitBranch, Package } from 'lucide-react';
import { useState } from 'react';

interface ApprovalModalProps {
    isOpen: boolean;
    onClose: () => void;
    onApprove: () => void;
    onReject: () => void;
    changes: ChangeRequest[];
}

interface ChangeRequest {
    id: string;
    type: 'code_change' | 'validator_trigger';
    lines_changed: number;
    files_modified: number;
    dependencies_affected: number;
    risk_score: number;
    reason: string;
}

const ApprovalModal = ({
    isOpen,
    onClose,
    onApprove,
    onReject,
    changes
}: ApprovalModalProps) => {
    const [isBatchProcessing, setIsBatchProcessing] = useState(false);

    const totalLinesChanged = changes.reduce((sum, change) => sum + change.lines_changed, 0);
    const totalFilesModified = changes.reduce((sum, change) => sum + change.files_modified, 0);
    const totalDependenciesAffected = changes.reduce((sum, change) => sum + change.dependencies_affected, 0);
    const maxRiskScore = Math.max(...changes.map(c => c.risk_score), 0);
    const avgRiskScore = Math.round(changes.reduce((sum, c) => sum + c.risk_score, 0) / changes.length);

    const handleApprove = () => {
        setIsBatchProcessing(true);
        onApprove();
        setTimeout(() => setIsBatchProcessing(false), 500);
    };

    const handleReject = () => {
        setIsBatchProcessing(true);
        onReject();
        setTimeout(() => setIsBatchProcessing(false), 500);
    };

    const getRiskColor = (score: number) => {
        if (score < 30) return 'text-green-500';
        if (score < 60) return 'text-yellow-500';
        return 'text-red-500';
    };

    const getRiskLabel = (score: number) => {
        if (score < 30) return 'LOW';
        if (score < 60) return 'MEDIUM';
        return 'HIGH';
    };

    return (
        <AnimatePresence>
            {isOpen && (
                <>
                    {/* Backdrop */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50"
                        onClick={onClose}
                    />
                    
                    {/* Modal */}
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95, y: 20 }}
                        transition={{ duration: 0.2, ease: 'easeOut' }}
                        className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-[90vw] max-w-2xl"
                    >
                        <div className="bg-[#0a0a0a] border border-white/10 rounded-xl shadow-2xl overflow-hidden">
                            {/* Header */}
                            <div className="px-6 py-4 border-b border-white/10 bg-white/[0.02]">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <div className="p-2 bg-yellow-500/10 rounded-lg">
                                            <AlertTriangle className="w-5 h-5 text-[#facc15]" />
                                        </div>
                                        <div>
                                            <h2 className="text-white font-mono text-[15px] font-semibold">
                                                SPECTER APPROVAL REQUIRED
                                            </h2>
                                            <p className="text-white/40 text-[13px] font-mono mt-0.5">
                                                {changes.length} change{changes.length !== 1 ? 's' : ''} pending validation
                                            </p>
                                        </div>
                                    </div>
                                    <button 
                                        onClick={onClose}
                                        className="p-2 hover:bg-white/5 rounded-lg transition-colors text-white/40 hover:text-white"
                                    >
                                        <XCircle className="w-5 h-5" />
                                    </button>
                                </div>
                            </div>

                            {/* Content */}
                            <div className="p-6 space-y-6">
                                {/* Summary Metrics */}
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                    <div className="bg-white/[0.02] border border-white/5 rounded-lg p-3">
                                        <div className="flex items-center gap-2 text-white/40 text-[13px] mb-1">
                                            <FileCode className="w-3.5 h-3.5" />
                                            <span>Lines Changed</span>
                                        </div>
                                        <div className="text-white font-mono text-[15px]">
                                            {totalLinesChanged.toLocaleString()}
                                        </div>
                                    </div>
                                    
                                    <div className="bg-white/[0.02] border border-white/5 rounded-lg p-3">
                                        <div className="flex items-center gap-2 text-white/40 text-[13px] mb-1">
                                            <GitBranch className="w-3.5 h-3.5" />
                                            <span>Files Modified</span>
                                        </div>
                                        <div className="text-white font-mono text-[15px]">
                                            {totalFilesModified.toLocaleString()}
                                        </div>
                                    </div>
                                    
                                    <div className="bg-white/[0.02] border border-white/5 rounded-lg p-3">
                                        <div className="flex items-center gap-2 text-white/40 text-[13px] mb-1">
                                            <Package className="w-3.5 h-3.5" />
                                            <span>Dependencies</span>
                                        </div>
                                        <div className="text-white font-mono text-[15px]">
                                            {totalDependenciesAffected.toLocaleString()}
                                        </div>
                                    </div>
                                    
                                    <div className="bg-white/[0.02] border border-white/5 rounded-lg p-3">
                                        <div className="flex items-center gap-2 text-white/40 text-[13px] mb-1">
                                            <AlertTriangle className="w-3.5 h-3.5" />
                                            <span>Risk Score</span>
                                        </div>
                                        <div className={`font-mono text-[15px] ${getRiskColor(avgRiskScore)}`}>
                                            {avgRiskScore} ({getRiskLabel(avgRiskScore)})
                                        </div>
                                    </div>
                                </div>

                                {/* Change Details */}
                                <div className="space-y-3 max-h-[300px] overflow-y-auto pr-2 custom-scrollbar">
                                    {changes.map((change, index) => (
                                        <motion.div
                                            key={change.id}
                                            initial={{ opacity: 0, y: 10 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ delay: index * 0.05 }}
                                            className="bg-white/[0.02] border border-white/5 rounded-lg p-4"
                                        >
                                            <div className="flex items-start justify-between">
                                                <div className="flex-1">
                                                    <div className="flex items-center gap-2 mb-2">
                                                        <span className={`px-2 py-0.5 rounded text-[11px] font-mono ${
                                                            change.type === 'code_change' 
                                                                ? 'bg-blue-500/10 text-blue-400' 
                                                                : 'bg-yellow-500/10 text-[#facc15]'
                                                        }`}>
                                                            {change.type.replace('_', ' ').toUpperCase()}
                                                        </span>
                                                        <span className="text-white/40 text-[12px] font-mono">
                                                            ID: {change.id.substring(0, 8)}
                                                        </span>
                                                    </div>
                                                    
                                                    <p className="text-white/80 text-[13px] leading-relaxed">
                                                        {change.reason}
                                                    </p>
                                                </div>
                                                
                                                <div className="flex flex-col items-end gap-1 ml-3">
                                                    <div className={`text-[13px] font-mono ${getRiskColor(change.risk_score)}`}>
                                                        Risk: {change.risk_score}
                                                    </div>
                                                    <div className="text-white/40 text-[11px] font-mono">
                                                        {change.lines_changed} lines • {change.files_modified} files
                                                    </div>
                                                </div>
                                            </div>
                                        </motion.div>
                                    ))}
                                </div>

                                {/* Trigger Reason Highlight */}
                                <div className="bg-yellow-500/5 border border-yellow-500/10 rounded-lg p-4">
                                    <div className="flex items-start gap-3">
                                        <AlertTriangle className="w-4 h-4 text-[#facc15] mt-0.5 flex-shrink-0" />
                                        <div>
                                            <h4 className="text-[#facc15] text-[13px] font-mono font-semibold mb-1">
                                                TRIGGER REASON
                                            </h4>
                                            <p className="text-white/60 text-[13px] leading-relaxed">
                                                {changes.some(c => c.type === 'validator_trigger') 
                                                    ? "Validator detected changes exceeding safety thresholds. Manual approval required before proceeding."
                                                    : "Automated analysis identified significant changes requiring explicit user confirmation."}
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Footer */}
                            <div className="px-6 py-4 border-t border-white/10 bg-white/[0.02] flex justify-end gap-3">
                                <button
                                    onClick={handleReject}
                                    disabled={isBatchProcessing}
                                    className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-white/5 hover:bg-white/10 text-white/60 hover:text-white transition-all disabled:opacity-50 disabled:cursor-not-allowed font-mono text-[13px]"
                                >
                                    <XCircle className="w-4 h-4" />
                                    Reject
                                </button>
                                
                                <button
                                    onClick={handleApprove}
                                    disabled={isBatchProcessing}
                                    className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-[#facc15] hover:bg-[#facc15]/90 text-black font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed font-mono text-[13px] shadow-[0_0_15px_rgba(250,204,21,0.3)] hover:shadow-[0_0_20px_rgba(250,204,21,0.4)]"
                                >
                                    {isBatchProcessing ? (
                                        <>
                                            <div className="w-4 h-4 border-2 border-black/30 border-t-black rounded-full animate-spin" />
                                            Processing...
                                        </>
                                    ) : (
                                        <>
                                            <CheckCircle2 className="w-4 h-4" />
                                            Approve All
                                        </>
                                    )}
                                </button>
                            </div>
                        </div>
                    </motion.div>
                </>
            )}
        </AnimatePresence>
    );
};

export default ApprovalModal;
