/**
 * HistoryPanel — Detection history list with filtering, pagination & delete.
 *
 * Extracted from App.tsx to keep the main file focused and enable
 * future reuse (e.g. dashboard widgets, export views).
 */

import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
  History, Trash2, Scan, Camera, Volume2,
  Type, MessageSquare, FileCheck,
  ChevronLeft, ChevronRight, Loader2,
} from 'lucide-react';

// ── Types ──

export interface HistoryItem {
  id: number;
  detection_type: string;
  filename: string;
  risk_score: number;
  verdict: string;
  confidence: number;
  findings: string[];
  kenya_warnings: { type: string; severity: string; warning: string; action: string }[];
  details: Record<string, any>;
  created_at: string;
}

interface Pagination {
  page: number;
  per_page: number;
  total: number;
  pages: number;
}

// ── Helpers ──

export function getVerdictColor(verdict: string): string {
  if (verdict === 'AUTHENTIC' || verdict === 'APPEARS_GENUINE')
    return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20';
  if (verdict === 'LIKELY_DEEPFAKE' || verdict === 'LIKELY_MISINFORMATION')
    return 'text-rose-400 bg-rose-500/10 border-rose-500/20';
  return 'text-amber-400 bg-amber-500/10 border-amber-500/20';
}

export function getTypeIcon(type: string): React.ReactNode {
  switch (type) {
    case 'image':    return <Camera size={14} />;
    case 'audio':    return <Volume2 size={14} />;
    case 'text':     return <Type size={14} />;
    case 'forward':  return <MessageSquare size={14} />;
    case 'document': return <FileCheck size={14} />;
    default:         return <Scan size={14} />;
  }
}

export function formatDate(d: string | null): string {
  if (!d) return 'Never';
  try {
    return new Date(d).toLocaleDateString('en-KE', {
      year: 'numeric', month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });
  } catch {
    return d;
  }
}

function riskColor(score: number): string {
  if (score > 65) return 'text-rose-400';
  if (score > 40) return 'text-amber-400';
  return 'text-emerald-400';
}

// ── Component ──

const PER_PAGE = 20;

export default function HistoryPanel() {
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [filter, setFilter] = useState('');
  const [page, setPage] = useState(1);
  const [pagination, setPagination] = useState<Pagination | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ page: String(page), per_page: String(PER_PAGE) });
      if (filter) params.set('type', filter);
      const res = await fetch(`/api/history?${params}`, { credentials: 'include' });
      if (res.ok) {
        const data = await res.json();
        setItems(data.history ?? []);
        setPagination(data.pagination ?? null);
      }
    } catch (err) {
      console.error('Failed to load history:', err);
    } finally {
      setLoading(false);
    }
  }, [page, filter]);

  useEffect(() => { load(); }, [load]);

  // Reset to page 1 when filter changes
  useEffect(() => { setPage(1); }, [filter]);

  const handleDelete = async (id: number) => {
    try {
      await fetch(`/api/history/${id}`, { method: 'DELETE', credentials: 'include' });
      setItems(prev => prev.filter(h => h.id !== id));
      if (pagination) setPagination({ ...pagination, total: pagination.total - 1 });
    } catch { /* swallow */ }
  };

  return (
    <motion.div
      key="history-panel"
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 20 }}
      className="analysis-card"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-bold text-white flex items-center gap-2">
          <History size={20} className="text-violet-400" />
          Detection History
          {pagination && (
            <span className="ml-2 text-xs font-normal text-slate-500">
              {pagination.total} total
            </span>
          )}
        </h3>

        <select
          value={filter}
          onChange={e => setFilter(e.target.value)}
          className="bg-white/[0.04] border border-white/[0.08] rounded-xl px-3 py-2 text-sm text-slate-300 outline-none focus:border-violet-500/40"
        >
          <option value="">All Types</option>
          <option value="image">Image</option>
          <option value="audio">Audio</option>
          <option value="text">Text</option>
          <option value="forward">WhatsApp</option>
          <option value="document">Document</option>
        </select>
      </div>

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 size={28} className="animate-spin text-violet-400" />
        </div>
      )}

      {/* Empty */}
      {!loading && items.length === 0 && (
        <div className="text-center py-16">
          <Scan size={48} className="text-slate-700 mx-auto mb-4" />
          <h4 className="text-lg font-bold text-slate-400 mb-2">No Scans Yet</h4>
          <p className="text-sm text-slate-600">
            Your detection history will appear here after you analyze content.
          </p>
        </div>
      )}

      {/* Items */}
      {!loading && items.length > 0 && (
        <div className="space-y-3">
          {items.map((item, i) => (
            <motion.div
              key={item.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.03 }}
              className="flex flex-col sm:flex-row items-start sm:items-center gap-2.5 sm:gap-4 p-3 sm:p-4 bg-white/[0.02] border border-white/[0.04] rounded-xl hover:bg-white/[0.04] transition-all duration-300 group"
            >
              {/* Icon */}
              <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-white/[0.04] border border-white/[0.06] flex items-center justify-center text-slate-400 flex-shrink-0">
                {getTypeIcon(item.detection_type)}
              </div>

              {/* Details */}
              <div className="flex-1 min-w-0 w-full">
                <div className="flex items-center gap-2 mb-1 flex-wrap">
                  <span className="text-xs sm:text-sm font-semibold text-white truncate max-w-[150px] sm:max-w-none">
                    {item.filename}
                  </span>
                  <span className={`text-[10px] px-2 py-0.5 rounded-full font-semibold uppercase border ${getVerdictColor(item.verdict)}`}>
                    {item.verdict.replace(/_/g, ' ')}
                  </span>
                </div>
                <div className="flex items-center flex-wrap gap-2 sm:gap-4 text-[10px] sm:text-xs text-slate-500">
                  <span className="capitalize">{item.detection_type}</span>
                  <span>Risk: {item.risk_score?.toFixed(1)}%</span>
                  <span className="hidden sm:inline">{formatDate(item.created_at)}</span>
                </div>
                <div className="sm:hidden text-[10px] text-slate-600 mt-1">
                  {formatDate(item.created_at)}
                </div>
              </div>

              {/* Score + delete */}
              <div className="flex items-center gap-2 self-end sm:self-center">
                <div className={`text-base sm:text-lg font-bold ${riskColor(item.risk_score)}`}>
                  {item.risk_score?.toFixed(0)}
                </div>
                <button
                  onClick={() => handleDelete(item.id)}
                  className="p-1.5 rounded-lg text-slate-600 hover:text-rose-400 hover:bg-rose-500/10 transition-all sm:opacity-0 sm:group-hover:opacity-100"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            </motion.div>
          ))}
        </div>
      )}

      {/* Pagination */}
      {pagination && pagination.pages > 1 && (
        <div className="flex items-center justify-between mt-6 pt-4 border-t border-white/[0.06]">
          <button
            disabled={page <= 1}
            onClick={() => setPage(p => Math.max(1, p - 1))}
            className="flex items-center gap-1 text-xs text-slate-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            <ChevronLeft size={14} /> Previous
          </button>
          <span className="text-xs text-slate-500">
            Page {page} of {pagination.pages}
          </span>
          <button
            disabled={page >= pagination.pages}
            onClick={() => setPage(p => p + 1)}
            className="flex items-center gap-1 text-xs text-slate-400 hover:text-white disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            Next <ChevronRight size={14} />
          </button>
        </div>
      )}
    </motion.div>
  );
}
