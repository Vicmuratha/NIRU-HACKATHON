import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import { motion, AnimatePresence, useMotionValue, useTransform } from 'framer-motion';
import {
  Shield, ShieldCheck, ShieldAlert, ShieldX,
  Upload, Image, Mic, FileText, Zap, Eye,
  ChevronRight, X, Check, AlertTriangle, Info,
  BarChart3, Lock, Globe, Cpu, Sparkles,
  Menu, ArrowRight, ExternalLink, Activity,
  Sun, Moon, Volume2, Type, Camera, Search,
  TrendingUp, Users, FileWarning, BrainCircuit,
  Scan, Fingerprint, AudioLines, Bot,
  CircleCheck, CircleX, CircleDot, Loader2
} from 'lucide-react';

// ─── TYPES ───
interface AnalysisResult {
  risk_score: number;
  verdict: string;
  confidence: number;
  findings: string[];
  kenya_warnings?: { type: string; severity: string; warning: string; action: string }[];
  details?: Record<string, any>;
  is_authentic?: boolean;
}

type AnalysisTab = 'image' | 'audio' | 'text';
type AppView = 'home' | 'analyze';

// ─── API URL ───
const API_BASE = '/api';

// ─── PARTICLES ───
const Particles: React.FC = () => {
  const particles = useMemo(() =>
    Array.from({ length: 40 }, (_, i) => ({
      id: i,
      left: Math.random() * 100,
      size: Math.random() * 3 + 1,
      delay: Math.random() * 15,
      duration: Math.random() * 10 + 15,
      color: ['#8b5cf6', '#06b6d4', '#f43f5e', '#10b981'][Math.floor(Math.random() * 4)],
      opacity: Math.random() * 0.4 + 0.1,
    })), []);

  return (
    <div className="fixed inset-0 z-0 pointer-events-none overflow-hidden">
      {particles.map(p => (
        <div
          key={p.id}
          className="particle"
          style={{
            left: `${p.left}%`,
            width: p.size,
            height: p.size,
            background: p.color,
            opacity: p.opacity,
            animationDelay: `${p.delay}s`,
            animationDuration: `${p.duration}s`,
            boxShadow: `0 0 ${p.size * 3}px ${p.color}`,
          }}
        />
      ))}
    </div>
  );
};

// ─── ANIMATED COUNTER ───
const AnimatedCounter: React.FC<{ value: number; suffix?: string; decimals?: number }> = ({
  value, suffix = '', decimals = 0
}) => {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    let start = 0;
    const duration = 1500;
    const startTime = performance.now();
    const step = (time: number) => {
      const elapsed = time - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(eased * value);
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, [value]);
  return <span className="stat-number">{display.toFixed(decimals)}{suffix}</span>;
};

// ─── RISK GAUGE ───
const RiskGauge: React.FC<{ score: number; size?: number }> = ({ score, size = 180 }) => {
  const radius = (size - 20) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;
  const color = score < 40 ? '#10b981' : score < 65 ? '#fbbf24' : '#f43f5e';

  return (
    <div className="risk-gauge" style={{ width: size, height: size }}>
      <svg width={size} height={size}>
        <circle cx={size / 2} cy={size / 2} r={radius} className="risk-gauge-circle risk-gauge-bg" strokeWidth="8" />
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          className="risk-gauge-circle risk-gauge-fill"
          strokeWidth="8"
          stroke={color}
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1.5, ease: [0.4, 0, 0.2, 1] }}
          style={{ '--gauge-color': color } as React.CSSProperties}
        />
      </svg>
      <div className="risk-gauge-label">
        <motion.span
          className="text-3xl font-bold"
          style={{ color }}
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.5, type: 'spring', stiffness: 200 }}
        >
          <AnimatedCounter value={score} decimals={1} />
        </motion.span>
        <span className="text-xs text-slate-400 mt-1 uppercase tracking-wider">Risk Score</span>
      </div>
    </div>
  );
};

// ─── VERDICT BADGE ───
const VerdictBadge: React.FC<{ verdict: string }> = ({ verdict }) => {
  const config: Record<string, { icon: React.ReactNode; class: string; label: string }> = {
    AUTHENTIC: { icon: <CircleCheck size={18} />, class: 'badge-authentic', label: 'Authentic' },
    LIKELY_DEEPFAKE: { icon: <CircleX size={18} />, class: 'badge-deepfake', label: 'Likely Deepfake' },
    REVIEW_REQUIRED: { icon: <CircleDot size={18} />, class: 'badge-review', label: 'Review Required' },
  };
  const c = config[verdict] || config.REVIEW_REQUIRED;
  return (
    <motion.div
      initial={{ scale: 0, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: 'spring', stiffness: 300, delay: 0.3 }}
      className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-semibold ${c.class}`}
    >
      {c.icon}
      {c.label}
    </motion.div>
  );
};

// ─── NAVBAR ───
const Navbar: React.FC<{ view: AppView; setView: (v: AppView) => void }> = ({ view, setView }) => {
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const handler = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handler);
    return () => window.removeEventListener('scroll', handler);
  }, []);

  return (
    <motion.nav
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ type: 'spring', stiffness: 100, damping: 20 }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
        scrolled ? 'nav-glass shadow-2xl' : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between relative">
        <div className="flex items-center gap-3 cursor-pointer" onClick={() => setView('home')}>
          <div className="relative">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-violet-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-violet-500/25">
              <Shield size={18} className="text-white" />
            </div>
            <div className="absolute -top-0.5 -right-0.5 w-3 h-3 rounded-full bg-emerald-400 border-2 border-[#050816] pulse-dot" />
          </div>
          <span className="text-lg font-bold tracking-tight">
            Safe<span className="text-aurora">Eye</span>
          </span>
        </div>

        <div className="hidden md:flex items-center gap-1 bg-white/[0.03] rounded-xl p-1 border border-white/[0.06] absolute left-1/2 -translate-x-1/2">
          {(['home', 'analyze'] as AppView[]).map(v => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`px-5 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${
                view === v
                  ? 'bg-violet-500/15 text-violet-300 shadow-inner'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-white/[0.04]'
              }`}
            >
              {v === 'home' ? 'Home' : 'Analyze'}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-3">
          <div className="hidden sm:flex items-center gap-2 text-xs text-slate-500 bg-white/[0.03] px-3 py-1.5 rounded-full border border-white/[0.06]">
            <Activity size={12} className="text-emerald-400" />
            <span>System Online</span>
          </div>
          <button
            onClick={() => setView('analyze')}
            className="btn-glow text-sm !px-5 !py-2.5"
          >
            <span className="flex items-center gap-2">
              <Scan size={14} />
              Scan Now
            </span>
          </button>
        </div>
      </div>
    </motion.nav>
  );
};

// ─── HERO SECTION ───
const HeroSection: React.FC<{ onAnalyze: () => void }> = ({ onAnalyze }) => {
  const stats = [
    { label: 'Detection Accuracy', value: 99.2, suffix: '%', icon: <Zap size={14} /> },
    { label: 'Scans Performed', value: 14820, suffix: '+', icon: <BarChart3 size={14} /> },
    { label: 'Threats Blocked', value: 3650, suffix: '+', icon: <ShieldCheck size={14} /> },
    { label: 'Response Time', value: 1.2, suffix: 's', icon: <Activity size={14} /> },
  ];

  return (
    <section className="relative min-h-screen flex items-center justify-center pt-16">
      {/* Radial accent blur */}
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-violet-600/[0.07] blur-[120px] pointer-events-none" />
      <div className="absolute top-1/2 right-1/4 w-[400px] h-[400px] rounded-full bg-cyan-500/[0.05] blur-[100px] pointer-events-none" />

      <div className="relative z-10 max-w-6xl mx-auto px-6 text-center">
        {/* Top badge */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="inline-flex items-center gap-2 mb-8 px-4 py-2 rounded-full bg-violet-500/[0.08] border border-violet-500/20 text-violet-300 text-sm"
        >
          <Sparkles size={14} className="text-violet-400" />
          AI-Powered Deepfake Detection Platform
          <ChevronRight size={14} />
        </motion.div>

        {/* Heading */}
        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.15 }}
          className="text-5xl md:text-7xl font-extrabold leading-tight mb-6"
        >
          Defend Against{' '}
          <span className="text-aurora">Digital</span>
          <br />
          <span className="text-aurora">Deception</span>
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="text-lg md:text-xl text-slate-400 max-w-2xl mx-auto mb-10 leading-relaxed"
        >
          Instantly detect deepfake images, cloned voices, and AI-generated text
          with military-grade neural analysis. Protect truth in the age of synthetic media.
        </motion.p>

        {/* CTA Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.45 }}
          className="flex flex-wrap items-center justify-center gap-4 mb-16"
        >
          <button onClick={onAnalyze} className="btn-glow text-base !px-8 !py-3.5">
            <span className="flex items-center gap-2">
              <Scan size={18} />
              Start Analyzing
              <ArrowRight size={16} />
            </span>
          </button>
          <button className="btn-outline-glow text-base !px-8 !py-3.5 flex items-center gap-2">
            <Eye size={18} />
            Watch Demo
          </button>
        </motion.div>

        {/* Stats Grid */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-3xl mx-auto"
        >
          {stats.map((stat, i) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.7 + i * 0.1 }}
              className="glass-card p-4 text-center"
            >
              <div className="flex items-center justify-center gap-1.5 text-violet-400 mb-2">
                {stat.icon}
                <span className="text-2xl font-bold text-white">
                  <AnimatedCounter value={stat.value} suffix={stat.suffix} decimals={stat.value % 1 !== 0 ? 1 : 0} />
                </span>
              </div>
              <div className="text-xs text-slate-500 uppercase tracking-wider">{stat.label}</div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};

// ─── FEATURES SECTION ───
const FeaturesSection: React.FC = () => {
  const features = [
    {
      icon: <Camera size={24} />,
      title: 'Image Forensics',
      desc: 'Multi-layered analysis: AI classification, Error Level Analysis, metadata verification, face texture mapping, and noise spectrum analysis.',
      gradient: 'from-violet-500 to-purple-600',
      glow: 'rgba(139, 92, 246, 0.2)',
    },
    {
      icon: <AudioLines size={24} />,
      title: 'Voice Authentication',
      desc: 'MFCC analysis, silence pattern detection, spectral analysis to identify cloned or synthesized voices with high precision.',
      gradient: 'from-cyan-500 to-blue-600',
      glow: 'rgba(6, 182, 212, 0.2)',
    },
    {
      icon: <Type size={24} />,
      title: 'Text Verification',
      desc: 'RoBERTa-based fake news classification with clickbait detection, source credibility scoring, and semantic consistency checks.',
      gradient: 'from-rose-500 to-pink-600',
      glow: 'rgba(244, 63, 94, 0.2)',
    },
    {
      icon: <BrainCircuit size={24} />,
      title: 'Neural Architecture',
      desc: 'State-of-the-art transformer models fine-tuned on millions of deepfake samples for unmatched accuracy.',
      gradient: 'from-emerald-500 to-teal-600',
      glow: 'rgba(16, 185, 129, 0.2)',
    },
    {
      icon: <Fingerprint size={24} />,
      title: 'Digital Forensics',
      desc: 'Deep pixel-level analysis examining compression artifacts, EXIF metadata, and generation fingerprints.',
      gradient: 'from-amber-500 to-orange-600',
      glow: 'rgba(245, 158, 11, 0.2)',
    },
    {
      icon: <Lock size={24} />,
      title: 'Privacy First',
      desc: 'All analysis runs on-device. Your uploads are processed and immediately deleted. Zero data retention.',
      gradient: 'from-indigo-500 to-violet-600',
      glow: 'rgba(99, 102, 241, 0.2)',
    },
  ];

  return (
    <section className="relative py-32 px-6">
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[500px] h-[500px] rounded-full bg-violet-600/[0.04] blur-[120px] pointer-events-none" />
      <div className="max-w-6xl mx-auto relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <div className="inline-flex items-center gap-2 mb-4 px-3 py-1.5 rounded-full bg-white/[0.04] border border-white/[0.08] text-xs text-slate-400 uppercase tracking-wider">
            <Cpu size={12} /> Capabilities
          </div>
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            Multi-Modal <span className="text-aurora">Detection Engine</span>
          </h2>
          <p className="text-slate-400 max-w-xl mx-auto">
            Six specialized AI modules working in concert to deliver forensic-grade authenticity verification.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
          {features.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.08 }}
              className="glass-card glow-border feature-card-glow p-6 group cursor-default"
              onMouseMove={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                e.currentTarget.style.setProperty('--mouse-x', `${e.clientX - rect.left}px`);
                e.currentTarget.style.setProperty('--mouse-y', `${e.clientY - rect.top}px`);
              }}
            >
              <div
                className={`w-12 h-12 rounded-xl bg-gradient-to-br ${f.gradient} flex items-center justify-center mb-4 shadow-lg group-hover:scale-110 transition-transform duration-300`}
                style={{ boxShadow: `0 8px 32px ${f.glow}` }}
              >
                {f.icon}
              </div>
              <h3 className="text-lg font-semibold mb-2 text-white">{f.title}</h3>
              <p className="text-sm text-slate-400 leading-relaxed">{f.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

// ─── ANALYSIS PANEL ───
const AnalysisPanel: React.FC = () => {
  const [tab, setTab] = useState<AnalysisTab>('image');
  const [file, setFile] = useState<File | null>(null);
  const [textInput, setTextInput] = useState('');
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [scanProgress, setScanProgress] = useState(0);
  const fileRef = useRef<HTMLInputElement>(null);

  const tabs: { key: AnalysisTab; label: string; icon: React.ReactNode; desc: string }[] = [
    { key: 'image', label: 'Image', icon: <Camera size={18} />, desc: 'Upload an image to check for deepfake manipulation' },
    { key: 'audio', label: 'Audio', icon: <Volume2 size={18} />, desc: 'Upload an audio clip to detect voice cloning' },
    { key: 'text', label: 'Text', icon: <Type size={18} />, desc: 'Paste text content to verify authenticity' },
  ];

  const resetState = useCallback(() => {
    setFile(null);
    setTextInput('');
    setResult(null);
    setError(null);
    setPreviewUrl(null);
    setScanProgress(0);
  }, []);

  const handleTabChange = useCallback((t: AnalysisTab) => {
    setTab(t);
    resetState();
  }, [resetState]);

  const handleFile = useCallback((f: File) => {
    setFile(f);
    setResult(null);
    setError(null);
    if (tab === 'image' && f.type.startsWith('image/')) {
      setPreviewUrl(URL.createObjectURL(f));
    } else {
      setPreviewUrl(null);
    }
  }, [tab]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const analyze = useCallback(async () => {
    setLoading(true);
    setResult(null);
    setError(null);
    setScanProgress(0);

    // Simulate progress
    const progressInterval = setInterval(() => {
      setScanProgress(prev => {
        if (prev >= 90) { clearInterval(progressInterval); return 90; }
        return prev + Math.random() * 15;
      });
    }, 300);

    try {
      let res: Response;
      if (tab === 'text') {
        res = await fetch(`${API_BASE}/analyze/text`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: textInput }),
        });
      } else {
        const fd = new FormData();
        fd.append('file', file!);
        res = await fetch(`${API_BASE}/analyze/${tab}`, { method: 'POST', body: fd });
      }
      if (!res.ok) throw new Error(`Analysis failed (${res.status})`);
      const data = await res.json();
      clearInterval(progressInterval);
      setScanProgress(100);
      setTimeout(() => setResult(data), 400);
    } catch (err: any) {
      clearInterval(progressInterval);
      setScanProgress(0);
      setError(err.message || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  }, [tab, file, textInput]);

  const canAnalyze = tab === 'text' ? textInput.trim().length > 10 : !!file;

  const tabConfig = tabs.find(t => t.key === tab)!;

  return (
    <section id="analyze" className="relative min-h-screen pt-24 pb-20 px-6">
      <div className="absolute top-20 right-1/4 w-[400px] h-[400px] rounded-full bg-cyan-500/[0.04] blur-[100px] pointer-events-none" />
      <div className="max-w-5xl mx-auto relative z-10">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <div className="inline-flex items-center gap-2 mb-4 px-3 py-1.5 rounded-full bg-white/[0.04] border border-white/[0.08] text-xs text-slate-400 uppercase tracking-wider">
            <Scan size={12} /> Detection Console
          </div>
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            Analyze <span className="text-aurora">Content</span>
          </h2>
          <p className="text-slate-400 max-w-lg mx-auto">
            Upload media or paste text to run our multi-model forensic analysis pipeline.
          </p>
        </motion.div>

        {/* Tab Selector */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="flex justify-center mb-8"
        >
          <div className="inline-flex gap-2 p-1.5 rounded-2xl bg-white/[0.03] border border-white/[0.06]">
            {tabs.map(t => (
              <button
                key={t.key}
                onClick={() => handleTabChange(t.key)}
                className={`flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-medium transition-all duration-300 ${
                  tab === t.key ? 'tab-active' : 'text-slate-400 hover:text-slate-200 hover:bg-white/[0.04]'
                }`}
              >
                {t.icon}
                {t.label}
              </button>
            ))}
          </div>
        </motion.div>

        {/* Main Analysis Card */}
        <motion.div
          layout
          className="glass-card p-8 overflow-hidden"
        >
          <AnimatePresence mode="wait">
            {!result ? (
              <motion.div
                key="input"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.3 }}
              >
                {/* Description */}
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500/20 to-cyan-500/20 border border-violet-500/20 flex items-center justify-center">
                    {tabConfig.icon}
                  </div>
                  <div>
                    <h3 className="font-semibold text-white">{tabConfig.label} Analysis</h3>
                    <p className="text-sm text-slate-400">{tabConfig.desc}</p>
                  </div>
                </div>

                {/* Upload or Text Input */}
                {tab !== 'text' ? (
                  <>
                    <input
                      ref={fileRef}
                      type="file"
                      accept={tab === 'image' ? 'image/*' : 'audio/*'}
                      className="hidden"
                      onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])}
                    />
                    <div
                      className={`upload-zone ${dragging ? 'dragging' : ''} ${file ? 'border-violet-500/40' : ''}`}
                      onDragOver={e => { e.preventDefault(); setDragging(true); }}
                      onDragLeave={() => setDragging(false)}
                      onDrop={onDrop}
                      onClick={() => fileRef.current?.click()}
                    >
                      {/* Scan line animation when loading */}
                      {loading && <div className="scan-line" />}

                      {previewUrl && tab === 'image' ? (
                        <div className="relative max-w-sm mx-auto">
                          <img
                            src={previewUrl}
                            alt="Preview"
                            className="rounded-xl max-h-64 mx-auto object-contain shadow-2xl"
                          />
                          {loading && (
                            <div className="absolute inset-0 bg-black/40 rounded-xl flex items-center justify-center">
                              <div className="spinner" />
                            </div>
                          )}
                        </div>
                      ) : file ? (
                        <div className="flex flex-col items-center gap-3 py-4">
                          <div className="w-16 h-16 rounded-2xl bg-violet-500/10 border border-violet-500/20 flex items-center justify-center">
                            {tab === 'image' ? <Image size={28} className="text-violet-400" /> : <Mic size={28} className="text-cyan-400" />}
                          </div>
                          <div>
                            <div className="font-medium text-white">{file.name}</div>
                            <div className="text-sm text-slate-400">{(file.size / 1024).toFixed(1)} KB</div>
                          </div>
                          <button
                            onClick={e => { e.stopPropagation(); resetState(); }}
                            className="text-xs text-slate-500 hover:text-rose-400 transition-colors flex items-center gap-1"
                          >
                            <X size={12} /> Remove
                          </button>
                        </div>
                      ) : (
                        <div className="flex flex-col items-center gap-4 py-6 relative z-10">
                          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-500/10 to-cyan-500/10 border border-white/[0.08] flex items-center justify-center">
                            <Upload size={28} className="text-violet-400" />
                          </div>
                          <div>
                            <p className="text-white font-medium mb-1">
                              Drop your {tab === 'image' ? 'image' : 'audio file'} here
                            </p>
                            <p className="text-sm text-slate-500">
                              or click to browse &middot; Max 50 MB
                            </p>
                          </div>
                          <div className="flex gap-2 text-xs text-slate-500">
                            {tab === 'image'
                              ? ['PNG', 'JPG', 'WEBP', 'BMP'].map(f => (
                                  <span key={f} className="px-2 py-0.5 rounded bg-white/[0.04] border border-white/[0.06]">{f}</span>
                                ))
                              : ['WAV', 'MP3', 'FLAC', 'OGG'].map(f => (
                                  <span key={f} className="px-2 py-0.5 rounded bg-white/[0.04] border border-white/[0.06]">{f}</span>
                                ))
                            }
                          </div>
                        </div>
                      )}
                    </div>
                  </>
                ) : (
                  <div className="relative">
                    <textarea
                      value={textInput}
                      onChange={e => setTextInput(e.target.value)}
                      placeholder="Paste the text content you want to verify for authenticity..."
                      rows={8}
                      className="w-full bg-white/[0.04] border border-white/[0.08] rounded-2xl p-5 text-white placeholder:text-slate-600 focus:outline-none focus:border-violet-500/40 focus:ring-2 focus:ring-violet-500/10 resize-none transition-all duration-300 text-sm leading-relaxed"
                    />
                    <div className="absolute bottom-3 right-3 text-xs text-slate-600">
                      {textInput.length} characters
                    </div>
                  </div>
                )}

                {/* Progress Bar */}
                {loading && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="mt-6"
                  >
                    <div className="flex justify-between text-xs text-slate-400 mb-2">
                      <span className="flex items-center gap-2">
                        <Loader2 size={12} className="animate-spin" />
                        Analyzing with neural models...
                      </span>
                      <span>{Math.round(scanProgress)}%</span>
                    </div>
                    <div className="h-1.5 bg-white/[0.04] rounded-full overflow-hidden">
                      <motion.div
                        className="h-full rounded-full bg-gradient-to-r from-violet-500 to-cyan-400"
                        style={{ width: `${scanProgress}%` }}
                        transition={{ duration: 0.3 }}
                      />
                    </div>
                  </motion.div>
                )}

                {/* Error */}
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-4 flex items-center gap-3 p-4 rounded-xl bg-rose-500/[0.08] border border-rose-500/20 text-rose-300 text-sm"
                  >
                    <AlertTriangle size={18} />
                    {error}
                  </motion.div>
                )}

                {/* Analyze Button */}
                <motion.button
                  whileHover={{ scale: 1.01 }}
                  whileTap={{ scale: 0.99 }}
                  onClick={analyze}
                  disabled={!canAnalyze || loading}
                  className={`w-full mt-6 py-4 rounded-2xl font-semibold text-base transition-all duration-300 flex items-center justify-center gap-3 ${
                    canAnalyze && !loading
                      ? 'bg-gradient-to-r from-violet-600 to-violet-500 hover:from-violet-500 hover:to-violet-400 text-white shadow-xl shadow-violet-500/25 hover:shadow-violet-500/40'
                      : 'bg-white/[0.04] text-slate-600 cursor-not-allowed'
                  }`}
                >
                  {loading ? (
                    <>
                      <Loader2 size={20} className="animate-spin" />
                      Running Forensic Analysis...
                    </>
                  ) : (
                    <>
                      <Scan size={20} />
                      Run Deep Analysis
                    </>
                  )}
                </motion.button>
              </motion.div>
            ) : (
              /* ─── RESULTS VIEW ─── */
              <motion.div
                key="results"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.4 }}
              >
                {/* Back button */}
                <button
                  onClick={resetState}
                  className="flex items-center gap-2 text-sm text-slate-400 hover:text-violet-300 transition-colors mb-6"
                >
                  <ChevronRight size={16} className="rotate-180" />
                  New Analysis
                </button>

                {/* Results Header */}
                <div className="flex flex-col md:flex-row items-center gap-8 mb-8">
                  {/* Risk Gauge */}
                  <motion.div
                    initial={{ scale: 0.5, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ type: 'spring', stiffness: 150 }}
                  >
                    <RiskGauge score={result.risk_score} />
                  </motion.div>

                  {/* Verdict & Confidence */}
                  <div className="flex-1 text-center md:text-left">
                    <VerdictBadge verdict={result.verdict || (result.is_authentic ? 'AUTHENTIC' : 'LIKELY_DEEPFAKE')} />
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.5 }}
                      className="mt-4"
                    >
                      <div className="text-sm text-slate-400 mb-1">Confidence Level</div>
                      <div className="flex items-center gap-3">
                        <div className="flex-1 h-2 bg-white/[0.04] rounded-full overflow-hidden max-w-xs">
                          <motion.div
                            className="h-full rounded-full bg-gradient-to-r from-violet-500 to-cyan-400"
                            initial={{ width: 0 }}
                            animate={{ width: `${(result.confidence || 0) * 100}%` }}
                            transition={{ duration: 1, delay: 0.6 }}
                          />
                        </div>
                        <span className="text-sm font-semibold text-white">
                          {((result.confidence || 0) * 100).toFixed(0)}%
                        </span>
                      </div>
                    </motion.div>

                    {/* Quick stats */}
                    {result.details && (
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.7 }}
                        className="flex flex-wrap gap-3 mt-4"
                      >
                        {Object.entries(result.details).map(([key, val]) => (
                          <div key={key} className="px-3 py-1.5 rounded-lg bg-white/[0.04] border border-white/[0.06] text-xs text-slate-400">
                            <span className="text-slate-500">{key.replace(/_/g, ' ')}: </span>
                            <span className="text-white font-medium">{typeof val === 'number' ? val.toFixed(1) : String(val)}</span>
                          </div>
                        ))}
                      </motion.div>
                    )}
                  </div>
                </div>

                {/* Findings */}
                {result.findings && result.findings.length > 0 && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.6 }}
                  >
                    <h4 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-2">
                      <Search size={14} /> Forensic Findings
                    </h4>
                    <div className="grid gap-2">
                      {result.findings.map((f, i) => (
                        <motion.div
                          key={i}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: 0.7 + i * 0.1 }}
                          className="finding-item"
                        >
                          {f}
                        </motion.div>
                      ))}
                    </div>
                  </motion.div>
                )}

                {/* Kenya Warnings */}
                {result.kenya_warnings && result.kenya_warnings.length > 0 && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.9 }}
                    className="mt-6"
                  >
                    <h4 className="text-sm font-semibold uppercase tracking-wider text-rose-400 mb-3 flex items-center gap-2">
                      <ShieldAlert size={14} /> Threat Alerts
                    </h4>
                    {result.kenya_warnings.map((w, i) => (
                      <div
                        key={i}
                        className="p-4 rounded-xl bg-rose-500/[0.06] border border-rose-500/20 mb-2"
                      >
                        <div className="flex items-center gap-2 mb-1">
                          <AlertTriangle size={14} className="text-rose-400" />
                          <span className="font-semibold text-rose-300 text-sm">{w.type}</span>
                          <span className={`ml-auto text-xs px-2 py-0.5 rounded-full ${
                            w.severity === 'CRITICAL'
                              ? 'bg-rose-500/20 text-rose-300'
                              : 'bg-amber-500/20 text-amber-300'
                          }`}>
                            {w.severity}
                          </span>
                        </div>
                        <p className="text-sm text-slate-400">{w.warning}</p>
                        <p className="text-xs text-slate-500 mt-1">Action: {w.action}</p>
                      </div>
                    ))}
                  </motion.div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>
    </section>
  );
};

// ─── HOW IT WORKS SECTION ───
const HowItWorks: React.FC = () => {
  const steps = [
    { num: '01', title: 'Upload', desc: 'Drag and drop any image, audio, or paste text content.', icon: <Upload size={22} /> },
    { num: '02', title: 'Analyze', desc: 'Our AI pipeline runs 6+ forensic checks in parallel.', icon: <BrainCircuit size={22} /> },
    { num: '03', title: 'Detect', desc: 'Neural models classify authenticity with confidence scores.', icon: <Scan size={22} /> },
    { num: '04', title: 'Report', desc: 'Get actionable insights, risk scores, and threat alerts.', icon: <BarChart3 size={22} /> },
  ];

  return (
    <section className="relative py-32 px-6">
      <div className="max-w-5xl mx-auto relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <div className="inline-flex items-center gap-2 mb-4 px-3 py-1.5 rounded-full bg-white/[0.04] border border-white/[0.08] text-xs text-slate-400 uppercase tracking-wider">
            <Zap size={12} /> Workflow
          </div>
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            How It <span className="text-aurora">Works</span>
          </h2>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {steps.map((s, i) => (
            <motion.div
              key={s.num}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.12 }}
              className="relative"
            >
              {/* Connector line */}
              {i < steps.length - 1 && (
                <div className="hidden md:block absolute top-10 left-full w-full h-px bg-gradient-to-r from-violet-500/30 to-transparent z-0" />
              )}
              <div className="glass-card p-6 text-center relative z-10">
                <div className="text-3xl font-black text-violet-500/20 mb-3">{s.num}</div>
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-violet-500/20 to-cyan-500/20 border border-violet-500/20 flex items-center justify-center mx-auto mb-4 text-violet-400">
                  {s.icon}
                </div>
                <h3 className="font-semibold text-white mb-2">{s.title}</h3>
                <p className="text-sm text-slate-400">{s.desc}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

// ─── FOOTER ───
const Footer: React.FC = () => (
  <footer className="relative py-16 px-6 footer-glow">
    <div className="max-w-6xl mx-auto relative z-10">
      <div className="flex flex-col md:flex-row items-center justify-between gap-8">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-violet-500 to-cyan-400 flex items-center justify-center">
            <Shield size={18} className="text-white" />
          </div>
          <span className="text-lg font-bold">
            Safe<span className="text-aurora">Eye</span>
          </span>
        </div>
        <div className="flex items-center gap-6 text-sm text-slate-500">
          <span>AI-Powered Deepfake Detection</span>
          <span className="hidden sm:inline">&middot;</span>
          <span className="hidden sm:inline">Built for Jaseci Hackathon 2026</span>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-600">
          <Lock size={12} />
          Privacy-First &middot; Zero Data Retention
        </div>
      </div>
    </div>
  </footer>
);

// ─── MAIN APP ───
const App: React.FC = () => {
  const [view, setView] = useState<AppView>('home');

  return (
    <div className="relative min-h-screen">
      {/* Background layers */}
      <div className="mesh-bg" />
      <div className="grid-pattern" />
      <Particles />

      {/* Navigation */}
      <Navbar view={view} setView={setView} />

      {/* Content */}
      <AnimatePresence mode="wait">
        {view === 'home' ? (
          <motion.div
            key="home"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.4 }}
          >
            <HeroSection onAnalyze={() => setView('analyze')} />
            <HowItWorks />
            <FeaturesSection />
            <Footer />
          </motion.div>
        ) : (
          <motion.div
            key="analyze"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.4 }}
          >
            <AnalysisPanel />
            <Footer />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default App;
