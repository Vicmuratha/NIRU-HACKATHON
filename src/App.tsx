import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import { motion, AnimatePresence, useScroll, useTransform } from 'framer-motion';
import {
  Shield, ShieldCheck, ShieldAlert,
  Upload, Image, Mic, Zap, Eye,
  ChevronRight, X, AlertTriangle,
  BarChart3, Lock, Cpu,
  ArrowRight, Activity,
  Volume2, Type, Camera, Search,
  BrainCircuit,
  Scan, AudioLines,
  CircleCheck, CircleX, CircleDot, Loader2,
  MessageSquare, FileCheck, Flag, Scale,
  Sparkles, Globe, Users, ChevronDown, ExternalLink,
  CheckCircle2, Clock, TrendingUp, Menu, XIcon
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
  forward_analysis?: any;
  document_analysis?: any;
  screenshot_analysis?: any;
  kenya_audio_context?: any;
  detection_note?: string;
}

interface UserInfo {
  name: string;
  email: string;
  picture?: string | null;
}

type AnalysisTab = 'image' | 'audio' | 'text' | 'forward' | 'document';
type AppView = 'home' | 'analyze';

// ─── API URL ───
const API_BASE = '/api';

// ─── FLOATING PARTICLES ───
const Particles: React.FC = () => {
  const particles = useMemo(() =>
    Array.from({ length: 50 }, (_, i) => ({
      id: i,
      left: Math.random() * 100,
      size: Math.random() * 2.5 + 0.5,
      delay: Math.random() * 20,
      duration: Math.random() * 15 + 20,
      color: ['#8b5cf6', '#06b6d4', '#f43f5e', '#10b981', '#a78bfa'][Math.floor(Math.random() * 5)],
      opacity: Math.random() * 0.3 + 0.05,
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
            boxShadow: `0 0 ${p.size * 4}px ${p.color}`,
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
    const duration = 2000;
    const startTime = performance.now();
    const step = (time: number) => {
      const elapsed = time - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 4);
      setDisplay(eased * value);
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, [value]);
  return <span className="stat-number">{display.toFixed(decimals)}{suffix}</span>;
};

// ─── RISK GAUGE ───
const RiskGauge: React.FC<{ score: number; size?: number }> = ({ score, size = 200 }) => {
  const radius = (size - 24) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;
  const color = score < 35 ? '#10b981' : score < 60 ? '#fbbf24' : '#f43f5e';
  const bgRingColor = score < 35 ? 'rgba(16,185,129,0.08)' : score < 60 ? 'rgba(251,191,36,0.08)' : 'rgba(244,63,94,0.08)';

  return (
    <div className="risk-gauge" style={{ width: size, height: size }}>
      <svg width={size} height={size}>
        <defs>
          <filter id="gaugeGlow">
            <feGaussianBlur stdDeviation="4" result="coloredBlur" />
            <feMerge><feMergeNode in="coloredBlur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>
        <circle cx={size / 2} cy={size / 2} r={radius} className="risk-gauge-circle risk-gauge-bg" strokeWidth="10" style={{ stroke: bgRingColor } as React.CSSProperties} />
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          className="risk-gauge-circle risk-gauge-fill"
          strokeWidth="10"
          stroke={color}
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 2, ease: [0.4, 0, 0.2, 1] }}
          style={{ '--gauge-color': color, filter: 'url(#gaugeGlow)' } as React.CSSProperties}
        />
      </svg>
      <div className="risk-gauge-label">
        <motion.span
          className="text-4xl font-black tracking-tight"
          style={{ color }}
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.6, type: 'spring', stiffness: 200 }}
        >
          <AnimatedCounter value={score} decimals={1} />
        </motion.span>
        <span className="text-[11px] text-slate-500 mt-1 uppercase tracking-[0.2em] font-semibold">Risk Score</span>
      </div>
    </div>
  );
};

// ─── VERDICT BADGE ───
const VerdictBadge: React.FC<{ verdict: string }> = ({ verdict }) => {
  const config: Record<string, { icon: React.ReactNode; cls: string; label: string }> = {
    AUTHENTIC: { icon: <CircleCheck size={18} />, cls: 'badge-authentic', label: 'Authentic' },
    LIKELY_DEEPFAKE: { icon: <CircleX size={18} />, cls: 'badge-deepfake', label: 'Likely Deepfake' },
    REVIEW_REQUIRED: { icon: <CircleDot size={18} />, cls: 'badge-review', label: 'Review Required' },
  };
  const c = config[verdict] || config.REVIEW_REQUIRED;
  return (
    <motion.div
      initial={{ scale: 0, opacity: 0, rotate: -10 }}
      animate={{ scale: 1, opacity: 1, rotate: 0 }}
      transition={{ type: 'spring', stiffness: 300, delay: 0.3 }}
      className={`verdict-badge ${c.cls}`}
    >
      {c.icon}
      {c.label}
    </motion.div>
  );
};

// ─── SCROLL INDICATOR ───
const ScrollIndicator: React.FC = () => (
  <motion.div
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    transition={{ delay: 1.5 }}
    className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
  >
    <span className="text-[10px] uppercase tracking-[0.3em] text-slate-600 font-medium">Scroll</span>
    <motion.div
      animate={{ y: [0, 8, 0] }}
      transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
    >
      <ChevronDown size={16} className="text-slate-600" />
    </motion.div>
  </motion.div>
);

// ─── NAVBAR ───
const Navbar: React.FC<{ view: AppView; setView: (v: AppView) => void; user: UserInfo | null }> = ({ view, setView, user }) => {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handler = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handler);
    return () => window.removeEventListener('scroll', handler);
  }, []);

  const handleLogout = () => { window.location.href = '/logout'; };
  const getInitials = (name: string) => name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2);

  return (
    <motion.nav
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ type: 'spring', stiffness: 80, damping: 20 }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-700 ${
        scrolled ? 'nav-glass shadow-2xl shadow-black/20' : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 h-[72px] flex items-center justify-between relative">
        <div className="flex items-center gap-3 cursor-pointer group" onClick={() => setView('home')}>
          <div className="relative">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-violet-500/30 group-hover:shadow-violet-500/50 transition-all duration-500 group-hover:scale-105">
              <Shield size={20} className="text-white" />
            </div>
            <div className="absolute -top-0.5 -right-0.5 w-3 h-3 rounded-full bg-emerald-400 border-2 border-[#050816] pulse-dot" />
          </div>
          <span className="text-xl font-extrabold tracking-tight">
            Safe<span className="text-aurora">Eye</span>
          </span>
        </div>

        <div className="hidden md:flex items-center gap-1 nav-pill absolute left-1/2 -translate-x-1/2">
          {(['home', 'analyze'] as AppView[]).map(v => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`nav-pill-item ${view === v ? 'nav-pill-active' : ''}`}
            >
              {v === 'home' ? 'Home' : 'Analyze'}
              {view === v && (
                <motion.div layoutId="navIndicator" className="nav-pill-indicator" transition={{ type: 'spring', stiffness: 300, damping: 30 }} />
              )}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-3">
          <div className="hidden lg:flex items-center gap-2 text-xs text-slate-500 bg-white/[0.03] px-3 py-1.5 rounded-full border border-white/[0.06]">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            <span>All Systems Active</span>
          </div>

          <button
            className="md:hidden p-2 rounded-lg hover:bg-white/[0.06] transition-colors text-slate-300"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          >
            {mobileMenuOpen ? <XIcon size={20} /> : <Menu size={20} />}
          </button>

          {user ? (
            <div className="relative hidden md:block">
              <button
                onClick={() => setMenuOpen(!menuOpen)}
                className="flex items-center gap-2.5 bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.08] hover:border-violet-500/30 rounded-full pl-1.5 pr-4 py-1.5 transition-all duration-300"
              >
                {user.picture ? (
                  <img src={user.picture} alt={user.name} className="w-8 h-8 rounded-full ring-2 ring-violet-500/30" />
                ) : (
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-500 to-cyan-400 flex items-center justify-center text-[11px] font-bold text-white ring-2 ring-violet-500/30">
                    {getInitials(user.name)}
                  </div>
                )}
                <span className="hidden sm:block text-sm font-medium text-slate-300 max-w-[100px] truncate">{user.name.split(' ')[0]}</span>
              </button>

              <AnimatePresence>
                {menuOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: 8, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 8, scale: 0.95 }}
                    transition={{ duration: 0.15 }}
                    className="absolute right-0 top-full mt-2 w-60 bg-[#0c1220]/95 backdrop-blur-2xl border border-white/[0.08] rounded-2xl shadow-2xl shadow-black/50 overflow-hidden"
                  >
                    <div className="px-4 py-3.5 border-b border-white/[0.06]">
                      <div className="text-sm font-semibold text-slate-200 truncate">{user.name}</div>
                      <div className="text-xs text-slate-500 truncate mt-0.5">{user.email}</div>
                    </div>
                    <div className="p-2">
                      <button
                        onClick={handleLogout}
                        className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-sm text-rose-400 hover:bg-rose-500/10 transition-all duration-200"
                      >
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>
                        Sign out
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          ) : (
            <button
              onClick={() => setView('analyze')}
              className="hidden md:flex btn-glow text-sm !px-6 !py-2.5"
            >
              <span className="flex items-center gap-2">
                <Scan size={14} />
                Scan Now
              </span>
            </button>
          )}
        </div>
      </div>

      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden overflow-hidden border-t border-white/[0.06] bg-[#050816]/95 backdrop-blur-2xl"
          >
            <div className="p-4 flex flex-col gap-2">
              {(['home', 'analyze'] as AppView[]).map(v => (
                <button
                  key={v}
                  onClick={() => { setView(v); setMobileMenuOpen(false); }}
                  className={`px-4 py-3 rounded-xl text-sm font-medium text-left transition-all ${
                    view === v ? 'bg-violet-500/15 text-violet-300' : 'text-slate-400 hover:bg-white/[0.04]'
                  }`}
                >
                  {v === 'home' ? 'Home' : 'Analyze Content'}
                </button>
              ))}
              {user && (
                <button
                  onClick={handleLogout}
                  className="px-4 py-3 rounded-xl text-sm font-medium text-left text-rose-400 hover:bg-rose-500/10 transition-all mt-2 border-t border-white/[0.06] pt-4"
                >
                  Sign out
                </button>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.nav>
  );
};

// ─── HERO SECTION ───
const HeroSection: React.FC<{ onAnalyze: () => void }> = ({ onAnalyze }) => {
  const { scrollY } = useScroll();
  const heroY = useTransform(scrollY, [0, 500], [0, 150]);
  const heroOpacity = useTransform(scrollY, [0, 400], [1, 0]);

  const stats = [
    { label: 'Kenyans at Risk', value: 54, suffix: 'M+', icon: <Users size={16} />, color: 'text-rose-400' },
    { label: 'Detection Modes', value: 5, suffix: '', icon: <Cpu size={16} />, color: 'text-violet-400' },
    { label: 'Election Countdown', value: Math.max(0, Math.ceil((new Date('2027-08-10').getTime() - Date.now()) / (1000 * 60 * 60 * 24))), suffix: 'd', icon: <Clock size={16} />, color: 'text-cyan-400' },
    { label: 'Response Time', value: 1.2, suffix: 's', icon: <Zap size={16} />, color: 'text-emerald-400' },
  ];

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] rounded-full bg-violet-600/[0.06] blur-[150px] pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/4 w-[500px] h-[500px] rounded-full bg-cyan-500/[0.05] blur-[120px] pointer-events-none" />
      <div className="absolute top-1/3 left-1/4 w-[300px] h-[300px] rounded-full bg-rose-500/[0.03] blur-[100px] pointer-events-none" />

      <motion.div
        style={{ y: heroY, opacity: heroOpacity }}
        className="relative z-10 max-w-6xl mx-auto px-6 text-center pt-24"
      >
        <motion.div
          initial={{ opacity: 0, y: 20, filter: 'blur(10px)' }}
          animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
          transition={{ duration: 0.8 }}
          className="hero-badge"
        >
          <span className="text-base">🇰🇪</span>
          <span>Kenya's AI-Powered Election & Media Integrity Shield</span>
          <ChevronRight size={14} className="text-violet-400" />
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 40, filter: 'blur(10px)' }}
          animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
          transition={{ duration: 1, delay: 0.15 }}
          className="text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-black leading-[0.95] mb-8 tracking-tight"
        >
          Protect Kenya
          <br />
          From <span className="text-aurora">Digital</span>
          <br />
          <span className="text-aurora">Deception</span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20, filter: 'blur(10px)' }}
          animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
          transition={{ duration: 0.8, delay: 0.35 }}
          className="text-base sm:text-lg md:text-xl text-slate-400 max-w-2xl mx-auto mb-4 leading-relaxed"
        >
          Detect deepfake images, manipulated audio, fake news screenshots,
          and WhatsApp misinformation — built for Kenya's unique threat landscape.
        </motion.p>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="text-sm text-slate-500/80 italic mb-12 font-light"
        >
          Kulinda Ukweli wa Kidijitali — Protecting Digital Truth
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.55 }}
          className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-20"
        >
          <button onClick={onAnalyze} className="btn-glow text-base !px-10 !py-4 group">
            <span className="flex items-center gap-3">
              <Scan size={20} />
              Start Analyzing
              <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform duration-300" />
            </span>
          </button>
          <button className="btn-outline-glow text-base !px-10 !py-4 flex items-center gap-3">
            <Eye size={20} />
            Watch Demo
          </button>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.7 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4 max-w-3xl mx-auto"
        >
          {stats.map((stat, i) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, scale: 0.8, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              transition={{ delay: 0.8 + i * 0.1, type: 'spring', stiffness: 200 }}
              className="stat-card group"
            >
              <div className={`flex items-center justify-center gap-2 mb-2 ${stat.color}`}>
                {stat.icon}
                <span className="text-2xl sm:text-3xl font-black text-white tracking-tight">
                  <AnimatedCounter value={stat.value} suffix={stat.suffix} decimals={stat.value % 1 !== 0 ? 1 : 0} />
                </span>
              </div>
              <div className="text-[10px] sm:text-xs text-slate-500 uppercase tracking-[0.15em] font-medium">{stat.label}</div>
            </motion.div>
          ))}
        </motion.div>
      </motion.div>

      <ScrollIndicator />
    </section>
  );
};

// ─── TRUSTED BY ───
const TrustedBySection: React.FC = () => (
  <section className="relative py-12 px-6 border-y border-white/[0.03]">
    <div className="max-w-5xl mx-auto">
      <motion.div
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        className="flex flex-col sm:flex-row items-center justify-center gap-6 sm:gap-12"
      >
        <span className="text-xs uppercase tracking-[0.2em] text-slate-600 font-medium whitespace-nowrap">Built for</span>
        <div className="flex flex-wrap items-center justify-center gap-6 sm:gap-10">
          {[
            { name: 'NIRU AI Hackathon', icon: <Sparkles size={16} /> },
            { name: 'Kenya Elections', icon: <Flag size={16} /> },
            { name: 'Media Integrity', icon: <ShieldCheck size={16} /> },
            { name: 'Community Safety', icon: <Users size={16} /> },
          ].map((item, i) => (
            <motion.div
              key={item.name}
              initial={{ opacity: 0, y: 10 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
              className="flex items-center gap-2 text-slate-500 hover:text-slate-300 transition-colors duration-300 cursor-default"
            >
              {item.icon}
              <span className="text-sm font-medium">{item.name}</span>
            </motion.div>
          ))}
        </div>
      </motion.div>
    </div>
  </section>
);

// ─── KENYA IMPACT STATS ───
const KenyaImpactStats: React.FC = () => {
  const impactData = [
    { value: '1,500+', label: 'Kenyans killed in 2007/08 PEV — incitement spread via media', color: 'from-rose-500 to-red-600', borderColor: 'border-rose-500/30', icon: <AlertTriangle size={20} /> },
    { value: '67%', label: 'Of Kenyans get news via WhatsApp (Reuters 2024)', color: 'from-emerald-500 to-teal-600', borderColor: 'border-emerald-500/30', icon: <MessageSquare size={20} /> },
    { value: '2027', label: 'Next general election — deepfake risk is rising fast', color: 'from-amber-500 to-orange-600', borderColor: 'border-amber-500/30', icon: <Clock size={20} /> },
    { value: 'Zero', label: "Existing tools built for Kenya's specific threat landscape", color: 'from-violet-500 to-purple-600', borderColor: 'border-violet-500/30', icon: <Globe size={20} /> },
  ];

  return (
    <section className="relative py-24 sm:py-32 px-6">
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-rose-500/[0.02] to-transparent pointer-events-none" />
      <div className="max-w-5xl mx-auto relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-14"
        >
          <div className="section-badge">
            <Flag size={12} /> Why Kenya Needs This
          </div>
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-black mb-4 tracking-tight">
            The <span className="text-aurora">Threat</span> Is Real
          </h2>
          <p className="text-slate-400 max-w-lg mx-auto text-sm sm:text-base">
            Misinformation has real consequences in Kenya. Here's what's at stake.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {impactData.map((item, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
              className={`impact-card ${item.borderColor}`}
            >
              <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${item.color} flex items-center justify-center mb-4 shadow-lg text-white`}>
                {item.icon}
              </div>
              <div className="text-3xl font-black text-white mb-2 tracking-tight">{item.value}</div>
              <div className="text-xs sm:text-sm text-slate-400 leading-relaxed">{item.label}</div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

// ─── FEATURES SECTION ───
const FeaturesSection: React.FC = () => {
  const features = [
    {
      icon: <Camera size={24} />,
      title: 'Deepfake Image Detector',
      desc: 'EfficientNet-B4 + ELA + metadata analysis. Detects manipulated images of politicians and fake campaign material.',
      gradient: 'from-violet-500 to-purple-600',
      glow: 'rgba(139, 92, 246, 0.15)',
      tag: 'Computer Vision',
    },
    {
      icon: <AudioLines size={24} />,
      title: 'Audio Forensics',
      desc: 'Detects audio splicing and manipulation — flags fabricated "leaked audio" recordings of political figures.',
      gradient: 'from-cyan-500 to-blue-600',
      glow: 'rgba(6, 182, 212, 0.15)',
      tag: 'Audio Analysis',
    },
    {
      icon: <Type size={24} />,
      title: 'Fake News Classifier',
      desc: 'RoBERTa NLP with clickbait detection. Identifies AI-generated fake articles targeting Kenyan audiences.',
      gradient: 'from-rose-500 to-pink-600',
      glow: 'rgba(244, 63, 94, 0.15)',
      tag: 'NLP',
    },
    {
      icon: <MessageSquare size={24} />,
      title: 'WhatsApp Forward Checker',
      desc: '67% of Kenyans get news via WhatsApp. Detects misinformation patterns — in English and Swahili.',
      gradient: 'from-emerald-500 to-teal-600',
      glow: 'rgba(16, 185, 129, 0.15)',
      tag: 'Social Media',
    },
    {
      icon: <FileCheck size={24} />,
      title: 'Document Verifier',
      desc: 'Catches forged KRA PINs, HELB letters, fake M-Pesa confirmations, and edited news screenshots.',
      gradient: 'from-amber-500 to-orange-600',
      glow: 'rgba(245, 158, 11, 0.15)',
      tag: 'Document AI',
    },
    {
      icon: <Scale size={24} />,
      title: 'Kenya Legal Framework',
      desc: 'Results reference CMCA 2018, NCIC Act, Elections Act. Direct reporting links to DCI, NCIC, and CA.',
      gradient: 'from-indigo-500 to-violet-600',
      glow: 'rgba(99, 102, 241, 0.15)',
      tag: 'Compliance',
    },
  ];

  return (
    <section className="relative py-24 sm:py-32 px-6">
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[600px] rounded-full bg-violet-600/[0.03] blur-[150px] pointer-events-none" />
      <div className="max-w-6xl mx-auto relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-14"
        >
          <div className="section-badge">
            <Cpu size={12} /> Capabilities
          </div>
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-black mb-4 tracking-tight">
            Multi-Modal <span className="text-aurora">Detection Engine</span>
          </h2>
          <p className="text-slate-400 max-w-xl mx-auto text-sm sm:text-base">
            Six specialised AI modules purpose-built for Kenya's threat landscape.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-5">
          {features.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.08 }}
              className="feature-card group"
              onMouseMove={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                e.currentTarget.style.setProperty('--mouse-x', `${e.clientX - rect.left}px`);
                e.currentTarget.style.setProperty('--mouse-y', `${e.clientY - rect.top}px`);
              }}
            >
              <div className="text-[10px] uppercase tracking-[0.15em] text-slate-500 font-semibold mb-4">{f.tag}</div>
              <div className="flex items-start gap-4">
                <div
                  className={`w-12 h-12 min-w-[48px] rounded-xl bg-gradient-to-br ${f.gradient} flex items-center justify-center shadow-lg group-hover:scale-110 group-hover:rotate-[-3deg] transition-all duration-500 text-white`}
                  style={{ boxShadow: `0 8px 32px ${f.glow}` }}
                >
                  {f.icon}
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-base sm:text-lg font-bold mb-1.5 text-white group-hover:text-violet-200 transition-colors duration-300">{f.title}</h3>
                  <p className="text-sm text-slate-400 leading-relaxed">{f.desc}</p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

// ─── HOW IT WORKS ───
const HowItWorks: React.FC = () => {
  const steps = [
    { num: '01', title: 'Upload', desc: 'Drag and drop any image, audio, or paste text content.', icon: <Upload size={24} />, color: 'from-violet-500 to-purple-600' },
    { num: '02', title: 'Analyze', desc: 'Our AI pipeline runs 6+ forensic checks in parallel.', icon: <BrainCircuit size={24} />, color: 'from-cyan-500 to-blue-600' },
    { num: '03', title: 'Detect', desc: 'Neural models classify authenticity with confidence scores.', icon: <Scan size={24} />, color: 'from-rose-500 to-pink-600' },
    { num: '04', title: 'Report', desc: 'Get actionable insights, risk scores, and threat alerts.', icon: <BarChart3 size={24} />, color: 'from-emerald-500 to-teal-600' },
  ];

  return (
    <section className="relative py-24 sm:py-32 px-6">
      <div className="max-w-5xl mx-auto relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-14"
        >
          <div className="section-badge">
            <Zap size={12} /> Workflow
          </div>
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-black mb-4 tracking-tight">
            How It <span className="text-aurora">Works</span>
          </h2>
          <p className="text-slate-400 max-w-lg mx-auto text-sm sm:text-base">
            Four simple steps from upload to actionable intelligence.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 sm:gap-6">
          {steps.map((s, i) => (
            <motion.div
              key={s.num}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.12 }}
              className="relative"
            >
              {i < steps.length - 1 && (
                <div className="hidden md:block absolute top-12 left-[calc(50%+32px)] w-[calc(100%-64px)] h-px">
                  <div className="w-full h-full bg-gradient-to-r from-violet-500/30 to-transparent" />
                </div>
              )}
              <div className="how-it-works-card text-center relative z-10">
                <div className="text-4xl font-black text-white/[0.04] mb-3 select-none">{s.num}</div>
                <div className={`w-14 h-14 rounded-2xl bg-gradient-to-br ${s.color} flex items-center justify-center mx-auto mb-5 shadow-lg text-white`}>
                  {s.icon}
                </div>
                <h3 className="text-lg font-bold text-white mb-2">{s.title}</h3>
                <p className="text-sm text-slate-400 leading-relaxed">{s.desc}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

// ─── CTA SECTION ───
const CTASection: React.FC<{ onAnalyze: () => void }> = ({ onAnalyze }) => (
  <section className="relative py-24 sm:py-32 px-6">
    <div className="absolute inset-0 bg-gradient-to-b from-transparent via-violet-500/[0.03] to-transparent pointer-events-none" />
    <div className="max-w-3xl mx-auto relative z-10">
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="cta-card text-center"
      >
        <div className="relative z-10">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-500 to-cyan-400 flex items-center justify-center mx-auto mb-6 shadow-2xl shadow-violet-500/30">
            <ShieldCheck size={30} className="text-white" />
          </div>
          <h2 className="text-3xl sm:text-4xl font-black mb-4 tracking-tight">
            Ready to Protect <span className="text-aurora">Truth</span>?
          </h2>
          <p className="text-slate-400 max-w-lg mx-auto mb-8 text-sm sm:text-base leading-relaxed">
            Upload any suspicious content and get instant AI-powered forensic analysis. Zero data retention, privacy-first.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <button onClick={onAnalyze} className="btn-glow text-base !px-10 !py-4 group">
              <span className="flex items-center gap-3">
                <Scan size={20} />
                Start Free Analysis
                <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform duration-300" />
              </span>
            </button>
          </div>
          <div className="flex flex-wrap items-center justify-center gap-6 mt-8">
            {[
              { icon: <Lock size={14} />, text: 'End-to-end encrypted' },
              { icon: <Zap size={14} />, text: 'Results in seconds' },
              { icon: <CheckCircle2 size={14} />, text: 'No signup needed' },
            ].map((item, i) => (
              <div key={i} className="flex items-center gap-2 text-xs text-slate-500">
                <span className="text-slate-600">{item.icon}</span>
                {item.text}
              </div>
            ))}
          </div>
        </div>
      </motion.div>
    </div>
  </section>
);

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
    { key: 'image', label: 'Image', icon: <Camera size={16} />, desc: 'Upload an image to check for deepfake manipulation' },
    { key: 'audio', label: 'Audio', icon: <Volume2 size={16} />, desc: 'Upload audio to detect splicing or manipulation' },
    { key: 'text', label: 'Text', icon: <Type size={16} />, desc: 'Paste text content to verify authenticity' },
    { key: 'forward', label: 'WhatsApp', icon: <MessageSquare size={16} />, desc: 'Paste a WhatsApp forward to check for misinformation' },
    { key: 'document', label: 'Document', icon: <FileCheck size={16} />, desc: 'Upload a document image or news screenshot to verify' },
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
    if ((tab === 'image' || tab === 'document') && f.type.startsWith('image/')) {
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

    const progressInterval = setInterval(() => {
      setScanProgress(prev => {
        if (prev >= 90) { clearInterval(progressInterval); return 90; }
        return prev + Math.random() * 12;
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
      } else if (tab === 'forward') {
        res = await fetch(`${API_BASE}/analyze/forward`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: textInput }),
        });
      } else if (tab === 'document') {
        const fd = new FormData();
        fd.append('file', file!);
        res = await fetch(`${API_BASE}/analyze/document`, { method: 'POST', body: fd });
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

  const canAnalyze = (tab === 'text' || tab === 'forward') ? textInput.trim().length > 10 : !!file;
  const tabConfig = tabs.find(t => t.key === tab)!;

  return (
    <section id="analyze" className="relative min-h-screen pt-24 pb-20 px-4 sm:px-6">
      <div className="absolute top-20 right-1/4 w-[500px] h-[500px] rounded-full bg-violet-500/[0.04] blur-[120px] pointer-events-none" />
      <div className="absolute bottom-20 left-1/4 w-[400px] h-[400px] rounded-full bg-cyan-500/[0.03] blur-[100px] pointer-events-none" />

      <div className="max-w-4xl mx-auto relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-10"
        >
          <div className="section-badge">
            <Scan size={12} /> Detection Console
          </div>
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-black mb-3 tracking-tight">
            Analyze <span className="text-aurora">Content</span>
          </h2>
          <p className="text-slate-400 max-w-lg mx-auto text-sm sm:text-base">
            Upload media or paste text to run multi-model forensic analysis.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="flex justify-center mb-8"
        >
          <div className="analysis-tabs">
            {tabs.map(t => (
              <button
                key={t.key}
                onClick={() => handleTabChange(t.key)}
                className={`analysis-tab ${tab === t.key ? 'analysis-tab-active' : ''}`}
              >
                {t.icon}
                <span className="hidden sm:inline">{t.label}</span>
                {tab === t.key && (
                  <motion.div layoutId="tabIndicator" className="analysis-tab-indicator" transition={{ type: 'spring', stiffness: 300, damping: 30 }} />
                )}
              </button>
            ))}
          </div>
        </motion.div>

        <motion.div layout className="analysis-card">
          <AnimatePresence mode="wait">
            {!result ? (
              <motion.div
                key="input"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.3 }}
              >
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-violet-500/20 to-cyan-500/20 border border-violet-500/20 flex items-center justify-center text-violet-400">
                    {tabConfig.icon}
                  </div>
                  <div>
                    <h3 className="font-bold text-white text-base">{tabConfig.label} Analysis</h3>
                    <p className="text-sm text-slate-500">{tabConfig.desc}</p>
                  </div>
                </div>

                {tab !== 'text' && tab !== 'forward' ? (
                  <>
                    <input
                      ref={fileRef}
                      type="file"
                      accept={tab === 'image' ? 'image/*' : tab === 'document' ? 'image/*' : 'audio/*'}
                      className="hidden"
                      onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])}
                    />
                    <div
                      className={`upload-zone ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
                      onDragOver={e => { e.preventDefault(); setDragging(true); }}
                      onDragLeave={() => setDragging(false)}
                      onDrop={onDrop}
                      onClick={() => fileRef.current?.click()}
                    >
                      {loading && <div className="scan-line" />}
                      {previewUrl && (tab === 'image' || tab === 'document') ? (
                        <div className="relative max-w-sm mx-auto">
                          <img src={previewUrl} alt="Preview" className="rounded-2xl max-h-64 mx-auto object-contain shadow-2xl shadow-black/30" />
                          {loading && (
                            <div className="absolute inset-0 bg-black/50 backdrop-blur-sm rounded-2xl flex items-center justify-center">
                              <div className="spinner" />
                            </div>
                          )}
                        </div>
                      ) : file ? (
                        <div className="flex flex-col items-center gap-4 py-6">
                          <div className="w-16 h-16 rounded-2xl bg-violet-500/10 border border-violet-500/20 flex items-center justify-center">
                            {tab === 'image' ? <Image size={28} className="text-violet-400" /> : tab === 'document' ? <FileCheck size={28} className="text-amber-400" /> : <Mic size={28} className="text-cyan-400" />}
                          </div>
                          <div className="text-center">
                            <div className="font-semibold text-white text-sm">{file.name}</div>
                            <div className="text-xs text-slate-500 mt-1">{(file.size / 1024).toFixed(1)} KB</div>
                          </div>
                          <button
                            onClick={e => { e.stopPropagation(); resetState(); }}
                            className="text-xs text-slate-500 hover:text-rose-400 transition-colors flex items-center gap-1 px-3 py-1.5 rounded-lg hover:bg-rose-500/10"
                          >
                            <X size={12} /> Remove file
                          </button>
                        </div>
                      ) : (
                        <div className="flex flex-col items-center gap-5 py-8 relative z-10">
                          <div className="upload-icon-wrapper">
                            <Upload size={28} className="text-violet-400" />
                          </div>
                          <div className="text-center">
                            <p className="text-white font-semibold mb-1.5">
                              Drop your {tab === 'image' ? 'image' : tab === 'document' ? 'document screenshot' : 'audio file'} here
                            </p>
                            <p className="text-sm text-slate-500">or click to browse &middot; Max 50 MB</p>
                          </div>
                          <div className="flex gap-2 text-[11px] text-slate-500">
                            {(tab === 'image' ? ['PNG', 'JPG', 'WEBP', 'BMP'] : tab === 'document' ? ['PNG', 'JPG', 'WEBP', 'Screenshots'] : ['WAV', 'MP3', 'FLAC', 'OGG']).map(f => (
                              <span key={f} className="format-badge">{f}</span>
                            ))}
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
                      placeholder={tab === 'forward'
                        ? 'Paste the WhatsApp forward message you want to check for misinformation... (works with English and Swahili)'
                        : 'Paste the text content you want to verify for authenticity...'
                      }
                      rows={8}
                      className="text-area-input"
                    />
                    <div className="absolute bottom-4 right-4 text-[11px] text-slate-600 font-medium tabular-nums">
                      {textInput.length} chars
                    </div>
                  </div>
                )}

                {loading && (
                  <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mt-6">
                    <div className="flex justify-between text-xs text-slate-400 mb-2.5">
                      <span className="flex items-center gap-2">
                        <Loader2 size={12} className="animate-spin text-violet-400" />
                        Running neural forensic analysis...
                      </span>
                      <span className="font-mono font-semibold text-white">{Math.round(scanProgress)}%</span>
                    </div>
                    <div className="progress-bar">
                      <motion.div className="progress-bar-fill" style={{ width: `${scanProgress}%` }} transition={{ duration: 0.3 }} />
                    </div>
                  </motion.div>
                )}

                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-5 flex items-center gap-3 p-4 rounded-2xl bg-rose-500/[0.08] border border-rose-500/20 text-rose-300 text-sm"
                  >
                    <AlertTriangle size={18} />
                    <span>{error}</span>
                  </motion.div>
                )}

                <motion.button
                  whileHover={canAnalyze && !loading ? { scale: 1.01 } : {}}
                  whileTap={canAnalyze && !loading ? { scale: 0.99 } : {}}
                  onClick={analyze}
                  disabled={!canAnalyze || loading}
                  className={`analyze-btn ${canAnalyze && !loading ? 'analyze-btn-active' : 'analyze-btn-disabled'}`}
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
                      <ArrowRight size={16} />
                    </>
                  )}
                </motion.button>
              </motion.div>
            ) : (
              <motion.div
                key="results"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.4 }}
              >
                <button onClick={resetState} className="flex items-center gap-2 text-sm text-slate-400 hover:text-violet-300 transition-all mb-8 group">
                  <ChevronRight size={16} className="rotate-180 group-hover:-translate-x-1 transition-transform" />
                  New Analysis
                </button>

                <div className="flex flex-col md:flex-row items-center gap-8 mb-10 pb-10 border-b border-white/[0.06]">
                  <motion.div
                    initial={{ scale: 0.5, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ type: 'spring', stiffness: 150 }}
                  >
                    <RiskGauge score={result.risk_score} />
                  </motion.div>

                  <div className="flex-1 text-center md:text-left">
                    <VerdictBadge verdict={result.verdict || (result.is_authentic ? 'AUTHENTIC' : 'LIKELY_DEEPFAKE')} />

                    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }} className="mt-5">
                      <div className="text-xs text-slate-500 mb-2 uppercase tracking-wider font-semibold">Confidence Level</div>
                      <div className="flex items-center gap-3">
                        <div className="flex-1 h-2.5 bg-white/[0.04] rounded-full overflow-hidden max-w-xs">
                          <motion.div
                            className="h-full rounded-full bg-gradient-to-r from-violet-500 to-cyan-400"
                            initial={{ width: 0 }}
                            animate={{ width: `${(result.confidence || 0) * 100}%` }}
                            transition={{ duration: 1.2, delay: 0.6, ease: [0.4, 0, 0.2, 1] }}
                          />
                        </div>
                        <span className="text-sm font-bold text-white min-w-[40px]">
                          {((result.confidence || 0) * 100).toFixed(0)}%
                        </span>
                      </div>
                    </motion.div>

                    {result.details && (
                      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.7 }} className="flex flex-wrap gap-2 mt-5">
                        {Object.entries(result.details).map(([key, val]) => (
                          <div key={key} className="detail-tag">
                            <span className="text-slate-500">{key.replace(/_/g, ' ')}</span>
                            <span className="text-white font-semibold">{typeof val === 'number' ? val.toFixed(1) : String(val)}</span>
                          </div>
                        ))}
                      </motion.div>
                    )}
                  </div>
                </div>

                {result.findings && result.findings.length > 0 && (
                  <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 }}>
                    <h4 className="text-xs font-bold uppercase tracking-[0.2em] text-slate-500 mb-4 flex items-center gap-2">
                      <Search size={14} /> Forensic Findings
                    </h4>
                    <div className="grid gap-2">
                      {result.findings.map((f, i) => (
                        <motion.div key={i} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.7 + i * 0.08 }} className="finding-item">
                          <CheckCircle2 size={14} className="text-violet-400 min-w-[14px] mt-0.5" />
                          <span>{f}</span>
                        </motion.div>
                      ))}
                    </div>
                  </motion.div>
                )}

                {result.kenya_warnings && result.kenya_warnings.length > 0 && (
                  <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.9 }} className="mt-8">
                    <h4 className="text-xs font-bold uppercase tracking-[0.2em] text-rose-400 mb-4 flex items-center gap-2">
                      <ShieldAlert size={14} /> Threat Alerts
                    </h4>
                    {result.kenya_warnings.map((w, i) => (
                      <div key={i} className="warning-card mb-3">
                        <div className="flex items-center gap-2 mb-2">
                          <AlertTriangle size={14} className="text-rose-400" />
                          <span className="font-bold text-rose-300 text-sm">{w.type}</span>
                          <span className={`ml-auto text-[10px] px-2.5 py-1 rounded-full font-semibold uppercase tracking-wider ${
                            w.severity === 'CRITICAL' ? 'bg-rose-500/20 text-rose-300 border border-rose-500/20' : 'bg-amber-500/20 text-amber-300 border border-amber-500/20'
                          }`}>
                            {w.severity}
                          </span>
                        </div>
                        <p className="text-sm text-slate-400 leading-relaxed">{w.warning}</p>
                        <p className="text-xs text-slate-500 mt-2 flex items-center gap-1.5">
                          <ArrowRight size={10} /> {w.action}
                        </p>
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

// ─── FOOTER ───
const Footer: React.FC = () => (
  <footer className="relative py-16 sm:py-20 px-6 footer-glow">
    <div className="max-w-6xl mx-auto relative z-10">
      <div className="flex flex-col gap-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-violet-500/20">
              <Shield size={20} className="text-white" />
            </div>
            <div>
              <span className="text-xl font-extrabold tracking-tight">
                Safe<span className="text-aurora">Eye</span>
              </span>
              <p className="text-xs text-slate-600 mt-0.5">AI-Powered Media Integrity</p>
            </div>
          </div>
          <div className="flex items-center gap-6 text-sm text-slate-500">
            <span className="flex items-center gap-2">🇰🇪 Built for Kenya</span>
            <span className="hidden sm:inline text-slate-700">&middot;</span>
            <span className="hidden sm:inline">NIRU AI Hackathon 2026</span>
          </div>
        </div>
        <div className="h-px bg-gradient-to-r from-transparent via-white/[0.06] to-transparent" />
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2 text-xs text-slate-600">
            <Lock size={12} />
            Privacy-First &middot; Zero Data Retention &middot; Open Source
          </div>
          <div className="text-xs text-slate-700">
            &copy; 2026 SafEye &middot; Kulinda Ukweli wa Kidijitali
          </div>
        </div>
      </div>
    </div>
  </footer>
);

// ─── MAIN APP ───
const App: React.FC = () => {
  const [view, setView] = useState<AppView>('home');
  const [user, setUser] = useState<UserInfo | null>(null);

  useEffect(() => {
    fetch('/api/me', { credentials: 'include' })
      .then(r => r.json())
      .then(data => { if (data.user) setUser(data.user); })
      .catch(() => {});
  }, []);

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [view]);

  return (
    <div className="relative min-h-screen">
      <div className="mesh-bg" />
      <div className="grid-pattern" />
      <Particles />
      <Navbar view={view} setView={setView} user={user} />
      <AnimatePresence mode="wait">
        {view === 'home' ? (
          <motion.div key="home" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.5 }}>
            <HeroSection onAnalyze={() => setView('analyze')} />
            <TrustedBySection />
            <KenyaImpactStats />
            <HowItWorks />
            <FeaturesSection />
            <CTASection onAnalyze={() => setView('analyze')} />
            <Footer />
          </motion.div>
        ) : (
          <motion.div key="analyze" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.5 }}>
            <AnalysisPanel />
            <Footer />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default App;
