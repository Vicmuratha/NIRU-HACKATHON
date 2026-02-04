import React, { useState, useRef, useEffect } from 'react';
import { Upload, Image, Mic, FileText, Video, AlertCircle, CheckCircle, XCircle, Info, Zap, Shield, TrendingUp } from 'lucide-react';

interface AnalysisResult {
  authentic: boolean;
  confidence: number;
  riskScore: number;
  findings: string[];
  details: Record<string, any>;
}

interface UserProfile {
  name: string;
  email?: string;
  picture?: string;
}

const SafEyePlatform = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [file, setFile] = useState<File | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [textInput, setTextInput] = useState('');
  const [user, setUser] = useState<UserProfile | null>(null);
  const [showProfile, setShowProfile] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const loadUser = async () => {
      try {
        const response = await fetch('/api/me', { credentials: 'include' });
        const data = await response.json();
        setUser(data.user || null);
      } catch (error) {
        console.error('Failed to load user:', error);
      }
    };

    loadUser();
  }, []);

  const analyzeContent = async (content: string | File, type: string) => {
    setAnalyzing(true);
    setResult(null);

    try {
      if (type === 'text') {
        // Handle text analysis
        const response = await fetch('/api/analyze/text', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ text: content }),
        });

        if (!response.ok) {
          // Try to get error message from response
          let errorMessage = `API error: ${response.status}`;
          try {
            const errorData = await response.json();
            errorMessage = errorData.error || errorData.message || errorMessage;
            console.error('API Error Details:', errorData);
          } catch (e) {
            // If response is not JSON, use status text
            errorMessage = response.statusText || errorMessage;
          }
          throw new Error(errorMessage);
        }

        const apiResult = await response.json();

        const result: AnalysisResult = {
          authentic: apiResult.is_authentic,
          confidence: apiResult.confidence || 0.8,
          riskScore: apiResult.risk_score || 0,
          findings: apiResult.findings || [],
          details: apiResult.details || {}
        };

        setResult(result);
      } else if (content instanceof File) {
        // Handle file upload (image or audio)
        const formData = new FormData();
        formData.append('file', content);

        const endpoint = content.type.startsWith('image/') ? '/api/analyze/image' : '/api/analyze/audio';
        const response = await fetch(endpoint, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          // Try to get error message from response
          let errorMessage = `API error: ${response.status}`;
          try {
            const errorData = await response.json();
            errorMessage = errorData.error || errorData.message || errorMessage;
            console.error('API Error Details:', errorData);
          } catch (e) {
            // If response is not JSON, use status text
            errorMessage = response.statusText || errorMessage;
          }
          throw new Error(errorMessage);
        }

        const apiResult = await response.json();

        const result: AnalysisResult = {
          authentic: apiResult.is_authentic,
          confidence: apiResult.confidence || 0.8,
          riskScore: apiResult.risk_score || 0,
          findings: apiResult.findings || [],
          details: apiResult.details || {}
        };

        setResult(result);
      }
    } catch (error) {
      console.error('Analysis error:', error);
      
      // Try to get more detailed error from response
      let errorMessage = 'Unknown error';
      if (error instanceof Error) {
        errorMessage = error.message;
        
        // If it's a fetch error, try to get the response body
        if (error.message.includes('API error')) {
          // The error message should contain the status code
          errorMessage = `Server error: ${error.message}`;
        }
      }
      
      // Show error to user with more details
      alert(`Analysis failed: ${errorMessage}\n\nPlease check the browser console for more details.`);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = e.target.files?.[0];
    if (uploadedFile) {
      setFile(uploadedFile);
      const fileType = uploadedFile.type.startsWith('image') ? 'image' :
                       uploadedFile.type.startsWith('audio') ? 'audio' : 'video';
      analyzeContent(uploadedFile, fileType);
    }
  };

  const handleTextAnalysis = () => {
    if (textInput.trim()) {
      analyzeContent(textInput, 'text');
    }
  };

  const getRiskColor = (score: number): string => {
    if (score < 30) return 'text-green-500';
    if (score < 60) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getRiskBg = (score: number): string => {
    if (score < 30) return 'bg-green-100 border-green-300';
    if (score < 60) return 'bg-yellow-100 border-yellow-300';
    return 'bg-red-100 border-red-300';
  };

  const stats = [
    { label: 'Content Analyzed', value: '12,847', icon: Image, accent: 'blue' },
    { label: 'Authentic', value: '9,234', icon: CheckCircle, accent: 'green' },
    { label: 'Manipulated', value: '3,613', icon: XCircle, accent: 'red' },
    { label: 'Accuracy Rate', value: '99.2%', icon: Shield, accent: 'purple' },
  ];

  const accentClassMap: Record<string, string> = {
    blue: 'bg-blue-400/20 text-blue-200',
    green: 'bg-emerald-400/20 text-emerald-200',
    red: 'bg-rose-400/20 text-rose-200',
    purple: 'bg-purple-400/20 text-purple-200'
  };

  const trustSignals = [
    'Real-time deepfake defense',
    'Financial scam protection',
    'Election integrity monitoring',
    'Media authenticity verification'
  ];

  const workflowSteps = [
    {
      title: 'Secure Upload',
      description: 'Drop files or paste text. We hash, sandbox, and isolate each scan.',
      icon: Upload
    },
    {
      title: 'Multi-Model Analysis',
      description: 'Ensembles of CNNs, audio spoofing models, and NLP verification.',
      icon: Zap
    },
    {
      title: 'Actionable Verdicts',
      description: 'Clear risk scores, evidence trails, and mitigation guidance.',
      icon: Shield
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-indigo-950 to-purple-950 text-white">
      {/* Ambient Glow */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -top-32 -left-24 h-96 w-96 rounded-full bg-blue-500/30 blur-3xl" />
        <div className="absolute top-24 right-10 h-80 w-80 rounded-full bg-purple-500/30 blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-72 w-72 rounded-full bg-cyan-400/20 blur-3xl" />
        <div className="absolute inset-0 grid-overlay" />
      </div>

      {/* Header */}
      <header className="relative z-10 border-b border-white/10 bg-black/30 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 py-5 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="bg-gradient-to-br from-blue-500 to-purple-600 p-3 rounded-2xl shadow-2xl">
              <Shield className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight">SafEye</h1>
              <p className="text-xs text-white/70">AI-Powered Deepfake Detection Platform</p>
            </div>
          </div>
          <div className="hidden md:flex items-center space-x-8 text-sm text-white/80">
            <div className="flex items-center space-x-2">
              <Zap className="w-4 h-4 text-emerald-400" />
              <span>Real-Time Analysis</span>
            </div>
            <div className="flex items-center space-x-2">
              <TrendingUp className="w-4 h-4 text-cyan-400" />
              <span>99.2% Accuracy</span>
            </div>
          </div>
          <div className="flex items-center space-x-3 relative">
            {user ? (
              <button
                onClick={() => setShowProfile((prev) => !prev)}
                className="flex items-center space-x-2 rounded-full bg-white/10 border border-white/20 px-3 py-2 hover:border-white/40 transition"
              >
                <img
                  src={user.picture || 'https://ui-avatars.com/api/?name=SafEye&background=0ea5e9&color=fff'}
                  alt="Profile"
                  className="h-8 w-8 rounded-full object-cover"
                />
                <span className="text-sm text-white/80 hidden md:inline">{user.name}</span>
              </button>
            ) : (
              <>
                <a
                  href="/login"
                  className="px-4 py-2 rounded-xl border border-white/20 text-white/80 hover:text-white hover:border-white/40 transition"
                >
                  Log in
                </a>
                <a
                  href="/signup"
                  className="px-4 py-2 rounded-xl bg-gradient-to-r from-cyan-400 to-blue-500 text-slate-900 font-semibold shadow-lg hover:shadow-cyan-500/30 transition"
                >
                  Sign up
                </a>
              </>
            )}

            {user && showProfile && (
              <div className="absolute right-0 top-12 w-64 rounded-2xl border border-white/10 bg-slate-950/90 backdrop-blur-xl shadow-2xl p-4 text-sm text-white/80">
                <div className="flex items-center space-x-3 pb-3 border-b border-white/10">
                  <img
                    src={user.picture || 'https://ui-avatars.com/api/?name=SafEye&background=0ea5e9&color=fff'}
                    alt="Profile"
                    className="h-10 w-10 rounded-full object-cover"
                  />
                  <div>
                    <p className="font-semibold text-white">{user.name}</p>
                    <p className="text-xs text-white/60">{user.email || 'Verified user'}</p>
                  </div>
                </div>
                <div className="py-3 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-white/60">Plan</span>
                    <span className="text-white">Pro Demo</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-white/60">Role</span>
                    <span className="text-white">Analyst</span>
                  </div>
                </div>
                <a
                  href="/logout"
                  className="block text-center mt-2 w-full rounded-xl bg-white text-slate-900 font-semibold py-2 hover:shadow-lg transition"
                >
                  Sign out
                </a>
              </div>
            )}
          </div>
        </div>
      </header>

      <main className="relative z-10">
        {/* Hero */}
        <section className="max-w-7xl mx-auto px-6 pt-14 pb-10 grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          <div className="space-y-6">
            <div className="inline-flex items-center gap-2 rounded-full bg-white/10 px-4 py-2 text-xs uppercase tracking-[0.2em] text-white/70">
              <span className="h-2 w-2 rounded-full bg-emerald-400" />
              Live threat defense for Kenya & beyond
            </div>
            <h2 className="text-4xl md:text-5xl font-semibold leading-tight">
              Detect deepfakes with <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-300 to-blue-400">extreme clarity</span> and instant decisions.
            </h2>
            <p className="text-lg text-white/70">
              SafEye combines vision, audio, and language intelligence to protect communities, institutions, and families from digital deception.
            </p>
            <div className="flex flex-wrap gap-3">
              <a
                href="/login"
                className="px-6 py-3 rounded-xl bg-white text-slate-900 font-semibold shadow-xl hover:shadow-white/30 transition"
              >
                Enter Detection Studio
              </a>
              <a
                href="#detection"
                className="px-6 py-3 rounded-xl border border-white/20 text-white/80 hover:text-white hover:border-white/40 transition"
              >
                Explore Demo
              </a>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-4">
              {trustSignals.map((signal) => (
                <div key={signal} className="flex items-center gap-2 text-sm text-white/70">
                  <span className="h-2 w-2 rounded-full bg-cyan-400" />
                  {signal}
                </div>
              ))}
            </div>
          </div>
          <div className="space-y-6">
            <div className="relative overflow-hidden bg-white/10 border border-white/10 rounded-3xl p-6 shadow-2xl backdrop-blur-xl neon-card">
              <div className="absolute inset-0 scanline" />
              <div className="absolute -right-10 -bottom-10 h-40 w-40 rounded-full bg-cyan-400/20 blur-3xl" />
              <div className="absolute inset-0 pointer-events-none">
                <div className="absolute left-1/2 top-1/2 h-52 w-52 -translate-x-1/2 -translate-y-1/2 rounded-full border border-white/10 orbit-slow" />
                <div className="absolute left-1/2 top-1/2 h-32 w-32 -translate-x-1/2 -translate-y-1/2 rounded-full border border-cyan-400/40 orbit-fast" />
                <div className="absolute left-[63%] top-[28%] h-2 w-2 rounded-full bg-cyan-300 shadow-[0_0_12px_rgba(34,211,238,0.9)]" />
                <div className="absolute left-[28%] top-[68%] h-2 w-2 rounded-full bg-purple-300 shadow-[0_0_12px_rgba(192,132,252,0.9)]" />
              </div>
              <div className="relative">
                <p className="text-xs uppercase tracking-[0.3em] text-white/50">Command Center</p>
                <h3 className="text-2xl font-semibold mt-2">Signal Integrity Grid</h3>
                <div className="mt-6 grid grid-cols-3 gap-3 text-xs text-white/70">
                  <div className="rounded-2xl border border-white/10 bg-white/5 px-3 py-3">
                    <p className="text-white/50">Latency</p>
                    <p className="text-lg font-semibold text-white">0.8s</p>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-white/5 px-3 py-3">
                    <p className="text-white/50">Alerts</p>
                    <p className="text-lg font-semibold text-white">142</p>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-white/5 px-3 py-3">
                    <p className="text-white/50">Coverage</p>
                    <p className="text-lg font-semibold text-white">96%</p>
                  </div>
                </div>
                <div className="mt-6">
                  <div className="flex items-center justify-between text-sm text-white/70">
                    <span>Neural confidence</span>
                    <span className="text-emerald-200 font-semibold">98.6%</span>
                  </div>
                  <div className="mt-2 h-2 rounded-full bg-white/10">
                    <div className="h-2 rounded-full bg-gradient-to-r from-emerald-300 to-cyan-300 w-[82%] animate-pulse" />
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white/10 border border-white/10 rounded-3xl p-6 shadow-2xl backdrop-blur-xl floaty">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.25em] text-white/50">Live Threat Index</p>
                  <h3 className="text-2xl font-semibold mt-2">Moderate Risk</h3>
                </div>
                <div className="h-14 w-14 rounded-2xl bg-amber-400/20 text-amber-200 flex items-center justify-center">
                  <AlertCircle className="w-7 h-7" />
                </div>
              </div>
              <div className="mt-6 space-y-3 text-sm text-white/70">
                <div className="flex justify-between">
                  <span>Political content scans</span>
                  <span className="text-emerald-200">+22%</span>
                </div>
                <div className="flex justify-between">
                  <span>Voice cloning reports</span>
                  <span className="text-amber-200">+12%</span>
                </div>
                <div className="flex justify-between">
                  <span>Image manipulation flags</span>
                  <span className="text-rose-200">+8%</span>
                </div>
              </div>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              {workflowSteps.map((step) => (
                <div key={step.title} className="bg-white/10 border border-white/10 rounded-2xl p-4 text-sm text-white/70 neon-card">
                  <step.icon className="w-5 h-5 text-cyan-300 mb-3" />
                  <h4 className="text-white font-semibold mb-2">{step.title}</h4>
                  <p>{step.description}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Stats Bar */}
        <section className="max-w-7xl mx-auto px-6 pb-10">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {stats.map((stat) => {
              const Icon = stat.icon;
              return (
                <div key={stat.label} className="bg-white/10 border border-white/10 rounded-2xl p-5 shadow-xl">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs text-white/60 uppercase tracking-[0.2em]">{stat.label}</p>
                      <p className="text-2xl font-semibold mt-2">{stat.value}</p>
                    </div>
                    <div className={`h-10 w-10 rounded-xl flex items-center justify-center ${accentClassMap[stat.accent]}`}>
                      <Icon className="w-5 h-5" />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {/* Trusted Marquee */}
        <section className="max-w-7xl mx-auto px-6 pb-12">
          <div className="rounded-3xl border border-white/10 bg-white/5 px-6 py-6 backdrop-blur-xl neon-card">
            <p className="text-xs uppercase tracking-[0.35em] text-white/50">Trusted by public safety teams</p>
            <div className="marquee mt-5">
              <div className="marquee-track">
                <span>National CERT</span>
                <span>Ministry of ICT</span>
                <span>Election Integrity Unit</span>
                <span>Financial Crimes Taskforce</span>
                <span>Open Media Alliance</span>
                <span>University AI Lab</span>
                <span>Telecom Security</span>
                <span>Digital Forensics Bureau</span>
                <span>National CERT</span>
                <span>Ministry of ICT</span>
                <span>Election Integrity Unit</span>
                <span>Financial Crimes Taskforce</span>
                <span>Open Media Alliance</span>
                <span>University AI Lab</span>
                <span>Telecom Security</span>
                <span>Digital Forensics Bureau</span>
              </div>
            </div>
          </div>
        </section>

        {/* Threat Intelligence */}
        <section className="max-w-7xl mx-auto px-6 pb-16">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 bg-white/10 border border-white/10 rounded-3xl p-6 shadow-2xl backdrop-blur-xl neon-card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.3em] text-white/50">Threat intelligence</p>
                  <h3 className="text-2xl font-semibold mt-2">Multi-signal fusion map</h3>
                </div>
                <div className="rounded-2xl bg-emerald-400/20 px-4 py-2 text-xs font-semibold text-emerald-200">Live</div>
              </div>
              <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-white/70">
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                  <p className="text-white/50">Vision Signal</p>
                  <p className="text-lg font-semibold text-white">87.4%</p>
                  <p className="text-xs text-white/50">ELA + CNN ensemble</p>
                </div>
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                  <p className="text-white/50">Audio Signal</p>
                  <p className="text-lg font-semibold text-white">91.1%</p>
                  <p className="text-xs text-white/50">Spoofing defense</p>
                </div>
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                  <p className="text-white/50">Language Signal</p>
                  <p className="text-lg font-semibold text-white">88.3%</p>
                  <p className="text-xs text-white/50">Claim verification</p>
                </div>
              </div>
              <div className="mt-6 h-40 rounded-2xl border border-white/10 bg-gradient-to-br from-white/5 to-white/0 p-6">
                <div className="flex h-full items-end gap-3">
                  {[32, 58, 46, 72, 61, 84, 66, 92].map((value, idx) => (
                    <div key={idx} className="flex-1">
                      <div className="w-full rounded-full bg-white/10" style={{ height: '100%' }}>
                        <div
                          className="w-full rounded-full bg-gradient-to-t from-cyan-300/80 to-blue-500/80"
                          style={{ height: `${value}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="bg-white/10 border border-white/10 rounded-3xl p-6 shadow-2xl backdrop-blur-xl neon-card">
              <h4 className="text-lg font-semibold">Response Playbooks</h4>
              <div className="mt-6 space-y-4 text-sm text-white/70">
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                  <p className="text-white font-semibold">Public Advisory</p>
                  <p className="text-xs text-white/50 mt-1">Auto-generated brief with evidence trail.</p>
                </div>
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                  <p className="text-white font-semibold">Stakeholder Alert</p>
                  <p className="text-xs text-white/50 mt-1">Notify regulators, media teams, and community leads.</p>
                </div>
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                  <p className="text-white font-semibold">Rapid Takedown</p>
                  <p className="text-xs text-white/50 mt-1">Coordinate platform reporting with audit logs.</p>
                </div>
              </div>
              <div className="mt-6 rounded-2xl border border-cyan-300/30 bg-cyan-400/10 p-4 text-xs text-cyan-100/80">
                Median time-to-action: <span className="font-semibold text-cyan-100">2m 14s</span>
              </div>
            </div>
          </div>
        </section>

        <section id="detection" className="max-w-7xl mx-auto px-6 pb-16">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Upload Section */}
          <div className="lg:col-span-2">
            <div className="bg-white/10 border border-white/10 rounded-3xl shadow-2xl p-6 backdrop-blur-xl">
              <h2 className="text-2xl font-semibold text-white mb-6 flex items-center">
                <Upload className="w-6 h-6 mr-2 text-cyan-300" />
                Detection Studio
              </h2>

              {/* Tab Navigation */}
              <div className="flex space-x-2 mb-6 border-b border-gray-200">
                <button
                  onClick={() => setActiveTab('upload')}
                  className={`px-4 py-2 font-medium transition-all ${
                    activeTab === 'upload'
                      ? 'text-white border-b-2 border-cyan-300'
                      : 'text-white/60 hover:text-white'
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <Upload className="w-4 h-4" />
                    <span>Upload File</span>
                  </div>
                </button>
                <button
                  onClick={() => setActiveTab('text')}
                  className={`px-4 py-2 font-medium transition-all ${
                    activeTab === 'text'
                      ? 'text-white border-b-2 border-cyan-300'
                      : 'text-white/60 hover:text-white'
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <FileText className="w-4 h-4" />
                    <span>Analyze Text</span>
                  </div>
                </button>
              </div>

              {activeTab === 'upload' ? (
                <div>
                  <div
                    onClick={() => fileInputRef.current?.click()}
                    className="border-2 border-dashed border-white/30 rounded-2xl p-12 text-center cursor-pointer hover:border-cyan-300 hover:bg-white/10 transition-all"
                  >
                    <Upload className="w-16 h-16 mx-auto text-white/60 mb-4" />
                    <p className="text-lg font-semibold text-white mb-2">
                      Drop files here or click to upload
                    </p>
                    <p className="text-sm text-white/60 mb-4">
                      Supports images, audio, and video files
                    </p>
                    <div className="flex justify-center space-x-4 text-xs text-white/60">
                      <span className="flex items-center">
                        <Image className="w-4 h-4 mr-1" /> Images
                      </span>
                      <span className="flex items-center">
                        <Mic className="w-4 h-4 mr-1" /> Audio
                      </span>
                      <span className="flex items-center">
                        <Video className="w-4 h-4 mr-1" /> Video
                      </span>
                    </div>
                  </div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    onChange={handleFileUpload}
                    accept="image/*,audio/*,video/*"
                    className="hidden"
                  />
                  {file && (
                    <div className="mt-4 p-4 bg-white/10 rounded-lg border border-white/10">
                      <p className="text-sm font-medium text-white">
                        Selected: {file.name}
                      </p>
                    </div>
                  )}
                </div>
              ) : (
                <div>
                  <textarea
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                    placeholder="Paste text content, social media posts, or messages to verify..."
                    className="w-full h-40 p-4 border-2 border-white/20 bg-white/5 text-white rounded-2xl focus:border-cyan-300 focus:outline-none resize-none"
                  />
                  <button
                    onClick={handleTextAnalysis}
                    disabled={!textInput.trim() || analyzing}
                    className="mt-4 w-full bg-gradient-to-r from-cyan-300 to-blue-500 text-slate-900 py-3 rounded-2xl font-semibold hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {analyzing ? 'Analyzing...' : 'Analyze Text'}
                  </button>
                </div>
              )}

              {/* Analysis Progress */}
              {analyzing && (
                <div className="mt-6 p-6 bg-white/10 rounded-2xl border border-white/10">
                  <div className="flex items-center space-x-3 mb-4">
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-cyan-300"></div>
                    <span className="font-semibold text-white">Analyzing content...</span>
                  </div>
                  <div className="w-full bg-white/20 rounded-full h-2">
                    <div className="bg-gradient-to-r from-cyan-300 to-blue-500 h-2 rounded-full animate-pulse" style={{ width: '70%' }}></div>
                  </div>
                  <div className="mt-3 text-sm text-white/70 space-y-1">
                    <p>✓ Extracting features...</p>
                    <p>✓ Running AI models...</p>
                    <p className="animate-pulse">→ Generating report...</p>
                  </div>
                </div>
              )}

              {/* Results */}
              {result && !analyzing && (
                <div className="mt-6 space-y-4">
                  <div className={`p-6 rounded-2xl border-2 ${getRiskBg(result.riskScore)} text-slate-900`}>
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h3 className="text-xl font-bold text-gray-800 mb-1">
                          {result.authentic ? 'Likely Authentic' : 'Manipulation Detected'}
                        </h3>
                        <p className="text-sm text-gray-600">
                          Confidence: {(result.confidence * 100).toFixed(1)}%
                        </p>
                      </div>
                      {result.authentic ? (
                        <CheckCircle className="w-12 h-12 text-green-500" />
                      ) : (
                        <AlertCircle className="w-12 h-12 text-red-500" />
                      )}
                    </div>
                    <div className="mb-4">
                      <div className="flex justify-between text-sm mb-2">
                        <span className="font-medium">Risk Score</span>
                        <span className={`font-bold ${getRiskColor(result.riskScore)}`}>
                          {result.riskScore.toFixed(1)}/100
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-3">
                        <div
                          className={`h-3 rounded-full ${
                            result.riskScore < 30 ? 'bg-green-500' :
                            result.riskScore < 60 ? 'bg-yellow-500' : 'bg-red-500'
                          }`}
                          style={{ width: `${result.riskScore}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-white/10 p-6 rounded-2xl border border-white/10">
                    <h4 className="font-bold text-white mb-3 flex items-center">
                      <Info className="w-5 h-5 mr-2 text-cyan-300" />
                      Analysis Findings
                    </h4>
                    <ul className="space-y-2">
                      {result.findings.map((finding: string, idx: number) => (
                        <li key={idx} className="flex items-start space-x-2 text-sm">
                          <span className={result.authentic ? 'text-green-500' : 'text-red-500'}>
                            {result.authentic ? '✓' : '⚠'}
                          </span>
                          <span className="text-white/80">{finding}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="bg-white/10 p-6 rounded-2xl border border-white/10">
                    <h4 className="font-bold text-white mb-3">Technical Details</h4>
                    <div className="grid grid-cols-2 gap-4 text-sm text-white/80">
                      {Object.entries(result.details).map(([key, value]) => (
                        <div key={key}>
                          <span className="text-white/60 capitalize">
                            {key.replace(/_/g, ' ')}:
                          </span>
                          <span className="ml-2 font-semibold text-white">{String(value)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Info Sidebar */}
          <div className="space-y-6">
            <div className="bg-gradient-to-br from-cyan-400/20 via-blue-500/20 to-purple-500/20 rounded-3xl shadow-2xl p-6 border border-white/10">
              <h3 className="text-xl font-semibold mb-4">How SafEye Works</h3>
              <div className="space-y-4 text-sm text-white/70">
                <div className="flex items-start space-x-3">
                  <div className="bg-white/10 rounded-full p-2 mt-1">
                    <Image className="w-4 h-4" />
                  </div>
                  <div>
                    <p className="font-semibold text-white mb-1">Image Analysis</p>
                    <p>ELA, CNN classifiers, and metadata verification</p>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="bg-white/10 rounded-full p-2 mt-1">
                    <Mic className="w-4 h-4" />
                  </div>
                  <div>
                    <p className="font-semibold text-white mb-1">Audio Detection</p>
                    <p>Spectrogram analysis and anti-spoofing models</p>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="bg-white/10 rounded-full p-2 mt-1">
                    <FileText className="w-4 h-4" />
                  </div>
                  <div>
                    <p className="font-semibold text-white mb-1">Text Verification</p>
                    <p>NLP models and claim verification</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white/10 rounded-3xl shadow-2xl p-6 border border-white/10">
              <h3 className="text-lg font-semibold mb-4">Recent Detections</h3>
              <div className="space-y-3 text-sm">
                <div className="flex items-center justify-between p-3 bg-red-500/10 rounded-xl">
                  <span className="font-medium">Deepfake Audio</span>
                  <span className="text-red-300 font-bold">87% Risk</span>
                </div>
                <div className="flex items-center justify-between p-3 bg-green-500/10 rounded-xl">
                  <span className="font-medium">Authentic Image</span>
                  <span className="text-emerald-300 font-bold">12% Risk</span>
                </div>
                <div className="flex items-center justify-between p-3 bg-yellow-500/10 rounded-xl">
                  <span className="font-medium">Suspicious Text</span>
                  <span className="text-yellow-200 font-bold">56% Risk</span>
                </div>
              </div>
            </div>

            <div className="bg-amber-500/10 border border-amber-400/30 rounded-2xl p-4">
              <div className="flex items-start space-x-2">
                <AlertCircle className="w-5 h-5 text-amber-300 mt-0.5" />
                <div className="text-sm">
                  <p className="font-semibold text-amber-200 mb-1">Stay Safe Online</p>
                  <p className="text-amber-100/80">Always verify suspicious content before sharing. Trust but verify.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <section className="max-w-7xl mx-auto px-6 pb-20">
        <div className="relative overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-br from-cyan-400/20 via-blue-500/20 to-purple-500/20 p-8 shadow-2xl backdrop-blur-xl neon-card">
          <div className="absolute -top-16 -right-16 h-40 w-40 rounded-full bg-cyan-400/30 blur-3xl" />
          <div className="absolute -bottom-20 -left-10 h-48 w-48 rounded-full bg-purple-400/30 blur-3xl" />
          <div className="relative z-10 grid grid-cols-1 lg:grid-cols-3 gap-6 items-center">
            <div className="lg:col-span-2">
              <p className="text-xs uppercase tracking-[0.3em] text-white/60">Deploy in minutes</p>
              <h3 className="text-3xl md:text-4xl font-semibold mt-3">Activate a national-grade deepfake shield.</h3>
              <p className="text-white/70 mt-3">
                Integrate SafEye with your newsroom, hotline, or security command center. Deliver verified evidence in real time.
              </p>
            </div>
            <div className="flex flex-col sm:flex-row lg:flex-col gap-3">
              <a
                href="/signup"
                className="px-6 py-3 rounded-2xl bg-white text-slate-900 font-semibold shadow-xl hover:shadow-white/30 transition text-center"
              >
                Start a free pilot
              </a>
              <a
                href="/login"
                className="px-6 py-3 rounded-2xl border border-white/30 text-white/80 hover:text-white hover:border-white/60 transition text-center"
              >
                Speak with security
              </a>
            </div>
          </div>
        </div>
      </section>
      </main>
    </div>
  );
};

export default SafEyePlatform;
