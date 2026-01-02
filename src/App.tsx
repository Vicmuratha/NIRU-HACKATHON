import React, { useState, useRef } from 'react';
import { Upload, Image, Mic, FileText, Video, AlertCircle, CheckCircle, XCircle, Info, Zap, Shield, TrendingUp } from 'lucide-react';

interface AnalysisResult {
  authentic: boolean;
  confidence: number;
  riskScore: number;
  findings: string[];
  details: Record<string, any>;
}

const SafEyePlatform = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [file, setFile] = useState<File | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [textInput, setTextInput] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const analyzeContent = async (content: string | File, type: string) => {
    setAnalyzing(true);
    setResult(null);

    try {
      const API_BASE_URL = 'http://localhost:5000/api';

      if (type === 'text') {
        // Analyze text
        const response = await fetch(`${API_BASE_URL}/analyze/text`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ text: content })
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const apiResult = await response.json();
        setResult({
          authentic: apiResult.is_authentic,
          confidence: apiResult.confidence,
          riskScore: apiResult.risk_score,
          findings: apiResult.findings,
          details: apiResult.details
        });
      } else {
        // Analyze file
        const formData = new FormData();
        formData.append('file', content);

        let endpoint;
        if (type === 'image') {
          endpoint = '/analyze/image';
        } else if (type === 'audio') {
          endpoint = '/analyze/audio';
        } else {
          throw new Error('Unsupported file type');
        }

        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const apiResult = await response.json();
        setResult({
          authentic: apiResult.is_authentic,
          confidence: apiResult.confidence,
          riskScore: apiResult.risk_score,
          findings: apiResult.findings,
          details: apiResult.details
        });
      }
    } catch (error) {
      console.error('Analysis error:', error);
      alert('Error analyzing content. Please try again.');
    }

    setAnalyzing(false);
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

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-md border-b-4 border-blue-600">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="bg-gradient-to-br from-blue-600 to-purple-600 p-3 rounded-xl shadow-lg">
                <Shield className="w-8 h-8 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  SafEye
                </h1>
                <p className="text-sm text-gray-600">AI-Powered Deepfake Detection Platform</p>
              </div>
            </div>
            <div className="hidden md:flex items-center space-x-6">
              <div className="flex items-center space-x-2 text-sm">
                <Zap className="w-4 h-4 text-green-500" />
                <span className="text-gray-700 font-medium">Real-Time Analysis</span>
              </div>
              <div className="flex items-center space-x-2 text-sm">
                <TrendingUp className="w-4 h-4 text-blue-500" />
                <span className="text-gray-700 font-medium">99.2% Accuracy</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Stats Bar */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-white p-4 rounded-xl shadow-md border-l-4 border-blue-500">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 font-medium">Content Analyzed</p>
                <p className="text-2xl font-bold text-gray-800">12,847</p>
              </div>
              <Image className="w-10 h-10 text-blue-500 opacity-20" />
            </div>
          </div>
          <div className="bg-white p-4 rounded-xl shadow-md border-l-4 border-green-500">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 font-medium">Authentic</p>
                <p className="text-2xl font-bold text-gray-800">9,234</p>
              </div>
              <CheckCircle className="w-10 h-10 text-green-500 opacity-20" />
            </div>
          </div>
          <div className="bg-white p-4 rounded-xl shadow-md border-l-4 border-red-500">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 font-medium">Manipulated</p>
                <p className="text-2xl font-bold text-gray-800">3,613</p>
              </div>
              <XCircle className="w-10 h-10 text-red-500 opacity-20" />
            </div>
          </div>
          <div className="bg-white p-4 rounded-xl shadow-md border-l-4 border-purple-500">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600 font-medium">Accuracy Rate</p>
                <p className="text-2xl font-bold text-gray-800">99.2%</p>
              </div>
              <Shield className="w-10 h-10 text-purple-500 opacity-20" />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Upload Section */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-200">
              <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
                <Upload className="w-6 h-6 mr-2 text-blue-600" />
                Verify Content
              </h2>

              {/* Tab Navigation */}
              <div className="flex space-x-2 mb-6 border-b border-gray-200">
                <button
                  onClick={() => setActiveTab('upload')}
                  className={`px-4 py-2 font-medium transition-all ${
                    activeTab === 'upload'
                      ? 'text-blue-600 border-b-2 border-blue-600'
                      : 'text-gray-600 hover:text-gray-800'
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
                      ? 'text-blue-600 border-b-2 border-blue-600'
                      : 'text-gray-600 hover:text-gray-800'
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
                    className="border-3 border-dashed border-gray-300 rounded-xl p-12 text-center cursor-pointer hover:border-blue-500 hover:bg-blue-50 transition-all"
                  >
                    <Upload className="w-16 h-16 mx-auto text-gray-400 mb-4" />
                    <p className="text-lg font-semibold text-gray-700 mb-2">
                      Drop files here or click to upload
                    </p>
                    <p className="text-sm text-gray-500 mb-4">
                      Supports images, audio, and video files
                    </p>
                    <div className="flex justify-center space-x-4 text-xs text-gray-600">
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
                    <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
                      <p className="text-sm font-medium text-blue-900">
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
                    className="w-full h-40 p-4 border-2 border-gray-300 rounded-xl focus:border-blue-500 focus:outline-none resize-none"
                  />
                  <button
                    onClick={handleTextAnalysis}
                    disabled={!textInput.trim() || analyzing}
                    className="mt-4 w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 rounded-xl font-semibold hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {analyzing ? 'Analyzing...' : 'Analyze Text'}
                  </button>
                </div>
              )}

              {/* Analysis Progress */}
              {analyzing && (
                <div className="mt-6 p-6 bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl border border-blue-200">
                  <div className="flex items-center space-x-3 mb-4">
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                    <span className="font-semibold text-gray-800">Analyzing content...</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div className="bg-gradient-to-r from-blue-600 to-purple-600 h-2 rounded-full animate-pulse" style={{ width: '70%' }}></div>
                  </div>
                  <div className="mt-3 text-sm text-gray-600 space-y-1">
                    <p>✓ Extracting features...</p>
                    <p>✓ Running AI models...</p>
                    <p className="animate-pulse">→ Generating report...</p>
                  </div>
                </div>
              )}

              {/* Results */}
              {result && !analyzing && (
                <div className="mt-6 space-y-4">
                  <div className={`p-6 rounded-xl border-2 ${getRiskBg(result.riskScore)}`}>
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

                  <div className="bg-white p-6 rounded-xl border border-gray-200">
                    <h4 className="font-bold text-gray-800 mb-3 flex items-center">
                      <Info className="w-5 h-5 mr-2 text-blue-600" />
                      Analysis Findings
                    </h4>
                    <ul className="space-y-2">
                      {result.findings.map((finding: string, idx: number) => (
                        <li key={idx} className="flex items-start space-x-2 text-sm">
                          <span className={result.authentic ? 'text-green-500' : 'text-red-500'}>
                            {result.authentic ? '✓' : '⚠'}
                          </span>
                          <span className="text-gray-700">{finding}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="bg-gray-50 p-6 rounded-xl border border-gray-200">
                    <h4 className="font-bold text-gray-800 mb-3">Technical Details</h4>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      {Object.entries(result.details).map(([key, value]) => (
                        <div key={key}>
                          <span className="text-gray-600 capitalize">
                            {key.replace(/_/g, ' ')}:
                          </span>
                          <span className="ml-2 font-semibold text-gray-800">{String(value)}</span>
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
            <div className="bg-gradient-to-br from-blue-600 to-purple-600 rounded-2xl shadow-xl p-6 text-white">
              <h3 className="text-xl font-bold mb-4">How SafEye Works</h3>
              <div className="space-y-4 text-sm">
                <div className="flex items-start space-x-3">
                  <div className="bg-white bg-opacity-20 rounded-full p-2 mt-1">
                    <Image className="w-4 h-4" />
                  </div>
                  <div>
                    <p className="font-semibold mb-1">Image Analysis</p>
                    <p className="text-blue-100">ELA, CNN classifiers, and metadata verification</p>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="bg-white bg-opacity-20 rounded-full p-2 mt-1">
                    <Mic className="w-4 h-4" />
                  </div>
                  <div>
                    <p className="font-semibold mb-1">Audio Detection</p>
                    <p className="text-blue-100">Spectrogram analysis and anti-spoofing models</p>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="bg-white bg-opacity-20 rounded-full p-2 mt-1">
                    <FileText className="w-4 h-4" />
                  </div>
                  <div>
                    <p className="font-semibold mb-1">Text Verification</p>
                    <p className="text-blue-100">NLP models and claim verification</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-200">
              <h3 className="text-lg font-bold text-gray-800 mb-4">Recent Detections</h3>
              <div className="space-y-3 text-sm">
                <div className="flex items-center justify-between p-3 bg-red-50 rounded-lg">
                  <span className="font-medium text-gray-800">Deepfake Audio</span>
                  <span className="text-red-600 font-bold">87% Risk</span>
                </div>
                <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                  <span className="font-medium text-gray-800">Authentic Image</span>
                  <span className="text-green-600 font-bold">12% Risk</span>
                </div>
                <div className="flex items-center justify-between p-3 bg-yellow-50 rounded-lg">
                  <span className="font-medium text-gray-800">Suspicious Text</span>
                  <span className="text-yellow-600 font-bold">56% Risk</span>
                </div>
              </div>
            </div>

            <div className="bg-orange-50 border-2 border-orange-200 rounded-xl p-4">
              <div className="flex items-start space-x-2">
                <AlertCircle className="w-5 h-5 text-orange-600 mt-0.5" />
                <div className="text-sm">
                  <p className="font-semibold text-orange-900 mb-1">Stay Safe Online</p>
                  <p className="text-orange-800">Always verify suspicious content before sharing. Trust but verify.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SafEyePlatform;
