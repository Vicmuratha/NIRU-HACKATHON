// ═══════════════════════════════════════════════════════════
//  SafEye — Shared TypeScript Types
// ═══════════════════════════════════════════════════════════

export interface AnalysisResult {
  risk_score: number;
  verdict: string;
  confidence: number;
  findings: string[];
  kenya_warnings?: KenyaWarning[];
  details?: Record<string, any>;
  is_authentic?: boolean;
  forward_analysis?: any;
  document_analysis?: any;
  screenshot_analysis?: any;
  kenya_audio_context?: any;
  detection_note?: string;
}

export interface KenyaWarning {
  type: string;
  severity: string;
  warning: string;
  action: string;
}

export interface UserInfo {
  name: string;
  email: string;
  picture?: string | null;
}

export interface ProfileData {
  id: number;
  name: string;
  email: string;
  bio: string;
  phone: string;
  location: string;
  organization: string;
  role: string;
  profile_picture: string;
  auth_provider: string;
  created_at: string;
  updated_at: string;
  last_login: string;
}

export interface ProfileStats {
  total_scans: number;
  threats_detected: number;
  authentic_count: number;
  image_scans: number;
  audio_scans: number;
  text_scans: number;
  forward_scans: number;
  document_scans: number;
  avg_risk_score: number;
}

export interface HistoryItem {
  id: number;
  detection_type: string;
  filename: string;
  risk_score: number;
  verdict: string;
  confidence: number;
  findings: string[];
  kenya_warnings: KenyaWarning[];
  details: Record<string, any>;
  created_at: string;
}

export interface AllUser {
  id: number;
  name: string;
  email: string;
  bio: string;
  location: string;
  organization: string;
  role: string;
  profile_picture: string;
  auth_provider: string;
  created_at: string;
  last_login: string;
  total_scans: number;
}

export type AnalysisTab = 'image' | 'audio' | 'text' | 'forward' | 'document';
export type AppView = 'home' | 'analyze' | 'profile';

export interface HealthResponse {
  status: string;
  timestamp: string;
  models: {
    image_loaded: boolean;
    audio_loaded: boolean;
    text_loaded: boolean;
  };
  platform: string;
  version: string;
  environment: string;
}
