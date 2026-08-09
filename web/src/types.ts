export interface ModelOption {
  id: string;
  label: string;
  repo: string;
}

export interface LoraPreset {
  id: string;
  label: string;
  spec: string;
  scale: number;
  custom?: boolean;
}

export interface PresetOption {
  id: string;
  label: string;
  width?: number;
  height?: number;
  seconds?: number;
  num_frames?: number;
}

export interface Config {
  server_connected: boolean;
  server_url: string;
  preferred_model: string;
  default_model?: string;
  models: ModelOption[];
  resolution_presets: PresetOption[];
  duration_presets: PresetOption[];
  generation_modes: { id: string; label: string }[];
  chain_methods?: { id: string; label: string; description?: string }[];
  pipeline_profiles?: { id: string; label: string }[];
  defaults: {
    num_frames: number;
    width: number;
    height: number;
    num_steps: number;
    fps: number;
  };
  model_note: string;
  clip_multiplier_max?: number;
  embedded?: boolean;
  web_url?: string;
  active_model?: string;
  lora_presets?: LoraPreset[];
  default_lora_preset_id?: string;
  preferred_lora_preset_ids?: string[];
  ic_lora_preset_id?: string;
  ic_lora_motion_preset_id?: string;
  ic_lora_default_spec?: string;
  ic_lora_union_motion_spec?: string;
  face_swap_preset_id?: string;
  face_swap_default_spec?: string;
  id_lora_preset_id?: string;
  id_lora_celebvhq_preset_id?: string;
  id_lora_talkvid_preset_id?: string;
  id_lora_default_spec?: string;
  id_lora_talkvid_spec?: string;
  id_lora_prompt_placeholder?: string;
  lipdub_preset_id?: string;
  lipdub_default_spec?: string;
  lipdub_official_gated_spec?: string;
  lipdub_official_hf_url?: string;
  lipdub_env_var?: string;
  pose_control_available?: boolean;
  pyav_available?: boolean;
  audio_trim_available?: boolean;
}

export interface Clip {
  id: string;
  prompt: string;
  label: string;
  video_url: string;
  filename: string;
  chain_id: string;
  clip_index: number;
  mode: string;
  status: string;
  created_at: string;
  elapsed_s?: number;
  bytes?: number;
  error?: string;
  num_frames?: number;
  width?: number;
  height?: number;
  seed?: number;
  num_steps?: number;
  duration_seconds?: number;
  clip_count?: number;
  autocontinue?: boolean;
  autoconcat?: boolean;
  audiocontinue?: boolean;
}

export interface LibraryFrame {
  id: string;
  label: string;
  path: string;
  image_url: string;
  filename: string;
  width?: number;
  height?: number;
  source_clip_id?: string;
  time_s?: number;
  created_at: string;
}

export interface ModelProgress {
  stage?: string;
  step?: number;
  total?: number;
  pct?: number;
  eta_s?: number;
  avg_step_s?: number;
  elapsed_s?: number;
  label?: string;
}

export interface ProgressState {
  phase: string;
  message: string;
  elapsed_s?: number;
  kb?: number;
  stage?: string;
  step?: number;
  total?: number;
  pct?: number;
  eta_s?: number;
}
