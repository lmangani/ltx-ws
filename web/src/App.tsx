import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { AudioTrimControl } from "./AudioTrimControl";
import { clipDisplayPrompt, snapshotFromClip } from "./clipEditor";
import { applyProgressEvent } from "./progress";
import { captureVideoFrame, formatVideoTime } from "./frameCapture";
import type { Clip, Config, LibraryFrame, LoraPreset, ProgressState } from "./types";

const API = "";
const MODEL_PREF_KEY = "ltx-ws-preferred-model";
const LORA_SEL_KEY = "ltx-ws-lora-preset-ids";
const LORA_ENSURED_KEY = "ltx-ws-lora-ensured-specs";
const BLOB_VIDEO_PREFIX = "blob:";

function readEnsuredLoraSpecs(): Set<string> {
  try {
    const raw = sessionStorage.getItem(LORA_ENSURED_KEY);
    if (!raw) return new Set();
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return new Set();
    return new Set(parsed.map(String).filter(Boolean));
  } catch {
    return new Set();
  }
}

function persistEnsuredLoraSpec(spec: string) {
  const next = readEnsuredLoraSpecs();
  next.add(spec);
  try {
    sessionStorage.setItem(LORA_ENSURED_KEY, JSON.stringify([...next]));
  } catch {
    /* ignore quota / private mode */
  }
}

function removeEnsuredLoraSpec(spec: string) {
  if (!spec) return;
  const next = readEnsuredLoraSpecs();
  next.delete(spec);
  try {
    sessionStorage.setItem(LORA_ENSURED_KEY, JSON.stringify([...next]));
  } catch {
    /* ignore quota / private mode */
  }
}

type LoraActivity =
  | { phase: "idle" }
  | {
      phase: "working";
      label: string;
      index: number;
      total: number;
      downloading: boolean;
    }
  | { phase: "ready"; message: string }
  | { phase: "error"; message: string };

async function cacheVideoAsBlobUrl(serverUrl: string): Promise<string> {
  const res = await fetch(serverUrl);
  if (!res.ok) throw new Error(`Video download failed (${res.status})`);
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

function preserveBlobVideoUrls(prev: Clip[], incoming: Clip[]): Clip[] {
  const blobById = new Map(
    prev
      .filter((c) => c.video_url?.startsWith(BLOB_VIDEO_PREFIX))
      .map((c) => [c.id, c.video_url] as const),
  );
  return incoming.map((c) => {
    const blob = blobById.get(c.id);
    return blob ? { ...c, video_url: blob } : c;
  });
}

function revokeBlobVideoUrls(clips: Clip[]) {
  for (const clip of clips) {
    revokeClipBlob(clip);
  }
}

function revokeClipBlob(clip: Clip) {
  const url = clip.video_url;
  if (url?.startsWith(BLOB_VIDEO_PREFIX)) {
    URL.revokeObjectURL(url);
  }
}

function replaceClipsFromServer(prev: Clip[], incoming: Clip[]): Clip[] {
  const next = preserveBlobVideoUrls(prev, incoming);
  const nextIds = new Set(next.map((c) => c.id));
  for (const clip of prev) {
    if (!nextIds.has(clip.id)) {
      revokeClipBlob(clip);
    }
  }
  return next;
}

async function fetchConfig(): Promise<Config> {
  const r = await fetch(`${API}/api/config`);
  if (!r.ok) throw new Error("Failed to load config");
  return r.json();
}

async function fetchClips(chainId?: string): Promise<Clip[]> {
  const q = chainId ? `?chain_id=${encodeURIComponent(chainId)}` : "";
  const r = await fetch(`${API}/api/clips${q}`);
  if (!r.ok) throw new Error("Failed to load clips");
  const data = await r.json();
  return data.clips as Clip[];
}

async function fetchFrames(): Promise<LibraryFrame[]> {
  const r = await fetch(`${API}/api/frames`);
  if (!r.ok) throw new Error("Failed to load frames");
  const data = await r.json();
  return (data.frames ?? []) as LibraryFrame[];
}

const IMAGE_INPUT_MODES = new Set(["generate", "i2v", "a2v", "keyframe", "id_lora"]);
const ID_LORA_PROMPT_PLACEHOLDER =
  "[VISUAL]: A close-up of a person speaking to camera. " +
  "[SPEECH]: Hello, nice to meet you. " +
  "[SOUNDS]: Clear speech, quiet room.";

/** Merge server clip lists into local state (by id); used when refreshing one chain. */
function mergeClips(prev: Clip[], incoming: Clip[]): Clip[] {
  const byId = new Map(prev.map((c) => [c.id, c]));
  for (const c of incoming) {
    byId.set(c.id, c);
  }
  return Array.from(byId.values());
}

/** Replace all clips for one chain (e.g. after autoconcat removes fragments). */
function replaceChainClips(prev: Clip[], chainId: string, chainClips: Clip[]): Clip[] {
  const rest = prev.filter((c) => c.chain_id !== chainId);
  return [...rest, ...preserveBlobVideoUrls(prev, chainClips)];
}

function formatBytes(n?: number) {
  if (!n) return "";
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDuration(frames?: number, fps = 24) {
  if (!frames) return "";
  const s = frames / fps;
  return `${s.toFixed(1)}s`;
}

function pickPlaybackClip(clips: Clip[], chainId: string): string | null {
  const chain = clips.filter(
    (c) => c.chain_id === chainId && c.status === "done" && c.video_url,
  );
  const merged = chain.find((c) => c.label === "MERGED");
  const current = chain.find((c) => c.label === "CURRENT");
  const latest = [...chain].sort((a, b) => b.clip_index - a.clip_index)[0];
  return merged?.id ?? current?.id ?? latest?.id ?? null;
}

function ChainMethodPicker({
  chainMethod,
  onChange,
  className,
}: {
  chainMethod: string;
  onChange: (method: string) => void;
  className?: string;
}) {
  return (
    <div className={`chain-method-panel${className ? ` ${className}` : ""}`}>
      <span className="chain-method-label">Chain</span>
      <div className="chain-method-radios">
        <label className="check chain-method-option">
          <input
            type="radio"
            name="chainMethod"
            value="autocontinue"
            checked={chainMethod === "autocontinue"}
            onChange={() => onChange("autocontinue")}
          />
          Autocontinue
        </label>
        <label className="check chain-method-option">
          <input
            type="radio"
            name="chainMethod"
            value="native_extend"
            checked={chainMethod === "native_extend"}
            onChange={() => onChange("native_extend")}
          />
          Extend video
        </label>
      </div>
    </div>
  );
}

function loraSelectionSummary(presets: LoraPreset[], selectedIds: string[]) {
  const selected = presets.filter((p) => selectedIds.includes(p.id));
  if (selected.length === 0) return "None";
  if (selected.length === 1) {
    const raw = selected[0].label.replace(/\s*\(default\)\s*$/i, "").trim();
    return raw.length > 20 ? `${raw.slice(0, 19)}…` : raw;
  }
  return `${selected.length} LoRAs`;
}

function loraSelectionTitle(presets: LoraPreset[], selectedIds: string[]) {
  const selected = presets.filter((p) => selectedIds.includes(p.id));
  if (!selected.length) return "No LoRA selected";
  return selected.map((p) => p.label).join(", ");
}

function LoraMultiSelect({
  presets,
  selectedIds,
  disabled,
  onToggle,
  onRemovePreset,
}: {
  presets: LoraPreset[];
  selectedIds: string[];
  disabled?: boolean;
  onToggle: (id: string, checked: boolean) => void;
  onRemovePreset: (preset: LoraPreset) => void;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  const summary = loraSelectionSummary(presets, selectedIds);
  const title = loraSelectionTitle(presets, selectedIds);

  return (
    <div className={`multi-select${open ? " is-open" : ""}`} ref={rootRef}>
      <button
        type="button"
        className="multi-select-trigger"
        disabled={disabled}
        aria-expanded={open}
        title={title}
        onClick={() => setOpen((v) => !v)}
      >
        <span className="multi-select-trigger-text">{summary}</span>
      </button>
      {open && (
        <div className="multi-select-menu" role="listbox" aria-label="LoRA presets">
          {presets.map((p) => {
            const checked = selectedIds.includes(p.id);
            return (
              <div
                key={p.id}
                className={`multi-select-item${checked ? " is-selected" : ""}`}
                role="option"
                aria-selected={checked}
              >
                <input
                  type="checkbox"
                  checked={checked}
                  disabled={disabled}
                  aria-label={p.label}
                  onChange={(e) => onToggle(p.id, e.target.checked)}
                />
                <span
                  className="multi-select-item-label"
                  title={p.label}
                  onClick={() => {
                    if (!disabled) onToggle(p.id, !checked);
                  }}
                >
                  {p.label}
                </span>
                <button
                  type="button"
                  className="lora-remove"
                  title="Remove from list"
                  disabled={disabled}
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    onRemovePreset(p);
                  }}
                >
                  ×
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [config, setConfig] = useState<Config | null>(null);
  const [clips, setClips] = useState<Clip[]>([]);
  const [frameLibrary, setFrameLibrary] = useState<LibraryFrame[]>([]);
  const [chainId, setChainId] = useState<string | null>(null);
  const [selectedClipId, setSelectedClipId] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [showNegativePrompt, setShowNegativePrompt] = useState(false);
  const [busy, setBusy] = useState(false);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [progress, setProgress] = useState<ProgressState | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Options
  const [model, setModel] = useState("auto");
  const [mode, setMode] = useState("generate");
  const [resolutionId, setResolutionId] = useState("704x480");
  const [durationId, setDurationId] = useState("5s");
  const [clipMultiplier, setClipMultiplier] = useState(1);
  const [numSteps, setNumSteps] = useState(8);
  const [seed, setSeed] = useState<string>("");
  const [autocontinue, setAutocontinue] = useState(false);
  const [autoconcat, setAutoconcat] = useState(false);
  const [audiocontinue, setAudiocontinue] = useState(false);
  const [chainMethod, setChainMethod] = useState("autocontinue");
  const [enhancePrompt, setEnhancePrompt] = useState(false);
  const [pipelineProfile, setPipelineProfile] = useState("distilled");
  const [imagePath, setImagePath] = useState<string | null>(null);
  const [imageName, setImageName] = useState<string | null>(null);
  const [endImagePath, setEndImagePath] = useState<string | null>(null);
  const [endImageName, setEndImageName] = useState<string | null>(null);
  const [audioPath, setAudioPath] = useState<string | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioPreviewUrl, setAudioPreviewUrl] = useState<string | null>(null);
  const [audioName, setAudioName] = useState<string | null>(null);
  const [audioDurationSeconds, setAudioDurationSeconds] = useState<number | null>(null);
  const [audioStartSeconds, setAudioStartSeconds] = useState(0);
  const [videoPath, setVideoPath] = useState<string | null>(null);
  const [videoName, setVideoName] = useState<string | null>(null);
  const [conditioningVideoPath, setConditioningVideoPath] = useState<string | null>(null);
  const [conditioningVideoName, setConditioningVideoName] = useState<string | null>(null);
  const [conditioningClipId, setConditioningClipId] = useState<string | null>(null);
  const [conditioningVideoScale, setConditioningVideoScale] = useState(1.0);
  const [referenceStrength, setReferenceStrength] = useState(1.0);
  const [skipStage2, setSkipStage2] = useState(false);
  const [sourceClipId, setSourceClipId] = useState<string | null>(null);
  const [retakeStart, setRetakeStart] = useState(0);
  const [retakeEnd, setRetakeEnd] = useState(12);
  const [extendFrames, setExtendFrames] = useState(15);
  const [extendDirection, setExtendDirection] = useState("after");
  const [showOptions, setShowOptions] = useState(true);
  const [loraPresetIds, setLoraPresetIds] = useState<string[]>([]);
  const [loraActivity, setLoraActivity] = useState<LoraActivity>({ phase: "idle" });
  const [loraBusy, setLoraBusy] = useState(false);
  const [customLoraUrl, setCustomLoraUrl] = useState("");
  const [customLoraLabel, setCustomLoraLabel] = useState("");
  const [customLoraScale, setCustomLoraScale] = useState("1.0");
  const [addingCustomLora, setAddingCustomLora] = useState(false);
  const [savingFrame, setSavingFrame] = useState(false);

  const imageRef = useRef<HTMLInputElement>(null);
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const playerVideoRef = useRef<HTMLVideoElement>(null);
  const endImageRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLInputElement>(null);
  const conditioningVideoRef = useRef<HTMLInputElement>(null);
  const ensuredLoraSpecsRef = useRef<Set<string>>(readEnsuredLoraSpecs());
  const loraPresetsRef = useRef<LoraPreset[]>([]);
  const ensurePromisesRef = useRef<Map<string, Promise<void>>>(new Map());
  const runEventSourceRef = useRef<EventSource | null>(null);
  const preIcLoraPresetIdsRef = useRef<string[] | null>(null);
  const prevModeRef = useRef(mode);

  const libraryClips = useMemo(() => {
    return clips
      .filter((c) => c.status === "done" && (c.video_url || c.filename))
      .sort((a, b) => b.created_at.localeCompare(a.created_at))
      .slice(0, 48);
  }, [clips]);

  const videoLibraryClips = useMemo(
    () => libraryClips.filter((c) => c.filename),
    [libraryClips],
  );

  const hasVideoSource = Boolean(videoPath || sourceClipId);
  const hasConditioningVideo = Boolean(conditioningVideoPath || conditioningClipId);

  const icLoraHdrId = config?.ic_lora_preset_id ?? "ic_lora_hdr";
  const icLoraUnionId = config?.ic_lora_motion_preset_id ?? "ic_lora_union_motion";
  const icLoraUnionSelected = loraPresetIds.includes(icLoraUnionId);

  // Upstream hdr-ic-lora: video and image are independently optional.
  // Union Control is opt-in via the LoRA picker — never auto-switched by image.
  const icLoraSubMode = useMemo(() => {
    if (icLoraUnionSelected) return "motion_transfer" as const;
    if (!hasConditioningVideo && imagePath) return "i2v" as const;
    if (hasConditioningVideo && imagePath) return "i2v_v2v" as const;
    if (hasConditioningVideo) return "v2v" as const;
    return "t2v" as const;
  }, [hasConditioningVideo, icLoraUnionSelected, imagePath]);

  const icLoraModeLabel = useMemo(() => {
    switch (icLoraSubMode) {
      case "motion_transfer":
        return "Union Control — motion / pose transfer (needs reference video)";
      case "i2v_v2v":
        return "HDR IC-LoRA — image-to-video + reference video";
      case "i2v":
        return "HDR IC-LoRA — image-to-video (optional video)";
      case "v2v":
        return "HDR IC-LoRA — video-to-video (SDR → HDR)";
      default:
        return "HDR IC-LoRA — text-to-video (optional image / video)";
    }
  }, [icLoraSubMode]);

  const faceSwapPresetId = config?.face_swap_preset_id ?? "face_swap_head";
  const lipDubPresetId = config?.lipdub_preset_id ?? "lipdub_ic_lora";
  const idLoraPresetId = config?.id_lora_preset_id ?? "id_lora_celebvhq";
  const idLoraCelebPresetId = config?.id_lora_celebvhq_preset_id ?? "id_lora_celebvhq";
  const idLoraTalkvidPresetId = config?.id_lora_talkvid_preset_id ?? "id_lora_talkvid";
  const idLoraPromptPlaceholder =
    config?.id_lora_prompt_placeholder ?? ID_LORA_PROMPT_PLACEHOLDER;

  const chainParts = useMemo(() => {
    if (!chainId || !selectedClipId) return [];
    return clips
      .filter((c) => c.chain_id === chainId && c.status === "done" && c.video_url)
      .sort((a, b) => a.clip_index - b.clip_index);
  }, [clips, chainId, selectedClipId]);

  const showChainPicker = chainParts.length > 1;

  const activeClip = useMemo(() => {
    if (!selectedClipId) return null;
    return clips.find((x) => x.id === selectedClipId) ?? null;
  }, [clips, selectedClipId]);

  const closeRunSubscription = useCallback(() => {
    runEventSourceRef.current?.close();
    runEventSourceRef.current = null;
  }, []);

  /** Drop chain/clip editor context after library removal so generate stays usable. */
  const releaseClipContext = useCallback(() => {
    setSelectedClipId(null);
    setChainId(null);
    setSourceClipId(null);
    setMode((current) => {
      if (["retake", "extend", "lipdub", "face_swap"].includes(current)) return "generate";
      if (current === "i2v" && !imagePath) return "generate";
      if (current === "a2v" && !audioPath && !audioFile) return "generate";
      if (current === "keyframe" && (!imagePath || !endImagePath)) return "generate";
      return current;
    });
    setAutocontinue(false);
    setAutoconcat(false);
    setAudiocontinue(false);
    setClipMultiplier(1);
  }, [audioFile, audioPath, endImagePath, imagePath]);

  const syncClipSelection = useCallback(
    (all: Clip[], deleted?: Clip) => {
      const ids = new Set(all.map((c) => c.id));
      let nextSelected = selectedClipId;
      let nextChain = chainId;

      if (nextSelected && !ids.has(nextSelected)) {
        nextSelected = null;
      }
      if (sourceClipId && !ids.has(sourceClipId)) {
        setSourceClipId(null);
      }
      if (nextChain && !all.some((c) => c.chain_id === nextChain)) {
        nextChain = null;
        nextSelected = null;
      } else if (deleted && selectedClipId === deleted.id && nextChain) {
        const chainClips = all.filter(
          (c) => c.chain_id === nextChain && c.status === "done" && c.video_url,
        );
        nextSelected = nextChain ? pickPlaybackClip(chainClips, nextChain) : null;
        if (!nextSelected) {
          nextChain = null;
        }
      }

      setSelectedClipId(nextSelected);
      setChainId(nextChain);

      const lostActive =
        Boolean(deleted && selectedClipId === deleted.id) ||
        Boolean(selectedClipId && !ids.has(selectedClipId)) ||
        Boolean(chainId && !all.some((c) => c.chain_id === chainId));

      if (lostActive && !nextSelected) {
        releaseClipContext();
      } else if (deleted && sourceClipId === deleted.id) {
        setMode((current) =>
          ["retake", "extend", "lipdub", "face_swap"].includes(current) ? "generate" : current,
        );
      }
    },
    [chainId, releaseClipContext, selectedClipId, sourceClipId],
  );

  const ensureLoraPresets = useCallback(async (
    presetIds: string[],
    presetsOverride?: LoraPreset[],
    options?: { interactive?: boolean },
  ) => {
    const interactive = options?.interactive ?? true;
    const presets = presetsOverride ?? loraPresetsRef.current;
    const selected = presetIds
      .map((id) => presets.find((p) => p.id === id))
      .filter((p): p is NonNullable<typeof p> => Boolean(p?.spec));
    if (!selected.length) {
      setLoraActivity({ phase: "idle" });
      return;
    }

    const pending = selected.filter((p) => !ensuredLoraSpecsRef.current.has(p.spec));
    if (!pending.length) {
      setLoraActivity(
        interactive
          ? { phase: "ready", message: `${selected.length} LoRA(s) ready` }
          : { phase: "idle" },
      );
      return;
    }

    if (interactive) {
      setLoraBusy(true);
    }
    try {
      for (let i = 0; i < pending.length; i++) {
        const preset = pending[i];
        const existing = ensurePromisesRef.current.get(preset.spec);
        if (existing) {
          await existing;
          continue;
        }

        const ensureOne = (async () => {
          if (interactive) {
            setLoraActivity({
              phase: "working",
              label: preset.label,
              index: i + 1,
              total: pending.length,
              downloading: true,
            });
          }
          const r = await fetch(`${API}/api/loras/ensure`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ spec: preset.spec }),
          });
          if (!r.ok) {
            const err = await r.json().catch(() => ({}));
            throw new Error(err.detail || `LoRA download failed: ${preset.label}`);
          }
          const data = (await r.json()) as { cached?: boolean };
          ensuredLoraSpecsRef.current.add(preset.spec);
          persistEnsuredLoraSpec(preset.spec);
          if (interactive) {
            setLoraActivity({
              phase: "working",
              label: preset.label,
              index: i + 1,
              total: pending.length,
              downloading: !data.cached,
            });
          }
        })();

        ensurePromisesRef.current.set(preset.spec, ensureOne);
        try {
          await ensureOne;
        } finally {
          ensurePromisesRef.current.delete(preset.spec);
        }
      }
      setLoraActivity(
        interactive
          ? { phase: "ready", message: `${selected.length} LoRA(s) ready` }
          : { phase: "idle" },
      );
    } catch (e) {
      setLoraActivity({ phase: "error", message: String(e) });
    } finally {
      if (interactive) {
        setLoraBusy(false);
      }
    }
  }, []);

  const load = useCallback(async () => {
    try {
      const cfg = await fetchConfig();
      setConfig(cfg);
      loraPresetsRef.current = cfg.lora_presets ?? [];
      const preferredModel =
        cfg.preferred_model ||
        localStorage.getItem(MODEL_PREF_KEY) ||
        cfg.default_model ||
        "auto";
      setModel(preferredModel);
      localStorage.setItem(MODEL_PREF_KEY, preferredModel);
      setNumSteps(cfg.defaults.num_steps);
      let loraIds: string[] = [];
      try {
        const stored = localStorage.getItem(LORA_SEL_KEY);
        if (stored) {
          const parsed = JSON.parse(stored) as unknown;
          if (Array.isArray(parsed)) {
            loraIds = parsed.map(String).filter(Boolean);
          }
        }
      } catch {
        /* ignore */
      }
      if (!loraIds.length) {
        loraIds =
          cfg.preferred_lora_preset_ids?.length
            ? cfg.preferred_lora_preset_ids
            : cfg.default_lora_preset_id && cfg.default_lora_preset_id !== "none"
              ? [cfg.default_lora_preset_id]
              : [];
      }
      setLoraPresetIds(loraIds);
      if (loraIds.length) {
        void ensureLoraPresets(loraIds, cfg.lora_presets, { interactive: false });
      } else {
        setLoraActivity({ phase: "idle" });
      }
      const all = await fetchClips();
      setClips(all);
      const frames = await fetchFrames();
      setFrameLibrary(frames);
    } catch (e) {
      setError(String(e));
    }
  }, [ensureLoraPresets]);

  useEffect(() => {
    const entered = mode === "ic_lora" && prevModeRef.current !== "ic_lora";
    const left = mode !== "ic_lora" && prevModeRef.current === "ic_lora";
    const enteredV2v = mode === "v2v" && prevModeRef.current !== "v2v";
    prevModeRef.current = mode;

    if (entered) {
      preIcLoraPresetIdsRef.current = loraPresetIds;
      // Enter IC-LoRA on HDR by default (upstream hdr-ic-lora). Union is opt-in.
      setLoraPresetIds([icLoraHdrId]);
      void ensureLoraPresets([icLoraHdrId], config?.lora_presets, { interactive: true });
    } else if (left && preIcLoraPresetIdsRef.current !== null) {
      setLoraPresetIds(preIcLoraPresetIdsRef.current);
      preIcLoraPresetIdsRef.current = null;
    }
    if (enteredV2v) {
      // V2V is custom-LoRA oriented — drop built-in HDR/Union so they are not applied.
      setLoraPresetIds((prev) => prev.filter((id) => id !== icLoraHdrId && id !== icLoraUnionId));
      // Avoid stale I2V image leaking into V2V jobs.
      setImagePath(null);
      setImageName(null);
      if (imageRef.current) imageRef.current.value = "";
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- snapshot on mode enter only
  }, [mode, icLoraHdrId, icLoraUnionId]);

  useEffect(() => {
    if (mode !== "face_swap" || !faceSwapPresetId) return;
    setLoraPresetIds([faceSwapPresetId]);
    void ensureLoraPresets([faceSwapPresetId], config?.lora_presets, { interactive: true });
  }, [mode, faceSwapPresetId, config?.lora_presets, ensureLoraPresets]);

  useEffect(() => {
    if (mode !== "lipdub" || !lipDubPresetId) return;
    const preset = config?.lora_presets?.find((p) => p.id === lipDubPresetId && p.spec);
    if (!preset) return;
    setLoraPresetIds([lipDubPresetId]);
    void ensureLoraPresets([lipDubPresetId], config?.lora_presets, { interactive: true });
  }, [mode, lipDubPresetId, config?.lora_presets, ensureLoraPresets]);

  useEffect(() => {
    if (mode !== "id_lora" || !idLoraPresetId) return;
    const allowed = new Set([idLoraCelebPresetId, idLoraTalkvidPresetId]);
    setLoraPresetIds((prev) => {
      const current = prev[0];
      const next = current && allowed.has(current) ? current : idLoraPresetId;
      void ensureLoraPresets([next], config?.lora_presets, { interactive: true });
      return [next];
    });
    setNumSteps((prev) => (prev < 20 ? 30 : prev));
  }, [
    mode,
    idLoraPresetId,
    idLoraCelebPresetId,
    idLoraTalkvidPresetId,
    config?.lora_presets,
    ensureLoraPresets,
  ]);

  const persistLoraSelection = useCallback(async (ids: string[]) => {
    try {
      localStorage.setItem(LORA_SEL_KEY, JSON.stringify(ids));
    } catch {
      /* ignore */
    }
    try {
      await fetch(`${API}/api/config/loras`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ preset_ids: ids }),
      });
    } catch (err) {
      console.warn("Could not persist LoRA selection", err);
    }
  }, []);

  const toggleLoraPreset = useCallback(
    (presetId: string, checked: boolean) => {
      if (loraBusy) return;
      setLoraPresetIds((prev) => {
        let next = checked
          ? [...prev.filter((id) => id !== presetId), presetId]
          : prev.filter((id) => id !== presetId);
        if (mode === "ic_lora") {
          const hdrId = config?.ic_lora_preset_id ?? "ic_lora_hdr";
          const motionId = config?.ic_lora_motion_preset_id ?? "ic_lora_union_motion";
          const isBuiltin = presetId === hdrId || presetId === motionId;
          if (checked && isBuiltin) {
            // One IC-LoRA primary: HDR or Union alone (no stacked community adapters).
            next = [presetId];
          } else if (checked && !isBuiltin) {
            // Custom IC-LoRA (CrossView, etc.) replaces HDR/Union as the primary.
            next = next.filter((id) => id !== hdrId && id !== motionId);
          }
        }
        if (mode === "face_swap" && checked) {
          next = [presetId];
        }
        if (mode === "lipdub" && checked) {
          next = [presetId];
        }
        if (mode === "id_lora" && checked) {
          next = [presetId];
        }
        void persistLoraSelection(next);
        void ensureLoraPresets(next, undefined, { interactive: true });
        return next;
      });
    },
    [
      config?.ic_lora_motion_preset_id,
      config?.ic_lora_preset_id,
      ensureLoraPresets,
      loraBusy,
      mode,
      persistLoraSelection,
    ],
  );

  const updateLoraPresetScale = useCallback(
    (presetId: string, scale: number) => {
      const nextScale = Math.min(2, Math.max(0, scale));
      const preset =
        loraPresetsRef.current.find((p) => p.id === presetId) ??
        config?.lora_presets?.find((p) => p.id === presetId);
      setConfig((c) => {
        if (!c?.lora_presets) return c;
        const nextPresets = c.lora_presets.map((p) =>
          p.id === presetId ? { ...p, scale: nextScale } : p,
        );
        loraPresetsRef.current = nextPresets;
        return { ...c, lora_presets: nextPresets };
      });
      if (preset?.custom && preset.spec) {
        void fetch(`${API}/api/loras/custom`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            spec: preset.spec,
            label: preset.label,
            scale: nextScale,
          }),
        }).catch((err) => console.warn("Could not persist LoRA strength", err));
      }
    },
    [config?.lora_presets],
  );

  async function addCustomLora() {
    const spec = customLoraUrl.trim();
    if (!spec || addingCustomLora) return;
    setAddingCustomLora(true);
    try {
      const lower = spec.toLowerCase();
      const isCrossView = lower.includes("crossview") || lower.includes("cross-view");
      let scale = parseFloat(customLoraScale) || 1.0;
      if (isCrossView && scale < 1.2) {
        scale = 1.25;
        setCustomLoraScale("1.25");
      }
      const r = await fetch(`${API}/api/loras/custom`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          spec,
          label: customLoraLabel.trim() || undefined,
          scale,
        }),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.detail || "Could not add custom LoRA");
      }
      const data = await r.json();
      const nextPresets: LoraPreset[] = data.lora_presets ?? [];
      const newId = typeof data.id === "string" ? data.id : "";
      const added: LoraPreset | undefined =
        (data.preset as LoraPreset | undefined) ??
        nextPresets.find((p) => p.id === newId);
      if (!newId || !added) {
        throw new Error("Custom LoRA saved but missing from the LoRA menu catalog");
      }
      loraPresetsRef.current = nextPresets;
      setConfig((c) =>
        c
          ? {
              ...c,
              lora_presets: nextPresets,
              preferred_lora_preset_ids:
                data.preferred_lora_preset_ids ?? c.preferred_lora_preset_ids,
            }
          : c,
      );
      // Select the new adapter so it shows checked in the menu and is used on generate.
      setLoraPresetIds((prev) => {
        let nextIds: string[];
        if (mode === "ic_lora" || mode === "lipdub" || mode === "face_swap") {
          nextIds = [newId];
        } else if (mode === "v2v") {
          // V2V allows stacking multiple community LoRAs.
          nextIds = [...prev.filter((id) => id !== newId), newId];
        } else {
          nextIds = prev.filter((id) => id !== newId);
          nextIds.push(newId);
        }
        void persistLoraSelection(nextIds);
        return nextIds;
      });
      setCustomLoraUrl("");
      setCustomLoraLabel("");
      setCustomLoraScale(isCrossView ? "1.25" : "1.0");
      await ensureLoraPresets([newId], nextPresets, { interactive: true });
      setLoraActivity({
        phase: "ready",
        message: data.reused
          ? `LoRA updated & selected: ${added.label}`
          : `LoRA ready & selected: ${added.label}`,
      });
    } catch (e) {
      setLoraActivity({ phase: "error", message: String(e) });
    } finally {
      setAddingCustomLora(false);
    }
  }

  async function removeLoraPreset(preset: LoraPreset) {
    if (loraBusy) return;
    const msg = preset.custom
      ? `Remove custom LoRA "${preset.label}"?`
      : `Hide "${preset.label}" from the LoRA list?`;
    if (!confirm(msg)) return;
    try {
      const r = await fetch(`${API}/api/loras/preset/${encodeURIComponent(preset.id)}`, {
        method: "DELETE",
      });
      if (!r.ok) throw new Error("Delete failed");
      const data = await r.json();
      const nextPresets = data.lora_presets ?? [];
      loraPresetsRef.current = nextPresets;
      ensuredLoraSpecsRef.current.delete(preset.spec);
      removeEnsuredLoraSpec(preset.spec);
      setConfig((c) =>
        c
          ? {
              ...c,
              lora_presets: nextPresets,
              preferred_lora_preset_ids:
                data.preferred_lora_preset_ids ?? c.preferred_lora_preset_ids,
            }
          : c,
      );
      setLoraPresetIds(data.preferred_lora_preset_ids ?? []);
      setLoraActivity({ phase: "idle" });
    } catch (e) {
      setLoraActivity({ phase: "error", message: String(e) });
    }
  }

  useEffect(() => {
    load();
    return () => {
      closeRunSubscription();
    };
  }, [load, closeRunSubscription]);

  useEffect(() => {
    if (!selectedClipId) return;
    if (!clips.some((c) => c.id === selectedClipId)) {
      releaseClipContext();
    } else if (chainId && !clips.some((c) => c.chain_id === chainId)) {
      releaseClipContext();
    }
  }, [clips, chainId, releaseClipContext, selectedClipId]);

  useEffect(() => {
    if (clipMultiplier > 1) {
      setAutocontinue(true);
      setAutoconcat(true);
    }
  }, [clipMultiplier]);

  useEffect(() => {
    if (audiocontinue) {
      setAutocontinue(true);
      setAutoconcat(true);
    }
  }, [audiocontinue]);

  useEffect(() => {
    if (mode !== "a2v") {
      setAudiocontinue(false);
    }
  }, [mode]);

  const resolution = useMemo(() => {
    if (!config) return { width: 704, height: 480 };
    const p = config.resolution_presets.find((r) => r.id === resolutionId);
    return { width: p?.width ?? 704, height: p?.height ?? 480 };
  }, [config, resolutionId]);

  const durationSeconds = useMemo(() => {
    if (!config) return 5;
    const p = config.duration_presets.find((d) => d.id === durationId);
    return p?.seconds ?? 5;
  }, [config, durationId]);

  const totalDurationSeconds = durationSeconds * clipMultiplier;
  const isMultiClip = clipMultiplier > 1;
  const audioClipDurationSeconds = useMemo(() => {
    if (audiocontinue || isMultiClip) return totalDurationSeconds;
    return durationSeconds;
  }, [audiocontinue, isMultiClip, totalDurationSeconds, durationSeconds]);
  const audioStartSliderMax = useMemo(() => {
    if (audioDurationSeconds && audioDurationSeconds > 0) {
      return Math.max(0.1, audioDurationSeconds - audioClipDurationSeconds);
    }
    return 300;
  }, [audioClipDurationSeconds, audioDurationSeconds]);
  const editingChain =
    !isMultiClip &&
    Boolean(chainId && activeClip?.status === "done");
  const willContinueChain = editingChain && autocontinue;

  function beginFreshGeneration() {
    releaseClipContext();
    setError(null);
  }

  function applyFrameAsInput(frame: LibraryFrame, target: "start" | "end" = "start") {
    if (target === "end") {
      setEndImagePath(frame.path);
      setEndImageName(frame.label);
      if (mode !== "keyframe") {
        setMode("keyframe");
      }
    } else {
      setImagePath(frame.path);
      setImageName(frame.label);
      if (!IMAGE_INPUT_MODES.has(mode)) {
        setMode("i2v");
      }
    }
    setError(null);
  }

  async function saveCurrentFrame() {
    const video = playerVideoRef.current;
    if (!video || !activeClip || savingFrame || busy) return;
    setSavingFrame(true);
    try {
      const blob = await captureVideoFrame(video);
      const timeS = video.currentTime;
      const fd = new FormData();
      fd.append("file", blob, `frame_${Date.now()}.png`);
      fd.append(
        "label",
        `${formatVideoTime(timeS)} · ${activeClip.label || "clip"}`,
      );
      fd.append("time_s", String(timeS));
      fd.append("source_clip_id", activeClip.id);
      const r = await fetch(`${API}/api/frames`, { method: "POST", body: fd });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.detail || "Could not save frame");
      }
      const frames = await fetchFrames();
      setFrameLibrary(frames);
    } catch (e) {
      setError(String(e));
    } finally {
      setSavingFrame(false);
    }
  }

  async function deleteFrame(frame: LibraryFrame) {
    if (busy) return;
    if (!confirm(`Delete frame "${frame.label}"?`)) return;
    try {
      const r = await fetch(`${API}/api/frames/${encodeURIComponent(frame.id)}`, {
        method: "DELETE",
      });
      if (!r.ok) throw new Error("Could not delete frame");
      const data = await r.json();
      setFrameLibrary(data.frames ?? []);
      if (imagePath === frame.path) {
        setImagePath(null);
        setImageName(null);
      }
      if (endImagePath === frame.path) {
        setEndImagePath(null);
        setEndImageName(null);
      }
    } catch (e) {
      setError(String(e));
    }
  }

  function applyClipSelection(clip: Clip) {
    const snap = snapshotFromClip(clip, config, {
      numSteps: config?.defaults.num_steps ?? 8,
    });
    setSelectedClipId(clip.id);
    setChainId(clip.chain_id);
    setPrompt(snap.prompt);
    setMode(snap.mode);
    setResolutionId(snap.resolutionId);
    setDurationId(snap.durationId);
    setClipMultiplier(snap.clipMultiplier);
    setNumSteps(snap.numSteps);
    setSeed(snap.seed);
    // Only restore chain continuation for true multi-clip runs — not passive library preview.
    const multiClipRun = (clip.clip_count ?? 1) > 1;
    setAutocontinue(multiClipRun);
    setAutoconcat(multiClipRun && snap.autoconcat);
    setAudiocontinue(snap.audiocontinue);
    setShowOptions(true);
    setError(null);
  }

  async function deleteGeneration(clip: Clip) {
    if (busy) return;
    if (!confirm("Delete this video from the library?")) return;

    try {
      const r = await fetch(`${API}/api/clips/${encodeURIComponent(clip.id)}`, {
        method: "DELETE",
      });
      if (!r.ok) {
        setError("Could not delete");
        return;
      }

      revokeClipBlob(clip);
      const all = await fetchClips();
      setClips((prev) => replaceClipsFromServer(prev, all));
      syncClipSelection(all, clip);
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }

  function revokeAudioPreviewUrl(url: string | null) {
    if (url?.startsWith(BLOB_VIDEO_PREFIX)) {
      URL.revokeObjectURL(url);
    }
  }

  function resetAudioSelection() {
    setAudioPreviewUrl((prev) => {
      revokeAudioPreviewUrl(prev);
      return null;
    });
    setAudioFile(null);
    setAudioPath(null);
    setAudioName(null);
    setAudioDurationSeconds(null);
    setAudioStartSeconds(0);
    if (audioRef.current) audioRef.current.value = "";
  }

  function handleAudioFileSelected(file: File) {
    setAudioPreviewUrl((prev) => {
      revokeAudioPreviewUrl(prev);
      return URL.createObjectURL(file);
    });
    setAudioFile(file);
    setAudioName(file.name);
    setAudioPath(null);
    setAudioStartSeconds(0);
    setAudioDurationSeconds(null);
    void probeAudioDuration(file).then(setAudioDurationSeconds);
  }

  function clearAllMedia() {
    setImagePath(null);
    setImageName(null);
    setEndImagePath(null);
    setEndImageName(null);
    resetAudioSelection();
    setVideoPath(null);
    setVideoName(null);
    setSourceClipId(null);
    setConditioningVideoPath(null);
    setConditioningVideoName(null);
    setConditioningClipId(null);
    setConditioningVideoScale(1.0);
    setReferenceStrength(1.0);
    if (imageRef.current) imageRef.current.value = "";
    if (endImageRef.current) endImageRef.current.value = "";
    if (videoRef.current) videoRef.current.value = "";
    if (conditioningVideoRef.current) conditioningVideoRef.current.value = "";
  }

  function clearMediaForMode(nextMode: string) {
    if (
      !["i2v", "generate", "a2v", "keyframe", "ic_lora", "v2v", "lipdub", "face_swap", "id_lora"].includes(
        nextMode,
      )
    ) {
      setImagePath(null);
      setImageName(null);
      if (imageRef.current) imageRef.current.value = "";
    }
    if (nextMode !== "keyframe") {
      setEndImagePath(null);
      setEndImageName(null);
      if (endImageRef.current) endImageRef.current.value = "";
    }
    if (nextMode !== "a2v" && nextMode !== "lipdub" && nextMode !== "id_lora") {
      resetAudioSelection();
      setAudiocontinue(false);
    }
    if (!["retake", "extend", "lipdub", "face_swap"].includes(nextMode)) {
      setVideoPath(null);
      setVideoName(null);
      setSourceClipId(null);
      if (videoRef.current) videoRef.current.value = "";
    }
    if (nextMode !== "ic_lora" && nextMode !== "v2v") {
      setConditioningVideoPath(null);
      setConditioningVideoName(null);
      setConditioningClipId(null);
      setConditioningVideoScale(1.0);
      setReferenceStrength(1.0);
      if (conditioningVideoRef.current) conditioningVideoRef.current.value = "";
    }
    if (nextMode === "a2v") {
      setChainMethod("autocontinue");
    }
    if (
      nextMode === "ic_lora" ||
      nextMode === "v2v" ||
      nextMode === "face_swap" ||
      nextMode === "lipdub" ||
      nextMode === "id_lora"
    ) {
      setClipMultiplier(1);
      setAutocontinue(false);
      setAutoconcat(false);
    }
  }

  async function startNewProject() {
    setClips((prev) => {
      revokeBlobVideoUrls(prev);
      return [];
    });
    setChainId(null);
    setSelectedClipId(null);
    setPrompt("");
    setClipMultiplier(1);
    setAudiocontinue(false);
    setBusy(false);
    setProgress(null);
    setError(null);
    setLoraPresetIds(
      config?.preferred_lora_preset_ids?.length
        ? config.preferred_lora_preset_ids
        : config?.default_lora_preset_id && config.default_lora_preset_id !== "none"
          ? [config.default_lora_preset_id]
          : [],
    );
    clearAllMedia();
    try {
      await fetch(`${API}/api/session/clear`, { method: "POST" });
    } catch (err) {
      console.warn("Session clear failed", err);
    }
  }

  const needsImageUpload = mode === "i2v" || mode === "keyframe";
  const isA2v = mode === "a2v";
  const isV2v = mode === "v2v";
  const isIcLora = mode === "ic_lora";
  const isFaceSwap = mode === "face_swap";
  const isLipDub = mode === "lipdub";
  const isIdLora = mode === "id_lora";
  const pyavAvailable =
    config?.pyav_available ?? config?.audio_trim_available ?? false;
  const audioTrimAvailable = pyavAvailable;
  const needsEndImageUpload = mode === "keyframe";
  const needsVideoUpload =
    mode === "retake" || mode === "extend" || mode === "lipdub";
  const isT2vLike = mode === "generate" || mode === "i2v";
  const showChainMethodChoice =
    isT2vLike && !audiocontinue && isMultiClip;
  const chainMethodLabel =
    chainMethod === "native_extend" ? "extend video" : "autocontinue";
  const showChainedImageHint =
    isA2v &&
    (autocontinue || isMultiClip || audiocontinue) &&
    Boolean(imagePath);

  useEffect(() => {
    if (audioStartSeconds > audioStartSliderMax) {
      setAudioStartSeconds(audioStartSliderMax);
    }
  }, [audioStartSeconds, audioStartSliderMax]);

  useEffect(() => {
    return () => {
      revokeAudioPreviewUrl(audioPreviewUrl);
    };
  }, [audioPreviewUrl]);

  function probeAudioDuration(file: File): Promise<number | null> {
    return new Promise((resolve) => {
      const url = URL.createObjectURL(file);
      const el = document.createElement("audio");
      el.preload = "metadata";
      const cleanup = () => URL.revokeObjectURL(url);
      el.onloadedmetadata = () => {
        const duration = Number.isFinite(el.duration) ? el.duration : null;
        cleanup();
        resolve(duration);
      };
      el.onerror = () => {
        cleanup();
        resolve(null);
      };
      el.src = url;
    });
  }

  async function uploadFile(file: File, kind: string): Promise<string> {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch(`${API}/api/upload?kind=${kind}`, {
      method: "POST",
      body: fd,
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      const detail = err.detail;
      const message =
        typeof detail === "string"
          ? detail
          : Array.isArray(detail)
            ? detail.map((d: { msg?: string }) => d.msg).join("; ")
            : `Upload failed: ${kind}`;
      throw new Error(message);
    }
    const data = await r.json();
    return data.path as string;
  }

  async function changeModel(newModel: string, restart: boolean) {
    setModel(newModel);
    localStorage.setItem(MODEL_PREF_KEY, newModel);
    const r = await fetch(`${API}/api/config/model`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: newModel, restart_server: restart }),
    });
    if (!r.ok) return;
    const data = await r.json();
    setConfig((c) =>
      c
        ? {
            ...c,
            preferred_model: data.preferred_model ?? newModel,
            server_connected: data.server_connected ?? c.server_connected,
          }
        : c,
    );
  }

  function cacheClipVideoLocally(clipId: string, serverUrl: string) {
    cacheVideoAsBlobUrl(serverUrl)
      .then((blobUrl) => {
        setClips((prev) => {
          const existing = prev.find((c) => c.id === clipId);
          const oldUrl = existing?.video_url;
          if (oldUrl?.startsWith(BLOB_VIDEO_PREFIX)) {
            URL.revokeObjectURL(oldUrl);
          }
          return prev.map((c) =>
            c.id === clipId ? { ...c, video_url: blobUrl } : c,
          );
        });
      })
      .catch((err) => {
        console.warn("Failed to cache clip video locally", err);
      });
  }

  async function cancelActiveRun() {
    if (!activeRunId) return;
    try {
      await fetch(`${API}/api/runs/${activeRunId}/cancel`, { method: "POST" });
      setProgress({ phase: "cancelled", message: "Cancelling…" });
    } catch (err) {
      console.warn("Cancel request failed", err);
    }
  }

  async function subscribeRun(runId: string, runChainId: string) {
    closeRunSubscription();
    setActiveRunId(runId);
    let closed = false;
    let autoconcatRun = false;
    let audiocontinueRun = false;
    let streamFinalOnly = false;
    const es = new EventSource(`${API}/api/runs/${runId}/events`);
    runEventSourceRef.current = es;

    const finishRun = async () => {
      if (closed) return;
      closed = true;
      if (runEventSourceRef.current === es) {
        runEventSourceRef.current = null;
      }
      es.close();
      setActiveRunId(null);
      setBusy(false);
      setProgress(null);
      setChainId(runChainId);
      const chainClips = await fetchClips(runChainId);
      setClips((prev) => replaceChainClips(prev, runChainId, chainClips));
      setSelectedClipId(pickPlaybackClip(chainClips, runChainId));
    };

    const setFromProtocol = (e: Record<string, unknown>) => {
      if (e.type === "queue_status") {
        setProgress({
          phase: "queued",
          message: `Queue position ${e.position ?? "?"}`,
        });
      } else if (
        e.type === "generation_keepalive" ||
        e.type === "generation_status_ack"
      ) {
        setProgress((prev) => applyProgressEvent(prev, e));
      } else if (e.type === "gpu_assigned") {
        setProgress({ phase: "generating", message: "GPU assigned — starting…" });
      } else if (e.type === "ltx2_segment_start") {
        setProgress((prev) => ({
          phase: "generating",
          message: "Denoising…",
          ...prev,
        }));
      } else if (e.type === "error") {
        setError(String(e.message || "Generation error"));
      }
    };
    es.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "run_started") {
        autoconcatRun = Boolean(msg.autoconcat);
        audiocontinueRun = Boolean(msg.audiocontinue);
        const clipCount = Number(msg.clip_count ?? 0);
        streamFinalOnly =
          autoconcatRun || (Boolean(msg.autocontinue) && clipCount > 1);
        const chainLabel =
          msg.chain_method === "native_extend" ? "extend video" : "autocontinue";
        setProgress({
          phase: "starting",
          message: audiocontinueRun
            ? `Music video: ${msg.clip_count ?? "?"} clips (audiocontinue)…`
            : autoconcatRun
              ? `Generating ${msg.clip_count ?? "?"} clips (${chainLabel} + autoconcat)…`
              : streamFinalOnly
                ? `Generating ${msg.clip_count ?? "?"} clips (${chainLabel})…`
                : "Starting generation…",
        });
      } else if (msg.type === "clip_started") {
        const idx = typeof msg.index === "number" ? msg.index + 1 : "?";
        const total = msg.total_clips ?? "?";
        setProgress({
          phase: "running",
          message:
            (autoconcatRun || audiocontinueRun) && total !== "?"
              ? audiocontinueRun
                ? `Music video clip ${idx}/${total}…`
                : `Generating clip ${idx}/${total}…`
              : "Starting clip…",
        });
      } else if (msg.type === "generation_progress") {
        setProgress((prev) => applyProgressEvent(prev, msg));
      } else if (msg.type === "protocol") {
        setFromProtocol(msg.event as Record<string, unknown>);
      } else if (msg.type === "download_progress") {
        setProgress({
          phase: "downloading",
          message: `Receiving video ${msg.kb} KB`,
          kb: msg.kb,
        });
      } else if (msg.type === "clip_done") {
        const idx = typeof msg.index === "number" ? msg.index + 1 : "?";
        const total = msg.total_clips ?? "?";
        if (streamFinalOnly) {
          setProgress({
            phase: "clip_done",
            message:
              total !== "?"
                ? autoconcatRun
                  ? `Clip ${idx}/${total} done — ${idx === total ? "merging…" : "continuing…"}`
                  : `Clip ${idx}/${total} done — continuing…`
                : "Clip saved — continuing…",
          });
          // Autoconcat: video arrives on `merged`. Autocontinue: only the last clip has video_url.
          if (msg.clip_id && msg.video_url) {
            const clipId = msg.clip_id as string;
            const serverUrl = msg.video_url as string;
            setSelectedClipId(clipId);
            setClips((prev) => {
              const others = prev.filter((c) => c.id !== clipId);
              const existing = prev.find((c) => c.id === clipId);
              return [
                ...others,
                {
                  ...(existing ?? {}),
                  id: clipId,
                  video_url: serverUrl,
                  chain_id: runChainId,
                  status: "done",
                  label: existing?.label ?? "CURRENT",
                  prompt: existing?.prompt ?? "",
                  filename: existing?.filename ?? "",
                  clip_index: existing?.clip_index ?? 0,
                  mode: existing?.mode ?? "generate",
                  created_at: existing?.created_at ?? new Date().toISOString(),
                } as Clip,
              ];
            });
            cacheClipVideoLocally(clipId, serverUrl);
          }
        } else if (msg.clip_id && msg.video_url) {
          setProgress({ phase: "clip_done", message: "Clip saved" });
          const clipId = msg.clip_id as string;
          const serverUrl = msg.video_url as string;
          setSelectedClipId(clipId);
          setClips((prev) => {
            const others = prev.filter((c) => c.id !== clipId);
            const existing = prev.find((c) => c.id === clipId);
            return [
              ...others,
              {
                ...(existing ?? {}),
                id: clipId,
                video_url: serverUrl,
                chain_id: runChainId,
                status: "done",
                label: existing?.label ?? "CURRENT",
                prompt: existing?.prompt ?? "",
                filename: existing?.filename ?? "",
                clip_index: existing?.clip_index ?? 0,
                mode: existing?.mode ?? "generate",
                created_at: existing?.created_at ?? new Date().toISOString(),
              } as Clip,
            ];
          });
          cacheClipVideoLocally(clipId, serverUrl);
        } else {
          setProgress({ phase: "clip_done", message: "Clip saved" });
        }
      } else if (msg.type === "merged") {
        setProgress({ phase: "merged", message: "Clips merged" });
        const clipId = msg.clip_id as string | undefined;
        const videoUrl = msg.video_url as string | undefined;
        if (clipId && videoUrl) {
          setSelectedClipId(clipId);
          setClips((prev) => {
            const chainPrompt =
              prev.find((c) => c.chain_id === runChainId)?.prompt ?? "Merged clip";
            const others = prev.filter((c) => c.chain_id !== runChainId);
            return [
              ...others,
              {
                id: clipId,
                video_url: videoUrl,
                filename: (msg.filename as string) ?? "",
                chain_id: runChainId,
                label: "MERGED",
                status: "done",
                prompt: chainPrompt,
                clip_index: 0,
                mode: mode,
                created_at: new Date().toISOString(),
              } as Clip,
            ];
          });
          cacheClipVideoLocally(clipId, videoUrl);
        }
        fetchClips(runChainId).then((chainClips) => {
          setClips((prev) => replaceChainClips(prev, runChainId, chainClips));
          setSelectedClipId(pickPlaybackClip(chainClips, runChainId) ?? clipId ?? null);
        });
      } else if (msg.type === "run_cancelled") {
        setProgress({
          phase: "cancelled",
          message: String(msg.message || "Generation cancelled"),
        });
        finishRun();
      } else if (msg.type === "run_complete" || msg.type === "run_done") {
        finishRun();
      } else if (msg.type === "error" || msg.type === "clip_failed") {
        setError(msg.error || msg.message || "Failed");
        if (runEventSourceRef.current === es) {
          runEventSourceRef.current = null;
        }
        es.close();
        setActiveRunId(null);
        setBusy(false);
      }
    };
    es.onerror = () => {
      if (closed) return;
      closed = true;
      if (runEventSourceRef.current === es) {
        runEventSourceRef.current = null;
      }
      es.close();
      setBusy(false);
      setProgress(null);
      setError((prev) => prev ?? "Lost connection to server while waiting for progress.");
    };
  }

  async function handleGenerate() {
    if (!canSubmit || !prompt.trim() || busy) return;
    setError(null);
    setBusy(true);
    setProgress({ phase: "starting", message: "Submitting…" });

    const isChainEdit = willContinueChain;

    const durationPreset = config?.duration_presets.find((d) => d.id === durationId);

    const body: Record<string, unknown> = {
      prompt: prompt.trim(),
      mode,
      width: resolution.width,
      height: resolution.height,
      duration_seconds: durationSeconds,
      num_frames: durationPreset?.num_frames,
      clip_count: clipMultiplier,
      num_steps: numSteps,
      autocontinue: autocontinue || isMultiClip || audiocontinue,
      autoconcat: autoconcat || isMultiClip || audiocontinue,
      audiocontinue: audiocontinue && mode === "a2v",
      chain_method: chainMethod,
      enhance_prompt: enhancePrompt,
      pipeline_profile: pipelineProfile,
      chain_id: isChainEdit ? chainId : undefined,
      continue_from: isChainEdit ? activeClip?.id : undefined,
    };
    const neg = negativePrompt.trim();
    if (neg) {
      body.negative_prompt = neg;
    }
    if (mode === "retake") {
      body.retake_start = retakeStart;
      body.retake_end = retakeEnd;
    }
    if (mode === "extend") {
      body.extend_frames = extendFrames;
      body.extend_direction = extendDirection;
    }
    if (
      (mode === "i2v" ||
        mode === "generate" ||
        mode === "a2v" ||
        mode === "keyframe" ||
        mode === "ic_lora" ||
        mode === "v2v" ||
        mode === "lipdub" ||
        mode === "face_swap" ||
        mode === "id_lora") &&
      imagePath
    ) {
      body.image_path = imagePath;
    }
    if (mode === "keyframe" && endImagePath) {
      body.end_image_path = endImagePath;
    }
    let resolvedAudioPath = audioPath;
    if ((mode === "a2v" || mode === "lipdub" || mode === "id_lora") && !resolvedAudioPath && audioFile) {
      resolvedAudioPath = await uploadFile(audioFile, "audio");
      setAudioPath(resolvedAudioPath);
    }
    if (mode === "a2v" && resolvedAudioPath) {
      body.audio_path = resolvedAudioPath;
      if (mode === "a2v" && audioStartSeconds > 0) {
        body.audio_start_seconds = audioStartSeconds;
        if (audioDurationSeconds && audioDurationSeconds > 0) {
          body.audio_source_duration_seconds = audioDurationSeconds;
        }
      }
    }
    if ((mode === "lipdub" || mode === "id_lora") && resolvedAudioPath) {
      body.audio_path = resolvedAudioPath;
    }
    if (mode === "ic_lora" || mode === "v2v" || mode === "lipdub") {
      if (mode === "ic_lora" || mode === "v2v") {
        if (conditioningVideoPath) {
          body.conditioning_video_path = conditioningVideoPath;
          body.conditioning_video_scale = conditioningVideoScale;
        } else if (conditioningClipId) {
          body.conditioning_clip_id = conditioningClipId;
          body.conditioning_video_scale = conditioningVideoScale;
        }
      }
      body.reference_strength = referenceStrength;
    }
    if (mode === "ic_lora" && skipStage2) {
      body.skip_stage_2 = true;
    }
    if ((mode === "retake" || mode === "extend" || mode === "lipdub" || mode === "face_swap") && sourceClipId) {
      body.source_clip_id = sourceClipId;
    } else if ((mode === "retake" || mode === "extend" || mode === "lipdub" || mode === "face_swap") && videoPath) {
      body.video_path = videoPath;
    }
    if (seed.trim()) {
      body.seed = parseInt(seed, 10);
    } else {
      body.seed = -1;
    }

    const selectedLoras = (config?.lora_presets ?? []).filter(
      (p) => loraPresetIds.includes(p.id) && p.spec,
    );
    if (mode === "lipdub" && selectedLoras.length !== 1) {
      setError("LipDub requires exactly one LoRA — use the LipDub IC-LoRA preset.");
      setBusy(false);
      setProgress(null);
      return;
    }
    if (mode === "face_swap" && selectedLoras.length !== 1) {
      setError("Face swap requires exactly one LoRA — use the head-swap preset.");
      setBusy(false);
      setProgress(null);
      return;
    }
    if (mode === "id_lora" && selectedLoras.length !== 1) {
      setError("ID-LoRA requires exactly one adapter — use CelebV-HQ or TalkVid.");
      setBusy(false);
      setProgress(null);
      return;
    }
    if (mode === "id_lora" && !imagePath) {
      setError("ID-LoRA requires a first-frame reference image.");
      setBusy(false);
      setProgress(null);
      return;
    }
    if (mode === "id_lora" && !resolvedAudioPath) {
      setError("ID-LoRA requires reference audio (~5s identity sample).");
      setBusy(false);
      setProgress(null);
      return;
    }
    if (selectedLoras.length) {
      body.lora_specs = selectedLoras.map((p) => [p.spec, p.scale]);
    }

    try {
      const r = await fetch(`${API}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        const detail = err.detail;
        const message =
          typeof detail === "string"
            ? detail
            : Array.isArray(detail)
              ? detail.map((d: { msg?: string }) => d.msg).join("; ")
              : "Generate failed";
        throw new Error(message);
      }
      const data = await r.json();
      setChainId(data.chain_id);
      setSelectedClipId(null);
      setProgress(
        data.started_immediately
          ? { phase: "starting", message: "Starting…" }
          : { phase: "queued", message: "Queued — waiting for current job…" },
      );
      subscribeRun(data.run_id, data.chain_id);
      const chainClips = await fetchClips(data.chain_id);
      setClips((prev) => mergeClips(prev, chainClips));
    } catch (e) {
      setError(String(e));
      setBusy(false);
      setProgress(null);
    }
  }

  const serverOk = config?.server_connected;

  const endpointLabel = useMemo(() => {
    if (typeof window === "undefined") return config?.server_url ?? "";
    const ws = `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws`;
    return ws;
  }, [config?.server_url]);

  const canSubmit = useMemo(() => {
    if (!prompt.trim() || busy || !serverOk) return false;
    if ((isV2v || isIcLora) && loraBusy) return false;
    // V2V: reference video required; LoRA optional (pure motion transfer OK).
    if (isV2v && !hasConditioningVideo) return false;
    // IC-LoRA: need a selected LoRA (HDR default, or Union / custom).
    if (isIcLora && loraPresetIds.length === 0) return false;
    // Union Control needs a reference video; HDR allows T2V / I2V without video.
    if (isIcLora && icLoraUnionSelected && !hasConditioningVideo) return false;
    if (mode === "lipdub" && loraBusy) return false;
    if (mode === "face_swap" && loraBusy) return false;
    if (mode === "face_swap" && loraPresetIds.length !== 1) return false;
    if (mode === "id_lora" && loraBusy) return false;
    if (mode === "id_lora" && loraPresetIds.length !== 1) return false;
    if (mode === "id_lora" && !imagePath) return false;
    if (mode === "id_lora" && !audioPath && !audioFile) return false;
    const continuing = willContinueChain;
    if (mode === "i2v" && !imagePath && !continuing) return false;
    if (mode === "face_swap" && !imagePath) return false;
    if (mode === "a2v" && !audioPath && !audioFile) return false;
    if (mode === "a2v" && audioStartSeconds > 0 && !audioTrimAvailable) return false;
    if (audiocontinue && !pyavAvailable) return false;
    if ((mode === "retake" || mode === "extend" || mode === "lipdub" || mode === "face_swap") && !hasVideoSource) {
      return false;
    }
    if (mode === "retake" && retakeEnd <= retakeStart) return false;
    if (mode === "keyframe" && (!imagePath || !endImagePath)) return false;
    return true;
  }, [
    prompt,
    busy,
    serverOk,
    mode,
    isV2v,
    isIcLora,
    imagePath,
    endImagePath,
    audioPath,
    audioFile,
    audiocontinue,
    clipMultiplier,
    audioTrimAvailable,
    pyavAvailable,
    videoPath,
    sourceClipId,
    hasVideoSource,
    hasConditioningVideo,
    loraPresetIds,
    icLoraUnionSelected,
    autocontinue,
    activeClip,
    chainId,
    loraBusy,
    willContinueChain,
    audioStartSeconds,
    retakeStart,
    retakeEnd,
  ]);

  const fitPromptHeight = useCallback(() => {
    const el = promptRef.current;
    if (!el) return;
    el.style.height = "auto";
    const max = 200;
    el.style.height = `${Math.min(el.scrollHeight, max)}px`;
  }, []);

  useLayoutEffect(() => {
    fitPromptHeight();
  }, [prompt, fitPromptHeight]);

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="brand-mark">LTX-WS</span>
          <span className="brand-sub">Videofentanyl</span>
        </div>
        <div className="header-status">
          <button
            type="button"
            className="btn-secondary"
            title="Clear the library and reset the session"
            onClick={startNewProject}
          >
            New project
          </button>
          <span
            className={`status-dot ${serverOk ? "ok" : "off"}`}
            title={endpointLabel}
          />
          {serverOk ? "Server connected" : "Server offline"}
        </div>
      </header>

      <div className="app-body">
        <div className="app-main">
        <section className="player-section">
          <div className="player-wrap">
            {activeClip?.video_url ? (
              <video
                ref={playerVideoRef}
                className="player"
                src={activeClip.video_url}
                crossOrigin={
                  activeClip.video_url.startsWith(BLOB_VIDEO_PREFIX)
                    ? undefined
                    : "anonymous"
                }
                controls
                autoPlay
                loop
                playsInline
              />
            ) : (
              <div className="player placeholder">
                {busy ? progress?.message ?? "Generating…" : "Your video will appear here"}
              </div>
            )}
            {busy && (
              <div className="progress-overlay">
                <div className="progress-bar">
                  {progress?.pct != null ? (
                    <div
                      className="progress-fill"
                      style={{ width: `${Math.min(100, progress.pct)}%` }}
                    />
                  ) : (
                    <div className="progress-pulse" />
                  )}
                </div>
                <div className="progress-overlay-row">
                  <span>{progress?.message ?? "Working…"}</span>
                  {activeRunId && progress?.phase !== "cancelled" && (
                    <button
                      type="button"
                      className="btn-cancel"
                      onClick={() => void cancelActiveRun()}
                    >
                      Cancel
                    </button>
                  )}
                </div>
              </div>
            )}
            {activeClip?.video_url && !busy && (
              <button
                type="button"
                className="player-capture-btn"
                disabled={savingFrame}
                onClick={() => void saveCurrentFrame()}
                title="Save frame to library"
                aria-label="Save frame to library"
              >
                {savingFrame ? (
                  <span className="player-capture-spinner" aria-hidden />
                ) : (
                  <svg
                    className="player-capture-icon"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.75"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden
                  >
                    <path d="M4 7h2l2-3h8l2 3h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2z" />
                    <circle cx="12" cy="13" r="4" />
                  </svg>
                )}
              </button>
            )}
          </div>

          {error && <div className="error-banner">{error}</div>}
          {(showChainPicker || willContinueChain) && (
            <div className="player-context">
              <div className="player-context-body">
                {showChainPicker && (
                  <>
                    <label className="player-context-label">
                      Chain clip
                      <select
                        className="chain-picker-select"
                        value={selectedClipId ?? ""}
                        onChange={(e) => {
                          const c = chainParts.find((x) => x.id === e.target.value);
                          if (c) applyClipSelection(c);
                        }}
                      >
                        {chainParts.map((c) => (
                          <option key={c.id} value={c.id}>
                            {c.label}
                            {c.num_frames
                              ? ` · ${formatDuration(c.num_frames, config?.defaults.fps ?? 24)}`
                              : ""}
                          </option>
                        ))}
                      </select>
                    </label>
                    <p className="player-context-hint">
                      Multiple clips belong to one chained run (for example after
                      autocontinue or an edit). Pick which to preview.
                    </p>
                  </>
                )}
                {willContinueChain && (
                  <p className="player-context-hint player-context-hint-continue">
                    Autocontinue is on — your next prompt extends the selected clip.
                    Turn off autocontinue in options, or start a new generation for a
                    separate video.
                  </p>
                )}
              </div>
              <button
                type="button"
                className="btn-secondary btn-fresh-generation"
                onClick={beginFreshGeneration}
              >
                Start new generation
              </button>
            </div>
          )}
        </section>

        <section className="composer">
          <div className="prompt-row">
            <div className="prompt-field-wrap">
              <textarea
                ref={promptRef}
                className="prompt-input"
                rows={1}
                placeholder={
                  isIdLora
                    ? idLoraPromptPlaceholder
                    : willContinueChain
                      ? "What do you want to edit?"
                      : "What video do you want to create?"
                }
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void handleGenerate();
                  }
                }}
                disabled={busy}
              />
              {(showNegativePrompt || !!negativePrompt.trim()) && (
                <input
                  type="text"
                  className="negative-prompt-input"
                  placeholder="Negative prompt (optional)"
                  value={negativePrompt}
                  onChange={(e) => setNegativePrompt(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      void handleGenerate();
                    }
                  }}
                  disabled={busy}
                  aria-label="Negative prompt"
                />
              )}
              <div className="prompt-field-actions">
                <button
                  type="button"
                  className={`btn-prompt-action${showNegativePrompt || negativePrompt.trim() ? " is-active" : ""}`}
                  onClick={() => {
                    setShowNegativePrompt((v) => {
                      const next = !v;
                      if (!next) setNegativePrompt("");
                      return next;
                    });
                  }}
                  disabled={busy}
                  aria-pressed={showNegativePrompt || !!negativePrompt.trim()}
                  aria-label={
                    showNegativePrompt || negativePrompt.trim()
                      ? "Hide negative prompt"
                      : "Show negative prompt"
                  }
                  title="Show or hide negative prompt"
                >
                  Negative
                </button>
                <button
                  type="button"
                  className="btn-prompt-action"
                  onClick={() => setPrompt("")}
                  disabled={busy || !prompt}
                  aria-label="Clear prompt"
                  title="Clear prompt"
                >
                  Clear
                </button>
              </div>
            </div>
            <button
              type="button"
              className="btn-generate"
              onClick={handleGenerate}
              disabled={!canSubmit}
            >
              ↑
            </button>
          </div>

          <button
            type="button"
            className="options-toggle"
            onClick={() => setShowOptions((v) => !v)}
          >
            {showOptions ? "Hide options" : "Show options"}
          </button>

          {showOptions && config && (
            <div className="options-panel">
              <div className="options-row">
                <label>
                  Model
                  <select
                    value={model}
                    onChange={(e) => changeModel(e.target.value, false)}
                  >
                    {config.models.map((m) => (
                      <option key={m.id} value={m.id}>{m.label}</option>
                    ))}
                  </select>
                </label>
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => changeModel(model, true)}
                  disabled={config.embedded}
                >
                  Apply model (restart server)
                </button>
              </div>
              <p className="hint">
                {config.embedded && config.active_model
                  ? `Running: ${config.active_model}. `
                  : ""}
                {config.model_note}
              </p>

              <div className="options-grid options-grid-compact">
                <label className="opt-mode">
                  Mode
                  <select
                    value={mode}
                    onChange={(e) => {
                      const next = e.target.value;
                      setMode(next);
                      clearMediaForMode(next);
                    }}
                  >
                    {config.generation_modes.map((m) => (
                      <option key={m.id} value={m.id}>{m.label}</option>
                    ))}
                  </select>
                </label>
                <label className="opt-resolution">
                  Resolution
                  <select
                    value={resolutionId}
                    onChange={(e) => setResolutionId(e.target.value)}
                  >
                    {config.resolution_presets.map((r) => (
                      <option key={r.id} value={r.id}>{r.label}</option>
                    ))}
                  </select>
                </label>
                <label className="opt-narrow">
                  Duration
                  <select
                    value={durationId}
                    onChange={(e) => setDurationId(e.target.value)}
                    disabled={isLipDub}
                    title={
                      isLipDub
                        ? "LipDub length follows the reference video"
                        : undefined
                    }
                  >
                    {config.duration_presets.map((d) => (
                      <option key={d.id} value={d.id} title={d.label}>
                        {`~${d.seconds}s`}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="opt-narrow">
                  Clips
                  <select
                    value={clipMultiplier}
                    disabled={isV2v || isIcLora || isFaceSwap || isLipDub || isIdLora}
                    onChange={(e) => setClipMultiplier(Number(e.target.value))}
                  >
                    {Array.from(
                      { length: config.clip_multiplier_max ?? 10 },
                      (_, i) => i + 1,
                    ).map((n) => (
                      <option key={n} value={n}>
                        ×{n}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="opt-narrow">
                  Steps
                  <input
                    type="number"
                    min={1}
                    max={50}
                    value={numSteps}
                    onChange={(e) => setNumSteps(Number(e.target.value))}
                  />
                </label>
                <label className="opt-seed">
                  Seed
                  <input
                    type="text"
                    placeholder="random"
                    value={seed}
                    onChange={(e) => setSeed(e.target.value)}
                  />
                </label>
              </div>

              {(loraActivity.phase === "working" || loraActivity.phase === "error") && (
                <div
                  className={`lora-status-banner ${
                    loraActivity.phase === "error" ? "error" : "working"
                  }`}
                  role="status"
                  aria-live="polite"
                >
                  {loraActivity.phase === "working" && (
                    <>
                      <span className="lora-status-spinner" aria-hidden />
                      <span>
                        {loraActivity.downloading
                          ? `Downloading: ${loraActivity.label}`
                          : `Verifying: ${loraActivity.label}`}
                        {loraActivity.total > 1
                          ? ` (${loraActivity.index}/${loraActivity.total})`
                          : ""}
                      </span>
                    </>
                  )}
                  {loraActivity.phase === "error" && (
                    <span>{loraActivity.message}</span>
                  )}
                </div>
              )}

              <div className="lora-row">
                <div className="lora-row-select">
                  <span className="lora-field-label">LoRA</span>
                  <LoraMultiSelect
                    presets={(config.lora_presets ?? []).filter((p) => p.id !== "none")}
                    selectedIds={loraPresetIds}
                    disabled={loraBusy || addingCustomLora || isFaceSwap || isLipDub || isIdLora}
                    onToggle={(id, checked) => toggleLoraPreset(id, checked)}
                    onRemovePreset={(preset) => void removeLoraPreset(preset)}
                  />
                </div>
                <div className="lora-row-add">
                  <input
                    type="text"
                    className="lora-add-url"
                    placeholder="URL or path"
                    aria-label="LoRA URL or file path"
                    value={customLoraUrl}
                    disabled={addingCustomLora}
                    onChange={(e) => setCustomLoraUrl(e.target.value)}
                  />
                  <input
                    type="text"
                    className="lora-add-name"
                    placeholder="Label"
                    aria-label="LoRA display name"
                    value={customLoraLabel}
                    disabled={addingCustomLora}
                    onChange={(e) => setCustomLoraLabel(e.target.value)}
                  />
                  <input
                    type="number"
                    className="lora-add-scale"
                    min={0}
                    max={2}
                    step={0.05}
                    aria-label="LoRA strength"
                    title="Strength (0–2)"
                    placeholder="1.0"
                    value={customLoraScale}
                    disabled={addingCustomLora}
                    onChange={(e) => setCustomLoraScale(e.target.value)}
                  />
                  <button
                    type="button"
                    className="btn-secondary btn-compact lora-add-btn"
                    disabled={!customLoraUrl.trim() || addingCustomLora || loraBusy}
                    onClick={() => void addCustomLora()}
                  >
                    {addingCustomLora ? "…" : "Add"}
                  </button>
                </div>
              </div>

              {isMultiClip && !audiocontinue && (
                <p className="hint hint-inline">
                  ~{totalDurationSeconds}s total · {chainMethodLabel}
                </p>
              )}

              {showChainMethodChoice && (
                <ChainMethodPicker
                  chainMethod={chainMethod}
                  onChange={setChainMethod}
                />
              )}

              <div className="options-checks">
                {mode === "a2v" && (
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={audiocontinue}
                      onChange={(e) => {
                        const on = e.target.checked;
                        setAudiocontinue(on);
                        if (on && clipMultiplier < 2) {
                          setClipMultiplier(2);
                        }
                      }}
                      disabled={(!audioPath && !audioFile) || !pyavAvailable}
                    />
                    Audiocontinue
                  </label>
                )}
                {mode === "a2v" && !pyavAvailable && (
                  <p className="hint hint-inline">Audiocontinue requires PyAV (pip install av).</p>
                )}
                <label className="check">
                  <input
                    type="checkbox"
                    checked={enhancePrompt}
                    onChange={(e) => setEnhancePrompt(e.target.checked)}
                  />
                  Enhance prompt
                </label>
                {!isMultiClip && !audiocontinue && !isV2v && !isIcLora && !isFaceSwap && !isLipDub && !isIdLora && (
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={autocontinue}
                      onChange={(e) => setAutocontinue(e.target.checked)}
                    />
                    Chain clips
                  </label>
                )}
                <label className="check">
                  <input
                    type="checkbox"
                    checked={autoconcat}
                    onChange={(e) => setAutoconcat(e.target.checked)}
                    disabled={
                      isMultiClip || audiocontinue || isV2v || isIcLora || isFaceSwap || isLipDub || isIdLora
                    }
                  />
                  Autoconcat
                </label>
                <label className="opt-profile">
                  Profile
                  <select
                    value={pipelineProfile}
                    onChange={(e) => setPipelineProfile(e.target.value)}
                  >
                    {(config?.pipeline_profiles ?? [
                      { id: "distilled", label: "Distilled" },
                      { id: "two_stage", label: "Two-stage" },
                      { id: "hq", label: "HQ" },
                      { id: "one_stage", label: "One-stage" },
                    ]).map((p) => (
                      <option key={p.id} value={p.id}>{p.label}</option>
                    ))}
                  </select>
                </label>
              </div>

              {mode === "extend" && (
                <div className="options-grid">
                  <label>
                    Extend frames (latent)
                    <input
                      type="number"
                      min={1}
                      value={extendFrames}
                      onChange={(e) => setExtendFrames(Number(e.target.value))}
                    />
                    <span className="hint hint-inline">
                      ≈ ~{((extendFrames * 8) / 24).toFixed(1)}s added @ 24fps ({extendFrames * 8}{" "}
                      pixel frames)
                    </span>
                  </label>
                  <label>
                    Direction
                    <select
                      value={extendDirection}
                      onChange={(e) => setExtendDirection(e.target.value)}
                    >
                      <option value="after">After</option>
                      <option value="before">Before</option>
                    </select>
                  </label>
                </div>
              )}

              {(isA2v || isV2v || isIcLora || isFaceSwap || isLipDub || isIdLora || needsImageUpload || needsVideoUpload || needsEndImageUpload) && (
                <div className="media-panel">
                  {isIdLora && (
                    <>
                      <span className="media-panel-title">ID-LoRA inputs</span>
                      <p className="hint hint-inline">
                        First-frame face image + ~5s reference audio for <strong>identity</strong> only.
                        The model generates new speech from your{" "}
                        <code>[VISUAL] / [SPEECH] / [SOUNDS]</code> prompt — reference audio is not
                        remuxed as the soundtrack. Requires dev MLX weights. Defaults to CelebV-HQ
                        ID-LoRA (TalkVid available in the LoRA list).
                      </p>
                      <label className="media-upload">
                        <span className="media-upload-label">First-frame image (required)</span>
                        <input
                          ref={imageRef}
                          type="file"
                          accept="image/*"
                          onChange={async (e) => {
                            const f = e.target.files?.[0];
                            if (f) {
                              setImagePath(await uploadFile(f, "image"));
                              setImageName(f.name);
                            }
                          }}
                        />
                        <span className="media-upload-hint">
                          {imageName ?? "Choose face / first frame…"}
                        </span>
                      </label>
                      <label className="media-upload">
                        <span className="media-upload-label">
                          Reference audio (required, identity ~5s)
                        </span>
                        <input
                          ref={audioRef}
                          type="file"
                          accept="audio/*,.wav,.mp3,.m4a,.aac,.flac,.ogg"
                          onChange={(e) => {
                            const f = e.target.files?.[0];
                            if (f) handleAudioFileSelected(f);
                          }}
                        />
                        <span className="media-upload-hint">
                          {audioName ?? (audioPath ? "✓ audio selected" : "Choose reference WAV/MP3…")}
                        </span>
                      </label>
                      {(config?.lora_presets ?? [])
                        .filter((p) => loraPresetIds.includes(p.id) && p.spec)
                        .map((p) => (
                          <label key={p.id} className="v2v-lora-strength-row">
                            <span className="v2v-lora-strength-label" title={p.label}>
                              ID-LoRA strength
                            </span>
                            <input
                              type="number"
                              min={0}
                              max={2}
                              step={0.01}
                              value={p.scale}
                              disabled={busy || loraBusy}
                              title="ID-LoRA adapter strength"
                              aria-label={`LoRA strength for ${p.label}`}
                              onChange={(e) =>
                                updateLoraPresetScale(p.id, Number(e.target.value))
                              }
                            />
                          </label>
                        ))}
                    </>
                  )}
                  {isLipDub && (
                    <>
                      <span className="media-panel-title">LipDub inputs</span>
                      <p className="hint hint-inline">
                        Lip-sync / dub a reference clip: visual motion from the video,
                        voice tone from uploaded audio (or the video&apos;s audio track),
                        and new dialogue in your prompt. Frame count follows the reference
                        video (duration control disabled). Requires the LipDub IC-LoRA.
                      </p>
                      <label className="ic-lora-scale">
                        Reference attention strength
                        <input
                          type="number"
                          min={0}
                          max={1}
                          step={0.05}
                          value={referenceStrength}
                          title="LipDub reference_strength (0–1)"
                          onChange={(e) =>
                            setReferenceStrength(
                              Math.min(1, Math.max(0, Number(e.target.value) || 0)),
                            )
                          }
                        />
                      </label>
                      {!config?.lipdub_default_spec?.includes("buckets/audiohacking") && (
                        <p className="hint hint-inline">
                          LipDub weights are gated on Hugging Face — accept access at{" "}
                          <a
                            href={
                              config?.lipdub_official_hf_url ??
                              "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-LipDub"
                            }
                            target="_blank"
                            rel="noreferrer"
                          >
                            Lightricks/LTX-2.3-22b-IC-LoRA-LipDub
                          </a>
                          , set <code>HF_TOKEN</code> (or <code>huggingface-cli login</code>
                          ), then add a <strong>custom LoRA</strong> with the official resolve
                          URL or set <code>{config?.lipdub_env_var ?? "LTX_WS_LIPDUB_LORA"}</code>{" "}
                          to a local <code>.safetensors</code> path on the server.
                        </p>
                      )}
                      <label className="media-upload">
                        <span className="media-upload-label">
                          Voice tone audio (optional if video has audio)
                        </span>
                        <input
                          ref={audioRef}
                          type="file"
                          accept="audio/*"
                          onChange={(e) => {
                            const f = e.target.files?.[0];
                            if (f) handleAudioFileSelected(f);
                          }}
                        />
                        <span className="media-upload-hint">
                          {audioName ?? "Choose voice-tone audio…"}
                        </span>
                      </label>
                      <label className="media-upload">
                        <span className="media-upload-label">
                          I2V anchor image (optional)
                        </span>
                        <input
                          ref={imageRef}
                          type="file"
                          accept="image/*"
                          onChange={async (e) => {
                            const f = e.target.files?.[0];
                            if (f) {
                              setImagePath(await uploadFile(f, "image"));
                              setImageName(f.name);
                            }
                          }}
                        />
                        <span className="media-upload-hint">
                          {imageName ?? "Choose anchor image…"}
                        </span>
                      </label>
                    </>
                  )}
                  {isFaceSwap && (
                    <>
                      <span className="media-panel-title">Face swap inputs</span>
                      <p className="hint hint-inline">
                        Face identity image + reference performance video. Uses the BFS V3
                        composite guide (green side panel + face). Head-swap LoRA is locked
                        above; audio from the reference clip is preserved when present.{" "}
                        <a
                          href="https://www.runcomfy.com/comfyui-workflows/ltx-2-3-video-face-swap-in-comfyui-realistic-face-replacement-workflow"
                          target="_blank"
                          rel="noreferrer"
                        >
                          LTX 2.3 face swap workflow
                        </a>
                        .
                      </p>
                      <label className="media-upload">
                        <span className="media-upload-label">Face identity image (required)</span>
                        <input
                          ref={imageRef}
                          type="file"
                          accept="image/*"
                          onChange={async (e) => {
                            const f = e.target.files?.[0];
                            if (f) {
                              setImagePath(await uploadFile(f, "image"));
                              setImageName(f.name);
                            }
                          }}
                        />
                        <span className="media-upload-hint">
                          {imageName ?? "Choose face photo…"}
                        </span>
                      </label>
                      <div className="media-upload-row">
                        <label className="media-upload">
                          <span className="media-upload-label">Reference video (required)</span>
                          <input
                            ref={videoRef}
                            type="file"
                            accept="video/*"
                            onChange={async (e) => {
                              const f = e.target.files?.[0];
                              if (f) {
                                setSourceClipId(null);
                                setVideoPath(await uploadFile(f, "video"));
                                setVideoName(f.name);
                              }
                            }}
                          />
                          <span className="media-upload-hint">
                            {videoName ?? (videoPath ? "✓ file selected" : "Choose reference video…")}
                          </span>
                        </label>
                      </div>
                      {videoLibraryClips.length > 0 && (
                        <label className="clip-source-picker">
                          <span className="media-upload-label">Or from library</span>
                          <select
                            value={sourceClipId ?? ""}
                            onChange={(e) => {
                              const id = e.target.value || null;
                              setSourceClipId(id);
                              if (id) {
                                setVideoPath(null);
                                setVideoName(null);
                                if (videoRef.current) videoRef.current.value = "";
                              }
                            }}
                          >
                            <option value="">Select a clip…</option>
                            {videoLibraryClips.map((c) => {
                              const label = clipDisplayPrompt(c.prompt);
                              const meta = [
                                c.label,
                                c.width && c.height ? `${c.width}×${c.height}` : null,
                              ]
                                .filter(Boolean)
                                .join(" · ");
                              return (
                                <option key={c.id} value={c.id}>
                                  {meta ? `${label} (${meta})` : label}
                                </option>
                              );
                            })}
                          </select>
                        </label>
                      )}
                      {hasVideoSource && (
                        <p className="media-source-note">
                          {sourceClipId
                            ? "Using library clip as reference video (performance)."
                            : "Using uploaded file as reference video (performance)."}
                        </p>
                      )}
                      {(config?.lora_presets ?? [])
                        .filter((p) => loraPresetIds.includes(p.id) && p.spec)
                        .map((p) => (
                          <label key={p.id} className="v2v-lora-strength-row">
                            <span className="v2v-lora-strength-label" title={p.label}>
                              Head-swap LoRA strength
                            </span>
                            <input
                              type="number"
                              min={0}
                              max={2}
                              step={0.01}
                              value={p.scale}
                              disabled={busy || loraBusy}
                              title="Head-swap adapter strength (Comfy tip: ~0.98–1.0)"
                              aria-label={`LoRA strength for ${p.label}`}
                              onChange={(e) =>
                                updateLoraPresetScale(p.id, Number(e.target.value) || 0)
                              }
                            />
                          </label>
                        ))}
                    </>
                  )}
                  {isV2v && (
                    <>
                      <span className="media-panel-title">V2V inputs</span>
                      <p className="hint hint-inline">
                        Re-render from a reference clip. LoRA is optional: leave none
                        selected for pure motion / structure transfer, or add a community
                        IC-LoRA (e.g.{" "}
                        <a
                          href="https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Prompt"
                          target="_blank"
                          rel="noreferrer"
                        >
                          CrossView Prompt
                        </a>
                        ). No HDR/Union defaults. CrossView strength{" "}
                        <strong>1.2–1.5</strong> on distilled; prompts use the fixed
                        vocabulary, e.g.{" "}
                        <code>crossview. new camera angle: to the right, lower, closer.</code>
                      </p>
                      <div className="media-upload-row">
                        <label className="media-upload">
                          <span className="media-upload-label">Reference video (required)</span>
                          <input
                            ref={conditioningVideoRef}
                            type="file"
                            accept="video/*"
                            onChange={async (e) => {
                              const f = e.target.files?.[0];
                              if (f) {
                                setConditioningClipId(null);
                                setConditioningVideoPath(await uploadFile(f, "video"));
                                setConditioningVideoName(f.name);
                              }
                            }}
                          />
                          <span className="media-upload-hint">
                            {conditioningVideoName ?? "Choose reference video…"}
                          </span>
                        </label>
                      </div>
                      {videoLibraryClips.length > 0 && (
                        <label className="clip-source-picker">
                          <span className="media-upload-label">Or reference from library</span>
                          <select
                            value={conditioningClipId ?? ""}
                            onChange={(e) => {
                              const id = e.target.value || null;
                              setConditioningClipId(id);
                              if (id) {
                                setConditioningVideoPath(null);
                                setConditioningVideoName(null);
                                if (conditioningVideoRef.current) {
                                  conditioningVideoRef.current.value = "";
                                }
                              }
                            }}
                          >
                            <option value="">Select a clip…</option>
                            {videoLibraryClips.map((c) => {
                              const label = clipDisplayPrompt(c.prompt);
                              const meta = [
                                c.label,
                                c.width && c.height ? `${c.width}×${c.height}` : null,
                              ]
                                .filter(Boolean)
                                .join(" · ");
                              return (
                                <option key={c.id} value={c.id}>
                                  {meta ? `${label} (${meta})` : label}
                                </option>
                              );
                            })}
                          </select>
                        </label>
                      )}
                      <div className="v2v-lora-strengths">
                        <span className="media-upload-label">LoRA strength</span>
                        {loraPresetIds.length === 0 ? (
                          <p className="media-source-note">
                            No LoRA selected — pure reference-video conditioning. Add
                            CrossView or another IC-LoRA above for adapter-driven V2V
                            (CrossView tip: strength <strong>1.2–1.5</strong>).
                          </p>
                        ) : (
                          (config?.lora_presets ?? [])
                            .filter((p) => loraPresetIds.includes(p.id) && p.spec)
                            .map((p) => (
                              <label key={p.id} className="v2v-lora-strength-row">
                                <span className="v2v-lora-strength-label" title={p.label}>
                                  {p.label}
                                </span>
                                <input
                                  type="number"
                                  min={0}
                                  max={2}
                                  step={0.05}
                                  value={p.scale}
                                  disabled={busy || loraBusy}
                                  title="LoRA adapter strength (0–2). CrossView tip: 1.2–1.5"
                                  aria-label={`LoRA strength for ${p.label}`}
                                  onChange={(e) =>
                                    updateLoraPresetScale(
                                      p.id,
                                      Number(e.target.value) || 0,
                                    )
                                  }
                                />
                              </label>
                            ))
                        )}
                      </div>
                      <label className="ic-lora-scale">
                        Reference attention strength
                        <input
                          type="number"
                          min={0}
                          max={1}
                          step={0.05}
                          value={referenceStrength}
                          disabled={!hasConditioningVideo}
                          title="How strongly the reference video guides the result (0–1). Separate from LoRA strength."
                          onChange={(e) =>
                            setReferenceStrength(
                              Math.min(1, Math.max(0, Number(e.target.value) || 0)),
                            )
                          }
                        />
                      </label>
                      <p className="hint hint-inline">
                        LoRA strength is the adapter weight (often &gt;1). Reference attention
                        (0–1) controls how tightly the model follows the reference clip.
                      </p>
                      {hasConditioningVideo ? (
                        <p className="media-source-note">
                          {conditioningClipId
                            ? "Using library clip as V2V reference."
                            : "Using uploaded file as V2V reference."}
                        </p>
                      ) : (
                        <p className="media-source-note">
                          Upload a reference video (or pick one from the library) to enable Generate.
                        </p>
                      )}
                    </>
                  )}
                  {isIcLora && (
                    <>
                      <span className="media-panel-title">IC-LoRA inputs</span>
                      <p className="hint hint-inline ic-lora-active-mode">
                        Mode: <strong>{icLoraModeLabel}</strong>
                        {!config?.pose_control_available && icLoraSubMode === "motion_transfer" && (
                          <>
                            {" "}
                            (requires <code>pip install mediapipe</code>)
                          </>
                        )}
                      </p>
                      <p className="hint hint-inline">
                        Defaults to <strong>HDR IC-LoRA</strong> (upstream{" "}
                        <code>hdr-ic-lora</code>): text-to-video with optional start image
                        and/or SDR reference video. Pick <strong>Union Control</strong> in
                        the LoRA menu for pose motion transfer. Community adapters
                        (CrossView, etc.) belong in <strong>V2V</strong>.
                      </p>
                      <div className="media-upload-row">
                        <label className="media-upload">
                          <span className="media-upload-label">
                            Reference video
                            {icLoraUnionSelected
                              ? " (required for Union)"
                              : " (optional — V2V / SDR→HDR)"}
                          </span>
                          <input
                            ref={conditioningVideoRef}
                            type="file"
                            accept="video/*"
                            onChange={async (e) => {
                              const f = e.target.files?.[0];
                              if (f) {
                                setConditioningClipId(null);
                                setConditioningVideoPath(await uploadFile(f, "video"));
                                setConditioningVideoName(f.name);
                              }
                            }}
                          />
                          <span className="media-upload-hint">
                            {conditioningVideoName ?? "Choose reference video…"}
                          </span>
                        </label>
                        <label className="media-upload">
                          <span className="media-upload-label">
                            {icLoraUnionSelected
                              ? "Character image (optional for Union)"
                              : "Start image (optional I2V)"}
                          </span>
                          <input
                            ref={imageRef}
                            type="file"
                            accept="image/*"
                            onChange={async (e) => {
                              const f = e.target.files?.[0];
                              if (f) {
                                setImagePath(await uploadFile(f, "image"));
                                setImageName(f.name);
                              }
                            }}
                          />
                          <span className="media-upload-hint">
                            {imageName ??
                              (icLoraUnionSelected
                                ? "Choose character image…"
                                : "Choose start frame…")}
                          </span>
                        </label>
                      </div>
                      {videoLibraryClips.length > 0 && (
                        <label className="clip-source-picker">
                          <span className="media-upload-label">
                            Or reference video from library
                          </span>
                          <select
                            value={conditioningClipId ?? ""}
                            onChange={(e) => {
                              const id = e.target.value || null;
                              setConditioningClipId(id);
                              if (id) {
                                setConditioningVideoPath(null);
                                setConditioningVideoName(null);
                                if (conditioningVideoRef.current) {
                                  conditioningVideoRef.current.value = "";
                                }
                              }
                            }}
                          >
                            <option value="">Select a clip…</option>
                            {videoLibraryClips.map((c) => {
                              const label = clipDisplayPrompt(c.prompt);
                              const meta = [
                                c.label,
                                c.width && c.height ? `${c.width}×${c.height}` : null,
                              ]
                                .filter(Boolean)
                                .join(" · ");
                              return (
                                <option key={c.id} value={c.id}>
                                  {meta ? `${label} (${meta})` : label}
                                </option>
                              );
                            })}
                          </select>
                        </label>
                      )}
                      <label className="ic-lora-scale">
                        Motion conditioning strength
                        <input
                          type="number"
                          min={0}
                          max={2}
                          step={0.05}
                          value={conditioningVideoScale}
                          disabled={!hasConditioningVideo}
                          onChange={(e) =>
                            setConditioningVideoScale(
                              Math.min(2, Math.max(0, Number(e.target.value) || 0)),
                            )
                          }
                        />
                      </label>
                      <label className="ic-lora-scale">
                        Reference attention strength
                        <input
                          type="number"
                          min={0}
                          max={1}
                          step={0.05}
                          value={referenceStrength}
                          disabled={!hasConditioningVideo}
                          title="IC-LoRA conditioning_attention_strength (0–1)"
                          onChange={(e) =>
                            setReferenceStrength(
                              Math.min(1, Math.max(0, Number(e.target.value) || 0)),
                            )
                          }
                        />
                      </label>
                      <label className="check">
                        <input
                          type="checkbox"
                          checked={skipStage2}
                          onChange={(e) => setSkipStage2(e.target.checked)}
                        />
                        Skip stage 2 (half-res, faster)
                      </label>
                      {hasConditioningVideo ? (
                        <p className="media-source-note">
                          {conditioningClipId
                            ? "Using library clip as reference video."
                            : "Using uploaded file as reference video."}
                          {" "}
                          Motion strength scales the clip weight; attention strength
                          controls how tightly the model follows it (default 1.0).
                        </p>
                      ) : icLoraUnionSelected ? (
                        <p className="media-source-note">
                          Union Control needs a reference video (pose / motion).
                        </p>
                      ) : (
                        <p className="media-source-note">
                          No reference video — pure HDR text-to-video
                          {imagePath ? " with start image (I2V)" : ""}.
                          Output includes an SDR preview MP4 plus{" "}
                          <code>.hdr.npz</code> when available.
                        </p>
                      )}
                    </>
                  )}
                  {isA2v && (
                    <>
                      <span className="media-panel-title">Audio to video inputs</span>
                      <div className="media-upload-row">
                        <label className="media-upload">
                          <span className="media-upload-label">
                            Start image (optional, clip 1 only)
                          </span>
                          <input
                            ref={imageRef}
                            type="file"
                            accept="image/*"
                            onChange={async (e) => {
                              const f = e.target.files?.[0];
                              if (f) {
                                setImagePath(await uploadFile(f, "image"));
                                setImageName(f.name);
                              }
                            }}
                          />
                          <span className="media-upload-hint">
                            {imageName ?? "Choose image file…"}
                          </span>
                        </label>
                      </div>
                      <AudioTrimControl
                        fileName={audioName}
                        previewUrl={audioPreviewUrl}
                        durationSeconds={audioDurationSeconds}
                        startSeconds={audioStartSeconds}
                        clipDurationSeconds={audioClipDurationSeconds}
                        maxStart={audioStartSliderMax}
                        trimAvailable={audioTrimAvailable}
                        disabled={busy}
                        fileInputRef={audioRef}
                        onFileSelected={handleAudioFileSelected}
                        onStartChange={setAudioStartSeconds}
                      />
                      {showChainedImageHint && chainMethod === "autocontinue" && (
                        <p className="hint">
                          With autocontinue / audiocontinue, the start image is used for
                          clip 1 only; later clips use the last frame of the prior clip.
                        </p>
                      )}
                    </>
                  )}
                  {!isA2v && !isLipDub && needsImageUpload && (
                    <>
                      <span className="media-panel-title">Source media</span>
                      <label className="media-upload">
                      <span className="media-upload-label">
                        Source image (required)
                      </span>
                      <input
                        ref={imageRef}
                        type="file"
                        accept="image/*"
                        onChange={async (e) => {
                          const f = e.target.files?.[0];
                          if (f) {
                            setImagePath(await uploadFile(f, "image"));
                            setImageName(f.name);
                          }
                        }}
                      />
                      <span className="media-upload-hint">
                        {imageName ?? "Choose image file…"}
                      </span>
                    </label>
                    {needsEndImageUpload && (
                      <label className="media-upload">
                        <span className="media-upload-label">End image (required)</span>
                        <input
                          ref={endImageRef}
                          type="file"
                          accept="image/*"
                          onChange={async (e) => {
                            const f = e.target.files?.[0];
                            if (f) {
                              setEndImagePath(await uploadFile(f, "image"));
                              setEndImageName(f.name);
                            }
                          }}
                        />
                        <span className="media-upload-hint">
                          {endImageName ?? "Choose end frame…"}
                        </span>
                      </label>
                    )}
                    </>
                  )}
                  {needsVideoUpload && (
                    <>
                      {!isA2v && (
                        <span className="media-panel-title">
                          {isLipDub
                            ? "Reference video (required, visual motion)"
                            : "Source video"}
                        </span>
                      )}
                      <label className="media-upload">
                        <span className="media-upload-label">Upload from disk</span>
                        <input
                          ref={videoRef}
                          type="file"
                          accept="video/*"
                          onChange={async (e) => {
                            const f = e.target.files?.[0];
                            if (f) {
                              setSourceClipId(null);
                              setVideoPath(await uploadFile(f, "video"));
                              setVideoName(f.name);
                            }
                          }}
                        />
                        <span className="media-upload-hint">
                          {videoName ?? (videoPath ? "✓ file selected" : "Choose video file…")}
                        </span>
                      </label>
                      {videoLibraryClips.length > 0 && (
                        <label className="clip-source-picker">
                          <span className="media-upload-label">Or from library</span>
                          <select
                            value={sourceClipId ?? ""}
                            onChange={(e) => {
                              const id = e.target.value || null;
                              setSourceClipId(id);
                              if (id) {
                                setVideoPath(null);
                                setVideoName(null);
                                if (videoRef.current) videoRef.current.value = "";
                              }
                            }}
                          >
                            <option value="">Select a clip…</option>
                            {videoLibraryClips.map((c) => {
                              const label = clipDisplayPrompt(c.prompt);
                              const meta = [
                                c.label,
                                c.width && c.height ? `${c.width}×${c.height}` : null,
                              ]
                                .filter(Boolean)
                                .join(" · ");
                              return (
                                <option key={c.id} value={c.id}>
                                  {meta ? `${label} (${meta})` : label}
                                </option>
                              );
                            })}
                          </select>
                        </label>
                      )}
                      {hasVideoSource && (
                        <p className="media-source-note">
                          {sourceClipId
                            ? "Using library clip as reference video."
                            : isLipDub
                              ? "Using uploaded reference video. Add voice-tone audio above if the clip has no audio track."
                              : "Using uploaded file as source video."}
                        </p>
                      )}
                    </>
                  )}
                </div>
              )}

              {mode === "retake" && (
                <div className="options-grid">
                  <label>
                    Retake start (latent, inclusive)
                    <input
                      type="number"
                      min={0}
                      value={retakeStart}
                      onChange={(e) => setRetakeStart(Number(e.target.value))}
                    />
                  </label>
                  <label>
                    Retake end (latent, exclusive)
                    <input
                      type="number"
                      min={retakeStart + 1}
                      value={retakeEnd}
                      onChange={(e) => setRetakeEnd(Number(e.target.value))}
                    />
                  </label>
                  {retakeEnd <= retakeStart && (
                    <p className="hint hint-inline">End must be greater than start (exclusive end).</p>
                  )}
                </div>
              )}

              {activeClip && (
                <p className="meta">
                  Viewing: {formatDuration(activeClip.num_frames, config.defaults.fps)}
                  {activeClip.width && activeClip.height
                    ? ` · ${activeClip.width}×${activeClip.height}`
                    : ""}
                  {activeClip.bytes ? ` · ${formatBytes(activeClip.bytes)}` : ""}
                </p>
              )}
            </div>
          )}
        </section>
        </div>

        <aside className="library">
          <div className="library-header">
            <span className="library-title">Library</span>
            <span className="library-count">{libraryClips.length}</span>
          </div>
          <div className="library-grid">
            {libraryClips.map((clip) => (
              <div
                key={clip.id}
                className={`library-card-wrap ${
                  activeClip?.id === clip.id ? "active" : ""
                }`}
              >
                <button
                  type="button"
                  className="library-card"
                  onClick={() => applyClipSelection(clip)}
                  title={clip.prompt}
                >
                  {clip.video_url && (
                    <video
                      className="library-thumb"
                      src={clip.video_url}
                      muted
                      playsInline
                      preload="metadata"
                    />
                  )}
                  <span className={`library-label ${clip.label.toLowerCase()}`}>
                    {clip.label}
                  </span>
                  <span className="library-prompt">{clip.prompt}</span>
                </button>
                <button
                  type="button"
                  className="library-delete"
                  title="Delete"
                  disabled={busy}
                  onClick={(e) => {
                    e.stopPropagation();
                    void deleteGeneration(clip);
                  }}
                >
                  ×
                </button>
              </div>
            ))}
          </div>

          <div className="library-section">
            <div className="library-header">
              <span className="library-title">Frames</span>
              <span className="library-count">{frameLibrary.length}</span>
            </div>
            {frameLibrary.length === 0 ? (
              <p className="library-empty-hint">
                Pause a video and tap the camera icon on the player to capture
                stills for i2v, a2v, or keyframe inputs.
              </p>
            ) : (
              <div className="frame-library-grid">
                {frameLibrary.map((frame) => (
                  <div
                    key={frame.id}
                    className={`frame-card-wrap ${
                      imagePath === frame.path || endImagePath === frame.path
                        ? "active"
                        : ""
                    }`}
                  >
                    <button
                      type="button"
                      className="frame-card"
                      title={`Use as start image: ${frame.label}`}
                      onClick={() => applyFrameAsInput(frame, "start")}
                    >
                      <img
                        className="frame-thumb"
                        src={frame.image_url}
                        alt={frame.label}
                        loading="lazy"
                      />
                      <span className="frame-label">{frame.label}</span>
                    </button>
                    {mode === "keyframe" && (
                      <button
                        type="button"
                        className="frame-use-end"
                        title="Use as end image"
                        disabled={busy}
                        onClick={(e) => {
                          e.stopPropagation();
                          applyFrameAsInput(frame, "end");
                        }}
                      >
                        End
                      </button>
                    )}
                    <button
                      type="button"
                      className="library-delete"
                      title="Delete frame"
                      disabled={busy}
                      onClick={(e) => {
                        e.stopPropagation();
                        void deleteFrame(frame);
                      }}
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}
