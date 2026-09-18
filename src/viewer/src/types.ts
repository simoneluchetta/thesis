/**
 * Shape of public/scene.json: cameras, arena layers and people.
 */
// Poses follow COLMAP's convention and come out of its sparse model:
// https://colmap.github.io/format.html
// https://en.wikipedia.org/wiki/Pinhole_camera_model

/** How well one camera came out of the SfM, from the diag block of scene.json. */
interface CameraDiag {
  /** Unique sparse 3D points this camera sees. A low count means a shaky pose. */
  n_visible_points: number;
  /** Total 2D-3D observations. */
  n_observations: number;
  /** Median per-frame RMS reprojection error, px. null = no registered points. */
  reproj_error_px: number | null;
  /** How many frames of this camera made it into the SfM. */
  n_frames_registered: number;
}

/** One camera in public/scene.json. */
export interface SceneCameraEntry {
  id: number;
  /** Folder-style label: cam01, cam09L, cam09R, ... */
  name: string;
  /** World-frame camera centre, C = -R^T t. */
  position: [number, number, number];
  /** 3x3 cam-to-world rotation. COLMAP convention: the camera looks down +Z. */
  rotation_c2w: [
    [number, number, number],
    [number, number, number],
    [number, number, number]
  ];
  fovYDeg: number;
  aspect: number;
  /** Source sensor size [W,H] in px. */
  image_size?: [number, number];
  near: number;
  far: number;
  /** One eye of a stereo camera. Each eye has its own pose. */
  is_stereo?: boolean;
  n_frames_used: number;
  thumb: string | null;
  diag: CameraDiag;
  /** Second pose from a placed_cameras file, drawn by the PnP result tab. */
  pnp_position?: [number, number, number];
  pnp_rotation_c2w?: [
    [number, number, number],
    [number, number, number],
    [number, number, number]
  ];
}

/** The SfM point cloud of scene.json, packed flat for three.js. */
interface SparsePoints {
  /** Length 3*n, x,y,z packed. */
  positions: number[];
  /** Length 3*n, r,g,b in [0,1]. */
  colors: number[];
  n: number;
}

/** One toggleable PLY layer listed in scene.json. */
export interface PlyLayer {
  id: string;
  label: string;
  url: string;
  default_on: boolean;
}

/** One triangulated off-floor landmark: a net junction or a bench corner. */
export interface SceneTiePoint {
  /** Compound id, e.g. "vbnet/p1", "bench/fL". */
  id: string;
  /** Group key used for colouring: vbnet | bench. */
  group: string;
  /** Label from the annotation catalog. */
  label: string;
  /** Placer-world XYZ (Z up, metres), same frame as the cameras. */
  pos: [number, number, number];
  /** Triangulation rms over the used observations, px. */
  rms_px: number;
  /** How many camera observations were triangulated. */
  n_cams: number;
}

/** One line of the fitted FIBA/FIVB court, in placer-world XYZ at z~0. */
interface SceneCourtLine {
  /** "bb" (basketball, yellow) or "vb" (volleyball, cyan). */
  court: string;
  /** "line" | "arc" | "circle". */
  kind: string;
  pts3d: [number, number, number][];
}

/** Off-floor structure (net, bench) plus the court lines. */
export interface SceneTiePoints3D {
  points: SceneTiePoint[];
  n: number;
  /** Placement file the points were triangulated against. */
  poses_ref?: string | null;
  /** Filename of the triangulated tie-point JSON. */
  source?: string;
  court_lines?: SceneCourtLine[];
}

/** One fused person at one timestamp (people/fuse.py). */
export interface ScenePersonObs {
  id: number;
  /** Ground-contact position [X,Y,0] in the placer world frame. */
  pos: [number, number, number];
  /** How many cameras agreed on this position. */
  n_cams: number;
  /** Identity across timestamps. Absent on fuse-only exports. */
  track_id?: number;
  /** Spread (m) between the contributing cameras' ground intersections. */
  spread_m?: number;
  /** Label from people/roles.py: not every detection is an athlete. */
  role?: "player" | "peripheral" | "bystander" | "unknown";
  /** Appearance test from people/roles.py: the kit matches neither team. */
  off_team?: boolean;
}

/** Where the exported people block came from. */
interface ScenePeopleMeta {
  session?: string | null;
  label?: string | null;
  source?: string | null;
  calibration?: string | null;
  fps?: number;
  tracked?: boolean;
  n_tracks?: number | null;
  baked_at?: string;
  /** "pos" for the raw fused point, "pos_smooth" for the smoothed one. */
  pos_field?: string;
  /** True when the tracks carry a player / official role. */
  roles?: boolean;
}

/** People on the floor over a timeline, keyed by raw video frame index. */
export interface ScenePeople {
  /** Sorted timestamps (= raw video frame indices) that have detections. */
  timestamps: number[];
  /** timestamp (as string) -> the people fused at that instant. */
  frames: Record<string, ScenePersonObs[]>;
  /** Folder label of the camera the per-frame thumbnails come from, e.g. "cam13". */
  thumb_cam?: string;
  /** timestamp (as string) -> URL of that instant's thumbnail, when cached. */
  frame_thumbs?: Record<string, string>;
  /** Roster-locked exports only: track id -> athlete label ("A11"). */
  slots?: Record<string, string>;
  meta?: ScenePeopleMeta;
}

/** A camera the SfM never registered. */
interface UnregisteredCam {
  id: number;
  name: string;
}

/** Health numbers for the whole SfM run, shown in the HUD. */
interface SceneDiagnostics {
  n_registered_images: number;
  n_unique_cams: number;
  /** How many cams should have registered; each stereo eye counts separately. */
  n_expected_cams: number;
  n_unregistered_cams: number;
  unregistered_cams: UnregisteredCam[];
  n_sparse_points: number;
  cam_bbox: [
    [number, number, number],
    [number, number, number]
  ];
  mean_pairwise_baseline: number;
  min_pairwise_baseline: number;
  closest_camera_pairs?: Array<{
    a: string;
    b: string;
    distance: number;
  }>;
  opposite_side_violations?: Array<{
    a: string;
    b: string;
    distance: number;
    forward_angle_deg: number;
  }>;
  median_reproj_error_px: number;
  max_reproj_error_px: number;
}

/** Which `placed_cameras*.json` file was folded into this scene. */
interface SceneSource {
  /** Path of the source file at export time. */
  path: string;
  /** ISO8601 mtime at export time, for spotting a stale export. */
  mtime: string;
}

/** When scene.json was baked, and from which placement files. */
export interface SceneMeta {
  /** ISO8601 timestamp when this scene.json was written. */
  baked_at: string;
  placed_cameras_source: SceneSource | null;
  pnp_overlay_source: SceneSource | null;
}

/** The whole of public/scene.json. The people block comes from viewer/update_people_block.py. */
export interface SceneJson {
  version: string;
  world: {
    /** Smallest-variance axis of the camera centres (PCA hint). */
    up: [number, number, number];
    centroid: [number, number, number];
    /** RMS distance from the centroid to the cameras. */
    scale_hint: number;
  };
  cameras: SceneCameraEntry[];
  sparse_points: SparsePoints;
  ply_layers: PlyLayer[];
  /** Optional: people on the floor, written by update_people_block.py. */
  people?: ScenePeople;
  /** Optional: triangulated off-floor structure (net, bench). */
  tie_points_3d?: SceneTiePoints3D | null;
  diagnostics: SceneDiagnostics;
  /** Optional: absent on older exports. */
  meta?: SceneMeta;
}
