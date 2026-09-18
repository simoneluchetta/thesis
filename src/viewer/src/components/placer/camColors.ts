/** Dot colours for placer cameras, shared by the map, the panels and the legend. */

export const CAM_COLOR_SELECTED = "#ffcc66";     // amber, the focused cam
export const CAM_COLOR_BELOW_FLOOR = "#ff5050";  // red, Z < 0 and so probably wrong
export const CAM_COLOR_STEREO = "#ff8888";       // salmon, one eye of a stereo camera
export const CAM_COLOR_NORMAL = "#55ccff";       // cyan, nothing to flag

/** Priority: selection, then below-floor, then stereo, then the default. */
export function colorForPlacerCam(
  positionZ: number,
  isSelected: boolean,
  isStereo: boolean,
): string {
  if (isSelected) return CAM_COLOR_SELECTED;
  if (positionZ < 0) return CAM_COLOR_BELOW_FLOOR;
  if (isStereo) return CAM_COLOR_STEREO;
  return CAM_COLOR_NORMAL;
}
