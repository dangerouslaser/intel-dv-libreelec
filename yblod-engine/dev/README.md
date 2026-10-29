# Development tools

Used for the CB1 Master Reference work. Paths and hosts are those of the
test setup: Ollie builds, two boxes (an N150 and a Ryzen) dual-boot this
image beside Arch from btrfs subvolumes `@cb1` and `@cb1-storage`.

Build host (from the tree root, in the `libreelec-dv-build` container; each
retries with fewer jobs when the build runs out of memory):

- `run-vk.sh`: the full image and `dvvk-mr-compare`; log `build-vk.log`.
- `run-kodi.sh`: the YBLOD engines, Kodi and `dvvk-mr-compare` only; log `build-kodi.log`.
- `run-baseline.sh`: CroqueMr's image without these changes; log `build-baseline.log`.
- `run-compare.sh [options] FILE`: `dvvk-mr-compare` inside the built
  image's root (clips in `mr/clips`), on the host's render node.

Boxes (copy to `/storage`):

- `cb1-install.sh BTRFS_DEV ESP_DEV`: `/storage/SYSTEM.new` into `@cb1`, CB1
  the default boot entry, reboot. Refuses while something plays.
- `cb1-restore-default.sh ESP_DEV`: big-fella the default entry again.
- `kodi-dev.conf` (in `/storage/.config/system.d/kodi.service.d/`) and,
  from the workstation, `kodi-push.sh HOST...`: the latest built `kodi.bin`
  and engine libraries in `/storage/kodi-dev`, used by Kodi's service alone,
  Kodi restarted. `cb1-install.sh` clears them.
- `cb1test.sh ENGINE CLIP [SECONDS]`: Master Reference with ENGINE (0 CB1,
  1 YBLOD Vulkan, 2 YBLOD OpenCL), playback health and the engine in use.
- `dvwatch [SECONDS]`: the playback engine and health lines while playing.
