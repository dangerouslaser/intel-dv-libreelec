#!/bin/sh
# cb1-install.sh BTRFS_DEV ESP_DEV: /storage/SYSTEM.new into @cb1 and a reboot
# into it (CB1 the default boot entry until cb1-restore-default.sh). Refuses
# while something is playing.
set -e
DEV=$1; ESP=$2
P=$(printf '%s' '{"jsonrpc":"2.0","method":"Player.GetActivePlayers","id":1}' | nc -w 2 127.0.0.1 9090)
case "$P" in *'"result":[]'*) ;; *) echo "playing or Kodi not answering: not rebooting ($P)"; exit 1;; esac
mkdir -p /tmp/bt && mount -o subvolid=5 $DEV /tmp/bt
cp /storage/SYSTEM.new /tmp/bt/@cb1/SYSTEM.tmp && mv /tmp/bt/@cb1/SYSTEM.tmp /tmp/bt/@cb1/SYSTEM && sync && umount /tmp/bt
rm /storage/SYSTEM.new; rm -rf /storage/kodi-dev   # the image carries its own builds
mkdir -p /tmp/esp && mount $ESP /tmp/esp
[ -f /tmp/esp/loader/loader.conf.big-fella ] || cp /tmp/esp/loader/loader.conf /tmp/esp/loader/loader.conf.big-fella
sed -i "s/^default .*/default cb1.conf/" /tmp/esp/loader/loader.conf
sync; umount /tmp/esp
echo "rebooting into the new image"
systemctl reboot
