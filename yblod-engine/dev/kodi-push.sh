#!/bin/sh
# kodi-push.sh HOST...: Ollie's latest kodi.bin and engine libraries into
# /storage/kodi-dev on each box (see kodi-dev.conf), then Kodi restarted.
# Skips a box that is playing.
B=Projects/libreelec-x86-DV-vk/build.LibreELEC-Generic.x86_64-13.0-devel/install_pkg
for H in "$@"; do
  P=$(ssh root@$H "printf '%s' '{\"jsonrpc\":\"2.0\",\"method\":\"Player.GetActivePlayers\",\"id\":1}' | nc -w 2 127.0.0.1 9090")
  case "$P" in *'"result":[]'*) ;; *) echo "== $H: playing or Kodi not answering, skipped"; continue;; esac
  ssh root@$H 'mkdir -p /storage/kodi-dev'
  for F in "kodi-*/usr/lib/kodi/kodi.bin" "yblod-engine-vk-*/usr/lib/libdv_engine_vk.so.0" "yblod-engine-[0-9a-f]*/usr/lib/libdv_source_private.so.1"; do
    N=$(basename "$F")
    ssh ollie "cat \$(ls -t ~/$B/$F | head -1)" | ssh root@$H "cat > /storage/kodi-dev/$N.tmp && chmod +x /storage/kodi-dev/$N.tmp && mv /storage/kodi-dev/$N.tmp /storage/kodi-dev/$N"
  done
  ssh root@$H 'systemctl daemon-reload; systemctl restart kodi; md5sum /storage/kodi-dev/*' | sed "s/^/$H /"
done
