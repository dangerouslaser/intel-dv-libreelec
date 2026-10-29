#!/bin/sh
# cb1-restore-default.sh ESP_DEV: big-fella the default boot entry again.
ESP=$1
mkdir -p /tmp/esp && mount $ESP /tmp/esp || exit 1
[ -f /tmp/esp/loader/loader.conf.big-fella ] && mv /tmp/esp/loader/loader.conf.big-fella /tmp/esp/loader/loader.conf
head -1 /tmp/esp/loader/loader.conf; sync; umount /tmp/esp
