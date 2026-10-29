#!/bin/sh
# Capture DV tunnel frames from the HDMI encoder output (VDIN1 loopback) during playback.
# usage: cecapture.sh <file> <seek_s> <count> <interval_s> <outdir>
F=$1; SEEK=$2; N=$3; IV=$4; OUT=$5
# Kodi JSON-RPC over raw TCP (port 9090): needs no web-server password
J(){ (echo "$1"; sleep 0.4) | nc 127.0.0.1 9090 | grep -m1 '"id":1'; }
mkdir -p $OUT; rm -f $OUT/ce-*.raw $OUT/times.txt
J '{"jsonrpc":"2.0","id":1,"method":"Player.Open","params":{"item":{"file":"'$F'"},"options":{"resume":false}}}' >/dev/null
sleep 8
J '{"jsonrpc":"2.0","id":1,"method":"Player.SetSubtitle","params":{"playerid":1,"subtitle":"off"}}' >/dev/null
J '{"jsonrpc":"2.0","id":1,"method":"Player.Seek","params":{"playerid":1,"value":{"time":{"hours":0,"minutes":0,"seconds":'$SEEK',"milliseconds":0}}}}' >/dev/null
sleep 5   # let the seek bar time out
V=/sys/class/vdin/vdin1/attr
echo "cma_config_flag 101" > $V
i=0
while [ $i -lt $N ]; do
  echo "v4l2start venc0 3840 2160 24 0 0" > $V
  sleep 0.5
  ADDR=$(echo state > $V; dmesg | grep "buf\[0\]mem_start" | tail -1 | sed 's/.*mem_start = \(0x[0-9a-f]*\).*/\1/')
  echo freeze > $V
  R=$(J '{"jsonrpc":"2.0","id":1,"method":"Player.GetProperties","params":{"playerid":1,"properties":["time"]}}')
  T=$(echo "$R" | sed 's/.*"minutes":\([0-9]*\).*"seconds":\([0-9]*\).*/\1:\2/').$(echo "$R" | sed 's/.*"milliseconds":\([0-9]*\).*/\1/')
  dd if=/dev/mem of=$OUT/ce-$i.raw bs=4096 skip=$(( ADDR / 4096 )) count=6075 2>/dev/null
  echo "$i $T" >> $OUT/times.txt
  echo v4l2stop > $V
  i=$((i+1)); sleep $IV
done
echo v4l2stop > $V
echo "cma_config_flag 0" > $V
J '{"jsonrpc":"2.0","id":1,"method":"Player.Stop","params":{"playerid":1}}' >/dev/null
cat $OUT/times.txt
