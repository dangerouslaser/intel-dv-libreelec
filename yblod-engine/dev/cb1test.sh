#!/bin/sh
# cb1test.sh ENGINE CLIP [SECONDS]: Master Reference with ENGINE (0 CB1, 1 YBLOD
# Vulkan, 2 YBLOD OpenCL), CLIP played from the start for SECONDS (default 45).
# Prints CB1's playback health and which renderer drew the pictures.
E=$1; C=$2; T=${3:-45}; G=/storage/.kodi/userdata/guisettings.xml; L=/storage/.kodi/temp/kodi.log
systemctl stop kodi
sed -i -E "s|<setting id=\"dvbridge.masterengine\"[^>]*>[^<]*</setting>|<setting id=\"dvbridge.masterengine\">$E</setting>|; s|<setting id=\"dvbridge.outputmode\"[^>]*>[^<]*</setting>|<setting id=\"dvbridge.outputmode\">4</setting>|" $G
grep -q dvbridge.masterengine $G || sed -i "s|<setting id=\"dvbridge.outputmode\">4</setting>|&\n    <setting id=\"dvbridge.masterengine\">$E</setting>|" $G
systemctl start kodi; sleep 20
kodi-send --action="PlayMedia($C)" >/dev/null; sleep $T
kodi-send --action="PlayerControl(Stop)" >/dev/null; sleep 3
echo "== engine=$E clip=$C"
grep -E "YBLOD (Vulkan|OpenCL) engine (started|did not)|YBLOD .* engine (does not take|failed)" $L | head -3 | cut -c25-200
echo "CB1-rendered pictures logged: $(grep -c 'YBLOD engine: CB1 renders' $L)"
grep -E "DVBridge playback health" $L | tail -1 | grep -oE "(drop|skip)_total=[0-9]+|pts_s=[0-9.]+" | tr '\n' ' '; echo
grep -E "DVBridge renderer (summary|reset|closed)" $L | tail -1 | grep -oE "(prepared|direct|composed|presented|presentation_failures|stage_failures)=[0-9]+" | tr '\n' ' '; echo
grep -E "DVBridge output admission" $L | tail -1 | grep -oE "effective=[^=]*reason=[a-z-]+" | cut -c1-80
