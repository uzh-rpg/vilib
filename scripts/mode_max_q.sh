#!/bin/bash
echo "Enabling Max-Q mode"
echo " Denver cores: 0"
echo " ARM-A57 cores: 4, max 1.2 GHz"
echo " GPU: max 0.85 GHz"
sudo nvpmodel -m 1
if [ -f ~/jetson_clocks.sh ]; then
# For JetPack 3.3 and below
  sudo ~/jetson_clocks.sh --show
else
# For JetPack 4.2 and above
  sudo jetson_clocks --show
fi
