#!/bin/bash
echo "Enabling Max-P Denver"
echo " Denver cores: 1, max 2.0 GHz"
echo " ARM-A57 cores: 1, max 2.0 GHz"
echo " GPU: max 1.12 GHz"
sudo nvpmodel -m 4
if [ -f ~/jetson_clocks.sh ]; then
# For JetPack 3.3 and below
  sudo ~/jetson_clocks.sh --show
else
# For JetPack 4.2 and above
  sudo jetson_clocks --show
fi
