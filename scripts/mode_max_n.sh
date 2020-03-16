#!/bin/bash
echo "Enabling Mode-N (Boost mode)"
echo " Denver cores: 2, max 2.0 GHz"
echo " ARM-A57 cores: 4, max 2.0 GHz"
echo " GPU: max 1.3 GHz"
sudo nvpmodel -m 0
if [ -f ~/jetson_clocks.sh ]; then
# For JetPack 3.3 and below
  sudo ~/jetson_clocks.sh --show
else
# For JetPack 4.2 and above
  sudo jetson_clocks --show
fi
