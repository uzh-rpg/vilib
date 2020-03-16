#!/bin/bash
echo "Enabling Max-P core-all"
echo " Denver cores: 2, max 1.4 GHz"
echo " ARM-A57 cores: 4, max 1.4 GHz"
echo " GPU: max 1.12 GHz"
sudo nvpmodel -m 2
if [ -f ~/jetson_clocks.sh ]; then
# For JetPack 3.3 and below
  sudo ~/jetson_clocks.sh --show
else
# For JetPack 4.2 and above
  sudo jetson_clocks --show
fi
