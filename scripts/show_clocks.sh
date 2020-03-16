#!/bin/bash
if [ -f ~/jetson_clocks.sh ]; then
# For JetPack 3.3 and below
  sudo ~/jetson_clocks.sh --show
else
# For JetPack 4.2 and above
  sudo jetson_clocks --show
fi
