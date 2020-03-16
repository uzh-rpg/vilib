# Utility scripts

## Performance mode

As the Jetson TX2 supports multiple performance modes, one might want to switch between the different ones easily.

```bash
# To switch to a mode, simply just run the script (e.g. Max-N)
./mode_max_n.sh

# However, keep in mind, that the mode switching only changes the maximum
# ATTAINABLE clocks, the clocks are dynamically changed based on the load.
# In order to set the clocks statically to their highest setting WITHIN the mode,
# execute:
./max_clocks_within_mode.sh

# Therefore, to enter mode Max-N with highest clocks, execute both of the above.

# Show the actual CPU, GPU, and Memory clocks:
./show_clocks.sh
# Show the available modes:
./show_modes.sh
```

## WiFi connection

To easen the WiFi connection establishment on the Jetson TX2, run the appropriate script.
**NB**: this script can also be used on any platform that uses the *Network Manager* service.

```bash
./connect_to_wifi.sh <SSID> <Preshared Key>
```
