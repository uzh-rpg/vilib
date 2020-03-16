#!/bin/bash
if [ $# -ne 2 ]; then
	echo "Error: invalid number of arguments"
	echo "Usage: ./connect_to_wifi.sh <ssid> <password>"
	exit -1
fi
WIFI_SSID=$1
WIFI_PASSWORD=$2
sudo nmcli device wifi connect ${WIFI_SSID} password ${WIFI_PASSWORD}
