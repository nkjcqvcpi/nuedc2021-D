cd /home/pi/nudec2021-D/models || exit
killall python3 || true
python3 camera.py "192.168.0.20" "cam1"