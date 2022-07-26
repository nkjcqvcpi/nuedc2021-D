ssh pi@192.168.0.10 "sh /home/pi/nudec2021-D/notebooks/camera0.sh" &

ssh pi@192.168.0.20 "sh /home/pi/nudec2021-D/notebooks/camera1.sh" &

cd /home/pi/nudec2021-D || exit
killall python3 || true
python3 main.py
