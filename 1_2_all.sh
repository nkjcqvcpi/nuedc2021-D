ssh -t -Y pi@192.168.0.10 "sh /home/pi/nudec2021-D/notebooks/camera0_d.sh" &
echo camera0 set

ssh -t -Y pi@192.168.0.20 "sh /home/pi/nudec2021-D/notebooks/camera1_d.sh" &
echo camera1 set

cd /home/pi/nudec2021-D || exit
killall python3 || true
python3 main.py