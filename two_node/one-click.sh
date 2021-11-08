cam0() {
  ssh -t pi@192.168.0.20
  export DISPLAY=:0
  cd /home/pi/D/two_node || exit
  killall python3 || true
  python3 camera.py
}

cam0 & sleep 5

cd /home/pi/D/two_node || exit
killall python3 || true
python3 terminal.py

sleep 30
