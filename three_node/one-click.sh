cam0() {
  ssh -t pi@192.168.0.10
  sudo su
  export DISPLAY=:0
  cd /home/pi/D/three_node || exit
  killall python3 || true
  python3 camera0.py
}

cam0 & sleep 5

cam1() {
  ssh -t pi@192.168.0.20
  sudo su
  export DISPLAY=:0
  cd /home/pi/D/three_node || exit
  killall python3 || true
  python3 camera1.py
}

cam1 & sleep 5

cd /home/pi/D/three_node || exit
killall python3 || true
python3 terminal.py

sleep 30
