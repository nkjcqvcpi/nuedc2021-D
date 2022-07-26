import os
import time


class Sound:
    def __init__(self, length, theta):
        dt = int(int(theta) / 10)
        self.length = '%.2f' % length
        self.theta = '{}t.{}'.format(dt if dt > 1 else '', str(theta).split('.')[1][:2]) if int(
            theta) % 10 == 0 else str(int(theta))

    def __call__(self, local=False):
        command = 'afplay' if local else 'cvlc -q --play-and-exit'
        os.system(command + ' %s' % 'sound/wancheng.wav')
        for i in self.length:
            os.system(command + ' %s' % 'sound/' + str(i) + '.wav')
        os.system(command + ' %s' % 'sound/mi.wav')
        time.sleep(0.1)
        os.system(command + ' %s' % 'sound/jiaodushi.wav')
        for i in self.theta:
            os.system(command + ' %s' % 'sound/' + str(i) + '.wav')
        os.system(command + ' %s' % 'sound/d.wav')


if __name__ == '__main__':
    play = Sound(1.259090, 30.98760)
    play(local=True)
