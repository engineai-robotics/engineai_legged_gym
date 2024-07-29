# SPDX-License-Identifier: BSD-3-Clause
#
#
import time

from pynput import keyboard
from pynput.keyboard import Controller, Key
import threading


class KeyCommand:
    def __init__(self):
        self.stepTest = False
        self.stepNet = False
        self.stepCalibrate = True
        self.timestep = 0
        self.keyboardEvent = True
        self.listener = None

        self.keyboard = Controller()
        self.listening = True

    def start(self):
        # 定义一个函数来运行监听器
        def listener():
            with keyboard.Listener(on_press=self.on_press) as self.listener:
                self.listener.join()

        # 创建并启动线程
        self.thread = threading.Thread(target=listener, daemon=True)
        self.thread.start()

    def on_press(self, key):
        # print(f'key {key} is pressed')

        self.keyboardEvent = True

        if key == Key.up:  # str(key) == "'1'":
            self.timestep = 0
            print("std ", self.timestep)
            self.stepCalibrate = True
            # self.stepCalibrate = False
            self.stepTest = False
            self.stepNet = False
            print('!!!!!  静态归零模式 ！')
        elif key == Key.left:  # str(key) == "'2'":
            self.timestep = 0
            print("not ", self.timestep)
            self.stepTest = True
            self.stepCalibrate = False
            # self.stepTest = False
            self.stepNet = False
            print('!!!!!  挂起动腿模式 ！')
        elif key == Key.right:  # str(key) == "'3'":
            self.timestep = 0
            print("net ", self.timestep)
            self.stepNet = True
            self.stepCalibrate = False
            self.stepTest = False
            # self.stepNet = False
            print('!!!!!  神经网络模式 ！')
        else:
            print(f'key {key} is pressed, 停止监听按键.')
            self.stop()

    def on_release(self, key):
        pass

    def stop(self):
        # 停止监听器
        self.listening = False
        # 等待线程结束
        keyboard.Listener.stop(self.listener)


if __name__ == '__main__':
    # 创建KeyboardListener实例
    kc = KeyCommand()

    # 启动键盘监听器
    kc.start()

    try:
        while True:
            if kc.listening:
                time.sleep(0.001)
            else:
                break

    finally:
        kc.stop()
        print(f'程序结束')

