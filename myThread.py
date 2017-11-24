#-------------------------------------------------------------------------------
# Name:
# Purpose:
#
# Author:      BQH
#
# Created:     03/03/2016
# Copyright:   (c) BQH 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------


import threading

class MyThread(threading.Thread):
    def __init__(self,func,args,name=""):
        threading.Thread.__init__(self)
        self.name = name
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args)