# -*- coding: utf-8 -*-

def createDict(*args):
     return dict(((k, eval(k)) for k in args))