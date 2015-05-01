# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:54:02 2015

@author: Ziang
"""

def test(**kwargs):
    print(kwargs['key'])
    if 'key' in kwargs.keys():
        print('has key')

test(key='this is key')