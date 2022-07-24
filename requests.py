# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 17:32:39 2022

@author: Moksha Sri
"""
import requests

url = 'http://localhost:5000/predict for next 7 days_api'
r = requests.post(url,json={'datum':date })

print(r.json())
