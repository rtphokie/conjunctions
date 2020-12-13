import threading
from tqdm import tqdm
import logging
import itertools
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from skyfield import api
from scipy.signal import argrelextrema
import re
import simple_cache
import warnings

ts = api.load.timescale(builtin=True)
load = api.Loader('/var/data')
eph = load('de406.bsp')  # 3000 BCE to 3000 AD

outer_planets = ['jupiter', 'saturn', 'uranus', 'neptune', 'pluto']
planet_info = {'moon': {'radius_km': 1737, 'orbital inclination deg': 5.1},
               'mercury': {'radius_km': 2440, 'orbital inclination deg': 7.0},
               'venus': {'radius_km': 6052, 'orbital inclination deg': 3.4},
               'earth': {'radius_km': 6371, 'orbital inclination deg': 0.0},
               'mars': {'radius_km': 6371, 'orbital inclination deg': 1.9},
               'jupiter': {'radius_km': 69911, 'orbital inclination deg': 1.3},
               'saturn': {'radius_km': 58232, 'orbital inclination deg': 2.5},
               'uranus': {'radius_km': 25362, 'orbital inclination deg': 0.8},
               'neptune': {'radius_km': 24622, 'orbital inclination deg': 1.8},
               'pluto': {'radius_km': 1188, 'orbital inclination deg': 17.2}
               }


def threaded(f, daemon=False):
    import queue

    def wrapped_f(q, *args, **kwargs):
        '''this function calls the decorated function and puts the
        result in a queue'''
        ret = f(*args, **kwargs)
        q.put(ret)

    def wrap(*args, **kwargs):
        '''this is the function returned from the decorator. It fires off
        wrapped_f in a new thread and returns the thread object with
        the result queue attached'''

        q = queue.Queue()

        t = threading.Thread(target=wrapped_f, args=(q,)+args, kwargs=kwargs)
        t.daemon = daemon
        t.start()
        t.result_queue = q
        return t

    return wrap

def translatebody(body):
    body1=body.lower()
    if body1 in outer_planets:
        body1+=' barycenter'
    return body1

def getAngularSize(equatorialRadius, distancetocore):
    # from Stellarium planet.cpp line 2019
    foo = 2 * np.arctan((2*equatorialRadius)/(2*distancetocore))
    return np.degrees(foo)

def getAngularSize(equatorialRadius, distancetocore):
    # from Stellarium planet.cpp line 2019
    foo = 2 * np.arctan((2*equatorialRadius)/(2*distancetocore))
    return np.degrees(foo)

def yearfromisodate(txt):
    regexstr = '(-*\d+)\-\d+\-'
    result = None
    try:
        x = re.search(regexstr, txt)
        if x.group:
            result = int(x.group(1))
    except Exception as e:
        print(txt, e)
    return result

def build_timescale(start,end):
    resolution=None
    if start is None:
        resolution='jd'
        # base start and end dates on available dates in the spice kernel
        end=None
        for s in eph.spk.segments:
            if start != s.start_jd or end != s.end_jd:
                start, end = s.start_jd, s.end_jd
    daydelta = (end - start)

    if np.abs(daydelta) <= 7:
        # minute resolution because minute specified or date range is within a week
        resolution = 'minute'
        starttime=ts.tt_jd(start)
        t = ts.utc(starttime.utc.year,starttime.utc.month,starttime.utc.day,starttime.utc.hour, range(starttime.utc.minute-(24*60),starttime.utc.minute+(24*60)))
    elif np.abs(daydelta) < 366:
        # hour resolution because hour specified or date range is within a year
        resolution = 'hour'
        starttime=ts.tt_jd(start)
        t = ts.utc(starttime.utc.year,starttime.utc.month,starttime.utc.day,range(starttime.utc.hour-24,starttime.utc+24))
    else:
        resolution = 'day'
        t = ts.tt_jd(range(int(np.ceil(start)),int(np.floor(end))))
    return t, resolution
