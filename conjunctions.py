import unittest
import concurrent.futures
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
from skyfield import api
from scipy.signal import argrelextrema
from conjunction.utils import threaded, translatebody, build_timescale, getAngularSize, planet_info, yearfromisodate
pd.options.mode.chained_assignment = None

logger = logging.getLogger('conjunctions')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('conjunctions.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

maxprocesses = 75
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

ts = api.load.timescale(builtin=True)
load = api.Loader('/var/data')
eph = load('de406.bsp')  # 3000 BCE to 3000 AD

def minimum_separations(body1, body2, start, end):
    bodya=translatebody(body1)
    bodyb=translatebody(body2)

    sun=eph['sun']
    earth=eph['earth']
    t,precision = build_timescale(start,end)

    # observe the bodies (and the Sun to calculate elongation) from a geocentric position
    e = earth.at(t)
    b1 = e.observe(eph[bodya])
    b2 = e.observe(eph[bodyb])
    sun = e.observe(sun)

    distance_a = b1.distance().km
    distance_b = b2.distance().km

    appdiam_a = getAngularSize(planet_info[body2]['radius_km'], distance_a)
    appdiam_b = getAngularSize(planet_info[body1]['radius_km'], distance_a)

    df = pd.DataFrame({'date_tt': t.tt,
                       f'{body1} distance (km)': distance_a,
                       f'{body2} distance (km)': distance_b,
                       f'{body1} apparent diameter (deg)': appdiam_a,
                       f'{body2} apparent diameter (deg)': appdiam_b,
                       f'angular separation (deg)': b1.separation_from(b2).degrees,
                       'elongation': sun.separation_from(b1).degrees})
    # find minima
    step = df.iloc[1]['date_tt'] - df.iloc[0]['date_tt']
    if step >= 1.0:
        conjunctionindexes = argrelextrema(df['angular separation (deg)'].values, np.less)
        # filter out all but the local minimum separations
        df_working = df.iloc[conjunctionindexes[0]]
    else:
        # simple minimum
        df_working=pd.DataFrame([df.loc[df['angular separation (deg)'].idxmin()]])

    # filter out any minimum separations greater than the sum of in inclinations, uninteresting and not really a conjunctio
    df_working = df_working[df_working['angular separation (deg)'] <= planet_info[body1]['orbital inclination deg'] + planet_info[body2][ 'orbital inclination deg']]

    dffinal=pd.DataFrame()
    if precision in ['hour', 'minute']:# or True:
        dffinal=df_working
    else:
        finalrows=[]
        processes = []
        logger.info(f"calculating minute resolution for {df_working.shape[0]} {body1} - {body2} minima")
        with concurrent.futures.ProcessPoolExecutor(max_workers=maxprocesses) as executor:
            for row in df_working.iterrows():
                processes.append(executor.submit(minimum_separations, body1, body2, row[1]['date_tt']-.5, row[1]['date_tt']+.5))
        for future in concurrent.futures.as_completed(processes):
            for row2 in future.result().iterrows():
                finalrows.append(row2[1].to_dict())  # converting to a dictionary is more efficient and you get reindexing
        dffinal=pd.DataFrame(sorted(finalrows, key=lambda i: i['date_tt']))
    try:
        dffinal['year'] = ts.tt_jd(dffinal.date_tt).utc.year
        dffinal['date'] = ts.tt_jd(dffinal.date_tt).utc_jpl()
        dffinal=dffinal.sort_values('date_tt', axis=0)
    except:
        pass
    if dffinal.shape[0] > 1:
        logger.info(f"{dffinal.shape[0]:5}  {body1} - {body2} minima between {t[0].utc_jpl()} and {t[-1].utc_jpl()}")
    return dffinal

if __name__ == '__main__':
    planets = ['saturn', 'jupiter', 'venus', 'mercury', 'moon', 'mars', 'uranus', 'neptune', 'pluto']
    dataframes = {}
    for body1, body2 in tqdm(list(itertools.combinations(planets, 2)), unit=f'days'):
        df = minimum_separations(body1, body2, None, None)
        dataframes[f"{body1}-{body2}"] = df

        with pd.ExcelWriter('conjunctions.xlsx') as writer:
            for sheetname, df in dataframes.items():
                df.to_excel(writer, sheet_name=sheetname)