from multiprocessing import Pool

import argparse
import os
import pprint
import sys
sys.path.append('../')
import pyatmos
import time
import datetime
import numpy as np
from functools import partial
import thermonets as tn
import hwm93
from astropy.coordinates import get_sun
from astropy.time import Time
from pyatmos.jb2008.spaceweather import get_sw

#for MacOS users (see https://stackoverflow.com/questions/74217717/what-does-os-environkmp-duplicate-lib-ok-actually-do):
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# USAGE (example):
##  python generate_jb08_db.py --num_processes 1 --n_height_points 10 --n_lonlat_points 10

def compute_density(inputs):
    date,  alt, latitude, longitude, swdata, _, _, _, _, _, _, _, _, _, _, _ = inputs
    return pyatmos.jb2008(date,(latitude, longitude, alt), swdata).rho

def create_dir(dir_path):
    if os.path.exists(dir_path):
        pass
    else:
        dir = os.makedirs(dir_path)

def valid_date(s):
    try:
        return datetime.datetime.strptime(s, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        raise argparse.ArgumentTypeError('Not a valid date:' + s + '. Expecting YYYYMMDDHHMMSS.')

def main():
    parser = argparse.ArgumentParser(description='Script for generating JB-08 data (WARNING: much slower than NRLMSISE-00, due to lack of parallelization of JB08 python package):',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_lonlat_points', help='Number of points on the sphere', type=int, default = 100)
    parser.add_argument('--min_height', help='Minimum height (km)', type=float, default = 158.48931924611142) # 10**2.2
    parser.add_argument('--max_height', help='Minimum height (km)', type=float, default = 630.957344480193) # 10**2.8
    parser.add_argument('--n_height_points', help='Number of point to sample the altitude range (logarithmically)', type=int, default = 100)
    parser.add_argument('--num_processes', help='Number of processes to be spawn', type=int, default = 200)
    opt = parser.parse_args()
    #we download & load the space weather data needed for JB-08 (from Celestrack) only once here:
    swfile = pyatmos.download_sw_jb2008()
    swdata = pyatmos.read_sw_jb2008(swfile)
    # File name to log console output
    file_name_log = os.path.join('../dbs/jb08_db.log')
    te = open(file_name_log,'w')  # File where you need to keep the logs
    class Unbuffered:
       def __init__(self, stream):
           self.stream = stream

       def write(self, data):
           self.stream.write(data)
           self.stream.flush()
           te.write(data)    # Write the data of stdout here to a text file as well
           te.flush()

       def flush(self):
           self.stream.flush()
           te.flush()

    sys.stdout=Unbuffered(sys.stdout)

    print('JB-08 Script\n')
    print('Arguments:\n{}\n'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(opt), depth=2, width=50)

    years=[2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022]
    days = np.arange(365)


    pool = Pool(processes=opt.num_processes)
    #Generate random points on the sphere
    inputs=[]
    for u in np.linspace(0., 1., opt.n_lonlat_points):
        for v in np.linspace(0., 1., opt.n_lonlat_points):
            alts = np.logspace(np.log10(opt.min_height), np.log10(opt.max_height), opt.n_height_points)
            lon=2*np.pi*u
            lat=np.arccos(2*v-1)-np.pi/2
            year = np.random.choice(years)
            minutes = np.random.uniform(low=0.,high=1440.)
            day = np.random.choice(days)
            date = datetime.datetime(year, 1, 1, 0, 0, 0)+datetime.timedelta(days=int(day))+datetime.timedelta(minutes=minutes)
            #JB-08 also uses the Sun declination and right ascension:
            t = Time(date,location=(str(np.rad2deg(lon))+'d',str(np.rad2deg(lat))+'d'))
            sun = get_sun(t)
            #I also extract the space weather indices used by JB-08:
            f107,f107a,s107,s107a,m107,m107a,y107,y107a,dDstdT= get_sw(swdata,t.mjd)
            for alt in alts:
                inputs.append((date,  alt, np.rad2deg(lat), np.rad2deg(lon), swdata, np.rad2deg(sun.ra.rad), np.rad2deg(sun.dec.rad), f107, f107a, s107, s107a, m107, m107a, y107, y107a, dDstdT))
    print(f'Starting parallel pool with {len(inputs)}:')
    print(f'example element: {inputs[0], inputs[-1]}')
    p = pool.map(compute_density, inputs)
    print('Done ... writing to file')
    # Save inputs and outputs to a file
    output_file_name = f'../dbs/jb08_db.txt'  
    with open(output_file_name, 'w') as output_file:    
        output_file.write(f'day, month, year, hour, minute, second, microsecond, alt [km], lat [deg], lon [deg], sun ra [deg], sun dec [deg], f107, f107A, s107, s107A, m107, m107A, y107, y107A, dDstdT, density [kg/m^3]\n')
        for input_data, result in zip(inputs, p):
            day = input_data[0].day
            month = input_data[0].month
            year = input_data[0].year
            hour = input_data[0].hour
            minute = input_data[0].minute
            second = input_data[0].second
            microsecond = input_data[0].microsecond
            output_file.write(f'{day}, {month}, {year}, {hour}, {minute}, {second}, {microsecond}, {input_data[1]}, {input_data[2]}, {input_data[3]}, {input_data[4]}, {input_data[5]}, {input_data[6]}, {input_data[7]}, {input_data[8]}, {input_data[9]}, {input_data[10]}, {input_data[11]}, {input_data[12]}, {input_data[13]}, {input_data[14]}, {result}\n')
    print('Done')

if __name__ == "__main__":
    time_start = time.time()
    main()
    print('\nTotal duration: {}'.format(time.time() - time_start))
    sys.exit(0)