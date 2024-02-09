from multiprocessing import Pool

import argparse
import os
import pprint
import sys
from nrlmsise00 import msise_flat
import time
import datetime
import numpy as np
import spaceweather

#for MacOS users:
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def compute_density(inputs):
    date,  alt, latitude, longitude, f107A, f107, ap = inputs
    return msise_flat(date, alt, latitude, longitude, f107A, f107, ap)[5]*1e3

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
    parser = argparse.ArgumentParser(description='Differential drag project:',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_lonlat_points', help='Number of points on the sphere', type=int, default = 1024)
    parser.add_argument('--num_processes', help='Number of processes to be spawn', type=int, default = 200)
    opt = parser.parse_args()
    # File name to log console output
    file_name_log = os.path.join('log_{}'.format(datetime.datetime.now().timestamp()))
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

    print('Differential-Drag Script\n')
    print('Arguments:\n{}\n'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(opt), depth=2, width=50)

    years=[2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022]
    days = np.arange(365)
    sw_data=spaceweather.sw_daily()
    ap_data = sw_data[["Apavg"]]
    f107_data = sw_data[["f107_obs"]]
    f107a_data = sw_data[["f107_81ctr_obs"]]

    pool = Pool(processes=opt.num_processes)
    #Generate random points on the sphere
    inputs=[]
    for u in np.arange(0., 1., 0.01):
        for v in np.arange(0., 1., 0.01):
            alts = np.arange(10**2.2, 10**2.8, 5)
            lon=np.array((2*np.pi*u))
            lat=np.array((np.arccos(2*v-1)-np.pi/2))
            year = np.random.choice(years)
            minutes = np.random.uniform(low=0.,high=1440.)
            day = np.random.choice(days)
            date = datetime.datetime(year, 1, 1, 0, 0, 0)+datetime.timedelta(days=int(day))+datetime.timedelta(minutes=minutes)
            ap=ap_data.loc[f'{int(year)}-{int(date.month)}-{int(date.day)}'].values[0]
            f107=f107_data.loc[f'{int(year)}-{int(date.month)}-{int(date.day)}'].values[0]
            f107A=f107a_data.loc[f'{int(year)}-{int(date.month)}-{int(date.day)}'].values[0]
            if f107 > 0 and f107A and ap >= 0:
                inputs.append((date,  alts, np.rad2deg(lat), np.rad2deg(lon), f107A, f107, ap))
    print(f'Starting parallel pool with {len(inputs)}:')
    print(f'example element: {inputs[0], inputs[-1]}')
    p = pool.map(compute_density, inputs)
    print('Done')
    # Save inputs and outputs to a file
    output_file_name = f'output_nrlmsise00.txt'  
    with open(output_file_name, 'w') as output_file:       
        for input_data, result in zip(inputs, p):
            for alt in input_data[1]:
                day = input_data[0].day
                month = input_data[0].month
                year = input_data[0].year
                hour = input_data[0].hour
                minute = input_data[0].minute
                second = input_data[0].second
                microsecond = input_data[0].microsecond
                output_file.write(f'{day}, {month}, {year}, {hour}, {minute}, 
                                  {second}, {microsecond}, {alt}, {input_data[2]}, 
                                  {input_data[3]}, {input_data[4]}, {input_data[5]}, {input_data[6]}\n')

if __name__ == "__main__":
    time_start = time.time()
    main()
    print('\nTotal duration: {}'.format(time.time() - time_start))
    sys.exit(0)