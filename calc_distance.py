import pandas as pd
import numpy as np
from pykalman import KalmanFilter
import sys
import xml.etree.ElementTree as ET
from math import cos, asin, sqrt
import math

# pip install haversine
# pip install pykalman

def get_data(filename):
    df = pd.DataFrame(columns = ['lat', 'lon'])
    tree = ET.parse(filename) # Retrieved from https://docs.python.org/3/library/xml.etree.elementtree.html
    root = tree.getroot()
    i = 0
    for child in root[1][0]:
        #print(float(child.attrib['lat']), float(child.attrib['lon']))
        df.loc[i] = [float(child.attrib['lat']), float(child.attrib['lon'])]
        i += 1
    return df
 
def distance(data):
    shifted_data = data.shift(periods=-1) # Inspired from https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shift.html
    change = abs(data - shifted_data)
    total_change = change.sum()
    
    lat1 = data['lat'][0]
    lon1 = data['lon'][0]
    lat2 = data['lat'][0] + total_change['lat']
    lon2 = data['lon'][0] + total_change['lon']
    return distanceBetweenPoints(lat1, lon1, lat2, lon2)
    
def distanceBetweenPoints(lat1, lon1, lat2, lon2): # Retrieved this function from https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula/21623206
    p = math.pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return 12742 * asin(sqrt(a))

def smooth(data):
    kf = KalmanFilter(
        initial_state_mean = data.iloc[0],
        observation_covariance = np.diag([0.3, 0.3]) ** 2,
        transition_covariance = np.diag([0.1, 0.1]) ** 2,
        transition_matrices = [[1, 0], [0, 1]])
    kalman_smoothed, _ = kf.smooth(data)
    df = pd.DataFrame(kalman_smoothed, columns = ['lat', 'lon'])
    return df

def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.8f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.8f' % (pt['lon']))
        trkseg.appendChild(trkpt)
    
    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)
    
    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)
    
    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')

def main():
    #data = get_data('walk1.gpx')
    data = get_data(sys.argv[1])
    #print(distance(data))
    print('Unfiltered distance: %0.2f' % (distance(data)))
    
    smoothed_data = smooth(data)
    print('Filtered distance: %0.2f' % (distance(smoothed_data),))
    output_gpx(smoothed_data, 'out.gpx')

if __name__ == '__main__':
    main()