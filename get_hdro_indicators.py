import os
import json
import urllib

import numpy as np
import pandas as pd


def read_indicator_data():

    data = {'indicator':list(), 'value':list(), 'country3':list(), 'year':list()}

    base_url = 'http://ec2-54-174-131-205.compute-1.amazonaws.com/API/HDRO_API.php/indicator_id={}'
    df = pd.read_csv('data/hdro_indicators.tsv', delimiter='\t')
    for icode in df['indicator_id']:
        url = base_url.format(icode)
        with urllib.request.urlopen(url) as http_resp:
            print('Retrieving indicator: {}'.format(url))
            json_data = json.load(http_resp)
        if len(json_data) > 0:
            for c3,idata in json_data['indicator_value'].items():
                if str(icode) in idata:                    
                    for year,val in idata[str(icode)].items():
                        data['indicator'].append(icode)
                        data['value'].append(float(val))
                        data['country3'].append(c3)
                        data['year'].append(int(year))
                else:
                    print('Unexpected indicator code for country {}: {}'.format(c3, str(idata)))
        
    hdro_df = pd.DataFrame(data)
    hdro_df.sort_values(['indicator', 'country3', 'year'])
    hdro_df.to_csv('data/hdro_indices_flat.csv', header=True, index=False)


if __name__ == '__main__':
    read_indicator_data()
