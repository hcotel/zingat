import pandas as pd
import requests

url = "https://app.endeksa.com/dynamictrend/"

data = {'cityId': 34,
        'CountyId': 2051}

response = requests.post(url, data=data)
pass
