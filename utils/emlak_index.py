import pandas as pd
import requests
import json
emlak_csv = pd.read_csv("/home/huseyin/PycharmProjects/Zingat/data/endekse_index.csv")
# with open('/home/huseyin/PycharmProjects/Zingat/data/emlak.json') as json_file:
#     data = json.load(json_file)
#     data_df = pd.json_normalize(data['Trend'])
#     data_df['county'] = data_df['CityName'] + "-" + data_df['CountyName']
#     emlak_csv = pd.concat([emlak_csv, data_df], axis=0)
#     emlak_csv.to_csv("/home/huseyin/PycharmProjects/Zingat/data/endekse_index.csv")
#     pass

feats = ['county', 'UnitPriceForSale', 'DisplayName']
emlak_csv_2 = emlak_csv[feats]
emlak_csv_2.to_csv("/home/huseyin/PycharmProjects/Zingat/data/county_index.csv")
pass
