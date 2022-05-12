import pandas as pd

ci = pd.read_csv("../data/county_index.csv")
ci.fillna(method='ffill', inplace=True)
ci = ci.to_csv("../data/county_index_2.csv")
pass


