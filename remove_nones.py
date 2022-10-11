import pandas as pd
if __name__ == '__main__':
    df = pd.read_csv("/Users/gali.k/phd/phd_2021/data_analysis/rawdata/aggregated/agg_total_run_with_nulls.csv")
    df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))
    df.to_csv("/Users/gali.k/phd/phd_2021/data_analysis/rawdata/aggregated/agg_total_run_without_nulls.csv")