# German Credit data

`german_credit_data.csv` is the project's existing, human-readable version of the
Statlog German Credit dataset. It contains 1,000 records, 20 input attributes and
a `target` column (`good` / `bad`). The scripts encode good as 1 and bad as 0.

Source: Hofmann, H. (1994), *Statlog (German Credit Data)*, UCI Machine Learning
Repository, [DOI: 10.24432/C5NC77](https://doi.org/10.24432/C5NC77).
The [UCI dataset page](https://archive.ics.uci.edu/dataset/144/statlog-german-credit-data)
provides the original files and current dataset terms.

The project copy uses descriptive column names and categorical labels instead of
the original compact codes. It is preserved as used by the experiments; the original
conversion script was not included in the source folder.
