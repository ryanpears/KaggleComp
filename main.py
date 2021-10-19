import sys
import pandas
from sklearn import tree
import csv 

LABEL = ''
FEATURES = []
MAP = {}
MEDIAN_COLUMNS = ["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"]

def csv_parsing(file_path):
  # following columns are all contious
  # age, fnlwgt, education-num, 
  # capital-gain, capital-loss,
  # hours-per-week 
  df = pandas.read_csv(file_path) 
  # df2 = df.copy()
  df = df_map_numeric(df, train_map=True)
  # I think the map is ok
  # df2 = df_map_numeric(df2, train_map=False)
  # print(df)
  # print(df2)
  df = df_median_treatment(df, MEDIAN_COLUMNS)
  return df

# issue need a map between names and this
def df_map_numeric(df, train_map = False):
  global MAP
  for column in df.columns:
    targets = df[column].unique()
    if train_map:
      map_to_int = {name: n for n, name in enumerate(targets)}
      MAP[column] = map_to_int
    else: 
      map_to_int = MAP[column]
    df[column] = df[column].replace(map_to_int)
  print(MAP['age'])
  return df


def df_median_treatment(df, columns):
  # columns are the list of columns
  #  to have the median treatment 
  for column in columns:
    median = df[column].median()
    df[column] = df[column].apply(lambda x: "1" if x >= median else '-1')
  return df

def main(train_path, test_path):
  global LABEL
  
  train_df = csv_parsing(train_path)
  LABEL = train_df.columns[-1]
  FEATURES = train_df.columns[:-1]
  # print(FEATURES)
  Y = train_df[LABEL]
  X = train_df[FEATURES]
  # print(X)
  clf = tree.DecisionTreeClassifier(max_depth = 5)
  clf.fit(X,Y)

  test_df = csv_parsing(test_path)
  print(test_df)
  
  test_X = test_df[FEATURES]
  prediction = clf.predict(test_X)
  print(prediction)
  output = open("attempt.csv", "w", encoding='UTF8', newline='')
  writer = csv.writer(output)
  writer.writerow(['ID','Prediction'])
  for index, row in enumerate(prediction):
    print(f"{index+1}, {row}")
    writer.writerow([index+1, row])
  #   prediction = clf.predict(row)
  #   print(prediction)

if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2])