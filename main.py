import sys
import pandas
from sklearn import tree, ensemble
import csv 

LABEL = ''
FEATURES = []
MAP = {}
MEDIAN_COLUMNS = ["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"]
N = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 225, 250]

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
  print(len(X))
  best_rand_forest = None
  best_error = float('inf')
  best_n = 0
  for n in N:
    sample = train_df.sample(n=15000)
    sample_train = sample.sample(n=10000, replace=False)
    sample_test = sample.drop(sample_train.index)

    sample_train = sample_train.reset_index()
    sample_test = sample_test.reset_index()
    
    sample_Y = sample_train[LABEL]
    sample_X = sample_train[FEATURES]
    sample_test_Y = sample_test[LABEL]
    sample_test_X = sample_test[FEATURES]
    rand_forest = ensemble.RandomForestClassifier(n_estimators=n)
    rand_forest.fit(sample_X, sample_Y)
    sample_predictions = rand_forest.predict(sample_test_X)
    correct, incorrect = 0, 0

    for index, sample_prediction in enumerate(sample_predictions):
      if sample_prediction == sample_test_Y[index]:
        correct += 1
      else:
        incorrect += 1
    sample_error = incorrect / (correct + incorrect)
    print(f"sample error is ", sample_error)
    print(f"best error is ", best_error)
    if sample_error < best_error:
      best_rand_forest = rand_forest
      best_error = sample_error
      best_n = n




  # clf = tree.DecisionTreeClassifier(max_depth = 5)
  best_rand_forest.fit(X, Y)
  # clf.fit(X,Y)

  test_df = csv_parsing(test_path)
 
  
  test_X = test_df[FEATURES]
  # prediction = clf.predict(test_X)
  prediction = best_rand_forest.predict(test_X)
  print(prediction)
  output = open("attempt.csv", "w", encoding='UTF8', newline='')
  writer = csv.writer(output)
  writer.writerow(['ID','Prediction'])
  for index, row in enumerate(prediction):
    print(f"{index+1}, {row}")
    writer.writerow([index+1, row])
  #   prediction = clf.predict(row)
  #   print(prediction)
  print(best_n)
  print(best_error)

if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2])