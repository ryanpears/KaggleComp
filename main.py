import enum
import sys
import pandas
from sklearn import tree, ensemble
import csv 

LABEL = ''
FEATURES = []
MAP = {}
MEDIAN_COLUMNS = ["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"]
N = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 225, 250]

def csv_parsing(file_path, map):
  # following columns are all contious
  # age, fnlwgt, education-num, 
  # capital-gain, capital-loss,
  # hours-per-week 
  df = pandas.read_csv(file_path) 
  print(df.columns)
  if not map:
    df = df.drop(labels='ID', axis=1)
  # df2 = df.copy()
  df = df_map_numeric(df, train_map=map)
  # I think the map is ok
  # df2 = df_map_numeric(df2, train_map=False)
  # print(df)
  # print(df2)
  df = df_median_treatment(df, MEDIAN_COLUMNS)
  return df

# issue need a map between names and this
def df_map_numeric(df, train_map):
  global MAP
  for column in df.columns:
    #TODO just fucking hash
    if column not in MEDIAN_COLUMNS:
      df[column] = df[column].apply(hash)
  # print(MAP['age'])
  print(df['age'])
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
  
  train_df = csv_parsing(train_path, True)
  LABEL = train_df.columns[-1]
  FEATURES = train_df.columns[:-1]
  # print(FEATURES)
  Y = train_df[LABEL]
  X = train_df[FEATURES]

  #build a held out set
  held_out_train = train_df.sample(n=20000, replace=False)
  held_out_test = train_df.drop(held_out_train.index)

  held_out_train = held_out_train.reset_index()
  held_out_test = held_out_test.reset_index()
  print(held_out_test)
  held_out_X = held_out_train[FEATURES]
  held_out_Y = held_out_train[LABEL]

  held_test_X = held_out_test[FEATURES]
  held_test_Y = held_out_test[LABEL]

  best_error = float('inf')
  for n in range(1, len(FEATURES)):
    held_tree = tree.DecisionTreeClassifier(max_depth=n)
    held_tree.fit(held_out_X, held_out_Y)
    held_pred = held_tree.predict(held_test_X)
    incorrect = 0
    for index, pred in enumerate(held_pred):
      if pred != held_test_Y[index]:
        incorrect += 1

    error = incorrect/ len(held_pred)
    print(error)
    print(best_error)
    if error < best_error:
      best_depth = n
      best_error = error
    
  print(best_depth)
  clf = tree.DecisionTreeClassifier(max_depth=best_depth)

  clf.fit(X,Y)

  test_df = csv_parsing(test_path, False)
 
  
  test_X = test_df[FEATURES]
  prediction = clf.predict(test_X)

  # print(prediction)
  output = open("attempt.csv", "w", encoding='UTF8', newline='')
  writer = csv.writer(output)
  writer.writerow(['ID','Prediction'])
  for index, row in enumerate(prediction):
    # print(f"{index+1}, {row}")
    writer.writerow([index+1, row])
  

if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2])