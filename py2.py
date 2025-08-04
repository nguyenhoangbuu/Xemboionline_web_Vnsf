#import py1
import pandas as pd
#def hello():
    #print('hello world file 2')
if __name__ == '__main__':
    #hello()
    df = pd.read_csv('df.csv')
    a = df[df['Ten'] == 'Buu'].Luong_bia.values[0]
    print(a)
