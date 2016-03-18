from sklearn.neural_network import MLPClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelBinarizer
import numpy as np
#from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
try:
    from IPython.core.display import clear_output
    have_ipython = True
except ImportError:
    have_ipython = False
import sys
class SoftMaxMLPClassifier(MLPClassifier):
    def __init__(self, hidden_layer_sizes=(100,), activation="relu",
             algorithm='adam', alpha=0.0001,
             batch_size='auto', learning_rate="constant",
             learning_rate_init=0.001, power_t=0.5, max_iter=200,
             shuffle=True, random_state=None, tol=1e-4,
             verbose=False, warm_start=False, momentum=0.9,
             nesterovs_momentum=True, early_stopping=False,
             validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
             epsilon=1e-8):
        sup = super(MLPClassifier, self)
        sup.__init__(hidden_layer_sizes=hidden_layer_sizes,
                     activation=activation, algorithm=algorithm, alpha=alpha,
                     batch_size=batch_size, learning_rate=learning_rate,
                     learning_rate_init=learning_rate_init, power_t=power_t,
                     max_iter=max_iter, loss='log_loss', shuffle=shuffle,
                     random_state=random_state, tol=tol, verbose=verbose,
                     warm_start=warm_start, momentum=momentum,
                     nesterovs_momentum=nesterovs_momentum,
                     early_stopping=early_stopping,
                     validation_fraction=validation_fraction,
                     beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
        self.label_binarizer_ = LabelBinarizer()
    def predict(self, X):

        check_is_fitted(self, "coefs_")
        y_scores = self.predict_proba(X)
        maxs = np.max(y_scores,axis=1).reshape(y_scores.shape[0],1)
        return (y_scores == maxs)
   



class FootballDataHelper:
    def __init__ (self, recentNum=5):
        self.win_mapping = {'D':0, 'H':1,'A':2}
        self.recentNum = recentNum
        self.df = None
        #self.hiddensCount = 2
       
        
    def readFootBallData(self,filename): 
        df = pd.read_csv(filename)
        df = df.drop(df.columns[range(23,df.shape[1])], axis=1)
        df = df.drop("Div",axis=1)
        df['Date'] = pd.to_datetime(df['Date'],dayfirst=True)
        df['HTR'] = df['HTR'].map(self.win_mapping)
        df['FTR'] = df['FTR'].map(self.win_mapping)
        df= df.drop('Referee', 1)
        print(df.shape)
          #self.team = df['HomeTeam'].drop_duplicates()
        if self.df is None:
            self.df = df
        else:
            self.df = pd.concat([self.df,df])
            
        teams = self.df['HomeTeam'].drop_duplicates()
        teamMap = {}
        for index , v in enumerate(teams):
            teamMap[v] = index
        self.teamsMap = teamMap
 
    def getTeam(self,dataFrame, teamName):       
        return dataFrame[(dataFrame["HomeTeam"] == teamName) | (dataFrame["AwayTeam"] == teamName)]
    def formatData(self, X_train ):
        print("start format")
        X_train = X_train.sort_values(by="Date")
        X_train['Date'] = pd.to_numeric(X_train['Date'])/1e9/24/60/60
        
        sys.stdout.flush()
        res = []
        y =[]
        for v in X_train['FTR']:
            y.append(range(3)==v)
        resy=[]
        for i in range(X_train.shape[0]):
            print("\r progress {}".format(i))
            sys.stdout.flush()
            x = X_train.iloc[i]
            homeName = x['HomeTeam']
            awayName = x['AwayTeam']
            homeTeam = self.getTeam(X_train,homeName)
            awayTeam = self.getTeam(X_train,awayName)
            prevHome = self.previousRecords(homeTeam,x['Date'])
            prevAway = self.previousRecords(awayTeam,x['Date'])
            if prevHome is None or prevAway is None:
               # print("{} skip".format(i))
                continue
           # print("{} has enough sample".format(i))  
            prevHome=prevHome.copy()
            prevAway = prevAway.copy()
            prevHome['HomeTeam']=(prevHome['HomeTeam']==homeName)
            prevHome['AwayTeam']=(prevHome['AwayTeam']==homeName)
            prevAway['HomeTeam']=(prevAway['HomeTeam']==awayName)
            prevAway['AwayTeam']=(prevAway['AwayTeam']==awayName)
            homeDate = prevHome['Date'].values
            awayDate = prevAway['Date'].values
            #homeDate = homeDate.astype('uint64')/1e9/24/60/60
            homeDate = x['Date'] - homeDate 
            #awayDate = awayDate.astype('uint64')/1e9/24/60/60
            awayDate = x['Date'] - awayDate 
            #print(homeDate)
            hv = prevHome.drop('Date',axis=1).values
            av = prevAway.drop('Date',axis=1).values
            hv = np.column_stack([hv, homeDate])
            av = np.column_stack([av,awayDate])
            
            inData = np.ravel(np.array([hv,av]))
            res.append(inData)
            resy.append(y[i])
        
        Xres = np.array(res)
        
       
        X_train_std = Xres
        print("finish")

        return (X_train_std,np.array(resy))
            
    def fit(self,X=None,y=None):
        teams = self.df['HomeTeam'].drop_duplicates()
       
        if X is None or y is None:
            (X, y)=self.formatData(df)
        X_train,X_test_val, y_train, y_test_val =    train_test_split(X,y, test_size=0.4)
        X_val ,X_test,y_val,y_test = train_test_split(X_test_val,y_test_val, test_size=0.5)
        print(X_val)
        
        
     
        print("Start Training")
        self.nn.fit(X_train,y_train)
        print("fisish Training")
        return (X_val ,X_test,y_val,y_test)

    def validate(self, X_val, y_val):
            return self.nn.predict(X_val)
            
            
        
        
    def previousRecords(self,team, date , recentNum):
        prev = team[( team["Date"] < date)]
        
        if prev.shape[0] < recentNum :
            #print("less than min Num")
            return None
        else:
            return prev.iloc[-recentNum:]
    
    def getH1(self):
        #recent matches (only win/loss/draw)
       
        
       # print (self.df)      
        X  = self.df.sort_values(by="Date")
        y = []
        for v in X['FTR']:
            y.append(range(3)==v)
       # print(y)
        resy=[]
        resx=[]
        print("h1:start format")
        for i in range(X.shape[0]):
            
            sys.stdout.write("\r progress {}".format(i))
            sys.stdout.flush()
            x = X.iloc[i]
            homeName = x['HomeTeam']
            awayName = x['AwayTeam']
            homeTeam = self.getTeam(X,homeName)
            awayTeam = self.getTeam(X,awayName)
            prevHome = self.previousRecords(homeTeam,x['Date'])
            prevAway = self.previousRecords(awayTeam,x['Date'])
            if prevHome is None or prevAway is None:
               # print("{} skip".format(i))
                continue
            prevHomeWin = []
            prevAwayWin = []
            for v in prevHome['FTR']:
                prevHomeWin.append(range(3) == v)
            #print(prevHomeWin)
            for v in prevAway['FTR']:
                prevAwayWin.append(range(3) == v)
            pHHT=(prevHome['HomeTeam']==homeName).values
            pAHT=(prevAway['HomeTeam']==awayName).values
            tempX=[]
            for j in range(pHHT.shape[0]):
                tempX.append(np.append(prevHomeWin[j],pHHT[j]))
                
            for j in range(pAHT.shape[0]):
                tempX.append(np.append(prevAwayWin[j],pAHT[j]))
            resx.append(np.ravel(tempX))
            
            resy.append(y[i])
        print("finish")
        sys.stdout.flush()
        return np.array(resx), np.array(resy)
    def getH2(self):         
       #team based      
        X  = self.df.sort_values(by="Date")
        y = []
        for v in X['FTR']:
            y.append(range(3)==v)
       # print(y)
        resy=[]
        resx=[]
        print("h1:start format")
        X['HomeTeam'] = X['HomeTeam'].map(self.teamsMap)
        X['AwayTeam'] = X['AwayTeam'].map(self.teamsMap)
        nativeX = X[['HomeTeam','AwayTeam']].values
        #print(X)
        ohe = OneHotEncoder(categorical_features=[0,1])
        res = ohe.fit_transform(nativeX).toarray()
        #print(res)
        return res, np.array(y)
    def _getH3RecentMatches(self,x, X,teamName,recentNum):
        team = self.getTeam(X,teamName)
        prev = self.previousRecords(team,x['Date'],recentNum)
        if prev is None:
               return None
        prevHt=  prev['HomeTeam'].values
        prevAt=  prev['AwayTeam'].values   
        prevIsHome = []
        prevOther = []
        for i in range(recentNum):
            if prevHt[i] == teamName:
                prevIsHome.append(1)
                prevOther.append(prevAt[i])
            else:
                prevIsHome.append(0)
                prevOther.append(prevHt[i])
        wins = prev['FTR'].values       
        temp = np.array([prevIsHome,prevOther,wins]).T


        return np.ravel(temp)
        
    def getH3(self, recentNum):
        X  = self.df.sort_values(by="Date")
        X['HomeTeam'] = X['HomeTeam'].map(self.teamsMap)
        X['AwayTeam'] = X['AwayTeam'].map(self.teamsMap)
        y = []
        for v in X['FTR']:
            y.append(range(3)==v)
       # print(y)
        resy=[]
        resx=[]
        print("h3:start format")
        recents = []
        for i in range(X.shape[0]):
            
            sys.stdout.write("\r progress {}".format(i))
            sys.stdout.flush()
            x = X.iloc[i]
            homeName = x['HomeTeam']
            awayName = x['AwayTeam']
            homeRecent = self._getH3RecentMatches(x,X,homeName, recentNum)
            awayRecent =self._getH3RecentMatches(x,X,awayName, recentNum)
            if homeRecent is None or awayRecent is None:
                   continue 
        #    print(homeName)
         #   print(homeRecent)
        #    print(awayName)
         #   print(awayRecent)
         #   return
            recents.append(np.hstack([homeName,awayName, homeRecent,awayRecent]))        
            resy.append(y[i])
        cols =np.hstack([[0,1],list(range(3,len(recents[0]),3))])
        print(cols)
        ohe = OneHotEncoder(categorical_features=cols)
        res = ohe.fit_transform(recents).toarray()
        print("finish")
        sys.stdout.flush()
        return res, np.array(resy)