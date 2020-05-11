import datetime
import os
loc =locals()
path=os.getcwd()+'\\history\\test'+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
modelspath=path+'\\models\\'
cmodelspath=path+'\\cmodels\\'
picspath=path+'\\pics\\'
os.makedirs(path)
logname =path+'\\hyplog.txt'
logfile = open(logname,'w')

def pathcreate(x=0,y=0):
    os.makedirs(picspath)
    if x==1:
        os.makedirs(modelspath)
    if y==1:
        os.makedirs(cmodelspath)

def startlog():
    starttime = datetime.datetime.now()
    logfile.write(starttime.strftime('%Y-%m-%d %H:%M:%S') +'   实验开始\n')
    return starttime

def stoplog(time1):
    stoptime = datetime.datetime.now()
    logfile.write(stoptime.strftime('%Y-%m-%d %H:%M:%S') +'   实验结束\n')
    timing(time1, stoptime)
    logfile.close()
    
def record(string):
    logfile.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'   '+string+'\n')

def timing(time1, time2):
    dtime = time2 - time1
    record("本次实验共耗时" + str(dtime.days) + "天" + str(dtime.seconds) + "秒。")