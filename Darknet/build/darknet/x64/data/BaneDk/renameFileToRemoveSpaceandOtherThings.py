import os

path =  os.getcwd()
filenames = os.listdir(path)

for f in filenames:
    count = -1
    for tmp in f:
        if tmp == ".":
            count += 1
    newf = f.replace(".","",count)
    newf = newf.replace("hs_p_","hsp-")
    newf = newf.replace(" ","_")
    newf = newf.replace(",","")
    
    os.rename(f,newf)
    #os.rename(f, f.replace(" ", "_"))
    #os.rename(f, f.replace(",", ""))
    #os.rename(f, f.replace(".0", "0"))
    #os.rename(f, f.replace(".1", "1"))
    #os.rename(f, f.replace(".2", "2"))
    #os.rename(f, f.replace(".3", "3"))
    #os.rename(f, f.replace(".4", "4"))
    #os.rename(f, f.replace(".5", "5"))
    #os.rename(f, f.replace(".6", "6"))
    #os.rename(f, f.replace(".7", "7"))
    #os.rename(f, f.replace(".8", "8"))
    #os.rename(f, f.replace(".9", "9"))
