import pandas as pds
import matplotlib.pyplot as plts
import numpy as nps

def plotdata(x1Label,x1,x2Label,x2,x3Label,x3,x4Label,x4,x5Label,x5,x6Label,x6,ylabel,plotName,imgName):
    data = pds.DataFrame({x1Label: pds.Series(nps.mean(x1, axis=1)),
                         x2Label: pds.Series(nps.mean(x2, axis=1)),
                         x3Label: pds.Series(nps.mean(x3, axis=1)),
                         x4Label: pds.Series(nps.mean(x4, axis=1)),
                         x5Label: pds.Series(nps.mean(x5, axis=1)),
                         x6Label: pds.Series(nps.mean(x6, axis=1)), ylabel: pds.Series(train_sizes)})
    plts.plot(plotName, x1Label, data=data, label=x1Label)
    plts.plot(plotName, x2Label, data=data, label=x2Label)
    plts.plot(plotName, x3Label, data=data, label=x3Label)
    plts.plot(plotName, x4Label, data=data, label=x4Label)
    plts.plot(plotName, x5Label, data=data, label=x5Label)
    plts.plot(plotName, x6Label, data=data, label=x6Label)
    plts.xlabel(plotName)
    plts.ylabel(xlabel)
    plts.title(ylabel)
    plts.legend(loc="best")
    imgName = imgName + '.png'
    plts.savefig(imgName)
    plts.legend()
    plts.show()