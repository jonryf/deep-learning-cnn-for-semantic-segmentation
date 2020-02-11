import numpy as np
import matplotlib.pyplot as plt
from math import e

class PerformanceObj:
    def __init__(self):
        self.percentCorrect = []
        self.error = []
        self.params = {}

def plotMultiPerfObjs(setSetPerfObs, title="", labels=[]):
    correctFig, correctAx = plt.subplots()
    errorFig, errorAx = plt.subplots()
    colors = ["r","b","g","o","b","y","k","c"]

    for setPerfObs in setSetPerfObs:
        i = 0
        for perfObs in setPerfObs:
            corrects = []
            errors = []

            for perfObj in perfObs:
                corrects.append(perfObj.percentCorrect)
                errors.append(perfObj.error)
            
            corrects = np.array(corrects)
            errors = np.array(errors)

            meanCorrect = np.mean(corrects, axis=0)
            stdCorrect = np.std(corrects, axis=0)

            meanError = np.mean(errors, axis=0)
            stdError = np.std(errors, axis=0)

            plt.title(title)
            x = np.arange(meanError.shape[0])
            color = colors[i]
            correctAx.set_xlabel('Epochs')
            correctAx.set_ylabel('Percent Correct')
            correctAx.plot(x, meanCorrect, color=color)
            correctAx.tick_params(axis='y')
            correctAx.fill_between(x,meanCorrect - stdCorrect, meanCorrect + stdCorrect, alpha=0.35, color=color)
            correctAx.set_title('%s - Percent Correct' % title)
            color = colors[i]
            errorAx.set_xlabel('Epochs')
            errorAx.set_ylabel('Error')  # we already handled the x-label with ax1
            errorAx.plot(x, meanError, color=color)
            errorAx.tick_params(axis='y')
            errorAx.fill_between(x,meanError - stdError, meanError + stdError, alpha=0.35, color=color)
            errorAx.set_title('%s - Error' % title)
            i += 1
    correctAx.legend(labels)
    errorAx.legend(labels)
    correctFig.savefig('%s-MultiCorrect.pdf' % title.replace(" ",""))
    errorFig.savefig('%s-MultiError.pdf' % title.replace(" ",""))



def plotPerfObjs(perfObs, title=""):
    corrects = []
    errors = []

    for perfObj in perfObs:
        corrects.append(perfObj.percentCorrect)
        errors.append(perfObj.error)
    
    corrects = np.array(corrects)
    errors = np.array(errors)

    meanCorrect = np.mean(corrects, axis=0)
    stdCorrect = np.std(corrects, axis=0)

    meanError = np.mean(errors, axis=0)
    stdError = np.std(errors, axis=0)

    fig, ax1 = plt.subplots()
    plt.title(title)
    x = np.arange(meanError.shape[0])
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Percent Correct', color=color)
    ax1.plot(x, meanCorrect, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.fill_between(x,meanCorrect - stdCorrect, meanCorrect + stdCorrect, alpha=0.35, color=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Error', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, meanError, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.fill_between(x,meanError - stdError, meanError + stdError, alpha=0.35, color=color)
    plt.savefig('%s.pdf' % title.replace(" ",""))

# # Test Code
# temp = PerformanceObj()
# temp.percentCorrect = [4,5,100]
# temp.error = [1,2,10]
# plotMultiPerfObjs([[[temp]]])
# plt.show()


# alls = []
# for j in range(2):
#     row = []
#     for i in range(3):
#         temp = PerformanceObj()
#         temp.percentCorrect = np.arange(10)
#         temp.percentCorrect += i + (j * 10)
#         temp.percentCorrect = np.power(e, temp.percentCorrect)
#         temp.error = np.arange(10)
#         temp.error += i + 1 + (j * 10)
#         temp.error = np.log(temp.error)
#         row.append(temp)
#     alls.append(row)
# plotMultiPerfObjs([alls], labels=["one", "two"])
# plt.show()