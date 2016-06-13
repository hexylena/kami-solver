import numpy as np
import json


def fromArrays(cD, gD):
    finalBins = json.load(gD)

    # for row in range(len(cD)):
        # for col in range(len(cD[0])):
            # colourGroup = cD[row][col]

            # if colourGroup not in finalBins:
                # finalBins[colourGroup] = {}

            # if group not in finalBins[colourGroup]:
                # finalBins[colourGroup][group] = []

            # finalBins[colourGroup][group].append((row, col))
    return finalBins

def toArrays(finalBins, w, h):
    outputColour = np.empty((w, h), dtype=object)

    for colourGroup in finalBins:
        for group in finalBins[colourGroup]:
            if group == '_meta_':
                finalBins[colourGroup]['_meta_'] = map(int, finalBins[colourGroup]['_meta_'])
                continue

            points = finalBins[colourGroup][group]
            for point in points:
                outputColour[point] = colourGroup

    return outputColour, json.dumps(finalBins)
