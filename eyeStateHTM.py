import csv
import numpy
import os
import yaml
import time

from nupic.algorithms.sdr_classifier_factory import SDRClassifierFactory
from nupic.algorithms.spatial_pooler import SpatialPooler
from nupic.algorithms.temporal_memory import TemporalMemory
from nupic.encoders.random_distributed_scalar import \
    RandomDistributedScalarEncoder

_NUM_RECORDS = 14400

_WORK_DIR = os.getcwd()
_INPUT_FILE_PATH = os.path.join(_WORK_DIR, "dataset",
                                "eyeState14400.csv")
_PARAMS_PATH = os.path.join(_WORK_DIR, "params", "modelEyeState.yaml")

print _INPUT_FILE_PATH
print _PARAMS_PATH

with open(_PARAMS_PATH, "r") as f:
    modelParams = yaml.safe_load(f)["modelParams"]
    enParams = modelParams["sensorParams"]["encoders"]
    spParams = modelParams["spParams"]
    tmParams = modelParams["tmParams"]

encoderAF3 = RandomDistributedScalarEncoder(enParams["sensAF3"]["resolution"])
encoderF7 = RandomDistributedScalarEncoder(enParams["sensF7"]["resolution"])
encoderF3 = RandomDistributedScalarEncoder(enParams["sensF3"]["resolution"])
encoderFC5 = RandomDistributedScalarEncoder(enParams["sensFC5"]["resolution"])
encoderT7 = RandomDistributedScalarEncoder(enParams["sensT7"]["resolution"])
encoderP7 = RandomDistributedScalarEncoder(enParams["sensP7"]["resolution"])
encoderO1 = RandomDistributedScalarEncoder(enParams["sensO1"]["resolution"])
encoderO2 = RandomDistributedScalarEncoder(enParams["sensO2"]["resolution"])
encoderP8 = RandomDistributedScalarEncoder(enParams["sensP8"]["resolution"])
encoderT8 = RandomDistributedScalarEncoder(enParams["sensT8"]["resolution"])
encoderFC6 = RandomDistributedScalarEncoder(enParams["sensFC6"]["resolution"])
encoderF4 = RandomDistributedScalarEncoder(enParams["sensF4"]["resolution"])
encoderF8 = RandomDistributedScalarEncoder(enParams["sensF8"]["resolution"])
encoderAF4 = RandomDistributedScalarEncoder(enParams["sensAF4"]["resolution"])
encoderEyeDetection = RandomDistributedScalarEncoder(
    enParams["sensEyeDetection"]["resolution"])

encodingWidth = (encoderAF3.getWidth() + encoderF7.getWidth() +
                 encoderF3.getWidth() + encoderFC5.getWidth() +
                 encoderT7.getWidth() + encoderP7.getWidth() +
                 encoderO1.getWidth() + encoderO2.getWidth() +
                 encoderP8.getWidth() + encoderT8.getWidth() +
                 encoderFC6.getWidth() + encoderF4.getWidth() +
                 encoderF8.getWidth() + encoderAF4.getWidth() +
                 encoderEyeDetection.getWidth())

print encodingWidth

start = time.time()

sp = SpatialPooler(
    inputDimensions=(encodingWidth,),
    columnDimensions=(spParams["columnCount"],),
    potentialPct=spParams["potentialPct"],
    potentialRadius=encodingWidth,
    globalInhibition=spParams["globalInhibition"],
    localAreaDensity=spParams["localAreaDensity"],
    numActiveColumnsPerInhArea=spParams["numActiveColumnsPerInhArea"],
    synPermInactiveDec=spParams["synPermInactiveDec"],
    synPermActiveInc=spParams["synPermActiveInc"],
    synPermConnected=spParams["synPermConnected"],
    boostStrength=spParams["boostStrength"],
    seed=spParams["seed"],
    wrapAround=False
)

end = time.time()
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

tm = TemporalMemory(
    columnDimensions=(tmParams["columnCount"],),
    cellsPerColumn=tmParams["cellsPerColumn"],
    activationThreshold=tmParams["activationThreshold"],
    initialPermanence=tmParams["initialPerm"],
    connectedPermanence=spParams["synPermConnected"],
    minThreshold=tmParams["minThreshold"],
    maxNewSynapseCount=tmParams["newSynapseCount"],
    permanenceIncrement=tmParams["permanenceInc"],
    permanenceDecrement=tmParams["permanenceDec"],
    predictedSegmentDecrement=0.0,
    maxSegmentsPerCell=tmParams["maxSegmentsPerCell"],
    maxSynapsesPerSegment=tmParams["maxSynapsesPerSegment"],
    seed=tmParams["seed"]
)

classifier = SDRClassifierFactory.create()
postvDetect = 0

with open(_INPUT_FILE_PATH, "r") as fin:
    reader = csv.reader(fin)
    headers = reader.next()
    reader.next()
    reader.next()

    for count, record in enumerate(reader):

        if count >= _NUM_RECORDS:
            break

        sensAF3 = float(record[0])
        sensF7 = float(record[1])
        sensF3 = float(record[2])
        sensFC5 = float(record[3])
        sensT7 = float(record[4])
        sensP7 = float(record[5])
        sensO1 = float(record[6])
        sensO2 = float(record[7])
        sensP8 = float(record[8])
        sensT8 = float(record[9])
        sensFC6 = float(record[10])
        sensF4 = float(record[11])
        sensF8 = float(record[12])
        sensAF4 = float(record[13])
        sensEyeDetection = float(record[14])

        sensAF3_Bits = numpy.zeros(encoderAF3.getWidth())
        sensF7_Bits = numpy.zeros(encoderF7.getWidth())
        sensF3_Bits = numpy.zeros(encoderF3.getWidth())
        sensFC5_Bits = numpy.zeros(encoderFC5.getWidth())
        sensT7_Bits = numpy.zeros(encoderT7.getWidth())
        sensP7_Bits = numpy.zeros(encoderP7.getWidth())
        sensO1_Bits = numpy.zeros(encoderO1.getWidth())
        sensO2_Bits = numpy.zeros(encoderO2.getWidth())
        sensP8_Bits = numpy.zeros(encoderP8.getWidth())
        sensT8_Bits = numpy.zeros(encoderT8.getWidth())
        sensFC6_Bits = numpy.zeros(encoderFC6.getWidth())
        sensF4_Bits = numpy.zeros(encoderF4.getWidth())
        sensF8_Bits = numpy.zeros(encoderF8.getWidth())
        sensAF4_Bits = numpy.zeros(encoderAF4.getWidth())
        sensEyeDetection_Bits = numpy.zeros(encoderEyeDetection.getWidth())

        encoderAF3.encodeIntoArray(sensAF3, sensAF3_Bits)
        encoderF7.encodeIntoArray(sensF7, sensF7_Bits)
        encoderF3.encodeIntoArray(sensF3, sensF3_Bits)
        encoderFC5.encodeIntoArray(sensFC5, sensFC5_Bits)
        encoderT7.encodeIntoArray(sensT7, sensT7_Bits)
        encoderP7.encodeIntoArray(sensP7, sensP7_Bits)
        encoderO1.encodeIntoArray(sensO1, sensO1_Bits)
        encoderO2.encodeIntoArray(sensO2, sensO2_Bits)
        encoderP8.encodeIntoArray(sensP8, sensP8_Bits)
        encoderT8.encodeIntoArray(sensT8, sensT8_Bits)
        encoderFC6.encodeIntoArray(sensFC6, sensFC6_Bits)
        encoderF4.encodeIntoArray(sensF4, sensF4_Bits)
        encoderF8.encodeIntoArray(sensF8, sensF8_Bits)
        encoderAF4.encodeIntoArray(sensAF4, sensAF4_Bits)
        encoderEyeDetection.encodeIntoArray(sensEyeDetection,
                                            sensEyeDetection_Bits)

        encoding = numpy.concatenate(
            [sensAF3_Bits, sensF7_Bits, sensF3_Bits, sensFC5_Bits, sensT7_Bits,
             sensP7_Bits, sensO1_Bits, sensO2_Bits, sensP8_Bits, sensT8_Bits,
             sensFC6_Bits, sensF4_Bits, sensF8_Bits, sensAF4_Bits,
             sensEyeDetection_Bits]
        )

        activeColumns = numpy.zeros(spParams["columnCount"])

        sp.compute(encoding, True, activeColumns)
        activeColumnIndices = numpy.nonzero(activeColumns)[0]

        tm.compute(activeColumnIndices, learn=True)

        activeCells = tm.getActiveCells()

        bucketIdx = encoderEyeDetection.getBucketIndices(sensEyeDetection)[0]

        classifierResult = classifier.compute(
            recordNum=count,
            patternNZ=activeCells,
            classification={
                "bucketIdx": bucketIdx,
                "actValue": sensEyeDetection
            },
            learn=True,
            infer=True
        )

        oneStepConfidence, oneStep = sorted(
            zip(classifierResult[1], classifierResult["actualValues"]),
            reverse=True
        )[0]

        if (sensEyeDetection == oneStep):
            postvDetect += 1

        print("{:5}\t{:4}\t{:4}\t{:4.4}%".format(count, sensEyeDetection,
              oneStep, oneStepConfidence * 100))

print (float(postvDetect) / _NUM_RECORDS)
