# PyDatasets

Wrapper to download and work with datasets.

## Installation

  git clone git@github.com:perellonieto/PyDatasets.git
  cd PyDatasets
  python3.6 -m venv venv
  source venv/bin/activate
  pip install -r requirements

## How to load iris dataset

This is an example for loading the iris dataset (each feature is normalised
with mean 0 and standard deviation 1).

  from pydatasets.datasets import Data

  data = Data(dataset_names=['iris'])
  print(data.datasets['iris'])

Will output the following

  Name = iris
  Data shape = (150, 4)
  Target shape = (150,)
  Target classes = [0, 1, 2]
  Target labels = [1 2 3]
  Target counts = [50 50 50]

You can access then to the data and target

  data.datasets['iris'].data
  data.datasets['iris'].target

## How to load multiple non-binary datasets

  from pydatasets.datasets import Data
  from pydatasets.datasets import datasets_non_binary

  data = Data(dataset_names=datasets_non_binary)

  for name, dataset in data.datasets.items():
    print(name)
    print(dataset)

Will print the following

  optdigits
  Name = optdigits
  Data shape = (5620, 64)
  Target shape = (5620,)
  Target classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  Target labels = [0 1 2 3 4 5 6 7 8 9]
  Target counts = [554 571 557 572 568 558 558 566 554 562]
  libras-movement
  Name = libras-movement
  Data shape = (360, 90)
  Target shape = (360,)
  Target classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
  Target labels = [  1.   2.   3.   4.   5.   6.   7.   8.   9.  10.  11.  12.  13.  14.  15.]
  Target counts = [24 24 24 24 24 24 24 24 24 24 24 24 24 24 24]
  pendigits
  Name = pendigits
  Data shape = (10992, 16)
  Target shape = (10992,)
  Target classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  Target labels = [0 1 2 3 4 5 6 7 8 9]
  Target counts = [1143 1143 1144 1055 1144 1055 1056 1142 1055 1055]
  cleveland
  Name = cleveland
  Data shape = (297, 13)
  Target shape = (297,)
  Target classes = [0, 1, 2, 3, 4]
  Target labels = [0 1 2 3 4]
  Target counts = [160  54  35  35  13]
  dermatology
  Name = dermatology
  Data shape = (358, 34)
  Target shape = (358,)
  Target classes = [0, 1, 2, 3, 4, 5]
  Target labels = [1 2 3 4 5 6]
  Target counts = [111  60  71  48  48  20]
  landsat-satellite
  Name = landsat-satellite
  Data shape = (6435, 36)
  Target shape = (6435,)
  Target classes = [0, 1, 2, 3, 4, 5]
  Target labels = [ 1.  2.  3.  4.  5.  7.]
  Target counts = [1533  703 1358  626  707 1508]
  zoo
  Name = zoo
  Data shape = (101, 16)
  Target shape = (101,)
  Target classes = [0, 1, 2, 3, 4, 5, 6]
  Target labels = [array([u'amphibian'],
        dtype='<U9')
   array([u'bird'],
        dtype='<U4') array([u'fish'],
        dtype='<U4')
   array([u'insect'],
        dtype='<U6')
   array([u'invertebrate'],
        dtype='<U12')
   array([u'mammal'],
        dtype='<U6')
   array([u'reptile'],
        dtype='<U7')]
  Target counts = [ 4 20 13  8 10 41  5]
  vehicle
  Name = vehicle
  Data shape = (846, 18)
  Target shape = (846,)
  Target classes = [0, 1, 2, 3]
  Target labels = [1 2 3 4]
  Target counts = [212 217 218 199]
  shuttle
  Name = shuttle
  Data shape = (101500, 9)
  Target shape = (101500,)
  Target classes = [0, 1, 2, 3, 4, 5, 6]
  Target labels = [1 2 3 4 5 6 7]
  Target counts = [79694    87   303 15651  5725    16    24]
  letter
  Name = letter
  Data shape = (35000, 16)
  Target shape = (35000,)
  Target classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
  Target labels = [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
   26]
  Target counts = [1383 1333 1290 1403 1333 1340 1320 1272 1322 1322 1311 1321 1399 1388 1327
   1400 1377 1326 1311 1395 1419 1345 1328 1370 1388 1277]
  waveform-5000
  Name = waveform-5000
  Data shape = (5000, 40)
  Target shape = (5000,)
  Target classes = [0, 1, 2]
  Target labels = [0 1 2]
  Target counts = [1692 1653 1655]
  yeast
  Name = yeast
  Data shape = (1484, 8)
  Target shape = (1484,)
  Target classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  Target labels = ['CYT' 'ERL' 'EXC' 'ME1' 'ME2' 'ME3' 'MIT' 'NUC' 'POX' 'VAC']
  Target counts = [463   5  35  44  51 163 244 429  20  30]
  ecoli
  Name = ecoli
  Data shape = (336, 7)
  Target shape = (336,)
  Target classes = [0, 1, 2, 3, 4, 5, 6, 7]
  Target labels = [array([u'cp'],
        dtype='<U2') array([u'im'],
        dtype='<U2')
   array([u'imL'],
        dtype='<U3') array([u'imS'],
        dtype='<U3')
   array([u'imU'],
        dtype='<U3') array([u'om'],
        dtype='<U2')
   array([u'omL'],
        dtype='<U3') array([u'pp'],
        dtype='<U2')]
  Target counts = [143  77   2   2  35  20   5  52]
  page-blocks
  Name = page-blocks
  Data shape = (5473, 10)
  Target shape = (5473,)
  Target classes = [0, 1, 2, 3, 4]
  Target labels = [ 1.  2.  3.  4.  5.]
  Target counts = [4913  329   28   88  115]
  autos
  Name = autos
  Data shape = (159, 25)
  Target shape = (159,)
  Target classes = [0, 1, 2, 3, 4, 5]
  Target labels = [-2 -1  0  1  2  3]
  Target counts = [ 3 20 48 46 29 13]
  abalone
  Name = abalone
  Data shape = (4177, 8)
  Target shape = (4177,)
  Target classes = [0, 1, 2]
  Target labels = [array([u'F'],
        dtype='<U1') array([u'I'],
        dtype='<U1')
   array([u'M'],
        dtype='<U1')]
  Target counts = [1307 1342 1528]
  vowel
  Name = vowel
  Data shape = (990, 10)
  Target shape = (990,)
  Target classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  Target labels = [array([u'hAd'],
        dtype='<U3') array([u'hEd'],
        dtype='<U3')
   array([u'hId'],
        dtype='<U3') array([u'hOd'],
        dtype='<U3')
   array([u'hUd'],
        dtype='<U3') array([u'hYd'],
        dtype='<U3')
   array([u'had'],
        dtype='<U3') array([u'hed'],
        dtype='<U3')
   array([u'hid'],
        dtype='<U3') array([u'hod'],
        dtype='<U3')
   array([u'hud'],
        dtype='<U3')]
  Target counts = [90 90 90 90 90 90 90 90 90 90 90]
  segment
  Name = segment
  Data shape = (2310, 19)
  Target shape = (2310,)
  Target classes = [0, 1, 2, 3, 4, 5, 6]
  Target labels = [array([u'brickface'],
        dtype='<U9')
   array([u'cement'],
        dtype='<U6')
   array([u'foliage'],
        dtype='<U7')
   array([u'grass'],
        dtype='<U5') array([u'path'],
        dtype='<U4')
   array([u'sky'],
        dtype='<U3') array([u'window'],
        dtype='<U6')]
  Target counts = [330 330 330 330 330 330 330]
  mfeat-morphological
  Name = mfeat-morphological
  Data shape = (2000, 6)
  Target shape = (2000,)
  Target classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  Target labels = [ 1  2  3  4  5  6  7  8  9 10]
  Target counts = [200 200 200 200 200 200 200 200 200 200]
  iris
  Name = iris
  Data shape = (150, 4)
  Target shape = (150,)
  Target classes = [0, 1, 2]
  Target labels = [1 2 3]
  Target counts = [50 50 50]
  glass
  Name = glass
  Data shape = (214, 9)
  Target shape = (214,)
  Target classes = [0, 1, 2, 3, 4, 5]
  Target labels = [1 2 3 5 6 7]
  Target counts = [70 76 17 13  9 29]
  car
  Name = car
  Data shape = (1728, 6)
  Target shape = (1728,)
  Target classes = [0, 1, 2, 3]
  Target labels = [array([u'acc'],
        dtype='<U3') array([u'good'],
        dtype='<U4')
   array([u'unacc'],
        dtype='<U5') array([u'vgood'],
        dtype='<U5')]
  Target counts = [ 384   69 1210   65]
  balance-scale
  Name = balance-scale
  Data shape = (625, 4)
  Target shape = (625,)
  Target classes = [0, 1, 2]
  Target labels = [array([u'B'],
        dtype='<U1') array([u'L'],
        dtype='<U1')
   array([u'R'],
        dtype='<U1')]
  Target counts = [ 49 288 288]
  mfeat-karhunen
  Name = mfeat-karhunen
  Data shape = (2000, 64)
  Target shape = (2000,)
  Target classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  Target labels = [ 1  2  3  4  5  6  7  8  9 10]
  Target counts = [200 200 200 200 200 200 200 200 200 200]
  mfeat-zernike
  Name = mfeat-zernike
  Data shape = (2000, 47)
  Target shape = (2000,)
  Target classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  Target labels = [ 1  2  3  4  5  6  7  8  9 10]
  Target counts = [200 200 200 200 200 200 200 200 200 200]
  flare
  Name = flare
  Data shape = (1389, 10)
  Target shape = (1389,)
  Target classes = [0, 1, 2, 3, 4, 5]
  Target labels = [array([u'B'],
        dtype='<U1') array([u'C'],
        dtype='<U1')
   array([u'D'],
        dtype='<U1') array([u'E'],
        dtype='<U1')
   array([u'F'],
        dtype='<U1') array([u'H'],
        dtype='<U1')]
  Target counts = [212 287 327 116  51 396]

# Saving datasets

By default all the datasets are downloaded once, and then stored in a folder
called datasets in your current folder. To change the folder just call the Data
function specifying the path. Eg.

  from pydatasets.datasets import Data

  data = Data(data_home='your/path/', dataset_names=['iris'])
