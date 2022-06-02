# ts-randomforest
![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg) 

TypeScript random-forest library created in rust.

# Usage
```jsx
  import {regressorTrain, regressorTest} from 'ts-random-forest/pkg'
  const trainFeatures = [
    [0.0, 2.0, 1.0, 0.0],
    [0.0, 2.0, 1.0, 1.0],
    [1.0, 2.0, 1.0, 0.0],
    [2.0, 1.0, 1.0, 0.0],
    [2.0, 0.0, 0.0, 0.0],
    [2.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [2.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
  ]

  const target = new Float64Array([25.0, 30.0, 46.0, 45.0, 52.0, 23.0, 43.0, 35.0, 38.0, 46.0, 48.0, 52.0])
  const flattenTrainFeatureLength = trainFeatures.length
  const flattenTrainFeatures = new Float64Array(trainFeatures.flat())
  const fileName = 'aaa.model'

  regressorTrain(flattenTrainFeatures, flattenTrainFeatureLength, target, fileName)
  
  const testFeatures = [1.0, 2.0, 0.0, 0.0]
  const result = regressorTest(flattenTrainFeatures, fileName)
  console.log('result', result)
```