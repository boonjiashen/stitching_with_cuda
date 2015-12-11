Run
---
```
>> cmake -DCMAKE_CXX_FLAGS=-std=c++11 .. && make && ./feature_matching ../testingImages/08.jpg ../testingImages/09.jpg ../testingImages/10.jpg
```


Output images
---
```
>> cd samples
>> find ../testingImages/* | xargs ./stitching_detailed
>> open result.jpg
```


Author
---
Jia-Shen Boon
