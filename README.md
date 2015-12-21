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

Slides [[link](https://drive.google.com/open?id=1amCMWRQyxwnu8AbeN6myjMdT64BYCb_nHBh_VRAnuIA)]
Author
---
Jia-Shen Boon
