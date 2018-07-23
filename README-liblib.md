# 人脸质量接口 (Face Quality Interface)

## 文件目录
|-- facequality.h
|-- facequality.cpp
|-- libfacequality.so
|-- test
|-- test.cpp
|-- test.txt
|-- 0.jpg
|-- 1.jpg
|-- README.md
|-- bsmodel
|   |-- deploy.prototxt
|   |-- mean.binaryproto
|   `-- network.caffemodel
`-- tnmodel
    |-- deploy.prototxt
    |-- mean.binaryproto
    `-- network.caffemodel

## 文件说明
**facequality.h** , **facequality.cpp** 是提供接口的源码文件。**facequality.h**中给出了接口的定义，接口定义如下：
```C
    extern "C" 
    {
        int txavatar_face_quality_init(const char *bs_model_dir, const char *in_model_dir, int gpu_id, int batch_size);
        int txavatar_face_quality(const void *p_cvmax, int num, void *p_blur_score, void *p_integrity);
        int txavatar_blurscore_init(const char *model_dir, int gpu_id, int batch_size);
        int txavatar_blurscore(const void *p_cvmax, int num, void *p_blur_score);
        int txavatar_integrity_init(const char *model_dir, int gpu_id, int batch_size);
        int txavatar_integrity(const void *p_cvmax, int num, void *p_integrity);
    }
```
调用 `***_init` 函数相当于初始化API, 没有`_init` 会返回对应的结果. <br>
**quality** 表示同时 **blurscore** 和 **integrity**.<br>

**libfacequality.so** 是接口生成的.so文件。生成该文件命令为：
```shell
gcc -fPIC -shared facequality.cpp -I/data/home/tyhyewang/caffe/include -I/data/cuda-8.0/include -L/usr/local/lib -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -lcudart -lcublas -lcurand -lglog -lgflags -lprotobuf -lleveldb -lsnappy -llmdb -lboost_system -lhdf5_hl -lhdf5 -lm -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_thread -lstdc++ -lcudnn -L/data/home/tyhyewang/caffe/build/lib/ -lcaffe -o libfacequality.so
```

`model_dir` 指的是模型存放的路径，该路径下需包含 `deploy.prototxt, network.caffemodel, mean.binaryproto` 三个文件，现在文件名需要确定。

**test.cpp** 是测试接口用的代码。**test**是该代码生成的可执行文件。生成 **test** 用的命令为：
```shell
gcc -rdynamic test.cpp -I/data/home/tyhyewang/caffe/include -I/data/cuda-8.0/include -L/usr/local/lib -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -lcudart -lcublas -lcurand -lglog -lgflags -lprotobuf -lleveldb -lsnappy -llmdb -lboost_system -lhdf5_hl -lhdf5 -lm -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_thread -lstdc++ -lcudnn -L/data/home/tyhyewang/caffe/build/lib/ -lcaffe -o test -ldl
```
调用 **test** 的命令为：
```shell
./test test.txt ./
```

## 功能说明
现在 **blurscore** 会正常返回结果， **integrity** 只会返回 0.0.
输入cv::Mat的序列，会返回的得分到 对应的vector中。


