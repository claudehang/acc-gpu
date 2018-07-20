#ifndef __PLUGIN_FACTORY_H__  
#define __PLUGIN_FACTORY_H__  
  
#include <algorithm>  
#include <cassert>  
#include <iostream>  
#include <cstring>  
#include <sys/stat.h>  
#include <map>  
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "upsample_layer.h"
#include <unordered_map>

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

//SSD Reshape layer : shape{0,-1,21}
template<int OutC>
class Reshape : public IPlugin
{
public:
    Reshape() {}
    Reshape(const void* buffer, size_t size)
    {
        assert(size == sizeof(mCopySize));
        mCopySize = *reinterpret_cast<const size_t*>(buffer);
    }

    int getNbOutputs() const override
    {
        return 1;
    }
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(nbInputDims == 1);
        assert(index == 0);
        assert(inputs[index].nbDims == 3);
        assert((inputs[0].d[0])*(inputs[0].d[1]) % OutC == 0);
        return DimsCHW(OutC, inputs[0].d[0] * inputs[0].d[1] / OutC, inputs[0].d[2]);
    }

    int initialize() override
    {
        return 0;
    }

    void terminate() override
    {
    }

    size_t getWorkspaceSize(int) const override
    {
        return 0;
    }

    // currently it is not possible for a plugin to execute "in place". Therefore we memcpy the data from the input to the output buffer
    int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override
    {
        CHECK(cudaMemcpyAsync(outputs[0], inputs[0], mCopySize * batchSize, cudaMemcpyDeviceToDevice, stream));
        return 0;
    }


    size_t getSerializationSize() override
    {
        return sizeof(mCopySize);
    }

    void serialize(void* buffer) override
    {
        *reinterpret_cast<size_t*>(buffer) = mCopySize;
    }

    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int)    override
    {
        mCopySize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2] * sizeof(float);
    }

protected:
    size_t mCopySize;
};



class SofaMaxChannelLayer: public IPlugin
{
public:
    SofaMaxChannelLayer(int axis): _axis(axis),inner_num_(1),outer_num_(3462){}
    
    SofaMaxChannelLayer(int axis,const void* buffer,size_t size)
    {
      _axis = axis;
    }

    inline int getNbOutputs() const override { return 1; };
    
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
   
        return DimsCHW(1,3462, 2);
    }

    int initialize() override
    {
        return 0;
    }
    inline void terminate() override
    {
    }

    inline size_t getWorkspaceSize(int) const override { return 0; }
    int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override
    {
 
        return 0;
    }


    size_t getSerializationSize() override
    {
        return 0;
    }

    void serialize(void* buffer) override
    {
     
    }

    void configure(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override
    {
         
    }

protected:
    int _axis;
    int _size;
    float* scale = new float [3462]; //scale
    int inner_num_;
    int outer_num_;
    
    
  
};


//SSD Flatten layer
class LReluLayer : public IPlugin
{
public:
    LReluLayer(){}
    LReluLayer(float para):para_(para)
    {
        std::cout<<"LReluLayer0"<<std::endl;
    }

    LReluLayer(const void* buffer,size_t sz, float para):para_(para)
    {
        assert(sz == 4 * sizeof(int));
        const int* d = reinterpret_cast<const int*>(buffer);
        float* p=(float*)(d+3);
        para_=p[0];
        channel_=d[0];
        w_=d[1];
        h_=d[2];
        //std::cout<<"LReluLayer1"<<para_ <<" " <<channel_<<" "<<w_ <<" "<<h_ <<std::endl;
        para_=0.1;
    }

    inline int getNbOutputs() const override { return 1; };
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
         std::cout << "index: " << index << ", Leaky ReLU ==> getOutputDimensions  channel"<<inputs[0].d[0]<<"h:"<<inputs[0].d[1]<<"w:"<<inputs[0].d[2]<<std::endl;

         channel_=inputs[0].d[0];
         w_=inputs[0].d[1];
         h_=inputs[0].d[2];
         
        return DimsCHW(inputs[0].d[0], inputs[0].d[1] , inputs[0].d[2] );
    }

    int initialize() override
    {
        return 0;
    }
    inline void terminate() override
    {
    }

 

    inline size_t getWorkspaceSize(int) const override { return 0; }
    int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override
    {
    

     //   std::cout<<"LReluLayer1 enqueue : "<<batchSize<<"c:"<<channel_<<"w:"<<w_<<"h:"<<h_<<"para_"<<para_<<std::endl;
    ReluForward_gpu((const float*)inputs[0],(float*)outputs[0],batchSize,channel_,w_,h_,para_);

     //ReLUForward1<<<CAFFE_GET_BLOCKS(batchSize*channel_*w_*h_), CAFFE_CUDA_NUM_THREADS>>>(batchSize*channel_*w_*h_,inputs[0],outputs[0],para_);

    //std::cout<<"flatten enqueue:"<<batchSize<<";"<<_size<<std::endl;
    //CHECK(cudaMemcpyAsync(outputs[0],inputs[0],batchSize*_size*sizeof(float),cudaMemcpyDeviceToDevice,stream));
    //Forward_gpu (
    //  (float*)inputs[0],1,channel_, w_, h_, stride_, (float*)outputs[0] );


        //CHECK(cudaMemcpyAsync(outputs[0], inputs[0], mCopySize * batchSize, cudaMemcpyDeviceToDevice, stream));

        return 0;
    }


    size_t getSerializationSize() override
    {
        return sizeof(int)*3+sizeof(float);
    }

    void serialize(void* buffer) override
    {
        
     
    //
    //write(q+3, (float)para_);

        float* q = reinterpret_cast<float*>(buffer);
        int* d = reinterpret_cast<int*>(buffer);
        d[0] = channel_; d[1] = w_; d[2] = h_;
        q[4]=para_;

    //serializeFromDevice(d, mKernelWeights);
    //serializeFromDevice(d, mBiasWeights);
     
 

    }

    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override
    {
      //  dimBottom = DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
         channel_=inputs[0].d[0];
         w_=inputs[0].d[1];
         h_=inputs[0].d[2];

    }

protected:
    float para_;
    int channel_;
    int w_;
    int h_;
};





//SSD Flatten layer
class UpsampleLayer : public IPlugin
{
public:
    UpsampleLayer(){}
    UpsampleLayer(size_t stride):stride_(stride)
    {
      std::cout<<"UpsampleLayer0"<<std::endl;


    }

    UpsampleLayer(const void* buffer,size_t sz, size_t stride):stride_(stride)
    {

        const int* d = reinterpret_cast<const int*>(buffer);
 
    channel_=d[0];
    w_=d[1];
    h_=d[2];


        std::cout<<"UpsampleLayer1"<<std::endl;

    }

    inline int getNbOutputs() const override { return 1; };
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
         std::cout<<"channel"<<inputs[0].d[0]<<"h:"<<inputs[0].d[1]<<"w:"<<inputs[0].d[2]<<std::endl;

         channel_=inputs[0].d[0];
         w_=inputs[0].d[1];
         h_=inputs[0].d[2];

        return DimsCHW(inputs[0].d[0], inputs[0].d[1]*stride_, inputs[0].d[2]*stride_);
    }

    int initialize() override
    {
        return 0;
    }
    inline void terminate() override
    {
    }



    inline size_t getWorkspaceSize(int) const override { return 0; }
    int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override
    {


    // std::cout<<"UpsampleLayer1 enqueue"<<std::endl;
        //std::cout<<"flatten enqueue:"<<batchSize<<";"<<_size<<std::endl;
        //CHECK(cudaMemcpyAsync(outputs[0],inputs[0],batchSize*_size*sizeof(float),cudaMemcpyDeviceToDevice,stream));
     Forward_gpu((float*)inputs[0],1,channel_, w_, h_, stride_, (float*)outputs[0] );




        return 0;
    }


    size_t getSerializationSize() override
    {
        return 4*sizeof(int);
    }

    void serialize(void* buffer) override
    {
   
        int* d = reinterpret_cast<int*>(buffer);
        d[0] = channel_; d[1] = w_; d[2] = h_;
        d[3]=stride_;
   
    }

    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override
    {
      //  dimBottom = DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
         channel_=inputs[0].d[0];
         w_=inputs[0].d[1];
         h_=inputs[0].d[2];

    }

protected:
    int stride_;
    int channel_;
    int w_;
    int h_;
};





//SSD Flatten layer
class FlattenLayer : public IPlugin
{
public:
    FlattenLayer(){}
    FlattenLayer(const void* buffer,size_t size)
    {
        assert(size == 3 * sizeof(int));
        const int* d = reinterpret_cast<const int*>(buffer);
        _size = d[0] * d[1] * d[2];
        dimBottom = DimsCHW{d[0], d[1], d[2]};
    }

    inline int getNbOutputs() const override { return 1; };
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(1 == nbInputDims);
        assert(0 == index);
        assert(3 == inputs[index].nbDims);
        _size = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
        return DimsCHW(_size, 1, 1);
    }

    int initialize() override
    {
        return 0;
    }
    inline void terminate() override
    {
    }

    inline size_t getWorkspaceSize(int) const override { return 0; }
    int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override
    {
        std::cout<<"flatten enqueue:"<<batchSize<<";"<<_size<<std::endl;
        CHECK(cudaMemcpyAsync(outputs[0],inputs[0],batchSize*_size*sizeof(float),cudaMemcpyDeviceToDevice,stream));
        return 0;
    }


    size_t getSerializationSize() override
    {
        return 3 * sizeof(int);
    }

    void serialize(void* buffer) override
    {
        int* d = reinterpret_cast<int*>(buffer);
        d[0] = dimBottom.c(); d[1] = dimBottom.h(); d[2] = dimBottom.w();
    }

    void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override
    {
        dimBottom = DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
    }

protected:
    DimsCHW dimBottom;
    int _size;
};



  
  
class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory {  
public:
    std::unordered_map<std::string, int> UpsampleIDs = {
        std::make_pair("Interp202", 0),
        std::make_pair("Interp227", 1)};

    std::unordered_map<std::string, int> LReLUIDs = {
        std::make_pair("ReLU2",0),
        std::make_pair("ReLU5",1),
        std::make_pair("ReLU8",2),  
        std::make_pair("ReLU11",3),  
        std::make_pair("ReLU15",4),  
        std::make_pair("ReLU18",5),
        std::make_pair("ReLU21",6),
        std::make_pair("ReLU25",7),  
        std::make_pair("ReLU28",8),  
        std::make_pair("ReLU32",9),  
        std::make_pair("ReLU35",10),  
        std::make_pair("ReLU38",11),  
        std::make_pair("ReLU42",12),  
        std::make_pair("ReLU45",13),  
        std::make_pair("ReLU49",14),  
        std::make_pair("ReLU52",15),  
        std::make_pair("ReLU56",16),  
        std::make_pair("ReLU59",17),  
        std::make_pair("ReLU63",18),  
        std::make_pair("ReLU66",19),  
        std::make_pair("ReLU70",20),


        std::make_pair("ReLU73",21),  
        std::make_pair("ReLU77",22),  
        std::make_pair("ReLU80",23),  
        std::make_pair("ReLU84",24),  
        std::make_pair("ReLU87",25),  
        std::make_pair("ReLU91",26),  
        std::make_pair("ReLU94",27),  
        std::make_pair("ReLU97",28),  
        std::make_pair("ReLU101",29),  
        std::make_pair("ReLU104",30),  
        std::make_pair("ReLU108",31),  
        std::make_pair("ReLU111",32),  
        std::make_pair("ReLU115",33),  
        std::make_pair("ReLU118",34),  
        std::make_pair("ReLU122",35),  
        std::make_pair("ReLU125",36),  
        std::make_pair("ReLU129",37),  
        std::make_pair("ReLU132",38),  
        std::make_pair("ReLU136",39),
        std::make_pair("ReLU139",40),


        std::make_pair("ReLU143",41),  
        std::make_pair("ReLU146",42),  
        std::make_pair("ReLU150",43),  
        std::make_pair("ReLU153",44),  
        std::make_pair("ReLU156",45),  
        std::make_pair("ReLU160",46),  
        std::make_pair("ReLU163",47),  
        std::make_pair("ReLU167",48),  
        std::make_pair("ReLU170",49),  
        std::make_pair("ReLU174",50),  
        std::make_pair("ReLU177",51),  
        std::make_pair("ReLU181",52),  
        std::make_pair("ReLU184",53),  
        std::make_pair("ReLU187",54),  
        std::make_pair("ReLU190",55),  
        std::make_pair("ReLU193",56),
        std::make_pair("ReLU201",57),
        std::make_pair("ReLU196",58), 
        std::make_pair("ReLU206",59),
        std::make_pair("ReLU209",60), 



        std::make_pair("ReLU212",61),  
        std::make_pair("ReLU215",62),  
        std::make_pair("ReLU218",63),  
        std::make_pair("ReLU226",64),  
        std::make_pair("ReLU221",65),  
        std::make_pair("ReLU231",66),
        std::make_pair("ReLU234",67),
        std::make_pair("ReLU237",68),
        std::make_pair("ReLU240",69),
        std::make_pair("ReLU243",70),
        std::make_pair("ReLU246",71)};
  
        // caffe parser plugin implementation  
        bool isPlugin(const char* layerName) override  
        {  
         printf("isPlugin %s\n",layerName);
        //  for (int layerNum = 1; layerNum < 150; layerNum++) {
        //     char matchLayer[40];
        //     if (strcmp(name, sprintf(matchLayer, "layer%d-act"))
        //         || strcmp(name, "Interp202")
        //         || strcmp(name, "Interp227")) { return true }
        //  }
        // return false;
        return ( !strcmp(layerName, "Interp202")
            || !strcmp(layerName, "Interp227")
            || !strcmp(layerName, "ReLU2")  
            || !strcmp(layerName, "ReLU5")  
            || !strcmp(layerName, "ReLU8")  
            || !strcmp(layerName, "ReLU11")  
            || !strcmp(layerName, "ReLU15")  
            || !strcmp(layerName, "ReLU18")
            || !strcmp(layerName, "ReLU21")
            || !strcmp(layerName, "ReLU25")  
            || !strcmp(layerName, "ReLU28")  
            || !strcmp(layerName, "ReLU32")  
            || !strcmp(layerName, "ReLU35")  
            || !strcmp(layerName, "ReLU38")  
            || !strcmp(layerName, "ReLU42")  
            || !strcmp(layerName, "ReLU45")  
            || !strcmp(layerName, "ReLU49")  
            || !strcmp(layerName, "ReLU52")  
            || !strcmp(layerName, "ReLU56")  
            || !strcmp(layerName, "ReLU59")  
            || !strcmp(layerName, "ReLU63")  
            || !strcmp(layerName, "ReLU66")  
            || !strcmp(layerName, "ReLU70")


            || !strcmp(layerName, "ReLU73")  
            || !strcmp(layerName, "ReLU77")  
            || !strcmp(layerName, "ReLU80")  
            || !strcmp(layerName, "ReLU84")  
            || !strcmp(layerName, "ReLU87")  
            || !strcmp(layerName, "ReLU91")  
            || !strcmp(layerName, "ReLU94")  
            || !strcmp(layerName, "ReLU97")  
            || !strcmp(layerName, "ReLU101")  
            || !strcmp(layerName, "ReLU104")  
            || !strcmp(layerName, "ReLU108")  
            || !strcmp(layerName, "ReLU111")  
            || !strcmp(layerName, "ReLU115")  
            || !strcmp(layerName, "ReLU118")  
            || !strcmp(layerName, "ReLU122")  
            || !strcmp(layerName, "ReLU125")  
            || !strcmp(layerName, "ReLU129")  
            || !strcmp(layerName, "ReLU132")  
            || !strcmp(layerName, "ReLU136")
            || !strcmp(layerName, "ReLU139")


            || !strcmp(layerName, "ReLU143")  
            || !strcmp(layerName, "ReLU146")  
            || !strcmp(layerName, "ReLU150")  
            || !strcmp(layerName, "ReLU153")  
            || !strcmp(layerName, "ReLU156")  
            || !strcmp(layerName, "ReLU160")  
            || !strcmp(layerName, "ReLU163")  
            || !strcmp(layerName, "ReLU167")  
            || !strcmp(layerName, "ReLU170")  
            || !strcmp(layerName, "ReLU174")  
            || !strcmp(layerName, "ReLU177")  
            || !strcmp(layerName, "ReLU181")  
            || !strcmp(layerName, "ReLU184")  
            || !strcmp(layerName, "ReLU187")  
            || !strcmp(layerName, "ReLU190")  
            || !strcmp(layerName, "ReLU193")
            || !strcmp(layerName, "ReLU201")
            || !strcmp(layerName, "ReLU196")  
            || !strcmp(layerName, "ReLU206")
            || !strcmp(layerName, "ReLU209") 



            || !strcmp(layerName, "ReLU212")  
            || !strcmp(layerName, "ReLU215")  
            || !strcmp(layerName, "ReLU218")  
            || !strcmp(layerName, "ReLU226")  
            || !strcmp(layerName, "ReLU221")  
            || !strcmp(layerName, "ReLU231")
            || !strcmp(layerName, "ReLU234")
            || !strcmp(layerName, "ReLU237")
            || !strcmp(layerName, "ReLU240")
            || !strcmp(layerName, "ReLU243")
            || !strcmp(layerName, "ReLU246")
        );  
  
        }  
  
        virtual IPlugin* createPlugin(const char* layerName, const Weights* weights, int nbWeights) override  
        {
            if(!strcmp(layerName, "Interp202") || !strcmp(layerName, "Interp227"))
            {
                const int i = UpsampleIDs[layerName];
                assert(mPluginUpsample[i] == nullptr);
                mPluginUpsample[i] = std::unique_ptr<UpsampleLayer>(new UpsampleLayer(2));
                return mPluginUpsample[i].get();
            }
            else if (  !strcmp(layerName, "ReLU2")  
            || !strcmp(layerName, "ReLU5")  
            || !strcmp(layerName, "ReLU8")  
            || !strcmp(layerName, "ReLU11")  
            || !strcmp(layerName, "ReLU15")  
            || !strcmp(layerName, "ReLU18")
            || !strcmp(layerName, "ReLU21")
            || !strcmp(layerName, "ReLU25")  
            || !strcmp(layerName, "ReLU28")  
            || !strcmp(layerName, "ReLU32")  
            || !strcmp(layerName, "ReLU35")  
            || !strcmp(layerName, "ReLU38")  
            || !strcmp(layerName, "ReLU42")  
            || !strcmp(layerName, "ReLU45")  
            || !strcmp(layerName, "ReLU49")  
            || !strcmp(layerName, "ReLU52")  
            || !strcmp(layerName, "ReLU56")  
            || !strcmp(layerName, "ReLU59")  
            || !strcmp(layerName, "ReLU63")  
            || !strcmp(layerName, "ReLU66")  
            || !strcmp(layerName, "ReLU70")


            || !strcmp(layerName, "ReLU73")  
            || !strcmp(layerName, "ReLU77")  
            || !strcmp(layerName, "ReLU80")  
            || !strcmp(layerName, "ReLU84")  
            || !strcmp(layerName, "ReLU87")  
            || !strcmp(layerName, "ReLU91")  
            || !strcmp(layerName, "ReLU94")  
            || !strcmp(layerName, "ReLU97")  
            || !strcmp(layerName, "ReLU101")  
            || !strcmp(layerName, "ReLU104")  
            || !strcmp(layerName, "ReLU108")  
            || !strcmp(layerName, "ReLU111")  
            || !strcmp(layerName, "ReLU115")  
            || !strcmp(layerName, "ReLU118")  
            || !strcmp(layerName, "ReLU122")  
            || !strcmp(layerName, "ReLU125")  
            || !strcmp(layerName, "ReLU129")  
            || !strcmp(layerName, "ReLU132")  
            || !strcmp(layerName, "ReLU136")
            || !strcmp(layerName, "ReLU139")


            || !strcmp(layerName, "ReLU143")  
            || !strcmp(layerName, "ReLU146")  
            || !strcmp(layerName, "ReLU150")  
            || !strcmp(layerName, "ReLU153")  
            || !strcmp(layerName, "ReLU156")  
            || !strcmp(layerName, "ReLU160")  
            || !strcmp(layerName, "ReLU163")  
            || !strcmp(layerName, "ReLU167")  
            || !strcmp(layerName, "ReLU170")  
            || !strcmp(layerName, "ReLU174")  
            || !strcmp(layerName, "ReLU177")  
            || !strcmp(layerName, "ReLU181")  
            || !strcmp(layerName, "ReLU184")  
            || !strcmp(layerName, "ReLU187")  
            || !strcmp(layerName, "ReLU190")  
            || !strcmp(layerName, "ReLU193")
            || !strcmp(layerName, "ReLU201")
            || !strcmp(layerName, "ReLU196")  
            || !strcmp(layerName, "ReLU206")
            || !strcmp(layerName, "ReLU209") 



            || !strcmp(layerName, "ReLU212")  
            || !strcmp(layerName, "ReLU215")  
            || !strcmp(layerName, "ReLU218")  
            || !strcmp(layerName, "ReLU226")  
            || !strcmp(layerName, "ReLU221")  
            || !strcmp(layerName, "ReLU231")
            || !strcmp(layerName, "ReLU234")
            || !strcmp(layerName, "ReLU237")
            || !strcmp(layerName, "ReLU240")
            || !strcmp(layerName, "ReLU243")
            || !strcmp(layerName, "ReLU246")
       ){
            const int i = LReLUIDs[layerName];
            assert(mPluginLReLU[i] == nullptr);
            mPluginLReLU[i] = std::unique_ptr<LReluLayer>(new LReluLayer(0.1));
            return mPluginLReLU[i].get();
        }else{  
             assert(0);  
             return nullptr;  
        }  
    }  
  
    // deserialization plugin implementation  
    IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override {                
        
        if(!strcmp(layerName, "Interp202") || !strcmp(layerName, "Interp227"))
        {
            const int i = UpsampleIDs[layerName];
            assert(mPluginUpsample[i] == nullptr);
            mPluginUpsample[i] = std::unique_ptr<UpsampleLayer>(new UpsampleLayer(serialData, serialLength,2));
            return mPluginUpsample[i].get();
        }
        else if ( !strcmp(layerName, "ReLU2")  
            || !strcmp(layerName, "ReLU5")  
            || !strcmp(layerName, "ReLU8")  
            || !strcmp(layerName, "ReLU11")  
            || !strcmp(layerName, "ReLU15")  
            || !strcmp(layerName, "ReLU18")
            || !strcmp(layerName, "ReLU21")
            || !strcmp(layerName, "ReLU25")  
            || !strcmp(layerName, "ReLU28")  
            || !strcmp(layerName, "ReLU32")  
            || !strcmp(layerName, "ReLU35")  
            || !strcmp(layerName, "ReLU38")  
            || !strcmp(layerName, "ReLU42")  
            || !strcmp(layerName, "ReLU45")  
            || !strcmp(layerName, "ReLU49")  
            || !strcmp(layerName, "ReLU52")  
            || !strcmp(layerName, "ReLU56")  
            || !strcmp(layerName, "ReLU59")  
            || !strcmp(layerName, "ReLU63")  
            || !strcmp(layerName, "ReLU66")  
            || !strcmp(layerName, "ReLU70")


            || !strcmp(layerName, "ReLU73")  
            || !strcmp(layerName, "ReLU77")  
            || !strcmp(layerName, "ReLU80")  
            || !strcmp(layerName, "ReLU84")  
            || !strcmp(layerName, "ReLU87")  
            || !strcmp(layerName, "ReLU91")  
            || !strcmp(layerName, "ReLU94")  
            || !strcmp(layerName, "ReLU97")  
            || !strcmp(layerName, "ReLU101")  
            || !strcmp(layerName, "ReLU104")  
            || !strcmp(layerName, "ReLU108")  
            || !strcmp(layerName, "ReLU111")  
            || !strcmp(layerName, "ReLU115")  
            || !strcmp(layerName, "ReLU118")  
            || !strcmp(layerName, "ReLU122")  
            || !strcmp(layerName, "ReLU125")  
            || !strcmp(layerName, "ReLU129")  
            || !strcmp(layerName, "ReLU132")  
            || !strcmp(layerName, "ReLU136")
            || !strcmp(layerName, "ReLU139")


            || !strcmp(layerName, "ReLU143")  
            || !strcmp(layerName, "ReLU146")  
            || !strcmp(layerName, "ReLU150")  
            || !strcmp(layerName, "ReLU153")  
            || !strcmp(layerName, "ReLU156")  
            || !strcmp(layerName, "ReLU160")  
            || !strcmp(layerName, "ReLU163")  
            || !strcmp(layerName, "ReLU167")  
            || !strcmp(layerName, "ReLU170")  
            || !strcmp(layerName, "ReLU174")  
            || !strcmp(layerName, "ReLU177")  
            || !strcmp(layerName, "ReLU181")  
            || !strcmp(layerName, "ReLU184")  
            || !strcmp(layerName, "ReLU187")  
            || !strcmp(layerName, "ReLU190")  
            || !strcmp(layerName, "ReLU193")
            || !strcmp(layerName, "ReLU201")
            || !strcmp(layerName, "ReLU196")  
            || !strcmp(layerName, "ReLU206")
            || !strcmp(layerName, "ReLU209") 



            || !strcmp(layerName, "ReLU212")  
            || !strcmp(layerName, "ReLU215")  
            || !strcmp(layerName, "ReLU218")  
            || !strcmp(layerName, "ReLU226")  
            || !strcmp(layerName, "ReLU221")  
            || !strcmp(layerName, "ReLU231")
            || !strcmp(layerName, "ReLU234")
            || !strcmp(layerName, "ReLU237")
            || !strcmp(layerName, "ReLU240")
            || !strcmp(layerName, "ReLU243")
            || !strcmp(layerName, "ReLU246")
       ){
            const int i = LReLUIDs[layerName];
            assert(mPluginLReLU[i] == nullptr);
            mPluginLReLU[i] = std::unique_ptr<LReluLayer>(new LReluLayer(serialData, serialLength,0.1));
            return mPluginLReLU[i].get();
        }else{  
            assert(0);  
            return nullptr;  
        }  
    }  
  
    void destroyPlugin()  
    {
        for (unsigned i = 0; i < LReLUIDs.size(); ++i) { mPluginLReLU[i].reset(); }
        for (unsigned j = 0; j < UpsampleIDs.size(); ++j) { mPluginUpsample[j].reset(); }
    }  
  
  
private:
    void (*nvPluginDeleter)(INvPlugin*){[](INvPlugin* ptr) { ptr->destroy(); }};
    std::unique_ptr<LReluLayer> mPluginLReLU[72];
    std::unique_ptr<UpsampleLayer> mPluginUpsample[2]{nullptr, nullptr};
};
  
#endif  

