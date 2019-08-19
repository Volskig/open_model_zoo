#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <ie_extension.h>
#include <ie_plugin_dispatcher.hpp>
#include <string>
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace InferenceEngine;

float confThreshold = 0.9;//*100%
int NUM_THREAD = -1; //
bool READ = true;// READ - true, WA - false
bool WA = !READ;

enum {
    REQ_READY_TO_START = 0,
    REQ_WORK_FIN = 1,
    REQ_WORK = 2
};

static char* get_cameras_list()
{
    return getenv("OPENCV_TEST_CAMERA_LIST");
}

void postprocess(Mat& frame, const Mat& outs)
{
    // Network produces output blob with a shape 1x1xNx7 where N is a number of
    // detections and an every detection is a vector of values
    // [batchId, classId, confidence, left, top, right, bottom]
    float* data = (float*)outs.data;
    for (size_t i = 0; i < outs.total(); i += 7)
    {
        float confidence = data[i + 2];
        if (confidence > confThreshold)
        {
            int left   = (int)(data[i + 3] * frame.cols);
            int top    = (int)(data[i + 4] * frame.rows);
            int right  = (int)(data[i + 5] * frame.cols);
            int bottom = (int)(data[i + 6] * frame.rows);
            rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));
        }
    }
}

int main(int argc, char* argv[])
{    
    bool ADAS = true;
    NUM_THREAD = atoi(argv[1]);
    READ = atoi(argv[2]);
    ADAS = atoi(argv[3]);
    std::cout <<"========================================================================"<< std::endl;
    if(ADAS)
        std::cout << "ADAS    ";
    else
        std::cout << "RETAIL    ";

    WA = !READ;
    if(WA)
        std::cout << "WA mode   ";
    if(READ)
        std::cout << "READ mode   ";
    std::cout << "NUM_THREAD: " << NUM_THREAD << std::endl;
    //char* datapath_dir = get_cameras_list(); //export OPENCV_TEST_CAMERA_LIST=...
    //std::vector<VideoCapture> cameras;
    //int step = 0; std::string path;
    //while(true)
    //{
    //    if(datapath_dir[step] == ':' || datapath_dir[step] == '\0')
    //    {
    //        cameras.emplace_back(VideoCapture(path, CAP_V4L));
    //
    //        std::cout << cameras.back().set(CAP_PROP_FRAME_WIDTH, 1280)<< std::endl;
    //        std::cout << cameras.back().set(CAP_PROP_FRAME_HEIGHT, 720)<< std::endl;
    //        std::cout << cameras.back().set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'))<< std::endl;
    //        std::cout << cameras.back().set(CAP_PROP_FPS, 60) << std::endl;
    //        path.clear();
    //        if(datapath_dir[step] != '\0')
    //            ++step;
    //    }
    //    if(datapath_dir[step] == '\0')
    //        break;
    //    path += datapath_dir[step];
    //    ++step;
    //}

    std::vector<VideoCapture> cameras;
    VideoCapture cap1(0);
    VideoCapture cap2(2);
    VideoCapture cap3(4);
    cameras.push_back(cap1);
    cameras.push_back(cap2);
    cameras.push_back(cap3);

    for(int i = 0; i < cameras.size(); ++i)
    {
        std::cout << cameras[i].set(CAP_PROP_FRAME_WIDTH, 1280)<< std::endl;
        std::cout <<  cameras[i].set(CAP_PROP_FRAME_HEIGHT, 720)<< std::endl;
        std::cout << cameras[i].set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'))<< std::endl;
        std::cout << cameras[i].set(CAP_PROP_FPS, 60) << std::endl;
        //VCM[i].read(forMAt[i]);
    }

    std::vector<InferRequest> vRequest;
    std::vector<int> state(cameras.size(), 0);
    std::vector<int> threadState(NUM_THREAD, 0);// 0 - ready to start, 1 - not ready, 2 - work
    std::vector<int> numCam(NUM_THREAD, -1);
    std::vector<std::string>cam_names;
    std::vector<BlobMap> inpBlobMap(NUM_THREAD);
    std::vector<BlobMap> outBlobMap(NUM_THREAD);
    std::vector<Mat> forImg(NUM_THREAD);
    std::vector<Mat> inpTen, outTen;

    std::string xmlPath;
    std::string binPath;
    if(ADAS)
    {
        xmlPath = "/home/volskig/src/weights/face-detection-adas-0001/FP32/face-detection-adas-0001.xml";
        binPath = "/home/volskig/src/weights/face-detection-adas-0001/FP32/face-detection-adas-0001.bin";
    }
    else
    {
        binPath = "/home/volskig/src/weights/face-detection-retail-0004/FP32/face-detection-retail-0004.bin";
        xmlPath = "/home/volskig/src/weights/face-detection-retail-0004/FP32/face-detection-retail-0004.xml";
    }
    CNNNetReader reader;
    reader.ReadNetwork(xmlPath);
    reader.ReadWeights(binPath);
    CNNNetwork net = reader.getNetwork();

    SizeVector  v_s = {1,3,1000, 1000};
    std::map<std::string, SizeVector> MPP;

    MPP.insert(std::pair<std::string, SizeVector>("data", v_s));

    if(!ADAS)
        net.reshape(MPP);

    InferenceEnginePluginPtr enginePtr;
    InferencePlugin plugin;
    ExecutableNetwork netExec;    
    try
    {
        auto dispatcher = InferenceEngine::PluginDispatcher({""});
        enginePtr = dispatcher.getPluginByDevice("CPU");
        IExtensionPtr extension = make_so_pointer<IExtension>("libcpu_extension.so");
        enginePtr->AddExtension(extension, 0);
        plugin = InferencePlugin(enginePtr);
        netExec = plugin.LoadNetwork(net, {});
    }
    catch (const std::exception& ex)
    {
        CV_Error(Error::StsAssert, format("Failed to initialize Inference Engine backend: %s", ex.what()));
    }   
    for(int nt = 0; nt < NUM_THREAD; ++nt)
    {

        if(ADAS)
            inpTen.emplace_back(Mat({1,3,384,672}, CV_32F));//adas
        else
            inpTen.emplace_back(Mat({1,3,1000,1000}, CV_32F));//retail

        outTen.emplace_back(Mat({1,1,200,7}, CV_32F));

        cam_names.emplace_back("Camera " + std::to_string(nt + 1));        
        if(ADAS)
            inpBlobMap[nt]["data"] = make_shared_blob<float>({Precision::FP32,  {1,3,384,672}, Layout::ANY}, (float*)inpTen[nt].data);//adas
        else
            inpBlobMap[nt]["data"] = make_shared_blob<float>({Precision::FP32,  {1,3,1000,1000}, Layout::ANY}, (float*)inpTen[nt].data);//retail

        outBlobMap[nt]["detection_out"] = make_shared_blob<float>({Precision::FP32, {1,1,200,7}, Layout::ANY}, (float*)outTen[nt].data);

        vRequest.emplace_back(netExec.CreateInferRequest());
        vRequest[nt].SetInput(inpBlobMap[nt]);
        vRequest[nt].SetOutput(outBlobMap[nt]);
        InferenceEngine::IInferRequest::Ptr infRequestPtr = vRequest[nt];
        int* pmsg = &threadState[nt];
        infRequestPtr->SetUserData(pmsg, 0);
        infRequestPtr->SetCompletionCallback(
            [](InferenceEngine::IInferRequest::Ptr reqst, InferenceEngine::StatusCode code)
            {
                int* ptr;
                reqst->GetUserData((void**)&ptr, 0);
                *ptr = REQ_WORK_FIN;                
            });
    }
    int start_frame = 100;
    bool flag = false;
    int NUM_CAM = 0;
    TickMeter tm; int frame_count = 0;
    std::vector<int> next_frame(cameras.size(), 0);
    std::vector<int> thread_frame(NUM_THREAD, 0);
    std::vector<int> launch_frame(cameras.size(), 0);

    std::ofstream out, asyn;
    out.open("read.txt");
    asyn.open("asyn.txt");


    //TickMeter tmM;
    std::vector<TickMeter> tmM(NUM_THREAD);
    std::vector<TickMeter> tmAS(NUM_THREAD);
    if(READ)//.read() method
    {
        while(true)
        {
            if(frame_count > start_frame && !flag)
            {
                tm.start();
                asyn<<"-------------------------------------------------------------"<<std::endl;
                out<<"-------------------------------------------------------------"<<std::endl;
                flag = true;
            }

            for(int nt = 0; nt < NUM_THREAD; ++nt)
            {
                if(threadState[nt] == REQ_WORK_FIN && thread_frame[nt] == next_frame[numCam[nt]] + 1)
                {
                     postprocess(forImg[nt], outTen[nt]);
                     //imshow(cam_names[numCam[nt]], forImg[nt]);
                     //tmAS[nt].stop();
                     //asyn<< "req " << numCam[nt] << " " << tmAS[nt].getTimeMilli() << std::endl;
                     threadState[nt] = REQ_READY_TO_START;
                     ++next_frame[numCam[nt]];
                     ++frame_count;
                }
                if(threadState[nt] == REQ_READY_TO_START)
                {
                    tmM[nt].reset();
                    tmM[nt].start();
                    cameras[NUM_CAM].read(forImg[nt]);
                    tmM[nt].stop();
                    //out<< "cam "<< NUM_CAM << " read " << tmM[nt].getTimeMilli() << " " << nt <<std::endl;
                    if(ADAS)
                        blobFromImage(forImg[nt], inpTen[nt], 1, Size(672, 384));
                    else
                        blobFromImage(forImg[nt], inpTen[nt], 1, Size(1000, 1000));
                    numCam[nt] = NUM_CAM;
                    thread_frame[nt] = ++launch_frame[NUM_CAM];
                    threadState[nt] = REQ_WORK;
                    //tmAS[nt].start();
                    vRequest[nt].StartAsync();
                    break;
                }
            }
            NUM_CAM = (NUM_CAM + 1) % cameras.size();
            if(/*(int)waitKey(1) == 27*/ frame_count >= 140)
            {
                break;
            }
        }
        tm.stop();
        std::cout << (frame_count - start_frame) / tm.getTimeSec() << std::endl;
        std::cout <<"-------------------------------------------------------------------"<< std::endl;
        return 0;
    }
    TickMeter forall;
    if(WA)//waitAny() method
    {
        forall.start();
        VideoCapture::waitAny(cameras, state);
        while(true)
        {
            if(frame_count > start_frame && !flag)
            {
                tm.reset();
                tm.start();
                asyn<<"-------------------------------------------------------------"<<std::endl;
                out<<"-------------------------------------------------------------"<<std::endl;
                flag = true;
            }
            for(int nt = 0; nt < NUM_THREAD; ++nt)
            {
                if(threadState[nt] == REQ_WORK_FIN  && thread_frame[nt] == next_frame[numCam[nt]] + 1)
                {
                    //postprocess(forImg[nt], outTen[nt]);
                    //imshow(cam_names[numCam[nt]], forImg[nt]);
                    //tmAS[nt].stop();
                    //forall.stop();
                    //asyn<< "req " << numCam[nt] << " " << tmAS[nt].getTimeMilli() << " " << nt << " ======"<< forall.getTimeMilli() << std::endl;
                    //forall.start();
                    threadState[nt] = REQ_READY_TO_START;
                    ++next_frame[numCam[nt]];
                    ++frame_count;
                }
                if(/*threadState[nt] == REQ_READY_TO_START*/true)
                {
                    for(unsigned int i = 0; i < state.size(); ++i)
                    {
                        if(state[i] == CAP_CAM_READY)
                        {
                            state[i] = 0;
                            cameras[i].retrieve(forImg[nt]);
                            //tmM[0].stop();
                            //forall.stop();
                            //out<< "cam "<< i << " read " << tmM[0].getTimeMilli() << " " << nt << " ======"<< forall.getTimeMilli()<<std::endl;
                            //forall.start();
                            numCam[nt] = i;
                            thread_frame[nt] = ++launch_frame[i];
                            //if(ADAS)
                            //    blobFromImage(forImg[nt], inpTen[nt], 1, Size(672, 384));
                            //else
                            //    blobFromImage(forImg[nt], inpTen[nt], 1, Size(1000, 1000));
                            //threadState[nt] = REQ_WORK;
                            //tmAS[nt].reset();
                            //tmAS[nt].start();
                            //vRequest[nt].StartAsync();
                            ++frame_count;
                            break;
                        }
                    }
                }
            }
            //tmM[0].reset();
            //tmM[0].start();
            VideoCapture::waitAny(cameras, state);
            if(/*(int)waitKey(1) == 27*/ frame_count >= 140)
            {
                break;
            }
        }
        tm.stop();
        std::cout << (frame_count - start_frame) / tm.getTimeSec() << std::endl;
        std::cout <<"-------------------------------------------------------------------"<< std::endl;
        return 0;
    }

    for(int nt = 0; nt < NUM_THREAD; ++nt)
    {
        vRequest[nt].Wait(InferenceEngine::IInferRequest::RESULT_READY);
    }
    return 0;
}

